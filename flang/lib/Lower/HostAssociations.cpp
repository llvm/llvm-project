//===-- HostAssociations.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/HostAssociations.h"
#include "flang/Evaluate/check-expression.h"
#include "flang/Lower/AbstractConverter.h"
#include "flang/Lower/Allocatable.h"
#include "flang/Lower/BoxAnalyzer.h"
#include "flang/Lower/CallInterface.h"
#include "flang/Lower/ConvertType.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/SymbolMap.h"
#include "flang/Lower/Todo.h"
#include "flang/Optimizer/Builder/Character.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Support/FatalError.h"
#include "flang/Semantics/tools.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "flang-host-assoc"

// Host association inside internal procedures is implemented by allocating an
// mlir tuple (a struct) inside the host containing the addresses and properties
// of variables that are accessed by internal procedures. The address of this
// tuple is passed as an argument by the host when calling internal procedures.
// Internal procedures propagate a reference to this tuple when calling other
// internal procedures of the host.
//
// This file defines how the type of the host tuple is built, how the tuple
// value is created inside the host, and how the host associated variables are
// instantiated inside the internal procedures from the tuple value. The
// CapturedXXX classes define each of these three actions for a specific
// kind of variables by providing a `getType`, a `instantiateHostTuple`, and a
// `getFromTuple` method. These classes are structured as follow:
//
//   class CapturedKindOfVar : public CapturedSymbols<CapturedKindOfVar> {
//     // Return the type of the tuple element for a host associated
//     // variable given its symbol inside the host. This is called when
//     // building function interfaces.
//     static mlir::Type getType();
//     // Build the tuple element value for a host associated variable given its
//     // value inside the host. This is called when lowering the host body.
//     static void instantiateHostTuple();
//     // Instantiate a host variable inside an internal procedure given its
//     // tuple element value. This is called when lowering internal procedure
//     // bodies.
//     static void getFromTuple();
//   };
//
// If a new kind of variable requires ad-hoc handling, a new CapturedXXX class
// should be added to handle it, and `walkCaptureCategories` should be updated
// to dispatch this new kind of variable to this new class.

/// Struct to be used as argument in walkCaptureCategories when building the
/// tuple element type for a host associated variable.
struct GetTypeInTuple {
  /// walkCaptureCategories must return a type.
  using Result = mlir::Type;
};

/// Struct to be used as argument in walkCaptureCategories when building the
/// tuple element value for a host associated variable.
struct InstantiateHostTuple {
  /// walkCaptureCategories returns nothing.
  using Result = void;
  /// Value of the variable inside the host procedure.
  fir::ExtendedValue hostValue;
  /// Address of the tuple element of the variable.
  mlir::Value addrInTuple;
  mlir::Location loc;
};

/// Struct to be used as argument in walkCaptureCategories when instantiating a
/// host associated variables from its tuple element value.
struct GetFromTuple {
  /// walkCaptureCategories returns nothing.
  using Result = void;
  /// Symbol map inside the internal procedure.
  Fortran::lower::SymMap &symMap;
  /// Value of the tuple element for the host associated variable.
  mlir::Value valueInTuple;
  mlir::Location loc;
};

/// Base class that must be inherited with CRTP by classes defining
/// how host association is implemented for a type of symbol.
/// It simply dispatches visit() calls to the implementations according
/// to the argument type.
template <typename SymbolCategory>
class CapturedSymbols {
public:
  template <typename T>
  static void visit(const T &, Fortran::lower::AbstractConverter &,
                    const Fortran::semantics::Symbol &,
                    const Fortran::lower::BoxAnalyzer &) {
    static_assert(!std::is_same_v<T, T> &&
                  "default visit must not be instantiated");
  }
  static mlir::Type visit(const GetTypeInTuple &,
                          Fortran::lower::AbstractConverter &converter,
                          const Fortran::semantics::Symbol &sym,
                          const Fortran::lower::BoxAnalyzer &) {
    return SymbolCategory::getType(converter, sym);
  }
  static void visit(const InstantiateHostTuple &args,
                    Fortran::lower::AbstractConverter &converter,
                    const Fortran::semantics::Symbol &sym,
                    const Fortran::lower::BoxAnalyzer &) {
    return SymbolCategory::instantiateHostTuple(args, converter, sym);
  }
  static void visit(const GetFromTuple &args,
                    Fortran::lower::AbstractConverter &converter,
                    const Fortran::semantics::Symbol &sym,
                    const Fortran::lower::BoxAnalyzer &ba) {
    return SymbolCategory::getFromTuple(args, converter, sym, ba);
  }
};

/// Class defining simple scalars are captured in internal procedures.
/// Simple scalars are non character intrinsic scalars. They are captured
/// as `!fir.ref<T>`, for example `!fir.ref<i32>` for `INTEGER*4`.
class CapturedSimpleScalars : public CapturedSymbols<CapturedSimpleScalars> {
public:
  static mlir::Type getType(Fortran::lower::AbstractConverter &converter,
                            const Fortran::semantics::Symbol &sym) {
    return fir::ReferenceType::get(converter.genType(sym));
  }

  static void instantiateHostTuple(const InstantiateHostTuple &args,
                                   Fortran::lower::AbstractConverter &converter,
                                   const Fortran::semantics::Symbol &) {
    fir::FirOpBuilder &builder = converter.getFirOpBuilder();
    mlir::Type typeInTuple = fir::dyn_cast_ptrEleTy(args.addrInTuple.getType());
    assert(typeInTuple && "addrInTuple must be an address");
    mlir::Value castBox = builder.createConvert(args.loc, typeInTuple,
                                                fir::getBase(args.hostValue));
    builder.create<fir::StoreOp>(args.loc, castBox, args.addrInTuple);
  }

  static void getFromTuple(const GetFromTuple &args,
                           Fortran::lower::AbstractConverter &,
                           const Fortran::semantics::Symbol &sym,
                           const Fortran::lower::BoxAnalyzer &) {
    args.symMap.addSymbol(sym, args.valueInTuple);
  }
};

/// Class defining how dummy procedures and procedure pointers
/// are captured in internal procedures.
class CapturedProcedure : public CapturedSymbols<CapturedProcedure> {
public:
  static mlir::Type getType(Fortran::lower::AbstractConverter &converter,
                            const Fortran::semantics::Symbol &sym) {
    if (Fortran::semantics::IsPointer(sym))
      TODO(converter.getCurrentLocation(),
           "capture procedure pointer in internal procedure");
    return Fortran::lower::getDummyProcedureType(sym, converter);
  }

  static void instantiateHostTuple(const InstantiateHostTuple &args,
                                   Fortran::lower::AbstractConverter &converter,
                                   const Fortran::semantics::Symbol &) {
    fir::FirOpBuilder &builder = converter.getFirOpBuilder();
    mlir::Type typeInTuple = fir::dyn_cast_ptrEleTy(args.addrInTuple.getType());
    assert(typeInTuple && "addrInTuple must be an address");
    mlir::Value castBox = builder.createConvert(args.loc, typeInTuple,
                                                fir::getBase(args.hostValue));
    builder.create<fir::StoreOp>(args.loc, castBox, args.addrInTuple);
  }

  static void getFromTuple(const GetFromTuple &args,
                           Fortran::lower::AbstractConverter &,
                           const Fortran::semantics::Symbol &sym,
                           const Fortran::lower::BoxAnalyzer &) {
    args.symMap.addSymbol(sym, args.valueInTuple);
  }
};

/// Class defining how character scalars are captured in internal procedures.
/// Character scalars are passed as !fir.boxchar<kind> in the tuple.
class CapturedCharacterScalars
    : public CapturedSymbols<CapturedCharacterScalars> {
public:
  // Note: so far, do not specialize constant length characters. They can be
  // implemented by only passing the address. This could be done later in
  // lowering or a CapturedStaticLenCharacterScalars class could be added here.

  static mlir::Type getType(Fortran::lower::AbstractConverter &converter,
                            const Fortran::semantics::Symbol &sym) {
    fir::KindTy kind =
        converter.genType(sym).cast<fir::CharacterType>().getFKind();
    return fir::BoxCharType::get(&converter.getMLIRContext(), kind);
  }

  static void instantiateHostTuple(const InstantiateHostTuple &args,
                                   Fortran::lower::AbstractConverter &converter,
                                   const Fortran::semantics::Symbol &) {
    const fir::CharBoxValue *charBox = args.hostValue.getCharBox();
    assert(charBox && "host value must be a fir::CharBoxValue");
    fir::FirOpBuilder &builder = converter.getFirOpBuilder();
    mlir::Value boxchar = fir::factory::CharacterExprHelper(builder, args.loc)
                              .createEmbox(*charBox);
    builder.create<fir::StoreOp>(args.loc, boxchar, args.addrInTuple);
  }

  static void getFromTuple(const GetFromTuple &args,
                           Fortran::lower::AbstractConverter &converter,
                           const Fortran::semantics::Symbol &sym,
                           const Fortran::lower::BoxAnalyzer &) {
    fir::factory::CharacterExprHelper charHelp(converter.getFirOpBuilder(),
                                               args.loc);
    std::pair<mlir::Value, mlir::Value> unboxchar =
        charHelp.createUnboxChar(args.valueInTuple);
    args.symMap.addCharSymbol(sym, unboxchar.first, unboxchar.second);
  }
};

/// Is \p sym a derived type entity with length parameters ?
static bool
isDerivedWithLengthParameters(const Fortran::semantics::Symbol &sym) {
  if (const auto *declTy = sym.GetType())
    if (const auto *derived = declTy->AsDerived())
      return Fortran::semantics::CountLenParameters(*derived) != 0;
  return false;
}

/// Class defining how allocatable and pointers entities are captured in
/// internal procedures. Allocatable and pointers are simply captured by placing
/// their !fir.ref<fir.box<>> address in the host tuple.
class CapturedAllocatableAndPointer
    : public CapturedSymbols<CapturedAllocatableAndPointer> {
public:
  static mlir::Type getType(Fortran::lower::AbstractConverter &converter,
                            const Fortran::semantics::Symbol &sym) {
    return fir::ReferenceType::get(converter.genType(sym));
  }
  static void instantiateHostTuple(const InstantiateHostTuple &args,
                                   Fortran::lower::AbstractConverter &converter,
                                   const Fortran::semantics::Symbol &) {
    assert(args.hostValue.getBoxOf<fir::MutableBoxValue>() &&
           "host value must be a fir::MutableBoxValue");
    fir::FirOpBuilder &builder = converter.getFirOpBuilder();
    mlir::Type typeInTuple = fir::dyn_cast_ptrEleTy(args.addrInTuple.getType());
    assert(typeInTuple && "addrInTuple must be an address");
    mlir::Value castBox = builder.createConvert(args.loc, typeInTuple,
                                                fir::getBase(args.hostValue));
    builder.create<fir::StoreOp>(args.loc, castBox, args.addrInTuple);
  }
  static void getFromTuple(const GetFromTuple &args,
                           Fortran::lower::AbstractConverter &converter,
                           const Fortran::semantics::Symbol &sym,
                           const Fortran::lower::BoxAnalyzer &ba) {
    fir::FirOpBuilder &builder = converter.getFirOpBuilder();
    mlir::Location loc = args.loc;
    // Non deferred type parameters impact the semantics of some statements
    // where allocatables/pointer can appear. For instance, assignment to a
    // scalar character allocatable with has a different semantics in F2003 and
    // later if the length is non deferred vs when it is deferred. So it is
    // important to keep track of the non deferred parameters here.
    llvm::SmallVector<mlir::Value> nonDeferredLenParams;
    if (ba.isChar()) {
      mlir::IndexType idxTy = builder.getIndexType();
      if (llvm::Optional<int64_t> len = ba.getCharLenConst()) {
        nonDeferredLenParams.push_back(
            builder.createIntegerConstant(loc, idxTy, *len));
      } else if (Fortran::semantics::IsAssumedLengthCharacter(sym) ||
                 ba.getCharLenExpr()) {
        // Read length from fir.box (explicit expr cannot safely be re-evaluated
        // here).
        auto readLength = [&]() {
          fir::BoxValue boxLoad =
              builder.create<fir::LoadOp>(loc, fir::getBase(args.valueInTuple))
                  .getResult();
          return fir::factory::readCharLen(builder, loc, boxLoad);
        };
        if (Fortran::semantics::IsOptional(sym)) {
          // It is not safe to unconditionally read boxes of optionals in case
          // they are absents. According to 15.5.2.12 3 (9), it is illegal to
          // inquire the length of absent optional, even if non deferred, so
          // it's fine to use undefOp in this case.
          auto isPresent = builder.create<fir::IsPresentOp>(
              loc, builder.getI1Type(), fir::getBase(args.valueInTuple));
          mlir::Value len =
              builder.genIfOp(loc, {idxTy}, isPresent, true)
                  .genThen([&]() {
                    builder.create<fir::ResultOp>(loc, readLength());
                  })
                  .genElse([&]() {
                    auto undef = builder.create<fir::UndefOp>(loc, idxTy);
                    builder.create<fir::ResultOp>(loc, undef.getResult());
                  })
                  .getResults()[0];
          nonDeferredLenParams.push_back(len);
        } else {
          nonDeferredLenParams.push_back(readLength());
        }
      }
    } else if (isDerivedWithLengthParameters(sym)) {
      TODO(loc, "host associated derived type allocatable or pointer with "
                "length parameters");
    }
    args.symMap.addSymbol(
        sym, fir::MutableBoxValue(args.valueInTuple, nonDeferredLenParams, {}));
  }
};

/// Class defining how arrays are captured inside internal procedures.
/// Array are captured via a `fir.box<fir.array<T>>` descriptor that belongs to
/// the host tuple. This allows capturing lower bounds, which can be done by
/// providing a ShapeShiftOp argument to the EmboxOp.
class CapturedArrays : public CapturedSymbols<CapturedArrays> {

  // Note: Constant shape arrays are not specialized (their base address would
  // be sufficient information inside the tuple). They could be specialized in
  // a later FIR pass, or a CapturedStaticShapeArrays could be added to deal
  // with them here.
public:
  static mlir::Type getType(Fortran::lower::AbstractConverter &converter,
                            const Fortran::semantics::Symbol &sym) {
    mlir::Type type = converter.genType(sym);
    assert(type.isa<fir::SequenceType>() && "must be a sequence type");
    return fir::BoxType::get(type);
  }

  static void instantiateHostTuple(const InstantiateHostTuple &args,
                                   Fortran::lower::AbstractConverter &converter,
                                   const Fortran::semantics::Symbol &sym) {
    fir::FirOpBuilder &builder = converter.getFirOpBuilder();
    mlir::Location loc = args.loc;
    fir::MutableBoxValue boxInTuple(args.addrInTuple, {}, {});
    if (args.hostValue.getBoxOf<fir::BoxValue>() &&
        Fortran::semantics::IsOptional(sym)) {
      // The assumed shape optional case need some care because it is illegal to
      // read the incoming box if it is absent (this would cause segfaults).
      // Pointer association requires reading the target box, so it can only be
      // done on present optional. For absent optionals, simply create a
      // disassociated pointer (it is illegal to inquire about lower bounds or
      // lengths of optional according to 15.5.2.12 3 (9) and 10.1.11 2 (7)b).
      auto isPresent = builder.create<fir::IsPresentOp>(
          loc, builder.getI1Type(), fir::getBase(args.hostValue));
      builder.genIfThenElse(loc, isPresent)
          .genThen([&]() {
            fir::factory::associateMutableBox(builder, loc, boxInTuple,
                                              args.hostValue,
                                              /*lbounds=*/llvm::None);
          })
          .genElse([&]() {
            fir::factory::disassociateMutableBox(builder, loc, boxInTuple);
          })
          .end();
    } else {
      fir::factory::associateMutableBox(builder, loc, boxInTuple,
                                        args.hostValue, /*lbounds=*/llvm::None);
    }
  }

  static void getFromTuple(const GetFromTuple &args,
                           Fortran::lower::AbstractConverter &converter,
                           const Fortran::semantics::Symbol &sym,
                           const Fortran::lower::BoxAnalyzer &ba) {
    fir::FirOpBuilder &builder = converter.getFirOpBuilder();
    mlir::Location loc = args.loc;
    mlir::Value box = args.valueInTuple;
    mlir::IndexType idxTy = builder.getIndexType();
    llvm::SmallVector<mlir::Value> lbounds;
    if (!ba.lboundIsAllOnes()) {
      if (ba.isStaticArray()) {
        for (std::int64_t lb : ba.staticLBound())
          lbounds.emplace_back(builder.createIntegerConstant(loc, idxTy, lb));
      } else {
        // Cannot re-evaluate specification expressions here.
        // Operands values may have changed. Get value from fir.box
        const unsigned rank = sym.Rank();
        for (unsigned dim = 0; dim < rank; ++dim) {
          mlir::Value dimVal = builder.createIntegerConstant(loc, idxTy, dim);
          auto dims = builder.create<fir::BoxDimsOp>(loc, idxTy, idxTy, idxTy,
                                                     box, dimVal);
          lbounds.emplace_back(dims.getResult(0));
        }
      }
    }

    if (canReadCapturedBoxValue(converter, sym)) {
      fir::BoxValue boxValue(box, lbounds, /*explicitParams=*/llvm::None);
      args.symMap.addSymbol(sym,
                            fir::factory::readBoxValue(builder, loc, boxValue));
    } else {
      // Keep variable as a fir.box.
      // If this is an optional that is absent, the fir.box needs to be an
      // AbsentOp result, otherwise it will not work properly with IsPresentOp
      // (absent boxes are null descriptor addresses, not descriptors containing
      // a null base address).
      if (Fortran::semantics::IsOptional(sym)) {
        auto boxTy = box.getType().cast<fir::BoxType>();
        auto eleTy = boxTy.getEleTy();
        if (!fir::isa_ref_type(eleTy))
          eleTy = builder.getRefType(eleTy);
        auto addr = builder.create<fir::BoxAddrOp>(loc, eleTy, box);
        mlir::Value isPresent = builder.genIsNotNull(loc, addr);
        auto absentBox = builder.create<fir::AbsentOp>(loc, boxTy);
        box = builder.create<mlir::arith::SelectOp>(loc, isPresent, box,
                                                    absentBox);
      }
      fir::BoxValue boxValue(box, lbounds, /*explicitParams=*/llvm::None);
      args.symMap.addSymbol(sym, boxValue);
    }
  }

private:
  /// Can the fir.box from the host link be read into simpler values ?
  /// Later, without the symbol information, it might not be possible
  /// to tell if the fir::BoxValue from the host link is contiguous.
  static bool
  canReadCapturedBoxValue(Fortran::lower::AbstractConverter &converter,
                          const Fortran::semantics::Symbol &sym) {
    bool isScalarOrContiguous =
        sym.Rank() == 0 || Fortran::evaluate::IsSimplyContiguous(
                               Fortran::evaluate::AsGenericExpr(sym).value(),
                               converter.getFoldingContext());
    const Fortran::semantics::DeclTypeSpec *type = sym.GetType();
    bool isPolymorphic = type && type->IsPolymorphic();
    return isScalarOrContiguous && !isPolymorphic &&
           !isDerivedWithLengthParameters(sym);
  }
};

/// Dispatch \p visitor to the CapturedSymbols which is handling how host
/// association is implemented for this kind of symbols. This ensures the same
/// dispatch decision is taken when building the tuple type, when creating the
/// tuple, and when instantiating host associated variables from it.
template <typename T>
typename T::Result
walkCaptureCategories(T visitor, Fortran::lower::AbstractConverter &converter,
                      const Fortran::semantics::Symbol &sym) {
  if (isDerivedWithLengthParameters(sym))
    // Should be boxed.
    TODO(converter.genLocation(sym.name()),
         "host associated derived type with length parameters");
  Fortran::lower::BoxAnalyzer ba;
  // Do not analyze procedures, they may be subroutines with no types that would
  // crash the analysis.
  if (Fortran::semantics::IsProcedure(sym))
    return CapturedProcedure::visit(visitor, converter, sym, ba);
  ba.analyze(sym);
  if (Fortran::evaluate::IsAllocatableOrPointer(sym))
    return CapturedAllocatableAndPointer::visit(visitor, converter, sym, ba);
  if (ba.isArray())
    return CapturedArrays::visit(visitor, converter, sym, ba);
  if (ba.isChar())
    return CapturedCharacterScalars::visit(visitor, converter, sym, ba);
  assert(ba.isTrivial() && "must be trivial scalar");
  return CapturedSimpleScalars::visit(visitor, converter, sym, ba);
}

// `t` should be the result of getArgumentType, which has a type of
// `!fir.ref<tuple<...>>`.
static mlir::TupleType unwrapTupleTy(mlir::Type t) {
  return fir::dyn_cast_ptrEleTy(t).cast<mlir::TupleType>();
}

static mlir::Value genTupleCoor(fir::FirOpBuilder &builder, mlir::Location loc,
                                mlir::Type varTy, mlir::Value tupleArg,
                                mlir::Value offset) {
  // fir.ref<fir.ref> and fir.ptr<fir.ref> are forbidden. Use
  // fir.llvm_ptr if needed.
  auto ty = varTy.isa<fir::ReferenceType>()
                ? mlir::Type(fir::LLVMPointerType::get(varTy))
                : mlir::Type(builder.getRefType(varTy));
  return builder.create<fir::CoordinateOp>(loc, ty, tupleArg, offset);
}

void Fortran::lower::HostAssociations::hostProcedureBindings(
    Fortran::lower::AbstractConverter &converter,
    Fortran::lower::SymMap &symMap) {
  if (symbols.empty())
    return;

  // Create the tuple variable.
  mlir::TupleType tupTy = unwrapTupleTy(getArgumentType(converter));
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::Location loc = converter.getCurrentLocation();
  auto hostTuple = builder.create<fir::AllocaOp>(loc, tupTy);
  mlir::IntegerType offTy = builder.getIntegerType(32);

  // Walk the list of symbols and update the pointers in the tuple.
  for (auto s : llvm::enumerate(symbols)) {
    auto indexInTuple = s.index();
    mlir::Value off = builder.createIntegerConstant(loc, offTy, indexInTuple);
    mlir::Type varTy = tupTy.getType(indexInTuple);
    mlir::Value eleOff = genTupleCoor(builder, loc, varTy, hostTuple, off);
    InstantiateHostTuple instantiateHostTuple{
        symMap.lookupSymbol(s.value()).toExtendedValue(), eleOff, loc};
    walkCaptureCategories(instantiateHostTuple, converter, *s.value());
  }

  converter.bindHostAssocTuple(hostTuple);
}

void Fortran::lower::HostAssociations::internalProcedureBindings(
    Fortran::lower::AbstractConverter &converter,
    Fortran::lower::SymMap &symMap) {
  if (symbols.empty())
    return;

  // Find the argument with the tuple type. The argument ought to be appended.
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::Type argTy = getArgumentType(converter);
  mlir::TupleType tupTy = unwrapTupleTy(argTy);
  mlir::Location loc = converter.getCurrentLocation();
  mlir::func::FuncOp func = builder.getFunction();
  mlir::Value tupleArg;
  for (auto [ty, arg] : llvm::reverse(llvm::zip(
           func.getFunctionType().getInputs(), func.front().getArguments())))
    if (ty == argTy) {
      tupleArg = arg;
      break;
    }
  if (!tupleArg)
    fir::emitFatalError(loc, "no host association argument found");

  converter.bindHostAssocTuple(tupleArg);

  mlir::IntegerType offTy = builder.getIntegerType(32);

  // Walk the list and add the bindings to the symbol table.
  for (auto s : llvm::enumerate(symbols)) {
    mlir::Value off = builder.createIntegerConstant(loc, offTy, s.index());
    mlir::Type varTy = tupTy.getType(s.index());
    mlir::Value eleOff = genTupleCoor(builder, loc, varTy, tupleArg, off);
    mlir::Value valueInTuple = builder.create<fir::LoadOp>(loc, eleOff);
    GetFromTuple getFromTuple{symMap, valueInTuple, loc};
    walkCaptureCategories(getFromTuple, converter, *s.value());
  }
}

mlir::Type Fortran::lower::HostAssociations::getArgumentType(
    Fortran::lower::AbstractConverter &converter) {
  if (symbols.empty())
    return {};
  if (argType)
    return argType;

  // Walk the list of Symbols and create their types. Wrap them in a reference
  // to a tuple.
  mlir::MLIRContext *ctxt = &converter.getMLIRContext();
  llvm::SmallVector<mlir::Type> tupleTys;
  for (const Fortran::semantics::Symbol *sym : symbols)
    tupleTys.emplace_back(
        walkCaptureCategories(GetTypeInTuple{}, converter, *sym));
  argType = fir::ReferenceType::get(mlir::TupleType::get(ctxt, tupleTys));
  return argType;
}
