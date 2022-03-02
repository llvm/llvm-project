//===-- ConvertVariable.cpp -- bridge to lower to MLIR --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/ConvertVariable.h"
#include "flang/Lower/AbstractConverter.h"
#include "flang/Lower/Allocatable.h"
#include "flang/Lower/BoxAnalyzer.h"
#include "flang/Lower/CallInterface.h"
#include "flang/Lower/ConvertExpr.h"
#include "flang/Lower/Mangler.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/StatementContext.h"
#include "flang/Lower/Support/Utils.h"
#include "flang/Lower/SymbolMap.h"
#include "flang/Lower/Todo.h"
#include "flang/Optimizer/Builder/Character.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/Derived.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Support/FIRContext.h"
#include "flang/Optimizer/Support/FatalError.h"
#include "flang/Semantics/tools.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "flang-lower-variable"

/// Helper to retrieve a copy of a character literal string from a SomeExpr.
/// Required to build character global initializers.
template <int KIND>
static llvm::Optional<std::tuple<std::string, std::size_t>>
getCharacterLiteralCopy(
    const Fortran::evaluate::Expr<
        Fortran::evaluate::Type<Fortran::common::TypeCategory::Character, KIND>>
        &x) {
  if (const auto *con =
          Fortran::evaluate::UnwrapConstantValue<Fortran::evaluate::Type<
              Fortran::common::TypeCategory::Character, KIND>>(x))
    if (auto val = con->GetScalarValue())
      return std::tuple<std::string, std::size_t>{
          std::string{(const char *)val->c_str(),
                      KIND * (std::size_t)con->LEN()},
          (std::size_t)con->LEN()};
  return llvm::None;
}
static llvm::Optional<std::tuple<std::string, std::size_t>>
getCharacterLiteralCopy(
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeCharacter> &x) {
  return std::visit([](const auto &e) { return getCharacterLiteralCopy(e); },
                    x.u);
}
static llvm::Optional<std::tuple<std::string, std::size_t>>
getCharacterLiteralCopy(const Fortran::lower::SomeExpr &x) {
  if (const auto *e = Fortran::evaluate::UnwrapExpr<
          Fortran::evaluate::Expr<Fortran::evaluate::SomeCharacter>>(x))
    return getCharacterLiteralCopy(*e);
  return llvm::None;
}
template <typename A>
static llvm::Optional<std::tuple<std::string, std::size_t>>
getCharacterLiteralCopy(const std::optional<A> &x) {
  if (x)
    return getCharacterLiteralCopy(*x);
  return llvm::None;
}

/// Helper to lower a scalar expression using a specific symbol mapping.
static mlir::Value genScalarValue(Fortran::lower::AbstractConverter &converter,
                                  mlir::Location loc,
                                  const Fortran::lower::SomeExpr &expr,
                                  Fortran::lower::SymMap &symMap,
                                  Fortran::lower::StatementContext &context) {
  // This does not use the AbstractConverter member function to override the
  // symbol mapping to be used expression lowering.
  return fir::getBase(Fortran::lower::createSomeExtendedExpression(
      loc, converter, expr, symMap, context));
}

/// Does this variable have a default initialization?
static bool hasDefaultInitialization(const Fortran::semantics::Symbol &sym) {
  if (sym.has<Fortran::semantics::ObjectEntityDetails>() && sym.size())
    if (!Fortran::semantics::IsAllocatableOrPointer(sym))
      if (const Fortran::semantics::DeclTypeSpec *declTypeSpec = sym.GetType())
        if (const Fortran::semantics::DerivedTypeSpec *derivedTypeSpec =
                declTypeSpec->AsDerived())
          return derivedTypeSpec->HasDefaultInitialization();
  return false;
}

//===----------------------------------------------------------------===//
// Global variables instantiation (not for alias and common)
//===----------------------------------------------------------------===//

/// Helper to generate expression value inside global initializer.
static fir::ExtendedValue
genInitializerExprValue(Fortran::lower::AbstractConverter &converter,
                        mlir::Location loc,
                        const Fortran::lower::SomeExpr &expr,
                        Fortran::lower::StatementContext &stmtCtx) {
  // Data initializer are constant value and should not depend on other symbols
  // given the front-end fold parameter references. In any case, the "current"
  // map of the converter should not be used since it holds mapping to
  // mlir::Value from another mlir region. If these value are used by accident
  // in the initializer, this will lead to segfaults in mlir code.
  Fortran::lower::SymMap emptyMap;
  return Fortran::lower::createSomeInitializerExpression(loc, converter, expr,
                                                         emptyMap, stmtCtx);
}

/// Can this symbol constant be placed in read-only memory?
static bool isConstant(const Fortran::semantics::Symbol &sym) {
  return sym.attrs().test(Fortran::semantics::Attr::PARAMETER) ||
         sym.test(Fortran::semantics::Symbol::Flag::ReadOnly);
}

/// Create the global op declaration without any initializer
static fir::GlobalOp declareGlobal(Fortran::lower::AbstractConverter &converter,
                                   const Fortran::lower::pft::Variable &var,
                                   llvm::StringRef globalName,
                                   mlir::StringAttr linkage) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  if (fir::GlobalOp global = builder.getNamedGlobal(globalName))
    return global;
  const Fortran::semantics::Symbol &sym = var.getSymbol();
  mlir::Location loc = converter.genLocation(sym.name());
  // Resolve potential host and module association before checking that this
  // symbol is an object of a function pointer.
  const Fortran::semantics::Symbol &ultimate = sym.GetUltimate();
  if (!ultimate.has<Fortran::semantics::ObjectEntityDetails>() &&
      !ultimate.has<Fortran::semantics::ProcEntityDetails>())
    mlir::emitError(loc, "lowering global declaration: symbol '")
        << toStringRef(sym.name()) << "' has unexpected details\n";
  return builder.createGlobal(loc, converter.genType(var), globalName, linkage,
                              mlir::Attribute{}, isConstant(ultimate));
}

/// Temporary helper to catch todos in initial data target lowering.
static bool
hasDerivedTypeWithLengthParameters(const Fortran::semantics::Symbol &sym) {
  if (const Fortran::semantics::DeclTypeSpec *declTy = sym.GetType())
    if (const Fortran::semantics::DerivedTypeSpec *derived =
            declTy->AsDerived())
      return Fortran::semantics::CountLenParameters(*derived) > 0;
  return false;
}

static mlir::Type unwrapElementType(mlir::Type type) {
  if (mlir::Type ty = fir::dyn_cast_ptrOrBoxEleTy(type))
    type = ty;
  if (auto seqType = type.dyn_cast<fir::SequenceType>())
    type = seqType.getEleTy();
  return type;
}

/// create initial-data-target fir.box in a global initializer region.
mlir::Value Fortran::lower::genInitialDataTarget(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    mlir::Type boxType, const Fortran::lower::SomeExpr &initialTarget) {
  Fortran::lower::SymMap globalOpSymMap;
  Fortran::lower::AggregateStoreMap storeMap;
  Fortran::lower::StatementContext stmtCtx;
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  if (Fortran::evaluate::UnwrapExpr<Fortran::evaluate::NullPointer>(
          initialTarget))
    return fir::factory::createUnallocatedBox(builder, loc, boxType,
                                              /*nonDeferredParams=*/llvm::None);
  // Pointer initial data target, and NULL(mold).
  if (const Fortran::semantics::Symbol *sym =
          Fortran::evaluate::GetFirstSymbol(initialTarget)) {
    // Length parameters processing will need care in global initializer
    // context.
    if (hasDerivedTypeWithLengthParameters(*sym))
      TODO(loc, "initial-data-target with derived type length parameters");

    auto var = Fortran::lower::pft::Variable(*sym, /*global=*/true);
    Fortran::lower::instantiateVariable(converter, var, globalOpSymMap,
                                        storeMap);
  }
  mlir::Value box;
  if (initialTarget.Rank() > 0) {
    box = fir::getBase(Fortran::lower::createSomeArrayBox(
        converter, initialTarget, globalOpSymMap, stmtCtx));
  } else {
    fir::ExtendedValue addr = Fortran::lower::createInitializerAddress(
        loc, converter, initialTarget, globalOpSymMap, stmtCtx);
    box = builder.createBox(loc, addr);
  }
  // box is a fir.box<T>, not a fir.box<fir.ptr<T>> as it should to be used
  // for pointers. A fir.convert should not be used here, because it would
  // not actually set the pointer attribute in the descriptor.
  // In a normal context, fir.rebox would be used to set the pointer attribute
  // while copying the projection from another fir.box. But fir.rebox cannot be
  // used in initializer because its current codegen expects that the input
  // fir.box is in memory, which is not the case in initializers.
  // So, just replace the fir.embox that created addr with one with
  // fir.box<fir.ptr<T>> result type.
  // Note that the descriptor cannot have been created with fir.rebox because
  // the initial-data-target cannot be a fir.box itself (it cannot be
  // assumed-shape, deferred-shape, or polymorphic as per C765). However the
  // case where the initial data target is a derived type with length parameters
  // will most likely be a bit trickier, hence the TODO above.

  mlir::Operation *op = box.getDefiningOp();
  if (!op || !mlir::isa<fir::EmboxOp>(*op))
    fir::emitFatalError(
        loc, "fir.box must be created with embox in global initializers");
  mlir::Type targetEleTy = unwrapElementType(box.getType());
  if (!fir::isa_char(targetEleTy))
    return builder.create<fir::EmboxOp>(loc, boxType, op->getOperands(),
                                        op->getAttrs());

  // Handle the character case length particularities: embox takes a length
  // value argument when the result type has unknown length, but not when the
  // result type has constant length. The type of the initial target must be
  // constant length, but the one of the pointer may not be. In this case, a
  // length operand must be added.
  auto targetLen = targetEleTy.cast<fir::CharacterType>().getLen();
  auto ptrLen = unwrapElementType(boxType).cast<fir::CharacterType>().getLen();
  if (ptrLen == targetLen)
    // Nothing to do
    return builder.create<fir::EmboxOp>(loc, boxType, op->getOperands(),
                                        op->getAttrs());
  auto embox = mlir::cast<fir::EmboxOp>(*op);
  auto ptrType = boxType.cast<fir::BoxType>().getEleTy();
  mlir::Value memref = builder.createConvert(loc, ptrType, embox.getMemref());
  if (targetLen == fir::CharacterType::unknownLen())
    // Drop the length argument.
    return builder.create<fir::EmboxOp>(loc, boxType, memref, embox.getShape(),
                                        embox.getSlice());
  // targetLen is constant and ptrLen is unknown. Add a length argument.
  mlir::Value targetLenValue =
      builder.createIntegerConstant(loc, builder.getIndexType(), targetLen);
  return builder.create<fir::EmboxOp>(loc, boxType, memref, embox.getShape(),
                                      embox.getSlice(),
                                      mlir::ValueRange{targetLenValue});
}

static mlir::Value genDefaultInitializerValue(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    const Fortran::semantics::Symbol &sym, mlir::Type symTy,
    Fortran::lower::StatementContext &stmtCtx) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::Type scalarType = symTy;
  fir::SequenceType sequenceType;
  if (auto ty = symTy.dyn_cast<fir::SequenceType>()) {
    sequenceType = ty;
    scalarType = ty.getEleTy();
  }
  // Build a scalar default value of the symbol type, looping through the
  // components to build each component initial value.
  auto recTy = scalarType.cast<fir::RecordType>();
  auto fieldTy = fir::FieldType::get(scalarType.getContext());
  mlir::Value initialValue = builder.create<fir::UndefOp>(loc, scalarType);
  const Fortran::semantics::DeclTypeSpec *declTy = sym.GetType();
  assert(declTy && "var with default initialization must have a type");
  Fortran::semantics::OrderedComponentIterator components(
      declTy->derivedTypeSpec());
  for (const auto &component : components) {
    // Skip parent components, the sub-components of parent types are part of
    // components and will be looped through right after.
    if (component.test(Fortran::semantics::Symbol::Flag::ParentComp))
      continue;
    mlir::Value componentValue;
    llvm::StringRef name = toStringRef(component.name());
    mlir::Type componentTy = recTy.getType(name);
    assert(componentTy && "component not found in type");
    if (const auto *object{
            component.detailsIf<Fortran::semantics::ObjectEntityDetails>()}) {
      if (const auto &init = object->init()) {
        // Component has explicit initialization.
        if (Fortran::semantics::IsPointer(component))
          // Initial data target.
          componentValue =
              genInitialDataTarget(converter, loc, componentTy, *init);
        else
          // Initial value.
          componentValue = fir::getBase(
              genInitializerExprValue(converter, loc, *init, stmtCtx));
      } else if (Fortran::semantics::IsAllocatableOrPointer(component)) {
        // Pointer or allocatable without initialization.
        // Create deallocated/disassociated value.
        // From a standard point of view, pointer without initialization do not
        // need to be disassociated, but for sanity and simplicity, do it in
        // global constructor since this has no runtime cost.
        componentValue = fir::factory::createUnallocatedBox(
            builder, loc, componentTy, llvm::None);
      } else if (hasDefaultInitialization(component)) {
        // Component type has default initialization.
        componentValue = genDefaultInitializerValue(converter, loc, component,
                                                    componentTy, stmtCtx);
      } else {
        // Component has no initial value.
        componentValue = builder.create<fir::UndefOp>(loc, componentTy);
      }
    } else if (const auto *proc{
                   component
                       .detailsIf<Fortran::semantics::ProcEntityDetails>()}) {
      if (proc->init().has_value())
        TODO(loc, "procedure pointer component default initialization");
      else
        componentValue = builder.create<fir::UndefOp>(loc, componentTy);
    }
    assert(componentValue && "must have been computed");
    componentValue = builder.createConvert(loc, componentTy, componentValue);
    // FIXME: type parameters must come from the derived-type-spec
    auto field = builder.create<fir::FieldIndexOp>(
        loc, fieldTy, name, scalarType,
        /*typeParams=*/mlir::ValueRange{} /*TODO*/);
    initialValue = builder.create<fir::InsertValueOp>(
        loc, recTy, initialValue, componentValue,
        builder.getArrayAttr(field.getAttributes()));
  }

  if (sequenceType) {
    // For arrays, duplicate the scalar value to all elements with an
    // fir.insert_range covering the whole array.
    auto arrayInitialValue = builder.create<fir::UndefOp>(loc, sequenceType);
    llvm::SmallVector<int64_t> rangeBounds;
    for (int64_t extent : sequenceType.getShape()) {
      if (extent == fir::SequenceType::getUnknownExtent())
        TODO(loc,
             "default initial value of array component with length parameters");
      rangeBounds.push_back(0);
      rangeBounds.push_back(extent - 1);
    }
    return builder.create<fir::InsertOnRangeOp>(
        loc, sequenceType, arrayInitialValue, initialValue,
        builder.getIndexVectorAttr(rangeBounds));
  }
  return initialValue;
}

/// Does this global already have an initializer ?
static bool globalIsInitialized(fir::GlobalOp global) {
  return !global.getRegion().empty() || global.getInitVal();
}

/// Call \p genInit to generate code inside \p global initializer region.
static void
createGlobalInitialization(fir::FirOpBuilder &builder, fir::GlobalOp global,
                           std::function<void(fir::FirOpBuilder &)> genInit) {
  mlir::Region &region = global.getRegion();
  region.push_back(new mlir::Block);
  mlir::Block &block = region.back();
  auto insertPt = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(&block);
  genInit(builder);
  builder.restoreInsertionPoint(insertPt);
}

/// Create the global op and its init if it has one
static fir::GlobalOp defineGlobal(Fortran::lower::AbstractConverter &converter,
                                  const Fortran::lower::pft::Variable &var,
                                  llvm::StringRef globalName,
                                  mlir::StringAttr linkage) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  const Fortran::semantics::Symbol &sym = var.getSymbol();
  mlir::Location loc = converter.genLocation(sym.name());
  bool isConst = isConstant(sym);
  fir::GlobalOp global = builder.getNamedGlobal(globalName);
  mlir::Type symTy = converter.genType(var);

  if (global && globalIsInitialized(global))
    return global;
  // If this is an array, check to see if we can use a dense attribute
  // with a tensor mlir type.  This optimization currently only supports
  // rank-1 Fortran arrays of integer, real, or logical. The tensor
  // type does not support nested structures which are needed for
  // complex numbers.
  // To get multidimensional arrays to work, we will have to use column major
  // array ordering with the tensor type (so it matches column major ordering
  // with the Fortran fir.array).  By default, tensor types assume row major
  // ordering. How to create this tensor type is to be determined.
  if (symTy.isa<fir::SequenceType>() && sym.Rank() == 1 &&
      !Fortran::semantics::IsAllocatableOrPointer(sym)) {
    mlir::Type eleTy = symTy.cast<fir::SequenceType>().getEleTy();
    if (eleTy.isa<mlir::IntegerType, mlir::FloatType, fir::LogicalType>()) {
      const auto *details =
          sym.detailsIf<Fortran::semantics::ObjectEntityDetails>();
      if (details->init()) {
        global = Fortran::lower::createDenseGlobal(
            loc, symTy, globalName, linkage, isConst, details->init().value(),
            converter);
        if (global) {
          global.setVisibility(mlir::SymbolTable::Visibility::Public);
          return global;
        }
      }
    }
  }
  if (!global)
    global = builder.createGlobal(loc, symTy, globalName, linkage,
                                  mlir::Attribute{}, isConst);
  if (Fortran::semantics::IsAllocatableOrPointer(sym)) {
    const auto *details =
        sym.detailsIf<Fortran::semantics::ObjectEntityDetails>();
    if (details && details->init()) {
      auto expr = *details->init();
      createGlobalInitialization(builder, global, [&](fir::FirOpBuilder &b) {
        mlir::Value box =
            Fortran::lower::genInitialDataTarget(converter, loc, symTy, expr);
        b.create<fir::HasValueOp>(loc, box);
      });
    } else {
      // Create unallocated/disassociated descriptor if no explicit init
      createGlobalInitialization(builder, global, [&](fir::FirOpBuilder &b) {
        mlir::Value box =
            fir::factory::createUnallocatedBox(b, loc, symTy, llvm::None);
        b.create<fir::HasValueOp>(loc, box);
      });
    }

  } else if (const auto *details =
                 sym.detailsIf<Fortran::semantics::ObjectEntityDetails>()) {
    if (details->init()) {
      if (fir::isa_char(symTy)) {
        // CHARACTER literal
        if (auto chLit = getCharacterLiteralCopy(details->init().value())) {
          mlir::StringAttr init =
              builder.getStringAttr(std::get<std::string>(*chLit));
          global->setAttr(global.getInitValAttrName(), init);
        } else {
          fir::emitFatalError(loc, "CHARACTER has unexpected initial value");
        }
      } else {
        createGlobalInitialization(
            builder, global, [&](fir::FirOpBuilder &builder) {
              Fortran::lower::StatementContext stmtCtx(
                  /*cleanupProhibited=*/true);
              fir::ExtendedValue initVal = genInitializerExprValue(
                  converter, loc, details->init().value(), stmtCtx);
              mlir::Value castTo =
                  builder.createConvert(loc, symTy, fir::getBase(initVal));
              builder.create<fir::HasValueOp>(loc, castTo);
            });
      }
    } else if (hasDefaultInitialization(sym)) {
      createGlobalInitialization(
          builder, global, [&](fir::FirOpBuilder &builder) {
            Fortran::lower::StatementContext stmtCtx(
                /*cleanupProhibited=*/true);
            mlir::Value initVal =
                genDefaultInitializerValue(converter, loc, sym, symTy, stmtCtx);
            mlir::Value castTo = builder.createConvert(loc, symTy, initVal);
            builder.create<fir::HasValueOp>(loc, castTo);
          });
    }
  } else if (sym.has<Fortran::semantics::CommonBlockDetails>()) {
    mlir::emitError(loc, "COMMON symbol processed elsewhere");
  } else {
    TODO(loc, "global"); // Procedure pointer or something else
  }
  // Creates undefined initializer for globals without initializers
  if (!globalIsInitialized(global))
    createGlobalInitialization(
        builder, global, [&](fir::FirOpBuilder &builder) {
          builder.create<fir::HasValueOp>(
              loc, builder.create<fir::UndefOp>(loc, symTy));
        });
  // Set public visibility to prevent global definition to be optimized out
  // even if they have no initializer and are unused in this compilation unit.
  global.setVisibility(mlir::SymbolTable::Visibility::Public);
  return global;
}

/// Return linkage attribute for \p var.
static mlir::StringAttr
getLinkageAttribute(fir::FirOpBuilder &builder,
                    const Fortran::lower::pft::Variable &var) {
  if (var.isModuleVariable())
    return {}; // external linkage
  // Otherwise, the variable is owned by a procedure and must not be visible in
  // other compilation units.
  return builder.createInternalLinkage();
}

/// Instantiate a global variable. If it hasn't already been processed, add
/// the global to the ModuleOp as a new uniqued symbol and initialize it with
/// the correct value. It will be referenced on demand using `fir.addr_of`.
static void instantiateGlobal(Fortran::lower::AbstractConverter &converter,
                              const Fortran::lower::pft::Variable &var,
                              Fortran::lower::SymMap &symMap) {
  const Fortran::semantics::Symbol &sym = var.getSymbol();
  assert(!var.isAlias() && "must be handled in instantiateAlias");
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  std::string globalName = Fortran::lower::mangle::mangleName(sym);
  mlir::Location loc = converter.genLocation(sym.name());
  fir::GlobalOp global = builder.getNamedGlobal(globalName);
  mlir::StringAttr linkage = getLinkageAttribute(builder, var);
  if (var.isModuleVariable()) {
    // A module global was or will be defined when lowering the module. Emit
    // only a declaration if the global does not exist at that point.
    global = declareGlobal(converter, var, globalName, linkage);
  } else {
    global = defineGlobal(converter, var, globalName, linkage);
  }
  auto addrOf = builder.create<fir::AddrOfOp>(loc, global.resultType(),
                                              global.getSymbol());
  Fortran::lower::StatementContext stmtCtx;
  mapSymbolAttributes(converter, var, symMap, stmtCtx, addrOf);
}

//===----------------------------------------------------------------===//
// Local variables instantiation (not for alias)
//===----------------------------------------------------------------===//

/// Create a stack slot for a local variable. Precondition: the insertion
/// point of the builder must be in the entry block, which is currently being
/// constructed.
static mlir::Value createNewLocal(Fortran::lower::AbstractConverter &converter,
                                  mlir::Location loc,
                                  const Fortran::lower::pft::Variable &var,
                                  mlir::Value preAlloc,
                                  llvm::ArrayRef<mlir::Value> shape = {},
                                  llvm::ArrayRef<mlir::Value> lenParams = {}) {
  if (preAlloc)
    return preAlloc;
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  std::string nm = Fortran::lower::mangle::mangleName(var.getSymbol());
  mlir::Type ty = converter.genType(var);
  const Fortran::semantics::Symbol &ultimateSymbol =
      var.getSymbol().GetUltimate();
  llvm::StringRef symNm = toStringRef(ultimateSymbol.name());
  bool isTarg = var.isTarget();
  // Let the builder do all the heavy lifting.
  return builder.allocateLocal(loc, ty, nm, symNm, shape, lenParams, isTarg);
}

/// Instantiate a local variable. Precondition: Each variable will be visited
/// such that if its properties depend on other variables, the variables upon
/// which its properties depend will already have been visited.
static void instantiateLocal(Fortran::lower::AbstractConverter &converter,
                             const Fortran::lower::pft::Variable &var,
                             Fortran::lower::SymMap &symMap) {
  assert(!var.isAlias());
  Fortran::lower::StatementContext stmtCtx;
  mapSymbolAttributes(converter, var, symMap, stmtCtx);
}

/// Helper to decide if a dummy argument must be tracked in an BoxValue.
static bool lowerToBoxValue(const Fortran::semantics::Symbol &sym,
                            mlir::Value dummyArg) {
  // Only dummy arguments coming as fir.box can be tracked in an BoxValue.
  if (!dummyArg || !dummyArg.getType().isa<fir::BoxType>())
    return false;
  // Non contiguous arrays must be tracked in an BoxValue.
  if (sym.Rank() > 0 && !sym.attrs().test(Fortran::semantics::Attr::CONTIGUOUS))
    return true;
  // Assumed rank and optional fir.box cannot yet be read while lowering the
  // specifications.
  if (Fortran::evaluate::IsAssumedRank(sym) ||
      Fortran::semantics::IsOptional(sym))
    return true;
  // Polymorphic entity should be tracked through a fir.box that has the
  // dynamic type info.
  if (const Fortran::semantics::DeclTypeSpec *type = sym.GetType())
    if (type->IsPolymorphic())
      return true;
  return false;
}

/// Compute extent from lower and upper bound.
static mlir::Value computeExtent(fir::FirOpBuilder &builder, mlir::Location loc,
                                 mlir::Value lb, mlir::Value ub) {
  mlir::IndexType idxTy = builder.getIndexType();
  // Let the folder deal with the common `ub - <const> + 1` case.
  auto diff = builder.create<mlir::arith::SubIOp>(loc, idxTy, ub, lb);
  mlir::Value one = builder.createIntegerConstant(loc, idxTy, 1);
  return builder.create<mlir::arith::AddIOp>(loc, idxTy, diff, one);
}

/// Lower explicit lower bounds into \p result. Does nothing if this is not an
/// array, or if the lower bounds are deferred, or all implicit or one.
static void lowerExplicitLowerBounds(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    const Fortran::lower::BoxAnalyzer &box,
    llvm::SmallVectorImpl<mlir::Value> &result, Fortran::lower::SymMap &symMap,
    Fortran::lower::StatementContext &stmtCtx) {
  if (!box.isArray() || box.lboundIsAllOnes())
    return;
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::IndexType idxTy = builder.getIndexType();
  if (box.isStaticArray()) {
    for (int64_t lb : box.staticLBound())
      result.emplace_back(builder.createIntegerConstant(loc, idxTy, lb));
    return;
  }
  for (const Fortran::semantics::ShapeSpec *spec : box.dynamicBound()) {
    if (auto low = spec->lbound().GetExplicit()) {
      auto expr = Fortran::lower::SomeExpr{*low};
      mlir::Value lb = builder.createConvert(
          loc, idxTy, genScalarValue(converter, loc, expr, symMap, stmtCtx));
      result.emplace_back(lb);
    } else if (!spec->lbound().isColon()) {
      // Implicit lower bound is 1 (Fortran 2018 section 8.5.8.3 point 3.)
      result.emplace_back(builder.createIntegerConstant(loc, idxTy, 1));
    }
  }
  assert(result.empty() || result.size() == box.dynamicBound().size());
}

/// Lower explicit extents into \p result if this is an explicit-shape or
/// assumed-size array. Does nothing if this is not an explicit-shape or
/// assumed-size array.
static void lowerExplicitExtents(Fortran::lower::AbstractConverter &converter,
                                 mlir::Location loc,
                                 const Fortran::lower::BoxAnalyzer &box,
                                 llvm::ArrayRef<mlir::Value> lowerBounds,
                                 llvm::SmallVectorImpl<mlir::Value> &result,
                                 Fortran::lower::SymMap &symMap,
                                 Fortran::lower::StatementContext &stmtCtx) {
  if (!box.isArray())
    return;
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::IndexType idxTy = builder.getIndexType();
  if (box.isStaticArray()) {
    for (int64_t extent : box.staticShape())
      result.emplace_back(builder.createIntegerConstant(loc, idxTy, extent));
    return;
  }
  for (const auto &spec : llvm::enumerate(box.dynamicBound())) {
    if (auto up = spec.value()->ubound().GetExplicit()) {
      auto expr = Fortran::lower::SomeExpr{*up};
      mlir::Value ub = builder.createConvert(
          loc, idxTy, genScalarValue(converter, loc, expr, symMap, stmtCtx));
      if (lowerBounds.empty())
        result.emplace_back(ub);
      else
        result.emplace_back(
            computeExtent(builder, loc, lowerBounds[spec.index()], ub));
    } else if (spec.value()->ubound().isStar()) {
      // Assumed extent is undefined. Must be provided by user's code.
      result.emplace_back(builder.create<fir::UndefOp>(loc, idxTy));
    }
  }
  assert(result.empty() || result.size() == box.dynamicBound().size());
}

/// Lower explicit character length if any. Return empty mlir::Value if no
/// explicit length.
static mlir::Value
lowerExplicitCharLen(Fortran::lower::AbstractConverter &converter,
                     mlir::Location loc, const Fortran::lower::BoxAnalyzer &box,
                     Fortran::lower::SymMap &symMap,
                     Fortran::lower::StatementContext &stmtCtx) {
  if (!box.isChar())
    return mlir::Value{};
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::Type lenTy = builder.getCharacterLengthType();
  if (llvm::Optional<int64_t> len = box.getCharLenConst())
    return builder.createIntegerConstant(loc, lenTy, *len);
  if (llvm::Optional<Fortran::lower::SomeExpr> lenExpr = box.getCharLenExpr())
    return genScalarValue(converter, loc, *lenExpr, symMap, stmtCtx);
  return mlir::Value{};
}

/// Treat negative values as undefined. Assumed size arrays will return -1 from
/// the front end for example. Using negative values can produce hard to find
/// bugs much further along in the compilation.
static mlir::Value genExtentValue(fir::FirOpBuilder &builder,
                                  mlir::Location loc, mlir::Type idxTy,
                                  long frontEndExtent) {
  if (frontEndExtent >= 0)
    return builder.createIntegerConstant(loc, idxTy, frontEndExtent);
  return builder.create<fir::UndefOp>(loc, idxTy);
}

/// Lower specification expressions and attributes of variable \p var and
/// add it to the symbol map.
/// For global and aliases, the address must be pre-computed and provided
/// in \p preAlloc.
/// Dummy arguments must have already been mapped to mlir block arguments
/// their mapping may be updated here.
void Fortran::lower::mapSymbolAttributes(
    AbstractConverter &converter, const Fortran::lower::pft::Variable &var,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx,
    mlir::Value preAlloc) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  const Fortran::semantics::Symbol &sym = var.getSymbol();
  const mlir::Location loc = converter.genLocation(sym.name());
  mlir::IndexType idxTy = builder.getIndexType();
  const bool isDummy = Fortran::semantics::IsDummy(sym);
  const bool isResult = Fortran::semantics::IsFunctionResult(sym);
  const bool replace = isDummy || isResult;
  fir::factory::CharacterExprHelper charHelp{builder, loc};
  Fortran::lower::BoxAnalyzer ba;
  ba.analyze(sym);

  // First deal with pointers an allocatables, because their handling here
  // is the same regardless of their rank.
  if (Fortran::semantics::IsAllocatableOrPointer(sym)) {
    // Get address of fir.box describing the entity.
    // global
    mlir::Value boxAlloc = preAlloc;
    // dummy or passed result
    if (!boxAlloc)
      if (Fortran::lower::SymbolBox symbox = symMap.lookupSymbol(sym))
        boxAlloc = symbox.getAddr();
    // local
    if (!boxAlloc)
      boxAlloc = createNewLocal(converter, loc, var, preAlloc);
    // Lower non deferred parameters.
    llvm::SmallVector<mlir::Value> nonDeferredLenParams;
    if (ba.isChar()) {
      if (mlir::Value len =
              lowerExplicitCharLen(converter, loc, ba, symMap, stmtCtx))
        nonDeferredLenParams.push_back(len);
      else if (Fortran::semantics::IsAssumedLengthCharacter(sym))
        TODO(loc, "assumed length character allocatable");
    } else if (const Fortran::semantics::DeclTypeSpec *declTy = sym.GetType()) {
      if (const Fortran::semantics::DerivedTypeSpec *derived =
              declTy->AsDerived())
        if (Fortran::semantics::CountLenParameters(*derived) != 0)
          TODO(loc,
               "derived type allocatable or pointer with length parameters");
    }
    fir::MutableBoxValue box = Fortran::lower::createMutableBox(
        converter, loc, var, boxAlloc, nonDeferredLenParams);
    symMap.addAllocatableOrPointer(var.getSymbol(), box, replace);
    return;
  }

  if (isDummy) {
    mlir::Value dummyArg = symMap.lookupSymbol(sym).getAddr();
    if (lowerToBoxValue(sym, dummyArg)) {
      llvm::SmallVector<mlir::Value> lbounds;
      llvm::SmallVector<mlir::Value> extents;
      llvm::SmallVector<mlir::Value> explicitParams;
      // Lower lower bounds, explicit type parameters and explicit
      // extents if any.
      if (ba.isChar())
        TODO(loc, "lowerToBoxValue character");
      // TODO: derived type length parameters.
      lowerExplicitLowerBounds(converter, loc, ba, lbounds, symMap, stmtCtx);
      lowerExplicitExtents(converter, loc, ba, lbounds, extents, symMap,
                           stmtCtx);
      symMap.addBoxSymbol(sym, dummyArg, lbounds, explicitParams, extents,
                          replace);
      return;
    }
  }

  // Helper to generate scalars for the symbol properties.
  auto genValue = [&](const Fortran::lower::SomeExpr &expr) {
    return genScalarValue(converter, loc, expr, symMap, stmtCtx);
  };

  // For symbols reaching this point, all properties are constant and can be
  // read/computed already into ssa values.

  // The origin must be \vec{1}.
  auto populateShape = [&](auto &shapes, const auto &bounds, mlir::Value box) {
    for (auto iter : llvm::enumerate(bounds)) {
      auto *spec = iter.value();
      assert(spec->lbound().GetExplicit() &&
             "lbound must be explicit with constant value 1");
      if (auto high = spec->ubound().GetExplicit()) {
        Fortran::lower::SomeExpr highEx{*high};
        mlir::Value ub = genValue(highEx);
        shapes.emplace_back(builder.createConvert(loc, idxTy, ub));
      } else if (spec->ubound().isColon()) {
        assert(box && "assumed bounds require a descriptor");
        mlir::Value dim =
            builder.createIntegerConstant(loc, idxTy, iter.index());
        auto dimInfo =
            builder.create<fir::BoxDimsOp>(loc, idxTy, idxTy, idxTy, box, dim);
        shapes.emplace_back(dimInfo.getResult(1));
      } else if (spec->ubound().isStar()) {
        shapes.emplace_back(builder.create<fir::UndefOp>(loc, idxTy));
      } else {
        llvm::report_fatal_error("unknown bound category");
      }
    }
  };

  // The origin is not \vec{1}.
  auto populateLBoundsExtents = [&](auto &lbounds, auto &extents,
                                    const auto &bounds, mlir::Value box) {
    for (auto iter : llvm::enumerate(bounds)) {
      auto *spec = iter.value();
      fir::BoxDimsOp dimInfo;
      mlir::Value ub, lb;
      if (spec->lbound().isColon() || spec->ubound().isColon()) {
        // This is an assumed shape because allocatables and pointers extents
        // are not constant in the scope and are not read here.
        assert(box && "deferred bounds require a descriptor");
        mlir::Value dim =
            builder.createIntegerConstant(loc, idxTy, iter.index());
        dimInfo =
            builder.create<fir::BoxDimsOp>(loc, idxTy, idxTy, idxTy, box, dim);
        extents.emplace_back(dimInfo.getResult(1));
        if (auto low = spec->lbound().GetExplicit()) {
          auto expr = Fortran::lower::SomeExpr{*low};
          mlir::Value lb = builder.createConvert(loc, idxTy, genValue(expr));
          lbounds.emplace_back(lb);
        } else {
          // Implicit lower bound is 1 (Fortran 2018 section 8.5.8.3 point 3.)
          lbounds.emplace_back(builder.createIntegerConstant(loc, idxTy, 1));
        }
      } else {
        if (auto low = spec->lbound().GetExplicit()) {
          auto expr = Fortran::lower::SomeExpr{*low};
          lb = builder.createConvert(loc, idxTy, genValue(expr));
        } else {
          TODO(loc, "assumed rank lowering");
        }

        if (auto high = spec->ubound().GetExplicit()) {
          auto expr = Fortran::lower::SomeExpr{*high};
          ub = builder.createConvert(loc, idxTy, genValue(expr));
          lbounds.emplace_back(lb);
          extents.emplace_back(computeExtent(builder, loc, lb, ub));
        } else {
          // An assumed size array. The extent is not computed.
          assert(spec->ubound().isStar() && "expected assumed size");
          lbounds.emplace_back(lb);
          extents.emplace_back(builder.create<fir::UndefOp>(loc, idxTy));
        }
      }
    }
  };

  // Lower length expression for non deferred and non dummy assumed length
  // characters.
  auto genExplicitCharLen =
      [&](llvm::Optional<Fortran::lower::SomeExpr> charLen) -> mlir::Value {
    if (!charLen)
      fir::emitFatalError(loc, "expected explicit character length");
    mlir::Value rawLen = genValue(*charLen);
    // If the length expression is negative, the length is zero. See
    // F2018 7.4.4.2 point 5.
    return genMaxWithZero(builder, loc, rawLen);
  };

  ba.match(
      //===--------------------------------------------------------------===//
      // Trivial case.
      //===--------------------------------------------------------------===//
      [&](const Fortran::lower::details::ScalarSym &) {
        if (isDummy) {
          // This is an argument.
          if (!symMap.lookupSymbol(sym))
            mlir::emitError(loc, "symbol \"")
                << toStringRef(sym.name()) << "\" must already be in map";
          return;
        } else if (isResult) {
          // Some Fortran results may be passed by argument (e.g. derived
          // types)
          if (symMap.lookupSymbol(sym))
            return;
        }
        // Otherwise, it's a local variable or function result.
        mlir::Value local = createNewLocal(converter, loc, var, preAlloc);
        symMap.addSymbol(sym, local);
      },

      //===--------------------------------------------------------------===//
      // The non-trivial cases are when we have an argument or local that has
      // a repetition value. Arguments might be passed as simple pointers and
      // need to be cast to a multi-dimensional array with constant bounds
      // (possibly with a missing column), bounds computed in the callee
      // (here), or with bounds from the caller (boxed somewhere else). Locals
      // have the same properties except they are never boxed arguments from
      // the caller and never having a missing column size.
      //===--------------------------------------------------------------===//

      [&](const Fortran::lower::details::ScalarStaticChar &x) {
        // type is a CHARACTER, determine the LEN value
        auto charLen = x.charLen();
        if (replace) {
          Fortran::lower::SymbolBox symBox = symMap.lookupSymbol(sym);
          std::pair<mlir::Value, mlir::Value> unboxchar =
              charHelp.createUnboxChar(symBox.getAddr());
          mlir::Value boxAddr = unboxchar.first;
          // Set/override LEN with a constant
          mlir::Value len = builder.createIntegerConstant(loc, idxTy, charLen);
          symMap.addCharSymbol(sym, boxAddr, len, true);
          return;
        }
        mlir::Value len = builder.createIntegerConstant(loc, idxTy, charLen);
        if (preAlloc) {
          symMap.addCharSymbol(sym, preAlloc, len);
          return;
        }
        mlir::Value local = createNewLocal(converter, loc, var, preAlloc);
        symMap.addCharSymbol(sym, local, len);
      },

      //===--------------------------------------------------------------===//

      [&](const Fortran::lower::details::ScalarDynamicChar &x) {
        TODO(loc, "ScalarDynamicChar variable lowering");
      },

      //===--------------------------------------------------------------===//

      [&](const Fortran::lower::details::StaticArray &x) {
        // object shape is constant, not a character
        mlir::Type castTy = builder.getRefType(converter.genType(var));
        mlir::Value addr = symMap.lookupSymbol(sym).getAddr();
        if (addr)
          addr = builder.createConvert(loc, castTy, addr);
        if (x.lboundAllOnes()) {
          // if lower bounds are all ones, build simple shaped object
          llvm::SmallVector<mlir::Value> shape;
          for (int64_t i : x.shapes)
            shape.push_back(genExtentValue(builder, loc, idxTy, i));
          mlir::Value local =
              isDummy ? addr : createNewLocal(converter, loc, var, preAlloc);
          symMap.addSymbolWithShape(sym, local, shape, isDummy);
          return;
        }
        // If object is an array process the lower bound and extent values by
        // constructing constants and populating the lbounds and extents.
        llvm::SmallVector<mlir::Value> extents;
        llvm::SmallVector<mlir::Value> lbounds;
        for (auto [fst, snd] : llvm::zip(x.lbounds, x.shapes)) {
          lbounds.emplace_back(builder.createIntegerConstant(loc, idxTy, fst));
          extents.emplace_back(genExtentValue(builder, loc, idxTy, snd));
        }
        mlir::Value local =
            isDummy ? addr
                    : createNewLocal(converter, loc, var, preAlloc, extents);
        assert(isDummy || Fortran::lower::isExplicitShape(sym));
        symMap.addSymbolWithBounds(sym, local, extents, lbounds, isDummy);
      },

      //===--------------------------------------------------------------===//

      [&](const Fortran::lower::details::DynamicArray &x) {
        // cast to the known constant parts from the declaration
        mlir::Type varType = converter.genType(var);
        mlir::Value addr = symMap.lookupSymbol(sym).getAddr();
        mlir::Value argBox;
        mlir::Type castTy = builder.getRefType(varType);
        if (addr) {
          if (auto boxTy = addr.getType().dyn_cast<fir::BoxType>()) {
            argBox = addr;
            mlir::Type refTy = builder.getRefType(boxTy.getEleTy());
            addr = builder.create<fir::BoxAddrOp>(loc, refTy, argBox);
          }
          addr = builder.createConvert(loc, castTy, addr);
        }
        if (x.lboundAllOnes()) {
          // if lower bounds are all ones, build simple shaped object
          llvm::SmallVector<mlir::Value> shapes;
          populateShape(shapes, x.bounds, argBox);
          if (isDummy) {
            symMap.addSymbolWithShape(sym, addr, shapes, true);
            return;
          }
          // local array with computed bounds
          assert(Fortran::lower::isExplicitShape(sym) ||
                 Fortran::semantics::IsAllocatableOrPointer(sym));
          mlir::Value local =
              createNewLocal(converter, loc, var, preAlloc, shapes);
          symMap.addSymbolWithShape(sym, local, shapes);
          return;
        }
        // if object is an array process the lower bound and extent values
        llvm::SmallVector<mlir::Value> extents;
        llvm::SmallVector<mlir::Value> lbounds;
        populateLBoundsExtents(lbounds, extents, x.bounds, argBox);
        if (isDummy) {
          symMap.addSymbolWithBounds(sym, addr, extents, lbounds, true);
          return;
        }
        // local array with computed bounds
        assert(Fortran::lower::isExplicitShape(sym));
        mlir::Value local =
            createNewLocal(converter, loc, var, preAlloc, extents);
        symMap.addSymbolWithBounds(sym, local, extents, lbounds);
      },

      //===--------------------------------------------------------------===//

      [&](const Fortran::lower::details::StaticArrayStaticChar &x) {
        // if element type is a CHARACTER, determine the LEN value
        auto charLen = x.charLen();
        mlir::Value addr;
        mlir::Value len;
        if (isDummy) {
          Fortran::lower::SymbolBox symBox = symMap.lookupSymbol(sym);
          std::pair<mlir::Value, mlir::Value> unboxchar =
              charHelp.createUnboxChar(symBox.getAddr());
          addr = unboxchar.first;
          // Set/override LEN with a constant
          len = builder.createIntegerConstant(loc, idxTy, charLen);
        } else {
          // local CHARACTER variable
          len = builder.createIntegerConstant(loc, idxTy, charLen);
        }

        // object shape is constant
        mlir::Type castTy = builder.getRefType(converter.genType(var));
        if (addr)
          addr = builder.createConvert(loc, castTy, addr);

        if (x.lboundAllOnes()) {
          // if lower bounds are all ones, build simple shaped object
          llvm::SmallVector<mlir::Value> shape;
          for (int64_t i : x.shapes)
            shape.push_back(genExtentValue(builder, loc, idxTy, i));
          mlir::Value local =
              isDummy ? addr : createNewLocal(converter, loc, var, preAlloc);
          symMap.addCharSymbolWithShape(sym, local, len, shape, isDummy);
          return;
        }

        // if object is an array process the lower bound and extent values
        llvm::SmallVector<mlir::Value> extents;
        llvm::SmallVector<mlir::Value> lbounds;
        // construct constants and populate `bounds`
        for (auto [fst, snd] : llvm::zip(x.lbounds, x.shapes)) {
          lbounds.emplace_back(builder.createIntegerConstant(loc, idxTy, fst));
          extents.emplace_back(genExtentValue(builder, loc, idxTy, snd));
        }

        if (isDummy) {
          symMap.addCharSymbolWithBounds(sym, addr, len, extents, lbounds,
                                         true);
          return;
        }
        // local CHARACTER array with computed bounds
        assert(Fortran::lower::isExplicitShape(sym));
        mlir::Value local =
            createNewLocal(converter, loc, var, preAlloc, extents);
        symMap.addCharSymbolWithBounds(sym, local, len, extents, lbounds);
      },

      //===--------------------------------------------------------------===//

      [&](const Fortran::lower::details::StaticArrayDynamicChar &x) {
        mlir::Value addr;
        mlir::Value len;
        [[maybe_unused]] bool mustBeDummy = false;
        auto charLen = x.charLen();
        // if element type is a CHARACTER, determine the LEN value
        if (isDummy) {
          Fortran::lower::SymbolBox symBox = symMap.lookupSymbol(sym);
          std::pair<mlir::Value, mlir::Value> unboxchar =
              charHelp.createUnboxChar(symBox.getAddr());
          addr = unboxchar.first;
          if (charLen) {
            // Set/override LEN with an expression
            len = genExplicitCharLen(charLen);
          } else {
            // LEN is from the boxchar
            len = unboxchar.second;
            mustBeDummy = true;
          }
        } else {
          // local CHARACTER variable
          len = genExplicitCharLen(charLen);
        }
        llvm::SmallVector<mlir::Value> lengths = {len};

        // cast to the known constant parts from the declaration
        mlir::Type castTy = builder.getRefType(converter.genType(var));
        if (addr)
          addr = builder.createConvert(loc, castTy, addr);

        if (x.lboundAllOnes()) {
          // if lower bounds are all ones, build simple shaped object
          llvm::SmallVector<mlir::Value> shape;
          for (int64_t i : x.shapes)
            shape.push_back(genExtentValue(builder, loc, idxTy, i));
          if (isDummy) {
            symMap.addCharSymbolWithShape(sym, addr, len, shape, true);
            return;
          }
          // local CHARACTER array with constant size
          mlir::Value local = createNewLocal(converter, loc, var, preAlloc,
                                             llvm::None, lengths);
          symMap.addCharSymbolWithShape(sym, local, len, shape);
          return;
        }

        // if object is an array process the lower bound and extent values
        llvm::SmallVector<mlir::Value> extents;
        llvm::SmallVector<mlir::Value> lbounds;

        // construct constants and populate `bounds`
        for (auto [fst, snd] : llvm::zip(x.lbounds, x.shapes)) {
          lbounds.emplace_back(builder.createIntegerConstant(loc, idxTy, fst));
          extents.emplace_back(genExtentValue(builder, loc, idxTy, snd));
        }
        if (isDummy) {
          symMap.addCharSymbolWithBounds(sym, addr, len, extents, lbounds,
                                         true);
          return;
        }
        // local CHARACTER array with computed bounds
        assert((!mustBeDummy) && (Fortran::lower::isExplicitShape(sym)));
        mlir::Value local =
            createNewLocal(converter, loc, var, preAlloc, llvm::None, lengths);
        symMap.addCharSymbolWithBounds(sym, local, len, extents, lbounds);
      },

      //===--------------------------------------------------------------===//

      [&](const Fortran::lower::details::DynamicArrayStaticChar &x) {
        TODO(loc, "DynamicArrayStaticChar variable lowering");
      },

      //===--------------------------------------------------------------===//

      [&](const Fortran::lower::details::DynamicArrayDynamicChar &x) {
        TODO(loc, "DynamicArrayDynamicChar variable lowering");
      },

      //===--------------------------------------------------------------===//

      [&](const Fortran::lower::BoxAnalyzer::None &) {
        mlir::emitError(loc, "symbol analysis failed on ")
            << toStringRef(sym.name());
      });
}

void Fortran::lower::defineModuleVariable(
    AbstractConverter &converter, const Fortran::lower::pft::Variable &var) {
  // Use empty linkage for module variables, which makes them available
  // for use in another unit.
  mlir::StringAttr externalLinkage;
  if (!var.isGlobal())
    fir::emitFatalError(converter.getCurrentLocation(),
                        "attempting to lower module variable as local");
  // Define aggregate storages for equivalenced objects.
  if (var.isAggregateStore()) {
    const mlir::Location loc = converter.genLocation(var.getSymbol().name());
    TODO(loc, "defineModuleVariable aggregateStore");
  }
  const Fortran::semantics::Symbol &sym = var.getSymbol();
  if (Fortran::semantics::FindCommonBlockContaining(var.getSymbol())) {
    const mlir::Location loc = converter.genLocation(sym.name());
    TODO(loc, "defineModuleVariable common block");
  } else if (var.isAlias()) {
    // Do nothing. Mapping will be done on user side.
  } else {
    std::string globalName = Fortran::lower::mangle::mangleName(sym);
    defineGlobal(converter, var, globalName, externalLinkage);
  }
}

void Fortran::lower::instantiateVariable(AbstractConverter &converter,
                                         const pft::Variable &var,
                                         SymMap &symMap,
                                         AggregateStoreMap &storeMap) {
  const Fortran::semantics::Symbol &sym = var.getSymbol();
  const mlir::Location loc = converter.genLocation(sym.name());
  if (var.isAggregateStore()) {
    TODO(loc, "instantiateVariable AggregateStore");
  } else if (Fortran::semantics::FindCommonBlockContaining(
                 var.getSymbol().GetUltimate())) {
    TODO(loc, "instantiateVariable Common");
  } else if (var.isAlias()) {
    TODO(loc, "instantiateVariable Alias");
  } else if (var.isGlobal()) {
    instantiateGlobal(converter, var, symMap);
  } else {
    instantiateLocal(converter, var, symMap);
  }
}

void Fortran::lower::mapCallInterfaceSymbols(
    AbstractConverter &converter, const Fortran::lower::CallerInterface &caller,
    SymMap &symMap) {
  Fortran::lower::AggregateStoreMap storeMap;
  const Fortran::semantics::Symbol &result = caller.getResultSymbol();
  for (Fortran::lower::pft::Variable var :
       Fortran::lower::pft::buildFuncResultDependencyList(result)) {
    if (var.isAggregateStore()) {
      instantiateVariable(converter, var, symMap, storeMap);
    } else {
      const Fortran::semantics::Symbol &sym = var.getSymbol();
      const auto *hostDetails =
          sym.detailsIf<Fortran::semantics::HostAssocDetails>();
      if (hostDetails && !var.isModuleVariable()) {
        // The callee is an internal procedure `A` whose result properties
        // depend on host variables. The caller may be the host, or another
        // internal procedure `B` contained in the same host.  In the first
        // case, the host symbol is obviously mapped, in the second case, it
        // must also be mapped because
        // HostAssociations::internalProcedureBindings that was called when
        // lowering `B` will have mapped all host symbols of captured variables
        // to the tuple argument containing the composite of all host associated
        // variables, whether or not the host symbol is actually referred to in
        // `B`. Hence it is possible to simply lookup the variable associated to
        // the host symbol without having to go back to the tuple argument.
        Fortran::lower::SymbolBox hostValue =
            symMap.lookupSymbol(hostDetails->symbol());
        assert(hostValue && "callee host symbol must be mapped on caller side");
        symMap.addSymbol(sym, hostValue.toExtendedValue());
        // The SymbolBox associated to the host symbols is complete, skip
        // instantiateVariable that would try to allocate a new storage.
        continue;
      }
      if (Fortran::semantics::IsDummy(sym) && sym.owner() == result.owner()) {
        // Get the argument for the dummy argument symbols of the current call.
        symMap.addSymbol(sym, caller.getArgumentValue(sym));
        // All the properties of the dummy variable may not come from the actual
        // argument, let instantiateVariable handle this.
      }
      // If this is neither a host associated or dummy symbol, it must be a
      // module or common block variable to satisfy specification expression
      // requirements in 10.1.11, instantiateVariable will get its address and
      // properties.
      instantiateVariable(converter, var, symMap, storeMap);
    }
  }
}
