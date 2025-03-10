//===-- ConvertCall.cpp ---------------------------------------------------===//
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

#include "flang/Lower/ConvertCall.h"
#include "flang/Lower/Allocatable.h"
#include "flang/Lower/ConvertExprToHLFIR.h"
#include "flang/Lower/ConvertProcedureDesignator.h"
#include "flang/Lower/ConvertVariable.h"
#include "flang/Lower/CustomIntrinsicCall.h"
#include "flang/Lower/HlfirIntrinsics.h"
#include "flang/Lower/StatementContext.h"
#include "flang/Lower/SymbolMap.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/Character.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Builder/IntrinsicCall.h"
#include "flang/Optimizer/Builder/LowLevelIntrinsics.h"
#include "flang/Optimizer/Builder/MutableBox.h"
#include "flang/Optimizer/Builder/Runtime/Derived.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/CUF/CUFOps.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include <optional>

#define DEBUG_TYPE "flang-lower-expr"

static llvm::cl::opt<bool> useHlfirIntrinsicOps(
    "use-hlfir-intrinsic-ops", llvm::cl::init(true),
    llvm::cl::desc("Lower via HLFIR transformational intrinsic operations such "
                   "as hlfir.sum"));

static constexpr char tempResultName[] = ".tmp.func_result";

/// Helper to package a Value and its properties into an ExtendedValue.
static fir::ExtendedValue toExtendedValue(mlir::Location loc, mlir::Value base,
                                          llvm::ArrayRef<mlir::Value> extents,
                                          llvm::ArrayRef<mlir::Value> lengths) {
  mlir::Type type = base.getType();
  if (mlir::isa<fir::BaseBoxType>(type))
    return fir::BoxValue(base, /*lbounds=*/{}, lengths, extents);
  type = fir::unwrapRefType(type);
  if (mlir::isa<fir::BaseBoxType>(type))
    return fir::MutableBoxValue(base, lengths, /*mutableProperties*/ {});
  if (auto seqTy = mlir::dyn_cast<fir::SequenceType>(type)) {
    if (seqTy.getDimension() != extents.size())
      fir::emitFatalError(loc, "incorrect number of extents for array");
    if (mlir::isa<fir::CharacterType>(seqTy.getEleTy())) {
      if (lengths.empty())
        fir::emitFatalError(loc, "missing length for character");
      assert(lengths.size() == 1);
      return fir::CharArrayBoxValue(base, lengths[0], extents);
    }
    return fir::ArrayBoxValue(base, extents);
  }
  if (mlir::isa<fir::CharacterType>(type)) {
    if (lengths.empty())
      fir::emitFatalError(loc, "missing length for character");
    assert(lengths.size() == 1);
    return fir::CharBoxValue(base, lengths[0]);
  }
  return base;
}

/// Lower a type(C_PTR/C_FUNPTR) argument with VALUE attribute into a
/// reference. A C pointer can correspond to a Fortran dummy argument of type
/// C_PTR with the VALUE attribute. (see 18.3.6 note 3).
static mlir::Value genRecordCPtrValueArg(fir::FirOpBuilder &builder,
                                         mlir::Location loc, mlir::Value rec,
                                         mlir::Type ty) {
  mlir::Value cAddr = fir::factory::genCPtrOrCFunptrAddr(builder, loc, rec, ty);
  mlir::Value cVal = builder.create<fir::LoadOp>(loc, cAddr);
  return builder.createConvert(loc, cAddr.getType(), cVal);
}

// Find the argument that corresponds to the host associations.
// Verify some assumptions about how the signature was built here.
[[maybe_unused]] static unsigned findHostAssocTuplePos(mlir::func::FuncOp fn) {
  // Scan the argument list from last to first as the host associations are
  // appended for now.
  for (unsigned i = fn.getNumArguments(); i > 0; --i)
    if (fn.getArgAttr(i - 1, fir::getHostAssocAttrName())) {
      // Host assoc tuple must be last argument (for now).
      assert(i == fn.getNumArguments() && "tuple must be last");
      return i - 1;
    }
  llvm_unreachable("anyFuncArgsHaveAttr failed");
}

mlir::Value
Fortran::lower::argumentHostAssocs(Fortran::lower::AbstractConverter &converter,
                                   mlir::Value arg) {
  if (auto addr = mlir::dyn_cast_or_null<fir::AddrOfOp>(arg.getDefiningOp())) {
    auto &builder = converter.getFirOpBuilder();
    if (auto funcOp = builder.getNamedFunction(addr.getSymbol()))
      if (fir::anyFuncArgsHaveAttr(funcOp, fir::getHostAssocAttrName()))
        return converter.hostAssocTupleValue();
  }
  return {};
}

static bool mustCastFuncOpToCopeWithImplicitInterfaceMismatch(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    mlir::FunctionType callSiteType, mlir::FunctionType funcOpType) {
  // Deal with argument number mismatch by making a function pointer so
  // that function type cast can be inserted. Do not emit a warning here
  // because this can happen in legal program if the function is not
  // defined here and it was first passed as an argument without any more
  // information.
  if (callSiteType.getNumResults() != funcOpType.getNumResults() ||
      callSiteType.getNumInputs() != funcOpType.getNumInputs())
    return true;

  // Implicit interface result type mismatch are not standard Fortran, but
  // some compilers are not complaining about it.  The front end is not
  // protecting lowering from this currently. Support this with a
  // discouraging warning.
  // Cast the actual function to the current caller implicit type because
  // that is the behavior we would get if we could not see the definition.
  if (callSiteType.getResults() != funcOpType.getResults()) {
    LLVM_DEBUG(mlir::emitWarning(
        loc, "a return type mismatch is not standard compliant and may "
             "lead to undefined behavior."));
    return true;
  }

  // In HLFIR, there is little attempt to cope with implicit interface
  // mismatch on the arguments. The argument are always prepared according
  // to the implicit interface. Cast the actual function if any of the
  // argument mismatch cannot be dealt with a simple fir.convert.
  if (converter.getLoweringOptions().getLowerToHighLevelFIR())
    for (auto [actualType, dummyType] :
         llvm::zip(callSiteType.getInputs(), funcOpType.getInputs()))
      if (actualType != dummyType &&
          !fir::ConvertOp::canBeConverted(actualType, dummyType))
        return true;
  return false;
}

static mlir::Value readDim3Value(fir::FirOpBuilder &builder, mlir::Location loc,
                                 mlir::Value dim3Addr, llvm::StringRef comp) {
  mlir::Type i32Ty = builder.getI32Type();
  mlir::Type refI32Ty = fir::ReferenceType::get(i32Ty);
  llvm::SmallVector<mlir::Value> lenParams;

  mlir::Value designate = builder.create<hlfir::DesignateOp>(
      loc, refI32Ty, dim3Addr, /*component=*/comp,
      /*componentShape=*/mlir::Value{}, hlfir::DesignateOp::Subscripts{},
      /*substring=*/mlir::ValueRange{}, /*complexPartAttr=*/std::nullopt,
      mlir::Value{}, lenParams);

  return hlfir::loadTrivialScalar(loc, builder, hlfir::Entity{designate});
}

static mlir::Value remapActualToDummyDescriptor(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    Fortran::lower::SymMap &symMap,
    const Fortran::lower::CallerInterface::PassedEntity &arg,
    Fortran::lower::CallerInterface &caller, bool isBindcCall) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::IndexType idxTy = builder.getIndexType();
  mlir::Value zero = builder.createIntegerConstant(loc, idxTy, 0);
  Fortran::lower::StatementContext localStmtCtx;
  auto lowerSpecExpr = [&](const auto &expr,
                           bool isAssumedSizeExtent) -> mlir::Value {
    mlir::Value convertExpr = builder.createConvert(
        loc, idxTy, fir::getBase(converter.genExprValue(expr, localStmtCtx)));
    if (isAssumedSizeExtent)
      return convertExpr;
    return fir::factory::genMaxWithZero(builder, loc, convertExpr);
  };
  bool mapSymbols = caller.mustMapInterfaceSymbolsForDummyArgument(arg);
  if (mapSymbols) {
    symMap.pushScope();
    const Fortran::semantics::Symbol *sym = caller.getDummySymbol(arg);
    assert(sym && "call must have explicit interface to map interface symbols");
    Fortran::lower::mapCallInterfaceSymbolsForDummyArgument(converter, caller,
                                                            symMap, *sym);
  }
  llvm::SmallVector<mlir::Value> extents;
  llvm::SmallVector<mlir::Value> lengths;
  mlir::Type dummyBoxType = caller.getDummyArgumentType(arg);
  mlir::Type dummyBaseType = fir::unwrapPassByRefType(dummyBoxType);
  if (mlir::isa<fir::SequenceType>(dummyBaseType))
    caller.walkDummyArgumentExtents(
        arg, [&](const Fortran::lower::SomeExpr &e, bool isAssumedSizeExtent) {
          extents.emplace_back(lowerSpecExpr(e, isAssumedSizeExtent));
        });
  mlir::Value shape;
  if (!extents.empty()) {
    if (isBindcCall) {
      // Preserve zero lower bounds (see F'2023 18.5.3).
      llvm::SmallVector<mlir::Value> lowerBounds(extents.size(), zero);
      shape = builder.genShape(loc, lowerBounds, extents);
    } else {
      shape = builder.genShape(loc, extents);
    }
  }

  hlfir::Entity explicitArgument = hlfir::Entity{caller.getInput(arg)};
  mlir::Type dummyElementType = fir::unwrapSequenceType(dummyBaseType);
  if (auto recType = llvm::dyn_cast<fir::RecordType>(dummyElementType))
    if (recType.getNumLenParams() > 0)
      TODO(loc, "sequence association of length parameterized derived type "
                "dummy arguments");
  if (fir::isa_char(dummyElementType))
    lengths.emplace_back(hlfir::genCharLength(loc, builder, explicitArgument));
  mlir::Value baseAddr =
      hlfir::genVariableRawAddress(loc, builder, explicitArgument);
  baseAddr = builder.createConvert(loc, fir::ReferenceType::get(dummyBaseType),
                                   baseAddr);
  mlir::Value mold;
  if (fir::isPolymorphicType(dummyBoxType))
    mold = explicitArgument;
  mlir::Value remapped =
      builder.create<fir::EmboxOp>(loc, dummyBoxType, baseAddr, shape,
                                   /*slice=*/mlir::Value{}, lengths, mold);
  if (mapSymbols)
    symMap.popScope();
  return remapped;
}

/// Create a descriptor for sequenced associated descriptor that are passed
/// by descriptor. Sequence association (F'2023 15.5.2.12) implies that the
/// dummy shape and rank need to not be the same as the actual argument. This
/// helper creates a descriptor based on the dummy shape and rank (sequence
/// association can only happen with explicit and assumed-size array) so that it
/// is safe to assume the rank of the incoming descriptor inside the callee.
/// This helper must be called once all the actual arguments have been lowered
/// and placed inside "caller". Copy-in/copy-out must already have been
/// generated if needed using the actual argument shape (the dummy shape may be
/// assumed-size).
static void remapActualToDummyDescriptors(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    Fortran::lower::SymMap &symMap,
    const Fortran::lower::PreparedActualArguments &loweredActuals,
    Fortran::lower::CallerInterface &caller, bool isBindcCall) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  for (auto [preparedActual, arg] :
       llvm::zip(loweredActuals, caller.getPassedArguments())) {
    if (arg.isSequenceAssociatedDescriptor()) {
      if (!preparedActual.value().handleDynamicOptional()) {
        mlir::Value remapped = remapActualToDummyDescriptor(
            loc, converter, symMap, arg, caller, isBindcCall);
        caller.placeInput(arg, remapped);
      } else {
        // Absent optional actual argument descriptor cannot be read and
        // remapped unconditionally.
        mlir::Type dummyType = caller.getDummyArgumentType(arg);
        mlir::Value isPresent = preparedActual.value().getIsPresent();
        auto &argLambdaCapture = arg;
        mlir::Value remapped =
            builder
                .genIfOp(loc, {dummyType}, isPresent,
                         /*withElseRegion=*/true)
                .genThen([&]() {
                  mlir::Value newBox = remapActualToDummyDescriptor(
                      loc, converter, symMap, argLambdaCapture, caller,
                      isBindcCall);
                  builder.create<fir::ResultOp>(loc, newBox);
                })
                .genElse([&]() {
                  mlir::Value absent =
                      builder.create<fir::AbsentOp>(loc, dummyType);
                  builder.create<fir::ResultOp>(loc, absent);
                })
                .getResults()[0];
        caller.placeInput(arg, remapped);
      }
    }
  }
}

std::pair<Fortran::lower::LoweredResult, bool>
Fortran::lower::genCallOpAndResult(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx,
    Fortran::lower::CallerInterface &caller, mlir::FunctionType callSiteType,
    std::optional<mlir::Type> resultType, bool isElemental) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  using PassBy = Fortran::lower::CallerInterface::PassEntityBy;
  bool mustPopSymMap = false;
  if (caller.mustMapInterfaceSymbolsForResult()) {
    symMap.pushScope();
    mustPopSymMap = true;
    Fortran::lower::mapCallInterfaceSymbolsForResult(converter, caller, symMap);
  }
  // If this is an indirect call, retrieve the function address. Also retrieve
  // the result length if this is a character function (note that this length
  // will be used only if there is no explicit length in the local interface).
  mlir::Value funcPointer;
  mlir::Value charFuncPointerLength;
  if (const Fortran::evaluate::ProcedureDesignator *procDesignator =
          caller.getIfIndirectCall()) {
    if (mlir::Value passedArg = caller.getIfPassedArg()) {
      // Procedure pointer component call with PASS argument. To avoid
      // "double" lowering of the ComponentRef, semantics only place the
      // ComponentRef in the ActualArguments, not in the ProcedureDesignator (
      // that is only the component symbol).
      // Fetch the passed argument and addresses of its procedure pointer
      // component.
      funcPointer = Fortran::lower::derefPassProcPointerComponent(
          loc, converter, *procDesignator, passedArg, symMap, stmtCtx);
    } else {
      Fortran::lower::SomeExpr expr{*procDesignator};
      fir::ExtendedValue loweredProc =
          converter.genExprAddr(loc, expr, stmtCtx);
      funcPointer = fir::getBase(loweredProc);
      // Dummy procedure may have assumed length, in which case the result
      // length was passed along the dummy procedure.
      // This is not possible with procedure pointer components.
      if (const fir::CharBoxValue *charBox = loweredProc.getCharBox())
        charFuncPointerLength = charBox->getLen();
    }
  }

  const bool isExprCall =
      converter.getLoweringOptions().getLowerToHighLevelFIR() &&
      callSiteType.getNumResults() == 1 &&
      llvm::isa<fir::SequenceType>(callSiteType.getResult(0));

  mlir::IndexType idxTy = builder.getIndexType();
  auto lowerSpecExpr = [&](const auto &expr) -> mlir::Value {
    mlir::Value convertExpr = builder.createConvert(
        loc, idxTy, fir::getBase(converter.genExprValue(expr, stmtCtx)));
    return fir::factory::genMaxWithZero(builder, loc, convertExpr);
  };
  llvm::SmallVector<mlir::Value> resultLengths;
  mlir::Value arrayResultShape;
  hlfir::EvaluateInMemoryOp evaluateInMemory;
  auto allocatedResult = [&]() -> std::optional<fir::ExtendedValue> {
    llvm::SmallVector<mlir::Value> extents;
    llvm::SmallVector<mlir::Value> lengths;
    if (!caller.callerAllocateResult())
      return {};
    mlir::Type type = caller.getResultStorageType();
    if (mlir::isa<fir::SequenceType>(type))
      caller.walkResultExtents(
          [&](const Fortran::lower::SomeExpr &e, bool isAssumedSizeExtent) {
            assert(!isAssumedSizeExtent && "result cannot be assumed-size");
            extents.emplace_back(lowerSpecExpr(e));
          });
    caller.walkResultLengths(
        [&](const Fortran::lower::SomeExpr &e, bool isAssumedSizeExtent) {
          assert(!isAssumedSizeExtent && "result cannot be assumed-size");
          lengths.emplace_back(lowerSpecExpr(e));
        });

    // Result length parameters should not be provided to box storage
    // allocation and save_results, but they are still useful information to
    // keep in the ExtendedValue if non-deferred.
    if (!mlir::isa<fir::BoxType>(type)) {
      if (fir::isa_char(fir::unwrapSequenceType(type)) && lengths.empty()) {
        // Calling an assumed length function. This is only possible if this
        // is a call to a character dummy procedure.
        if (!charFuncPointerLength)
          fir::emitFatalError(loc, "failed to retrieve character function "
                                   "length while calling it");
        lengths.push_back(charFuncPointerLength);
      }
      resultLengths = lengths;
    }

    if (!extents.empty())
      arrayResultShape = builder.genShape(loc, extents);

    if (isExprCall) {
      mlir::Type exprType = hlfir::getExprType(type);
      evaluateInMemory = builder.create<hlfir::EvaluateInMemoryOp>(
          loc, exprType, arrayResultShape, resultLengths);
      builder.setInsertionPointToStart(&evaluateInMemory.getBody().front());
      return toExtendedValue(loc, evaluateInMemory.getMemory(), extents,
                             lengths);
    }

    if ((!extents.empty() || !lengths.empty()) && !isElemental) {
      // Note: in the elemental context, the alloca ownership inside the
      // elemental region is implicit, and later pass in lowering (stack
      // reclaim) fir.do_loop will be in charge of emitting any stack
      // save/restore if needed.
      auto *bldr = &converter.getFirOpBuilder();
      mlir::Value sp = bldr->genStackSave(loc);
      stmtCtx.attachCleanup(
          [bldr, loc, sp]() { bldr->genStackRestore(loc, sp); });
    }
    mlir::Value temp =
        builder.createTemporary(loc, type, ".result", extents, resultLengths);
    return toExtendedValue(loc, temp, extents, lengths);
  }();

  if (mustPopSymMap)
    symMap.popScope();

  // Place allocated result
  if (allocatedResult) {
    if (std::optional<Fortran::lower::CallInterface<
            Fortran::lower::CallerInterface>::PassedEntity>
            resultArg = caller.getPassedResult()) {
      if (resultArg->passBy == PassBy::AddressAndLength)
        caller.placeAddressAndLengthInput(*resultArg,
                                          fir::getBase(*allocatedResult),
                                          fir::getLen(*allocatedResult));
      else if (resultArg->passBy == PassBy::BaseAddress)
        caller.placeInput(*resultArg, fir::getBase(*allocatedResult));
      else
        fir::emitFatalError(
            loc, "only expect character scalar result to be passed by ref");
    }
  }

  // In older Fortran, procedure argument types are inferred. This may lead
  // different view of what the function signature is in different locations.
  // Casts are inserted as needed below to accommodate this.

  // The mlir::func::FuncOp type prevails, unless it has a different number of
  // arguments which can happen in legal program if it was passed as a dummy
  // procedure argument earlier with no further type information.
  mlir::SymbolRefAttr funcSymbolAttr;
  bool addHostAssociations = false;
  if (!funcPointer) {
    mlir::FunctionType funcOpType = caller.getFuncOp().getFunctionType();
    mlir::SymbolRefAttr symbolAttr =
        builder.getSymbolRefAttr(caller.getMangledName());
    if (callSiteType.getNumResults() == funcOpType.getNumResults() &&
        callSiteType.getNumInputs() + 1 == funcOpType.getNumInputs() &&
        fir::anyFuncArgsHaveAttr(caller.getFuncOp(),
                                 fir::getHostAssocAttrName())) {
      // The number of arguments is off by one, and we're lowering a function
      // with host associations. Modify call to include host associations
      // argument by appending the value at the end of the operands.
      assert(funcOpType.getInput(findHostAssocTuplePos(caller.getFuncOp())) ==
             converter.hostAssocTupleValue().getType());
      addHostAssociations = true;
    }
    // When this is not a call to an internal procedure (where there is a
    // mismatch due to the extra argument, but the interface is otherwise
    // explicit and safe), handle interface mismatch due to F77 implicit
    // interface "abuse" with a function address cast if needed.
    if (!addHostAssociations &&
        mustCastFuncOpToCopeWithImplicitInterfaceMismatch(
            loc, converter, callSiteType, funcOpType))
      funcPointer = builder.create<fir::AddrOfOp>(loc, funcOpType, symbolAttr);
    else
      funcSymbolAttr = symbolAttr;

    // Issue a warning if the procedure name conflicts with
    // a runtime function name a call to which has been already
    // lowered (implying that the FuncOp has been created).
    // The behavior is undefined in this case.
    if (caller.getFuncOp()->hasAttrOfType<mlir::UnitAttr>(
            fir::FIROpsDialect::getFirRuntimeAttrName()))
      LLVM_DEBUG(mlir::emitWarning(
          loc,
          llvm::Twine("function name '") +
              llvm::Twine(symbolAttr.getLeafReference()) +
              llvm::Twine("' conflicts with a runtime function name used by "
                          "Flang - this may lead to undefined behavior")));
  }

  mlir::FunctionType funcType =
      funcPointer ? callSiteType : caller.getFuncOp().getFunctionType();
  llvm::SmallVector<mlir::Value> operands;
  // First operand of indirect call is the function pointer. Cast it to
  // required function type for the call to handle procedures that have a
  // compatible interface in Fortran, but that have different signatures in
  // FIR.
  if (funcPointer) {
    operands.push_back(
        mlir::isa<fir::BoxProcType>(funcPointer.getType())
            ? builder.create<fir::BoxAddrOp>(loc, funcType, funcPointer)
            : builder.createConvert(loc, funcType, funcPointer));
  }

  // Deal with potential mismatches in arguments types. Passing an array to a
  // scalar argument should for instance be tolerated here.
  bool callingImplicitInterface = caller.canBeCalledViaImplicitInterface();
  for (auto [fst, snd] : llvm::zip(caller.getInputs(), funcType.getInputs())) {
    // When passing arguments to a procedure that can be called by implicit
    // interface, allow any character actual arguments to be passed to dummy
    // arguments of any type and vice versa.
    mlir::Value cast;
    auto *context = builder.getContext();
    if (mlir::isa<fir::BoxProcType>(snd) &&
        mlir::isa<mlir::FunctionType>(fst.getType())) {
      auto funcTy =
          mlir::FunctionType::get(context, std::nullopt, std::nullopt);
      auto boxProcTy = builder.getBoxProcType(funcTy);
      if (mlir::Value host = argumentHostAssocs(converter, fst)) {
        cast = builder.create<fir::EmboxProcOp>(
            loc, boxProcTy, llvm::ArrayRef<mlir::Value>{fst, host});
      } else {
        cast = builder.create<fir::EmboxProcOp>(loc, boxProcTy, fst);
      }
    } else {
      mlir::Type fromTy = fir::unwrapRefType(fst.getType());
      if (fir::isa_builtin_cptr_type(fromTy) &&
          Fortran::lower::isCPtrArgByValueType(snd)) {
        cast = genRecordCPtrValueArg(builder, loc, fst, fromTy);
      } else if (fir::isa_derived(snd) && !fir::isa_derived(fst.getType())) {
        // TODO: remove this TODO once the old lowering is gone.
        TODO(loc, "derived type argument passed by value");
      } else {
        // With the lowering to HLFIR, box arguments have already been built
        // according to the attributes, rank, bounds, and type they should have.
        // Do not attempt any reboxing here that could break this.
        bool legacyLowering =
            !converter.getLoweringOptions().getLowerToHighLevelFIR();
        cast = builder.convertWithSemantics(loc, snd, fst,
                                            callingImplicitInterface,
                                            /*allowRebox=*/legacyLowering);
      }
    }
    operands.push_back(cast);
  }

  // Add host associations as necessary.
  if (addHostAssociations)
    operands.push_back(converter.hostAssocTupleValue());

  mlir::Value callResult;
  unsigned callNumResults;
  fir::FortranProcedureFlagsEnumAttr procAttrs =
      caller.getProcedureAttrs(builder.getContext());

  if (!caller.getCallDescription().chevrons().empty()) {
    // A call to a CUDA kernel with the chevron syntax.

    mlir::Type i32Ty = builder.getI32Type();
    mlir::Value one = builder.createIntegerConstant(loc, i32Ty, 1);

    mlir::Value grid_x, grid_y, grid_z;
    if (caller.getCallDescription().chevrons()[0].GetType()->category() ==
        Fortran::common::TypeCategory::Integer) {
      // If grid is an integer, it is converted to dim3(grid,1,1). Since z is
      // not used for the number of thread blocks, it is omitted in the op.
      grid_x = builder.createConvert(
          loc, i32Ty,
          fir::getBase(converter.genExprValue(
              caller.getCallDescription().chevrons()[0], stmtCtx)));
      grid_y = one;
      grid_z = one;
    } else {
      auto dim3Addr = converter.genExprAddr(
          caller.getCallDescription().chevrons()[0], stmtCtx);
      grid_x = readDim3Value(builder, loc, fir::getBase(dim3Addr), "x");
      grid_y = readDim3Value(builder, loc, fir::getBase(dim3Addr), "y");
      grid_z = readDim3Value(builder, loc, fir::getBase(dim3Addr), "z");
    }

    mlir::Value block_x, block_y, block_z;
    if (caller.getCallDescription().chevrons()[1].GetType()->category() ==
        Fortran::common::TypeCategory::Integer) {
      // If block is an integer, it is converted to dim3(block,1,1).
      block_x = builder.createConvert(
          loc, i32Ty,
          fir::getBase(converter.genExprValue(
              caller.getCallDescription().chevrons()[1], stmtCtx)));
      block_y = one;
      block_z = one;
    } else {
      auto dim3Addr = converter.genExprAddr(
          caller.getCallDescription().chevrons()[1], stmtCtx);
      block_x = readDim3Value(builder, loc, fir::getBase(dim3Addr), "x");
      block_y = readDim3Value(builder, loc, fir::getBase(dim3Addr), "y");
      block_z = readDim3Value(builder, loc, fir::getBase(dim3Addr), "z");
    }

    mlir::Value bytes; // bytes is optional.
    if (caller.getCallDescription().chevrons().size() > 2)
      bytes = builder.createConvert(
          loc, i32Ty,
          fir::getBase(converter.genExprValue(
              caller.getCallDescription().chevrons()[2], stmtCtx)));

    mlir::Value stream; // stream is optional.
    if (caller.getCallDescription().chevrons().size() > 3)
      stream = builder.createConvert(
          loc, i32Ty,
          fir::getBase(converter.genExprValue(
              caller.getCallDescription().chevrons()[3], stmtCtx)));

    builder.create<cuf::KernelLaunchOp>(
        loc, funcType.getResults(), funcSymbolAttr, grid_x, grid_y, grid_z,
        block_x, block_y, block_z, bytes, stream, operands,
        /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr);
    callNumResults = 0;
  } else if (caller.requireDispatchCall()) {
    // Procedure call requiring a dynamic dispatch. Call is created with
    // fir.dispatch.

    // Get the raw procedure name. The procedure name is not mangled in the
    // binding table, but there can be a suffix to distinguish bindings of
    // the same name (which happens only when PRIVATE bindings exist in
    // ancestor types in other modules).
    const auto &ultimateSymbol =
        caller.getCallDescription().proc().GetSymbol()->GetUltimate();
    std::string procName = ultimateSymbol.name().ToString();
    if (const auto &binding{
            ultimateSymbol.get<Fortran::semantics::ProcBindingDetails>()};
        binding.numPrivatesNotOverridden() > 0)
      procName += "."s + std::to_string(binding.numPrivatesNotOverridden());
    fir::DispatchOp dispatch;
    if (std::optional<unsigned> passArg = caller.getPassArgIndex()) {
      // PASS, PASS(arg-name)
      // Note that caller.getInputs is used instead of operands to get the
      // passed object because interface mismatch issues may have inserted a
      // cast to the operand with a different declared type, which would break
      // later type bound call resolution in the FIR to FIR pass.
      dispatch = builder.create<fir::DispatchOp>(
          loc, funcType.getResults(), builder.getStringAttr(procName),
          caller.getInputs()[*passArg], operands,
          builder.getI32IntegerAttr(*passArg), /*arg_attrs=*/nullptr,
          /*res_attrs=*/nullptr, procAttrs);
    } else {
      // NOPASS
      const Fortran::evaluate::Component *component =
          caller.getCallDescription().proc().GetComponent();
      assert(component && "expect component for type-bound procedure call.");

      fir::ExtendedValue dataRefValue = Fortran::lower::convertDataRefToValue(
          loc, converter, component->base(), symMap, stmtCtx);
      mlir::Value passObject = fir::getBase(dataRefValue);

      if (fir::isa_ref_type(passObject.getType()))
        passObject = builder.create<fir::LoadOp>(loc, passObject);
      dispatch = builder.create<fir::DispatchOp>(
          loc, funcType.getResults(), builder.getStringAttr(procName),
          passObject, operands, nullptr, /*arg_attrs=*/nullptr,
          /*res_attrs=*/nullptr, procAttrs);
    }
    callNumResults = dispatch.getNumResults();
    if (callNumResults != 0)
      callResult = dispatch.getResult(0);
  } else {
    // Standard procedure call with fir.call.
    auto call = builder.create<fir::CallOp>(
        loc, funcType.getResults(), funcSymbolAttr, operands,
        /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr, procAttrs);

    callNumResults = call.getNumResults();
    if (callNumResults != 0)
      callResult = call.getResult(0);
  }

  std::optional<Fortran::evaluate::DynamicType> retTy =
      caller.getCallDescription().proc().GetType();
  // With HLFIR lowering, isElemental must be set to true
  // if we are producing an elemental call. In this case,
  // the elemental results must not be destroyed, instead,
  // the resulting array result will be finalized/destroyed
  // as needed by hlfir.destroy.
  const bool mustFinalizeResult =
      !isElemental && callSiteType.getNumResults() > 0 &&
      !fir::isPointerType(callSiteType.getResult(0)) && retTy.has_value() &&
      (retTy->category() == Fortran::common::TypeCategory::Derived ||
       retTy->IsPolymorphic() || retTy->IsUnlimitedPolymorphic());

  if (caller.mustSaveResult()) {
    assert(allocatedResult.has_value());
    builder.create<fir::SaveResultOp>(loc, callResult,
                                      fir::getBase(*allocatedResult),
                                      arrayResultShape, resultLengths);
  }

  if (evaluateInMemory) {
    builder.setInsertionPointAfter(evaluateInMemory);
    mlir::Value expr = evaluateInMemory.getResult();
    fir::FirOpBuilder *bldr = &converter.getFirOpBuilder();
    if (!isElemental)
      stmtCtx.attachCleanup([bldr, loc, expr, mustFinalizeResult]() {
        bldr->create<hlfir::DestroyOp>(loc, expr,
                                       /*finalize=*/mustFinalizeResult);
      });
    return {LoweredResult{hlfir::EntityWithAttributes{expr}},
            mustFinalizeResult};
  }

  if (allocatedResult) {
    // The result must be optionally destroyed (if it is of a derived type
    // that may need finalization or deallocation of the components).
    // For an allocatable result we have to free the memory allocated
    // for the top-level entity. Note that the Destroy calls below
    // do not deallocate the top-level entity. The two clean-ups
    // must be pushed in reverse order, so that the final order is:
    //   Destroy(desc)
    //   free(desc->base_addr)
    allocatedResult->match(
        [&](const fir::MutableBoxValue &box) {
          if (box.isAllocatable()) {
            // 9.7.3.2 point 4. Deallocate allocatable results. Note that
            // finalization was done independently by calling
            // genDerivedTypeDestroy above and is not triggered by this inline
            // deallocation.
            fir::FirOpBuilder *bldr = &converter.getFirOpBuilder();
            stmtCtx.attachCleanup([bldr, loc, box]() {
              fir::factory::genFreememIfAllocated(*bldr, loc, box);
            });
          }
        },
        [](const auto &) {});

    // 7.5.6.3 point 5. Derived-type finalization for nonpointer function.
    bool resultIsFinalized = false;
    // Check if the derived-type is finalizable if it is a monomorphic
    // derived-type.
    // For polymorphic and unlimited polymorphic enities call the runtime
    // in any cases.
    if (mustFinalizeResult) {
      if (retTy->IsPolymorphic() || retTy->IsUnlimitedPolymorphic()) {
        auto *bldr = &converter.getFirOpBuilder();
        stmtCtx.attachCleanup([bldr, loc, allocatedResult]() {
          fir::runtime::genDerivedTypeDestroy(*bldr, loc,
                                              fir::getBase(*allocatedResult));
        });
        resultIsFinalized = true;
      } else {
        const Fortran::semantics::DerivedTypeSpec &typeSpec =
            retTy->GetDerivedTypeSpec();
        // If the result type may require finalization
        // or have allocatable components, we need to make sure
        // everything is properly finalized/deallocated.
        if (Fortran::semantics::MayRequireFinalization(typeSpec) ||
            // We can use DerivedTypeDestroy even if finalization is not needed.
            hlfir::mayHaveAllocatableComponent(funcType.getResults()[0])) {
          auto *bldr = &converter.getFirOpBuilder();
          stmtCtx.attachCleanup([bldr, loc, allocatedResult]() {
            mlir::Value box = bldr->createBox(loc, *allocatedResult);
            fir::runtime::genDerivedTypeDestroy(*bldr, loc, box);
          });
          resultIsFinalized = true;
        }
      }
    }
    return {LoweredResult{*allocatedResult}, resultIsFinalized};
  }

  // subroutine call
  if (!resultType)
    return {LoweredResult{fir::ExtendedValue{mlir::Value{}}},
            /*resultIsFinalized=*/false};

  // For now, Fortran return values are implemented with a single MLIR
  // function return value.
  assert(callNumResults == 1 && "Expected exactly one result in FUNCTION call");
  (void)callNumResults;

  // Call a BIND(C) function that return a char.
  if (caller.characterize().IsBindC() &&
      mlir::isa<fir::CharacterType>(funcType.getResults()[0])) {
    fir::CharacterType charTy =
        mlir::dyn_cast<fir::CharacterType>(funcType.getResults()[0]);
    mlir::Value len = builder.createIntegerConstant(
        loc, builder.getCharacterLengthType(), charTy.getLen());
    return {
        LoweredResult{fir::ExtendedValue{fir::CharBoxValue{callResult, len}}},
        /*resultIsFinalized=*/false};
  }

  return {LoweredResult{fir::ExtendedValue{callResult}},
          /*resultIsFinalized=*/false};
}

static hlfir::EntityWithAttributes genStmtFunctionRef(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx,
    const Fortran::evaluate::ProcedureRef &procRef) {
  const Fortran::semantics::Symbol *symbol = procRef.proc().GetSymbol();
  assert(symbol && "expected symbol in ProcedureRef of statement functions");
  const auto &details = symbol->get<Fortran::semantics::SubprogramDetails>();
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();

  // Statement functions have their own scope, we just need to associate
  // the dummy symbols to argument expressions. There are no
  // optional/alternate return arguments. Statement functions cannot be
  // recursive (directly or indirectly) so it is safe to add dummy symbols to
  // the local map here.
  symMap.pushScope();
  llvm::SmallVector<hlfir::AssociateOp> exprAssociations;
  for (auto [arg, bind] : llvm::zip(details.dummyArgs(), procRef.arguments())) {
    assert(arg && "alternate return in statement function");
    assert(bind && "optional argument in statement function");
    const auto *expr = bind->UnwrapExpr();
    // TODO: assumed type in statement function, that surprisingly seems
    // allowed, probably because nobody thought of restricting this usage.
    // gfortran/ifort compiles this.
    assert(expr && "assumed type used as statement function argument");
    // As per Fortran 2018 C1580, statement function arguments can only be
    // scalars.
    // The only care is to use the dummy character explicit length if any
    // instead of the actual argument length (that can be bigger).
    hlfir::EntityWithAttributes loweredArg = Fortran::lower::convertExprToHLFIR(
        loc, converter, *expr, symMap, stmtCtx);
    fir::FortranVariableOpInterface variableIface = loweredArg.getIfVariable();
    if (!variableIface) {
      // So far only FortranVariableOpInterface can be mapped to symbols.
      // Create an hlfir.associate to create a variable from a potential
      // value argument.
      mlir::Type argType = converter.genType(*arg);
      auto associate = hlfir::genAssociateExpr(
          loc, builder, loweredArg, argType, toStringRef(arg->name()));
      exprAssociations.push_back(associate);
      variableIface = associate;
    }
    const Fortran::semantics::DeclTypeSpec *type = arg->GetType();
    if (type &&
        type->category() == Fortran::semantics::DeclTypeSpec::Character) {
      // Instantiate character as if it was a normal dummy argument so that the
      // statement function dummy character length is applied and dealt with
      // correctly.
      symMap.addSymbol(*arg, variableIface.getBase());
      Fortran::lower::mapSymbolAttributes(converter, *arg, symMap, stmtCtx);
    } else {
      // No need to create an extra hlfir.declare otherwise for
      // numerical and logical scalar dummies.
      symMap.addVariableDefinition(*arg, variableIface);
    }
  }

  // Explicitly map statement function host associated symbols to their
  // parent scope lowered symbol box.
  for (const Fortran::semantics::SymbolRef &sym :
       Fortran::evaluate::CollectSymbols(*details.stmtFunction()))
    if (const auto *details =
            sym->detailsIf<Fortran::semantics::HostAssocDetails>())
      converter.copySymbolBinding(details->symbol(), sym);

  hlfir::Entity result = Fortran::lower::convertExprToHLFIR(
      loc, converter, details.stmtFunction().value(), symMap, stmtCtx);
  symMap.popScope();
  // The result must not be a variable.
  result = hlfir::loadTrivialScalar(loc, builder, result);
  if (result.isVariable())
    result = hlfir::Entity{builder.create<hlfir::AsExprOp>(loc, result)};
  for (auto associate : exprAssociations)
    builder.create<hlfir::EndAssociateOp>(loc, associate);
  return hlfir::EntityWithAttributes{result};
}

namespace {
// Structure to hold the information about the call and the lowering context.
// This structure is intended to help threading the information
// through the various lowering calls without having to pass every
// required structure one by one.
struct CallContext {
  CallContext(const Fortran::evaluate::ProcedureRef &procRef,
              std::optional<mlir::Type> resultType, mlir::Location loc,
              Fortran::lower::AbstractConverter &converter,
              Fortran::lower::SymMap &symMap,
              Fortran::lower::StatementContext &stmtCtx)
      : procRef{procRef}, converter{converter}, symMap{symMap},
        stmtCtx{stmtCtx}, resultType{resultType}, loc{loc} {}

  fir::FirOpBuilder &getBuilder() { return converter.getFirOpBuilder(); }

  std::string getProcedureName() const {
    if (const Fortran::semantics::Symbol *sym = procRef.proc().GetSymbol())
      return sym->GetUltimate().name().ToString();
    return procRef.proc().GetName();
  }

  /// Is this a call to an elemental procedure with at least one array argument?
  bool isElementalProcWithArrayArgs() const {
    if (procRef.IsElemental())
      for (const std::optional<Fortran::evaluate::ActualArgument> &arg :
           procRef.arguments())
        if (arg && arg->Rank() != 0)
          return true;
    return false;
  }

  /// Is this a statement function reference?
  bool isStatementFunctionCall() const {
    if (const Fortran::semantics::Symbol *symbol = procRef.proc().GetSymbol())
      if (const auto *details =
              symbol->detailsIf<Fortran::semantics::SubprogramDetails>())
        return details->stmtFunction().has_value();
    return false;
  }

  /// Is this a call to a BIND(C) procedure?
  bool isBindcCall() const {
    if (const Fortran::semantics::Symbol *symbol = procRef.proc().GetSymbol())
      return Fortran::semantics::IsBindCProcedure(*symbol);
    return false;
  }

  const Fortran::evaluate::ProcedureRef &procRef;
  Fortran::lower::AbstractConverter &converter;
  Fortran::lower::SymMap &symMap;
  Fortran::lower::StatementContext &stmtCtx;
  std::optional<mlir::Type> resultType;
  mlir::Location loc;
};

using ExvAndCleanup =
    std::pair<fir::ExtendedValue, std::optional<hlfir::CleanupFunction>>;
} // namespace

// Helper to transform a fir::ExtendedValue to an hlfir::EntityWithAttributes.
static hlfir::EntityWithAttributes
extendedValueToHlfirEntity(mlir::Location loc, fir::FirOpBuilder &builder,
                           const fir::ExtendedValue &exv,
                           llvm::StringRef name) {
  mlir::Value firBase = fir::getBase(exv);
  mlir::Type firBaseTy = firBase.getType();
  if (fir::isa_trivial(firBaseTy))
    return hlfir::EntityWithAttributes{firBase};
  if (auto charTy = mlir::dyn_cast<fir::CharacterType>(firBase.getType())) {
    // CHAR() intrinsic and BIND(C) procedures returning CHARACTER(1)
    // are lowered to a fir.char<kind,1> that is not in memory.
    // This tends to cause a lot of bugs because the rest of the
    // infrastructure is mostly tested with characters that are
    // in memory.
    // To avoid having to deal with this special case here and there,
    // place it in memory here. If this turns out to be suboptimal,
    // this could be fixed, but for now llvm opt -O1 is able to get
    // rid of the memory indirection in a = char(b), so there is
    // little incentive to increase the compiler complexity.
    hlfir::Entity storage{builder.createTemporary(loc, charTy)};
    builder.create<fir::StoreOp>(loc, firBase, storage);
    auto asExpr = builder.create<hlfir::AsExprOp>(
        loc, storage, /*mustFree=*/builder.createBool(loc, false));
    return hlfir::EntityWithAttributes{asExpr.getResult()};
  }
  return hlfir::genDeclare(loc, builder, exv, name,
                           fir::FortranVariableFlagsAttr{});
}
namespace {
/// Structure to hold the clean-up related to a dummy argument preparation
/// that may have to be done after a call (copy-out or temporary deallocation).
struct CallCleanUp {
  struct CopyIn {
    void genCleanUp(mlir::Location loc, fir::FirOpBuilder &builder) {
      builder.create<hlfir::CopyOutOp>(loc, tempBox, wasCopied, copyBackVar);
    }
    // address of the descriptor holding the temp if a temp was created.
    mlir::Value tempBox;
    // Boolean indicating if a copy was made or not.
    mlir::Value wasCopied;
    // copyBackVar may be null if copy back is not needed.
    mlir::Value copyBackVar;
  };
  struct ExprAssociate {
    void genCleanUp(mlir::Location loc, fir::FirOpBuilder &builder) {
      builder.create<hlfir::EndAssociateOp>(loc, tempVar, mustFree);
    }
    mlir::Value tempVar;
    mlir::Value mustFree;
  };
  void genCleanUp(mlir::Location loc, fir::FirOpBuilder &builder) {
    Fortran::common::visit([&](auto &c) { c.genCleanUp(loc, builder); },
                           cleanUp);
  }
  std::variant<CopyIn, ExprAssociate> cleanUp;
};

/// Structure representing a prepared dummy argument.
/// It holds the value to be passed in the call and any related
/// clean-ups to be done after the call.
struct PreparedDummyArgument {
  void pushCopyInCleanUp(mlir::Value tempBox, mlir::Value wasCopied,
                         mlir::Value copyBackVar) {
    cleanups.emplace_back(
        CallCleanUp{CallCleanUp::CopyIn{tempBox, wasCopied, copyBackVar}});
  }
  void pushExprAssociateCleanUp(mlir::Value tempVar, mlir::Value wasCopied) {
    cleanups.emplace_back(
        CallCleanUp{CallCleanUp::ExprAssociate{tempVar, wasCopied}});
  }
  void pushExprAssociateCleanUp(hlfir::AssociateOp associate) {
    mlir::Value hlfirBase = associate.getBase();
    mlir::Value firBase = associate.getFirBase();
    cleanups.emplace_back(CallCleanUp{CallCleanUp::ExprAssociate{
        hlfir::mayHaveAllocatableComponent(hlfirBase.getType()) ? hlfirBase
                                                                : firBase,
        associate.getMustFreeStrorageFlag()}});
  }

  mlir::Value dummy;
  // NOTE: the clean-ups are executed in reverse order.
  llvm::SmallVector<CallCleanUp, 2> cleanups;
};

/// Structure to help conditionally preparing a dummy argument based
/// on the actual argument presence.
/// It helps "wrapping" the dummy and the clean-up information in
/// an if (present) {...}:
///
///  %conditionallyPrepared = fir.if (%present) {
///    fir.result %preparedDummy
///  } else {
///    fir.result %absent
///  }
///
struct ConditionallyPreparedDummy {
  /// Create ConditionallyPreparedDummy from a preparedDummy that must
  /// be wrapped in a fir.if.
  ConditionallyPreparedDummy(PreparedDummyArgument &preparedDummy) {
    thenResultValues.push_back(preparedDummy.dummy);
    for (const CallCleanUp &c : preparedDummy.cleanups) {
      if (const auto *copyInCleanUp =
              std::get_if<CallCleanUp::CopyIn>(&c.cleanUp)) {
        thenResultValues.push_back(copyInCleanUp->wasCopied);
        if (copyInCleanUp->copyBackVar)
          thenResultValues.push_back(copyInCleanUp->copyBackVar);
      } else {
        const auto &exprAssociate =
            std::get<CallCleanUp::ExprAssociate>(c.cleanUp);
        thenResultValues.push_back(exprAssociate.tempVar);
        thenResultValues.push_back(exprAssociate.mustFree);
      }
    }
  }

  /// Get the result types of the wrapping fir.if that must be created.
  llvm::SmallVector<mlir::Type> getIfResulTypes() const {
    llvm::SmallVector<mlir::Type> types;
    for (mlir::Value res : thenResultValues)
      types.push_back(res.getType());
    return types;
  }

  /// Generate the "fir.result %preparedDummy" in the then branch of the
  /// wrapping fir.if.
  void genThenResult(mlir::Location loc, fir::FirOpBuilder &builder) const {
    builder.create<fir::ResultOp>(loc, thenResultValues);
  }

  /// Generate the "fir.result %absent" in the else branch of the
  /// wrapping fir.if.
  void genElseResult(mlir::Location loc, fir::FirOpBuilder &builder) const {
    llvm::SmallVector<mlir::Value> elseResultValues;
    mlir::Type i1Type = builder.getI1Type();
    for (mlir::Value res : thenResultValues) {
      mlir::Type type = res.getType();
      if (type == i1Type)
        elseResultValues.push_back(builder.createBool(loc, false));
      else
        elseResultValues.push_back(builder.genAbsentOp(loc, type));
    }
    builder.create<fir::ResultOp>(loc, elseResultValues);
  }

  /// Once the fir.if has been created, get the resulting %conditionallyPrepared
  /// dummy argument.
  PreparedDummyArgument
  getPreparedDummy(fir::IfOp ifOp,
                   const PreparedDummyArgument &unconditionalDummy) {
    PreparedDummyArgument preparedDummy;
    preparedDummy.dummy = ifOp.getResults()[0];
    for (const CallCleanUp &c : unconditionalDummy.cleanups) {
      if (const auto *copyInCleanUp =
              std::get_if<CallCleanUp::CopyIn>(&c.cleanUp)) {
        mlir::Value copyBackVar;
        if (copyInCleanUp->copyBackVar)
          copyBackVar = ifOp.getResults().back();
        // tempBox is an hlfir.copy_in argument created outside of the
        // fir.if region. It needs not to be threaded as a fir.if result.
        preparedDummy.pushCopyInCleanUp(copyInCleanUp->tempBox,
                                        ifOp.getResults()[1], copyBackVar);
      } else {
        preparedDummy.pushExprAssociateCleanUp(ifOp.getResults()[1],
                                               ifOp.getResults()[2]);
      }
    }
    return preparedDummy;
  }

  llvm::SmallVector<mlir::Value> thenResultValues;
};
} // namespace

/// Fix-up the fact that it is supported to pass a character procedure
/// designator to a non character procedure dummy procedure and vice-versa, even
/// in case of explicit interface. Uglier cases where an object is passed as
/// procedure designator or vice versa are handled only for implicit interfaces
/// (refused by semantics with explicit interface), and handled with a funcOp
/// cast like other implicit interface mismatches.
static hlfir::Entity fixProcedureDummyMismatch(mlir::Location loc,
                                               fir::FirOpBuilder &builder,
                                               hlfir::Entity actual,
                                               mlir::Type dummyType) {
  if (mlir::isa<fir::BoxProcType>(actual.getType()) &&
      fir::isCharacterProcedureTuple(dummyType)) {
    mlir::Value length =
        builder.create<fir::UndefOp>(loc, builder.getCharacterLengthType());
    mlir::Value tuple = fir::factory::createCharacterProcedureTuple(
        builder, loc, dummyType, actual, length);
    return hlfir::Entity{tuple};
  }
  assert(fir::isCharacterProcedureTuple(actual.getType()) &&
         mlir::isa<fir::BoxProcType>(dummyType) &&
         "unsupported dummy procedure mismatch with the actual argument");
  mlir::Value boxProc = fir::factory::extractCharacterProcedureTuple(
                            builder, loc, actual, /*openBoxProc=*/false)
                            .first;
  return hlfir::Entity{boxProc};
}

mlir::Value static getZeroLowerBounds(mlir::Location loc,
                                      fir::FirOpBuilder &builder,
                                      hlfir::Entity entity) {
  assert(!entity.isAssumedRank() &&
         "assumed-rank must use fir.rebox_assumed_rank");
  if (entity.getRank() < 1)
    return {};
  mlir::Value zero =
      builder.createIntegerConstant(loc, builder.getIndexType(), 0);
  llvm::SmallVector<mlir::Value> lowerBounds(entity.getRank(), zero);
  return builder.genShift(loc, lowerBounds);
}

static bool
isSimplyContiguous(const Fortran::evaluate::ActualArgument &arg,
                   Fortran::evaluate::FoldingContext &foldingContext) {
  if (const auto *expr = arg.UnwrapExpr())
    return Fortran::evaluate::IsSimplyContiguous(*expr, foldingContext);
  const Fortran::semantics::Symbol *sym = arg.GetAssumedTypeDummy();
  assert(sym &&
         "expect ActualArguments to be expression or assumed-type symbols");
  return sym->Rank() == 0 ||
         Fortran::evaluate::IsSimplyContiguous(*sym, foldingContext);
}

static bool isParameterObjectOrSubObject(hlfir::Entity entity) {
  mlir::Value base = entity;
  bool foundParameter = false;
  while (mlir::Operation *op = base ? base.getDefiningOp() : nullptr) {
    base =
        llvm::TypeSwitch<mlir::Operation *, mlir::Value>(op)
            .Case<hlfir::DeclareOp>([&](auto declare) -> mlir::Value {
              foundParameter |= hlfir::Entity{declare}.isParameter();
              return foundParameter ? mlir::Value{} : declare.getMemref();
            })
            .Case<hlfir::DesignateOp, hlfir::ParentComponentOp, fir::EmboxOp>(
                [&](auto op) -> mlir::Value { return op.getMemref(); })
            .Case<fir::ReboxOp>(
                [&](auto rebox) -> mlir::Value { return rebox.getBox(); })
            .Case<fir::ConvertOp>(
                [&](auto convert) -> mlir::Value { return convert.getValue(); })
            .Default([](mlir::Operation *) -> mlir::Value { return nullptr; });
  }
  return foundParameter;
}

/// When dummy is not ALLOCATABLE, POINTER and is not passed in register,
/// prepare the actual argument according to the interface. Do as needed:
/// - address element if this is an array argument in an elemental call.
/// - set dynamic type to the dummy type if the dummy is not polymorphic.
/// - copy-in into contiguous variable if the dummy must be contiguous
/// - copy into a temporary if the dummy has the VALUE attribute.
/// - package the prepared dummy as required (fir.box, fir.class,
///   fir.box_char...).
/// This function should only be called with an actual that is present.
/// The optional aspects must be handled by this function user.
static PreparedDummyArgument preparePresentUserCallActualArgument(
    mlir::Location loc, fir::FirOpBuilder &builder,
    const Fortran::lower::PreparedActualArgument &preparedActual,
    mlir::Type dummyType,
    const Fortran::lower::CallerInterface::PassedEntity &arg,
    CallContext &callContext) {

  Fortran::evaluate::FoldingContext &foldingContext =
      callContext.converter.getFoldingContext();

  // Step 1: get the actual argument, which includes addressing the
  // element if this is an array in an elemental call.
  hlfir::Entity actual = preparedActual.getActual(loc, builder);

  // Handle procedure arguments (procedure pointers should go through
  // prepareProcedurePointerActualArgument).
  if (hlfir::isFortranProcedureValue(dummyType)) {
    // Procedure pointer or function returns procedure pointer actual to
    // procedure dummy.
    if (actual.isProcedurePointer()) {
      actual = hlfir::derefPointersAndAllocatables(loc, builder, actual);
      return PreparedDummyArgument{actual, /*cleanups=*/{}};
    }
    // Procedure actual to procedure dummy.
    assert(actual.isProcedure());
    // Do nothing if this is a procedure argument. It is already a
    // fir.boxproc/fir.tuple<fir.boxproc, len> as it should.
    if (!mlir::isa<fir::BoxProcType>(actual.getType()) &&
        actual.getType() != dummyType)
      // The actual argument may be a procedure that returns character (a
      // fir.tuple<fir.boxproc, len>) while the dummy is not. Extract the tuple
      // in that case.
      actual = fixProcedureDummyMismatch(loc, builder, actual, dummyType);
    return PreparedDummyArgument{actual, /*cleanups=*/{}};
  }

  const bool ignoreTKRtype = arg.testTKR(Fortran::common::IgnoreTKR::Type);
  const bool passingPolymorphicToNonPolymorphic =
      actual.isPolymorphic() && !fir::isPolymorphicType(dummyType) &&
      !ignoreTKRtype;

  // When passing a CLASS(T) to TYPE(T), only the "T" part must be
  // passed. Unless the entity is a scalar passed by raw address, a
  // new descriptor must be made using the dummy argument type as
  // dynamic type. This must be done before any copy/copy-in because the
  // dynamic type matters to determine the contiguity.
  const bool mustSetDynamicTypeToDummyType =
      passingPolymorphicToNonPolymorphic &&
      (actual.isArray() || mlir::isa<fir::BaseBoxType>(dummyType));

  // The simple contiguity of the actual is "lost" when passing a polymorphic
  // to a non polymorphic entity because the dummy dynamic type matters for
  // the contiguity.
  const bool mustDoCopyInOut =
      actual.isArray() && arg.mustBeMadeContiguous() &&
      (passingPolymorphicToNonPolymorphic ||
       !isSimplyContiguous(*arg.entity, foldingContext));

  const bool actualIsAssumedRank = actual.isAssumedRank();
  // Create dummy type with actual argument rank when the dummy is an assumed
  // rank. That way, all the operation to create dummy descriptors are ranked if
  // the actual argument is ranked, which allows simple code generation.
  // Also do the same when the dummy is a sequence associated descriptor
  // because the actual shape/rank may mismatch with the dummy, and the dummy
  // may be an assumed-size array, so any descriptor manipulation should use the
  // actual argument shape information. A descriptor with the dummy shape
  // information will be created later when all actual arguments are ready.
  mlir::Type dummyTypeWithActualRank = dummyType;
  if (auto baseBoxDummy = mlir::dyn_cast<fir::BaseBoxType>(dummyType)) {
    if (baseBoxDummy.isAssumedRank() ||
        arg.testTKR(Fortran::common::IgnoreTKR::Rank) ||
        arg.isSequenceAssociatedDescriptor()) {
      mlir::Type actualTy =
          hlfir::getFortranElementOrSequenceType(actual.getType());
      dummyTypeWithActualRank = baseBoxDummy.getBoxTypeWithNewShape(actualTy);
    }
  }
  // Preserve the actual type in the argument preparation in case IgnoreTKR(t)
  // is set (descriptors must be created with the actual type in this case, and
  // copy-in/copy-out should be driven by the contiguity with regard to the
  // actual type).
  if (ignoreTKRtype) {
    if (auto boxCharType =
            mlir::dyn_cast<fir::BoxCharType>(dummyTypeWithActualRank)) {
      auto maybeActualCharType =
          mlir::dyn_cast<fir::CharacterType>(actual.getFortranElementType());
      if (!maybeActualCharType ||
          maybeActualCharType.getFKind() != boxCharType.getKind()) {
        // When passing to a fir.boxchar with ignore(tk), prepare the argument
        // as if only the raw address must be passed.
        dummyTypeWithActualRank =
            fir::ReferenceType::get(actual.getElementOrSequenceType());
      }
      // Otherwise, the actual is already a character with the same kind as the
      // dummy and can be passed normally.
    } else {
      dummyTypeWithActualRank = fir::changeElementType(
          dummyTypeWithActualRank, actual.getFortranElementType(),
          actual.isPolymorphic());
    }
  }

  PreparedDummyArgument preparedDummy;

  // Helpers to generate hlfir.copy_in operation and register the related
  // hlfir.copy_out creation.
  auto genCopyIn = [&](hlfir::Entity var, bool doCopyOut) -> hlfir::Entity {
    auto baseBoxTy = mlir::dyn_cast<fir::BaseBoxType>(var.getType());
    assert(baseBoxTy && "expect non simply contiguous variables to be boxes");
    // Create allocatable descriptor for the potential temporary.
    mlir::Type tempBoxType = baseBoxTy.getBoxTypeWithNewAttr(
        fir::BaseBoxType::Attribute::Allocatable);
    mlir::Value tempBox = builder.createTemporary(loc, tempBoxType);
    auto copyIn = builder.create<hlfir::CopyInOp>(
        loc, var, tempBox, /*var_is_present=*/mlir::Value{});
    // Register the copy-out after the call.
    preparedDummy.pushCopyInCleanUp(copyIn.getTempBox(), copyIn.getWasCopied(),
                                    doCopyOut ? copyIn.getVar()
                                              : mlir::Value{});
    return hlfir::Entity{copyIn.getCopiedIn()};
  };

  auto genSetDynamicTypeToDummyType = [&](hlfir::Entity var) -> hlfir::Entity {
    fir::BaseBoxType boxType = fir::BoxType::get(
        hlfir::getFortranElementOrSequenceType(dummyTypeWithActualRank));
    if (actualIsAssumedRank)
      return hlfir::Entity{builder.create<fir::ReboxAssumedRankOp>(
          loc, boxType, var, fir::LowerBoundModifierAttribute::SetToOnes)};
    // Use actual shape when creating descriptor with dummy type, the dummy
    // shape may be unknown in case of sequence association.
    mlir::Type actualTy =
        hlfir::getFortranElementOrSequenceType(actual.getType());
    boxType = boxType.getBoxTypeWithNewShape(actualTy);
    return hlfir::Entity{builder.create<fir::ReboxOp>(loc, boxType, var,
                                                      /*shape=*/mlir::Value{},
                                                      /*slice=*/mlir::Value{})};
  };

  // Step 2: prepare the storage for the dummy arguments, ensuring that it
  // matches the dummy requirements (e.g., must be contiguous or must be
  // a temporary).
  hlfir::Entity entity =
      hlfir::derefPointersAndAllocatables(loc, builder, actual);
  if (entity.isVariable()) {
    // Set dynamic type if needed before any copy-in or copy so that the dummy
    // is contiguous according to the dummy type.
    if (mustSetDynamicTypeToDummyType)
      entity = genSetDynamicTypeToDummyType(entity);
    if (arg.hasValueAttribute() ||
        // Constant expressions might be lowered as variables with
        // 'parameter' attribute. Even though the constant expressions
        // are not definable and explicit assignments to them are not
        // possible, we have to create a temporary copies when we pass
        // them down the call stack because of potential compiler
        // generated writes in copy-out.
        isParameterObjectOrSubObject(entity)) {
      // Make a copy in a temporary.
      auto copy = builder.create<hlfir::AsExprOp>(loc, entity);
      mlir::Type storageType = entity.getType();
      mlir::NamedAttribute byRefAttr = fir::getAdaptToByRefAttr(builder);
      hlfir::AssociateOp associate = hlfir::genAssociateExpr(
          loc, builder, hlfir::Entity{copy}, storageType, "", byRefAttr);
      entity = hlfir::Entity{associate.getBase()};
      // Register the temporary destruction after the call.
      preparedDummy.pushExprAssociateCleanUp(associate);
    } else if (mustDoCopyInOut) {
      // Copy-in non contiguous variables.
      // TODO: for non-finalizable monomorphic derived type actual
      // arguments associated with INTENT(OUT) dummy arguments
      // we may avoid doing the copy and only allocate the temporary.
      // The codegen would do a "mold" allocation instead of "sourced"
      // allocation for the temp in this case. We can communicate
      // this to the codegen via some CopyInOp flag.
      // This is a performance concern.
      entity = genCopyIn(entity, arg.mayBeModifiedByCall());
    }
  } else {
    const Fortran::lower::SomeExpr *expr = arg.entity->UnwrapExpr();
    assert(expr && "expression actual argument cannot be an assumed type");
    // The actual is an expression value, place it into a temporary
    // and register the temporary destruction after the call.
    mlir::Type storageType = callContext.converter.genType(*expr);
    mlir::NamedAttribute byRefAttr = fir::getAdaptToByRefAttr(builder);
    hlfir::AssociateOp associate = hlfir::genAssociateExpr(
        loc, builder, entity, storageType, "", byRefAttr);
    entity = hlfir::Entity{associate.getBase()};
    preparedDummy.pushExprAssociateCleanUp(associate);
    // Rebox the actual argument to the dummy argument's type, and make sure
    // that we pass a contiguous entity (i.e. make copy-in, if needed).
    //
    // TODO: this can probably be optimized by associating the expression with
    // properly typed temporary, but this needs either a new operation or
    // making the hlfir.associate more complex.
    if (mustSetDynamicTypeToDummyType) {
      entity = genSetDynamicTypeToDummyType(entity);
      entity = genCopyIn(entity, /*doCopyOut=*/false);
    }
  }

  // Step 3: now that the dummy argument storage has been prepared, package
  // it according to the interface.
  mlir::Value addr;
  if (mlir::isa<fir::BoxCharType>(dummyTypeWithActualRank)) {
    addr = hlfir::genVariableBoxChar(loc, builder, entity);
  } else if (mlir::isa<fir::BaseBoxType>(dummyTypeWithActualRank)) {
    entity = hlfir::genVariableBox(loc, builder, entity);
    // Ensures the box has the right attributes and that it holds an
    // addendum if needed.
    fir::BaseBoxType actualBoxType =
        mlir::cast<fir::BaseBoxType>(entity.getType());
    mlir::Type boxEleType = actualBoxType.getEleTy();
    // For now, assume it is not OK to pass the allocatable/pointer
    // descriptor to a non pointer/allocatable dummy. That is a strict
    // interpretation of 18.3.6 point 4 that stipulates the descriptor
    // has the dummy attributes in BIND(C) contexts.
    const bool actualBoxHasAllocatableOrPointerFlag =
        fir::isa_ref_type(boxEleType);
    // Fortran 2018 18.5.3, pp3: BIND(C) non pointer allocatable descriptors
    // must have zero lower bounds.
    bool needsZeroLowerBounds = callContext.isBindcCall() && entity.isArray();
    // On the callee side, the current code generated for unlimited
    // polymorphic might unconditionally read the addendum. Intrinsic type
    // descriptors may not have an addendum, the rebox below will create a
    // descriptor with an addendum in such case.
    const bool actualBoxHasAddendum = fir::boxHasAddendum(actualBoxType);
    const bool needToAddAddendum =
        fir::isUnlimitedPolymorphicType(dummyTypeWithActualRank) &&
        !actualBoxHasAddendum;
    if (needToAddAddendum || actualBoxHasAllocatableOrPointerFlag ||
        needsZeroLowerBounds) {
      if (actualIsAssumedRank) {
        auto lbModifier = needsZeroLowerBounds
                              ? fir::LowerBoundModifierAttribute::SetToZeroes
                              : fir::LowerBoundModifierAttribute::SetToOnes;
        entity = hlfir::Entity{builder.create<fir::ReboxAssumedRankOp>(
            loc, dummyTypeWithActualRank, entity, lbModifier)};
      } else {
        mlir::Value shift{};
        if (needsZeroLowerBounds)
          shift = getZeroLowerBounds(loc, builder, entity);
        entity = hlfir::Entity{builder.create<fir::ReboxOp>(
            loc, dummyTypeWithActualRank, entity, /*shape=*/shift,
            /*slice=*/mlir::Value{})};
      }
    }
    addr = entity;
  } else {
    addr = hlfir::genVariableRawAddress(loc, builder, entity);
  }

  // For ranked actual passed to assumed-rank dummy, the cast to assumed-rank
  // box is inserted when building the fir.call op. Inserting it here would
  // cause the fir.if results to be assumed-rank in case of OPTIONAL dummy,
  // causing extra runtime costs due to the unknown runtime size of assumed-rank
  // descriptors.
  preparedDummy.dummy =
      builder.createConvert(loc, dummyTypeWithActualRank, addr);
  return preparedDummy;
}

/// When dummy is not ALLOCATABLE, POINTER and is not passed in register,
/// prepare the actual argument according to the interface, taking care
/// of any optional aspect.
static PreparedDummyArgument prepareUserCallActualArgument(
    mlir::Location loc, fir::FirOpBuilder &builder,
    const Fortran::lower::PreparedActualArgument &preparedActual,
    mlir::Type dummyType,
    const Fortran::lower::CallerInterface::PassedEntity &arg,
    CallContext &callContext) {
  if (!preparedActual.handleDynamicOptional())
    return preparePresentUserCallActualArgument(loc, builder, preparedActual,
                                                dummyType, arg, callContext);

  // Conditional dummy argument preparation. The actual may be absent
  // at runtime, causing any addressing, copy, and packaging to have
  // undefined behavior.
  // To simplify the handling of this case, the "normal" dummy preparation
  // helper is used, except its generated code is wrapped inside a
  // fir.if(present).
  mlir::Value isPresent = preparedActual.getIsPresent();
  mlir::OpBuilder::InsertPoint insertPt = builder.saveInsertionPoint();

  // Code generated in a preparation block that will become the
  // "then" block in "if (present) then {} else {}". The reason
  // for this unusual if/then/else generation is that the number
  // and types of the if results will depend on how the argument
  // is prepared, and forecasting that here would be brittle.
  auto badIfOp = builder.create<fir::IfOp>(loc, dummyType, isPresent,
                                           /*withElseRegion=*/false);
  mlir::Block *preparationBlock = &badIfOp.getThenRegion().front();
  builder.setInsertionPointToStart(preparationBlock);
  PreparedDummyArgument unconditionalDummy =
      preparePresentUserCallActualArgument(loc, builder, preparedActual,
                                           dummyType, arg, callContext);
  builder.restoreInsertionPoint(insertPt);

  // TODO: when forwarding an optional to an optional of the same kind
  // (i.e, unconditionalDummy.dummy was not created in preparationBlock),
  // the if/then/else generation could be skipped to improve the generated
  // code.

  // Now that the result types of the ifOp can be deduced, generate
  // the "real" ifOp (operation result types cannot be changed, so
  // badIfOp cannot be modified and used here).
  llvm::SmallVector<mlir::Type> ifOpResultTypes;
  ConditionallyPreparedDummy conditionalDummy(unconditionalDummy);
  auto ifOp = builder.create<fir::IfOp>(loc, conditionalDummy.getIfResulTypes(),
                                        isPresent,
                                        /*withElseRegion=*/true);
  // Move "preparationBlock" into the "then" of the new
  // fir.if operation and create fir.result propagating
  // unconditionalDummy.
  preparationBlock->moveBefore(&ifOp.getThenRegion().back());
  ifOp.getThenRegion().back().erase();
  builder.setInsertionPointToEnd(&ifOp.getThenRegion().front());
  conditionalDummy.genThenResult(loc, builder);

  // Generate "else" branch with returning absent values.
  builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
  conditionalDummy.genElseResult(loc, builder);

  // Build dummy from IfOpResults.
  builder.setInsertionPointAfter(ifOp);
  PreparedDummyArgument result =
      conditionalDummy.getPreparedDummy(ifOp, unconditionalDummy);
  badIfOp->erase();
  return result;
}

/// Prepare actual argument for a procedure pointer dummy.
static PreparedDummyArgument prepareProcedurePointerActualArgument(
    mlir::Location loc, fir::FirOpBuilder &builder,
    const Fortran::lower::PreparedActualArgument &preparedActual,
    mlir::Type dummyType,
    const Fortran::lower::CallerInterface::PassedEntity &arg,
    CallContext &callContext) {

  // NULL() actual to procedure pointer dummy
  if (Fortran::evaluate::UnwrapExpr<Fortran::evaluate::NullPointer>(
          *arg.entity) &&
      fir::isBoxProcAddressType(dummyType)) {
    auto boxTy{Fortran::lower::getUntypedBoxProcType(builder.getContext())};
    auto tempBoxProc{builder.createTemporary(loc, boxTy)};
    hlfir::Entity nullBoxProc(
        fir::factory::createNullBoxProc(builder, loc, boxTy));
    builder.create<fir::StoreOp>(loc, nullBoxProc, tempBoxProc);
    return PreparedDummyArgument{tempBoxProc, /*cleanups=*/{}};
  }
  hlfir::Entity actual = preparedActual.getActual(loc, builder);
  if (actual.isProcedurePointer())
    return PreparedDummyArgument{actual, /*cleanups=*/{}};
  assert(actual.isProcedure());
  // Procedure actual to procedure pointer dummy.
  auto tempBoxProc{builder.createTemporary(loc, actual.getType())};
  builder.create<fir::StoreOp>(loc, actual, tempBoxProc);
  return PreparedDummyArgument{tempBoxProc, /*cleanups=*/{}};
}

/// Prepare arguments of calls to user procedures with actual arguments that
/// have been pre-lowered but not yet prepared according to the interface.
void prepareUserCallArguments(
    Fortran::lower::PreparedActualArguments &loweredActuals,
    Fortran::lower::CallerInterface &caller, mlir::FunctionType callSiteType,
    CallContext &callContext, llvm::SmallVector<CallCleanUp> &callCleanUps) {
  using PassBy = Fortran::lower::CallerInterface::PassEntityBy;
  mlir::Location loc = callContext.loc;
  bool mustRemapActualToDummyDescriptors = false;
  fir::FirOpBuilder &builder = callContext.getBuilder();
  for (auto [preparedActual, arg] :
       llvm::zip(loweredActuals, caller.getPassedArguments())) {
    mlir::Type argTy = callSiteType.getInput(arg.firArgument);
    if (!preparedActual) {
      // Optional dummy argument for which there is no actual argument.
      caller.placeInput(arg, builder.genAbsentOp(loc, argTy));
      continue;
    }

    switch (arg.passBy) {
    case PassBy::Value: {
      // True pass-by-value semantics.
      assert(!preparedActual->handleDynamicOptional() && "cannot be optional");
      hlfir::Entity actual = preparedActual->getActual(loc, builder);
      hlfir::Entity value = hlfir::loadTrivialScalar(loc, builder, actual);

      mlir::Type eleTy = value.getFortranElementType();
      if (fir::isa_builtin_cptr_type(eleTy)) {
        // Pass-by-value argument of type(C_PTR/C_FUNPTR).
        // Load the __address component and pass it by value.
        if (value.isValue()) {
          auto associate = hlfir::genAssociateExpr(loc, builder, value, eleTy,
                                                   "adapt.cptrbyval");
          value = hlfir::Entity{genRecordCPtrValueArg(
              builder, loc, associate.getFirBase(), eleTy)};
          builder.create<hlfir::EndAssociateOp>(loc, associate);
        } else {
          value =
              hlfir::Entity{genRecordCPtrValueArg(builder, loc, value, eleTy)};
        }
      } else if (fir::isa_derived(value.getFortranElementType()) ||
                 value.isCharacter()) {
        // BIND(C), VALUE derived type or character. The value must really
        // be loaded here.
        auto [exv, cleanup] = hlfir::convertToValue(loc, builder, value);
        mlir::Value loadedValue = fir::getBase(exv);
        // Character actual arguments may have unknown length or a length longer
        // than one. Cast the memory ref to the dummy type so that the load is
        // valid and only loads what is needed.
        if (mlir::Type baseTy = fir::dyn_cast_ptrEleTy(loadedValue.getType()))
          if (fir::isa_char(baseTy))
            loadedValue = builder.createConvert(
                loc, fir::ReferenceType::get(argTy), loadedValue);
        if (fir::isa_ref_type(loadedValue.getType()))
          loadedValue = builder.create<fir::LoadOp>(loc, loadedValue);
        caller.placeInput(arg, loadedValue);
        if (cleanup)
          (*cleanup)();
        break;
      }
      caller.placeInput(arg, builder.createConvert(loc, argTy, value));
    } break;
    case PassBy::BaseAddressValueAttribute:
    case PassBy::CharBoxValueAttribute:
    case PassBy::Box:
    case PassBy::BaseAddress:
    case PassBy::BoxChar: {
      PreparedDummyArgument preparedDummy = prepareUserCallActualArgument(
          loc, builder, *preparedActual, argTy, arg, callContext);
      callCleanUps.append(preparedDummy.cleanups.rbegin(),
                          preparedDummy.cleanups.rend());
      caller.placeInput(arg, preparedDummy.dummy);
      if (arg.passBy == PassBy::Box)
        mustRemapActualToDummyDescriptors |=
            arg.isSequenceAssociatedDescriptor();
    } break;
    case PassBy::BoxProcRef: {
      PreparedDummyArgument preparedDummy =
          prepareProcedurePointerActualArgument(loc, builder, *preparedActual,
                                                argTy, arg, callContext);
      callCleanUps.append(preparedDummy.cleanups.rbegin(),
                          preparedDummy.cleanups.rend());
      caller.placeInput(arg, preparedDummy.dummy);
    } break;
    case PassBy::AddressAndLength:
      // PassBy::AddressAndLength is only used for character results. Results
      // are not handled here.
      fir::emitFatalError(
          loc, "unexpected PassBy::AddressAndLength for actual arguments");
      break;
    case PassBy::CharProcTuple: {
      hlfir::Entity actual = preparedActual->getActual(loc, builder);
      if (actual.isProcedurePointer())
        actual = hlfir::derefPointersAndAllocatables(loc, builder, actual);
      if (!fir::isCharacterProcedureTuple(actual.getType()))
        actual = fixProcedureDummyMismatch(loc, builder, actual, argTy);
      caller.placeInput(arg, actual);
    } break;
    case PassBy::MutableBox: {
      const Fortran::lower::SomeExpr *expr = arg.entity->UnwrapExpr();
      // C709 and C710.
      assert(expr && "cannot pass TYPE(*) to POINTER or ALLOCATABLE");
      hlfir::Entity actual = preparedActual->getActual(loc, builder);
      if (Fortran::evaluate::UnwrapExpr<Fortran::evaluate::NullPointer>(
              *expr)) {
        // If expr is NULL(), the mutableBox created must be a deallocated
        // pointer with the dummy argument characteristics (see table 16.5
        // in Fortran 2018 standard).
        // No length parameters are set for the created box because any non
        // deferred type parameters of the dummy will be evaluated on the
        // callee side, and it is illegal to use NULL without a MOLD if any
        // dummy length parameters are assumed.
        mlir::Type boxTy = fir::dyn_cast_ptrEleTy(argTy);
        assert(boxTy && mlir::isa<fir::BaseBoxType>(boxTy) &&
               "must be a fir.box type");
        mlir::Value boxStorage =
            fir::factory::genNullBoxStorage(builder, loc, boxTy);
        caller.placeInput(arg, boxStorage);
        continue;
      }
      if (fir::isPointerType(argTy) &&
          !Fortran::evaluate::IsObjectPointer(*expr)) {
        // Passing a non POINTER actual argument to a POINTER dummy argument.
        // Create a pointer of the dummy argument type and assign the actual
        // argument to it.
        auto dataTy = llvm::cast<fir::BaseBoxType>(fir::unwrapRefType(argTy));
        fir::ExtendedValue actualExv = Fortran::lower::convertToAddress(
            loc, callContext.converter, actual, callContext.stmtCtx,
            hlfir::getFortranElementType(dataTy));
        // If the dummy is an assumed-rank pointer, allocate a pointer
        // descriptor with the actual argument rank (if it is not assumed-rank
        // itself).
        if (dataTy.isAssumedRank()) {
          dataTy =
              dataTy.getBoxTypeWithNewShape(fir::getBase(actualExv).getType());
        }
        mlir::Value irBox = builder.createTemporary(loc, dataTy);
        fir::MutableBoxValue ptrBox(irBox,
                                    /*nonDeferredParams=*/mlir::ValueRange{},
                                    /*mutableProperties=*/{});
        fir::factory::associateMutableBox(builder, loc, ptrBox, actualExv,
                                          /*lbounds=*/std::nullopt);
        caller.placeInput(arg, irBox);
        continue;
      }
      // Passing a POINTER to a POINTER, or an ALLOCATABLE to an ALLOCATABLE.
      assert(actual.isMutableBox() && "actual must be a mutable box");
      if (fir::isAllocatableType(argTy) && arg.isIntentOut() &&
          callContext.isBindcCall()) {
        // INTENT(OUT) allocatables are deallocated on the callee side,
        // but BIND(C) procedures may be implemented in C, so deallocation is
        // also done on the caller side (if the procedure is implemented in
        // Fortran, the deallocation attempt in the callee will be a no-op).
        auto [exv, cleanup] =
            hlfir::translateToExtendedValue(loc, builder, actual);
        const auto *mutableBox = exv.getBoxOf<fir::MutableBoxValue>();
        assert(mutableBox && !cleanup && "expect allocatable");
        Fortran::lower::genDeallocateIfAllocated(callContext.converter,
                                                 *mutableBox, loc);
      }
      caller.placeInput(arg, actual);
    } break;
    }
  }

  // Handle cases where caller must allocate the result or a fir.box for it.
  if (mustRemapActualToDummyDescriptors)
    remapActualToDummyDescriptors(loc, callContext.converter,
                                  callContext.symMap, loweredActuals, caller,
                                  callContext.isBindcCall());
}

/// Lower calls to user procedures with actual arguments that have been
/// pre-lowered but not yet prepared according to the interface.
/// This can be called for elemental procedures, but only with scalar
/// arguments: if there are array arguments, it must be provided with
/// the array argument elements value and will return the corresponding
/// scalar result value.
static std::optional<hlfir::EntityWithAttributes>
genUserCall(Fortran::lower::PreparedActualArguments &loweredActuals,
            Fortran::lower::CallerInterface &caller,
            mlir::FunctionType callSiteType, CallContext &callContext) {
  mlir::Location loc = callContext.loc;
  llvm::SmallVector<CallCleanUp> callCleanUps;
  fir::FirOpBuilder &builder = callContext.getBuilder();

  prepareUserCallArguments(loweredActuals, caller, callSiteType, callContext,
                           callCleanUps);

  // Prepare lowered arguments according to the interface
  // and map the lowered values to the dummy
  // arguments.
  auto [loweredResult, resultIsFinalized] = Fortran::lower::genCallOpAndResult(
      loc, callContext.converter, callContext.symMap, callContext.stmtCtx,
      caller, callSiteType, callContext.resultType,
      callContext.isElementalProcWithArrayArgs());

  /// Clean-up associations and copy-in.
  for (auto cleanUp : callCleanUps)
    cleanUp.genCleanUp(loc, builder);

  if (auto *entity = std::get_if<hlfir::EntityWithAttributes>(&loweredResult))
    return *entity;

  auto &result = std::get<fir::ExtendedValue>(loweredResult);

  // For procedure pointer function result, just return the call.
  if (callContext.resultType &&
      mlir::isa<fir::BoxProcType>(*callContext.resultType))
    return hlfir::EntityWithAttributes(fir::getBase(result));

  if (!fir::getBase(result))
    return std::nullopt; // subroutine call.

  if (fir::isPointerType(fir::getBase(result).getType()))
    return extendedValueToHlfirEntity(loc, builder, result, tempResultName);

  if (!resultIsFinalized) {
    hlfir::Entity resultEntity =
        extendedValueToHlfirEntity(loc, builder, result, tempResultName);
    resultEntity = loadTrivialScalar(loc, builder, resultEntity);
    if (resultEntity.isVariable()) {
      // If the result has no finalization, it can be moved into an expression.
      // In such case, the expression should not be freed after its use since
      // the result is stack allocated or deallocation (for allocatable results)
      // was already inserted in genCallOpAndResult.
      auto asExpr = builder.create<hlfir::AsExprOp>(
          loc, resultEntity, /*mustFree=*/builder.createBool(loc, false));
      return hlfir::EntityWithAttributes{asExpr.getResult()};
    }
    return hlfir::EntityWithAttributes{resultEntity};
  }
  // If the result has finalization, it cannot be moved because use of its
  // value have been created in the statement context and may be emitted
  // after the hlfir.expr destroy, so the result is kept as a variable in
  // HLFIR. This may lead to copies when passing the result to an argument
  // with VALUE, and this do not convey the fact that the result will not
  // change, but is correct, and using hlfir.expr without the move would
  // trigger a copy that may be avoided.

  // Load allocatable results before emitting the hlfir.declare and drop its
  // lower bounds: this is not a variable From the Fortran point of view, so
  // the lower bounds are ones when inquired on the caller side.
  const auto *allocatable = result.getBoxOf<fir::MutableBoxValue>();
  fir::ExtendedValue loadedResult =
      allocatable
          ? fir::factory::genMutableBoxRead(builder, loc, *allocatable,
                                            /*mayBePolymorphic=*/true,
                                            /*preserveLowerBounds=*/false)
          : result;
  return extendedValueToHlfirEntity(loc, builder, loadedResult, tempResultName);
}

/// Create an optional dummy argument value from an entity that may be
/// absent. \p actualGetter callback returns hlfir::Entity denoting
/// the lowered actual argument. \p actualGetter can only return numerical
/// or logical scalar entity.
/// If the entity is considered absent according to 15.5.2.12 point 1., the
/// returned value is zero (or false), otherwise it is the value of the entity.
/// \p eleType specifies the entity's Fortran element type.
template <typename T>
static ExvAndCleanup genOptionalValue(fir::FirOpBuilder &builder,
                                      mlir::Location loc, mlir::Type eleType,
                                      T actualGetter, mlir::Value isPresent) {
  return {builder
              .genIfOp(loc, {eleType}, isPresent,
                       /*withElseRegion=*/true)
              .genThen([&]() {
                hlfir::Entity entity = actualGetter(loc, builder);
                assert(eleType == entity.getFortranElementType() &&
                       "result type mismatch in genOptionalValue");
                assert(entity.isScalar() && fir::isa_trivial(eleType) &&
                       "must be a numerical or logical scalar");
                mlir::Value val =
                    hlfir::loadTrivialScalar(loc, builder, entity);
                builder.create<fir::ResultOp>(loc, val);
              })
              .genElse([&]() {
                mlir::Value zero =
                    fir::factory::createZeroValue(builder, loc, eleType);
                builder.create<fir::ResultOp>(loc, zero);
              })
              .getResults()[0],
          std::nullopt};
}

/// Create an optional dummy argument address from \p entity that may be
/// absent. If \p entity is considered absent according to 15.5.2.12 point 1.,
/// the returned value is a null pointer, otherwise it is the address of \p
/// entity.
static ExvAndCleanup genOptionalAddr(fir::FirOpBuilder &builder,
                                     mlir::Location loc, hlfir::Entity entity,
                                     mlir::Value isPresent) {
  auto [exv, cleanup] = hlfir::translateToExtendedValue(loc, builder, entity);
  // If it is an exv pointer/allocatable, then it cannot be absent
  // because it is passed to a non-pointer/non-allocatable.
  if (const auto *box = exv.getBoxOf<fir::MutableBoxValue>())
    return {fir::factory::genMutableBoxRead(builder, loc, *box), cleanup};
  // If this is not a POINTER or ALLOCATABLE, then it is already an OPTIONAL
  // address and can be passed directly.
  return {exv, cleanup};
}

/// Create an optional dummy argument address from \p entity that may be
/// absent. If \p entity is considered absent according to 15.5.2.12 point 1.,
/// the returned value is an absent fir.box, otherwise it is a fir.box
/// describing \p entity.
static ExvAndCleanup genOptionalBox(fir::FirOpBuilder &builder,
                                    mlir::Location loc, hlfir::Entity entity,
                                    mlir::Value isPresent) {
  auto [exv, cleanup] = hlfir::translateToExtendedValue(loc, builder, entity);

  // Non allocatable/pointer optional box -> simply forward
  if (exv.getBoxOf<fir::BoxValue>())
    return {exv, cleanup};

  fir::ExtendedValue newExv = exv;
  // Optional allocatable/pointer -> Cannot be absent, but need to translate
  // unallocated/diassociated into absent fir.box.
  if (const auto *box = exv.getBoxOf<fir::MutableBoxValue>())
    newExv = fir::factory::genMutableBoxRead(builder, loc, *box);

  // createBox will not do create any invalid memory dereferences if exv is
  // absent. The created fir.box will not be usable, but the SelectOp below
  // ensures it won't be.
  mlir::Value box = builder.createBox(loc, newExv);
  mlir::Type boxType = box.getType();
  auto absent = builder.create<fir::AbsentOp>(loc, boxType);
  auto boxOrAbsent = builder.create<mlir::arith::SelectOp>(
      loc, boxType, isPresent, box, absent);
  return {fir::BoxValue(boxOrAbsent), cleanup};
}

/// Lower calls to intrinsic procedures with custom optional handling where the
/// actual arguments have been pre-lowered
static std::optional<hlfir::EntityWithAttributes> genCustomIntrinsicRefCore(
    Fortran::lower::PreparedActualArguments &loweredActuals,
    const Fortran::evaluate::SpecificIntrinsic *intrinsic,
    CallContext &callContext) {
  auto &builder = callContext.getBuilder();
  const auto &loc = callContext.loc;
  assert(intrinsic &&
         Fortran::lower::intrinsicRequiresCustomOptionalHandling(
             callContext.procRef, *intrinsic, callContext.converter));

  // helper to get a particular prepared argument
  auto getArgument = [&](std::size_t i, bool loadArg) -> fir::ExtendedValue {
    if (!loweredActuals[i])
      return fir::getAbsentIntrinsicArgument();
    hlfir::Entity actual = loweredActuals[i]->getActual(loc, builder);
    if (loadArg && fir::conformsWithPassByRef(actual.getType())) {
      return hlfir::loadTrivialScalar(loc, builder, actual);
    }
    return Fortran::lower::translateToExtendedValue(loc, builder, actual,
                                                    callContext.stmtCtx);
  };
  // helper to get the isPresent flag for a particular prepared argument
  auto isPresent = [&](std::size_t i) -> std::optional<mlir::Value> {
    if (!loweredActuals[i])
      return {builder.createBool(loc, false)};
    if (loweredActuals[i]->handleDynamicOptional())
      return {loweredActuals[i]->getIsPresent()};
    return std::nullopt;
  };

  assert(callContext.resultType &&
         "the elemental intrinsics with custom handling are all functions");
  // if callContext.resultType is an array then this was originally an elemental
  // call. What we are lowering here is inside the kernel of the hlfir.elemental
  // so we should return the scalar type. If the return type is already a scalar
  // then it should be unchanged here.
  mlir::Type resTy = hlfir::getFortranElementType(*callContext.resultType);
  fir::ExtendedValue result = Fortran::lower::lowerCustomIntrinsic(
      builder, loc, callContext.getProcedureName(), resTy, isPresent,
      getArgument, loweredActuals.size(), callContext.stmtCtx);

  return {hlfir::EntityWithAttributes{extendedValueToHlfirEntity(
      loc, builder, result, ".tmp.custom_intrinsic_result")}};
}

/// Lower calls to intrinsic procedures with actual arguments that have been
/// pre-lowered but have not yet been prepared according to the interface.
static std::optional<hlfir::EntityWithAttributes>
genIntrinsicRefCore(Fortran::lower::PreparedActualArguments &loweredActuals,
                    const Fortran::evaluate::SpecificIntrinsic *intrinsic,
                    const fir::IntrinsicHandlerEntry &intrinsicEntry,
                    CallContext &callContext) {
  auto &converter = callContext.converter;
  if (intrinsic && Fortran::lower::intrinsicRequiresCustomOptionalHandling(
                       callContext.procRef, *intrinsic, converter))
    return genCustomIntrinsicRefCore(loweredActuals, intrinsic, callContext);
  llvm::SmallVector<fir::ExtendedValue> operands;
  llvm::SmallVector<hlfir::CleanupFunction> cleanupFns;
  auto addToCleanups = [&cleanupFns](std::optional<hlfir::CleanupFunction> fn) {
    if (fn)
      cleanupFns.emplace_back(std::move(*fn));
  };
  auto &stmtCtx = callContext.stmtCtx;
  fir::FirOpBuilder &builder = callContext.getBuilder();
  mlir::Location loc = callContext.loc;
  const fir::IntrinsicArgumentLoweringRules *argLowering =
      intrinsicEntry.getArgumentLoweringRules();
  for (auto arg : llvm::enumerate(loweredActuals)) {
    if (!arg.value()) {
      operands.emplace_back(fir::getAbsentIntrinsicArgument());
      continue;
    }
    if (!argLowering) {
      // No argument lowering instruction, lower by value.
      assert(!arg.value()->handleDynamicOptional() &&
             "should use genOptionalValue");
      hlfir::Entity actual = arg.value()->getActual(loc, builder);
      operands.emplace_back(
          Fortran::lower::convertToValue(loc, converter, actual, stmtCtx));
      continue;
    }
    // Helper to get the type of the Fortran expression in case it is a
    // computed value that must be placed in memory (logicals are computed as
    // i1, but must be placed in memory as fir.logical).
    auto getActualFortranElementType = [&]() -> mlir::Type {
      if (const Fortran::lower::SomeExpr *expr =
              callContext.procRef.UnwrapArgExpr(arg.index())) {

        mlir::Type type = converter.genType(*expr);
        return hlfir::getFortranElementType(type);
      }
      // TYPE(*): is already in memory anyway. Can return none
      // here.
      return builder.getNoneType();
    };
    // Ad-hoc argument lowering handling.
    fir::ArgLoweringRule argRules =
        fir::lowerIntrinsicArgumentAs(*argLowering, arg.index());
    if (arg.value()->handleDynamicOptional()) {
      mlir::Value isPresent = arg.value()->getIsPresent();
      switch (argRules.lowerAs) {
      case fir::LowerIntrinsicArgAs::Value: {
        // In case of elemental call, getActual() may produce
        // a designator denoting the array element to be passed
        // to the subprogram. If the actual array is dynamically
        // optional the designator must be generated under
        // isPresent check, because the box bounds reads will be
        // generated in the codegen. These reads are illegal,
        // if the dynamically optional argument is absent.
        auto getActualCb = [&](mlir::Location loc,
                               fir::FirOpBuilder &builder) -> hlfir::Entity {
          return arg.value()->getActual(loc, builder);
        };
        auto [exv, cleanup] =
            genOptionalValue(builder, loc, getActualFortranElementType(),
                             getActualCb, isPresent);
        addToCleanups(std::move(cleanup));
        operands.emplace_back(exv);
        continue;
      }
      case fir::LowerIntrinsicArgAs::Addr: {
        hlfir::Entity actual = arg.value()->getActual(loc, builder);
        auto [exv, cleanup] = genOptionalAddr(builder, loc, actual, isPresent);
        addToCleanups(std::move(cleanup));
        operands.emplace_back(exv);
        continue;
      }
      case fir::LowerIntrinsicArgAs::Box: {
        hlfir::Entity actual = arg.value()->getActual(loc, builder);
        auto [exv, cleanup] = genOptionalBox(builder, loc, actual, isPresent);
        addToCleanups(std::move(cleanup));
        operands.emplace_back(exv);
        continue;
      }
      case fir::LowerIntrinsicArgAs::Inquired: {
        hlfir::Entity actual = arg.value()->getActual(loc, builder);
        auto [exv, cleanup] =
            hlfir::translateToExtendedValue(loc, builder, actual);
        addToCleanups(std::move(cleanup));
        operands.emplace_back(exv);
        continue;
      }
      }
      llvm_unreachable("bad switch");
    }

    hlfir::Entity actual = arg.value()->getActual(loc, builder);
    switch (argRules.lowerAs) {
    case fir::LowerIntrinsicArgAs::Value:
      operands.emplace_back(
          Fortran::lower::convertToValue(loc, converter, actual, stmtCtx));
      continue;
    case fir::LowerIntrinsicArgAs::Addr:
      operands.emplace_back(Fortran::lower::convertToAddress(
          loc, converter, actual, stmtCtx, getActualFortranElementType()));
      continue;
    case fir::LowerIntrinsicArgAs::Box:
      operands.emplace_back(Fortran::lower::convertToBox(
          loc, converter, actual, stmtCtx, getActualFortranElementType()));
      continue;
    case fir::LowerIntrinsicArgAs::Inquired:
      if (const Fortran::lower::SomeExpr *expr =
              callContext.procRef.UnwrapArgExpr(arg.index())) {
        if (Fortran::evaluate::UnwrapExpr<Fortran::evaluate::NullPointer>(
                *expr)) {
          // NULL() pointer without a MOLD must be passed as a deallocated
          // pointer (see table 16.5 in Fortran 2018 standard).
          // !fir.box<!fir.ptr<none>> should always be valid in this context.
          mlir::Type noneTy = mlir::NoneType::get(builder.getContext());
          mlir::Type nullPtrTy = fir::PointerType::get(noneTy);
          mlir::Type boxTy = fir::BoxType::get(nullPtrTy);
          mlir::Value boxStorage =
              fir::factory::genNullBoxStorage(builder, loc, boxTy);
          hlfir::EntityWithAttributes nullBoxEntity =
              extendedValueToHlfirEntity(loc, builder, boxStorage,
                                         ".tmp.null_box");
          operands.emplace_back(Fortran::lower::translateToExtendedValue(
              loc, builder, nullBoxEntity, stmtCtx));
          continue;
        }
      }
      // Place hlfir.expr in memory, and unbox fir.boxchar. Other entities
      // are translated to fir::ExtendedValue without transformation (notably,
      // pointers/allocatable are not dereferenced).
      // TODO: once lowering to FIR retires, UBOUND and LBOUND can be simplified
      // since the fir.box lowered here are now guaranteed to contain the local
      // lower bounds thanks to the hlfir.declare (the extra rebox can be
      // removed).
      operands.emplace_back(Fortran::lower::translateToExtendedValue(
          loc, builder, actual, stmtCtx));
      continue;
    }
    llvm_unreachable("bad switch");
  }
  // genIntrinsicCall needs the scalar type, even if this is a transformational
  // procedure returning an array.
  std::optional<mlir::Type> scalarResultType;
  if (callContext.resultType)
    scalarResultType = hlfir::getFortranElementType(*callContext.resultType);
  const std::string intrinsicName = callContext.getProcedureName();
  // Let the intrinsic library lower the intrinsic procedure call.
  auto [resultExv, mustBeFreed] = genIntrinsicCall(
      builder, loc, intrinsicEntry, scalarResultType, operands, &converter);
  for (const hlfir::CleanupFunction &fn : cleanupFns)
    fn();
  if (!fir::getBase(resultExv))
    return std::nullopt;
  hlfir::EntityWithAttributes resultEntity = extendedValueToHlfirEntity(
      loc, builder, resultExv, ".tmp.intrinsic_result");
  // Move result into memory into an hlfir.expr since they are immutable from
  // that point, and the result storage is some temp. "Null" is special: it
  // returns a null pointer variable that should not be transformed into a value
  // (what matters is the memory address).
  if (resultEntity.isVariable() && intrinsicName != "null") {
    assert(!fir::isa_trivial(fir::unwrapRefType(resultEntity.getType())) &&
           "expect intrinsic scalar results to not be in memory");
    hlfir::AsExprOp asExpr;
    // Character/Derived MERGE lowering returns one of its argument address
    // (this is the only intrinsic implemented in that way so far). The
    // ownership of this address cannot be taken here since it may not be a
    // temp.
    if (intrinsicName == "merge")
      asExpr = builder.create<hlfir::AsExprOp>(loc, resultEntity);
    else
      asExpr = builder.create<hlfir::AsExprOp>(
          loc, resultEntity, builder.createBool(loc, mustBeFreed));
    resultEntity = hlfir::EntityWithAttributes{asExpr.getResult()};
  }
  return resultEntity;
}

/// Lower calls to intrinsic procedures with actual arguments that have been
/// pre-lowered but have not yet been prepared according to the interface.
static std::optional<hlfir::EntityWithAttributes> genHLFIRIntrinsicRefCore(
    Fortran::lower::PreparedActualArguments &loweredActuals,
    const Fortran::evaluate::SpecificIntrinsic *intrinsic,
    const fir::IntrinsicHandlerEntry &intrinsicEntry,
    CallContext &callContext) {
  // Try lowering transformational intrinsic ops to HLFIR ops if enabled
  // (transformational always have a result type)
  if (useHlfirIntrinsicOps && callContext.resultType) {
    fir::FirOpBuilder &builder = callContext.getBuilder();
    mlir::Location loc = callContext.loc;
    const std::string intrinsicName = callContext.getProcedureName();
    const fir::IntrinsicArgumentLoweringRules *argLowering =
        intrinsicEntry.getArgumentLoweringRules();
    std::optional<hlfir::EntityWithAttributes> res =
        Fortran::lower::lowerHlfirIntrinsic(builder, loc, intrinsicName,
                                            loweredActuals, argLowering,
                                            *callContext.resultType);
    if (res)
      return res;
  }

  // fallback to calling the intrinsic via fir.call
  return genIntrinsicRefCore(loweredActuals, intrinsic, intrinsicEntry,
                             callContext);
}

namespace {
template <typename ElementalCallBuilderImpl>
class ElementalCallBuilder {
public:
  std::optional<hlfir::EntityWithAttributes>
  genElementalCall(Fortran::lower::PreparedActualArguments &loweredActuals,
                   bool isImpure, CallContext &callContext) {
    mlir::Location loc = callContext.loc;
    fir::FirOpBuilder &builder = callContext.getBuilder();
    unsigned numArgs = loweredActuals.size();
    // Step 1: dereference pointers/allocatables and compute elemental shape.
    mlir::Value shape;
    Fortran::lower::PreparedActualArgument *optionalWithShape;
    // 10.1.4 p5. Impure elemental procedures must be called in element order.
    bool mustBeOrdered = isImpure;
    for (unsigned i = 0; i < numArgs; ++i) {
      auto &preparedActual = loweredActuals[i];
      if (preparedActual) {
        // Elemental procedure dummy arguments cannot be pointer/allocatables
        // (C15100), so it is safe to dereference any pointer or allocatable
        // actual argument now instead of doing this inside the elemental
        // region.
        preparedActual->derefPointersAndAllocatables(loc, builder);
        // Better to load scalars outside of the loop when possible.
        if (!preparedActual->handleDynamicOptional() &&
            impl().canLoadActualArgumentBeforeLoop(i))
          preparedActual->loadTrivialScalar(loc, builder);
        // TODO: merge shape instead of using the first one.
        if (!shape && preparedActual->isArray()) {
          if (preparedActual->handleDynamicOptional())
            optionalWithShape = &*preparedActual;
          else
            shape = preparedActual->genShape(loc, builder);
        }
        // 15.8.3 p1. Elemental procedure with intent(out)/intent(inout)
        // arguments must be called in element order.
        if (impl().argMayBeModifiedByCall(i))
          mustBeOrdered = true;
      }
    }
    if (!shape && optionalWithShape) {
      // If all array operands appear in optional positions, then none of them
      // is allowed to be absent as per 15.5.2.12 point 3. (6). Just pick the
      // first operand.
      shape = optionalWithShape->genShape(loc, builder);
      // TODO: There is an opportunity to add a runtime check here that
      // this array is present as required. Also, the optionality of all actual
      // could be checked and reset given the Fortran requirement.
      optionalWithShape->resetOptionalAspect();
    }
    assert(shape &&
           "elemental array calls must have at least one array arguments");

    // Evaluate the actual argument array expressions before the elemental
    // call of an impure subprogram or a subprogram with intent(out) or
    // intent(inout) arguments. Note that the scalar arguments are handled
    // above.
    if (mustBeOrdered) {
      for (auto &preparedActual : loweredActuals) {
        if (preparedActual) {
          if (hlfir::AssociateOp associate =
                  preparedActual->associateIfArrayExpr(loc, builder)) {
            fir::FirOpBuilder *bldr = &builder;
            callContext.stmtCtx.attachCleanup(
                [=]() { bldr->create<hlfir::EndAssociateOp>(loc, associate); });
          }
        }
      }
    }

    // Push a new local scope so that any temps made inside the elemental
    // iterations are cleaned up inside the iterations.
    if (!callContext.resultType) {
      // Subroutine case. Generate call inside loop nest.
      hlfir::LoopNest loopNest =
          hlfir::genLoopNest(loc, builder, shape, !mustBeOrdered);
      mlir::ValueRange oneBasedIndices = loopNest.oneBasedIndices;
      auto insPt = builder.saveInsertionPoint();
      builder.setInsertionPointToStart(loopNest.body);
      callContext.stmtCtx.pushScope();
      for (auto &preparedActual : loweredActuals)
        if (preparedActual)
          preparedActual->setElementalIndices(oneBasedIndices);
      impl().genElementalKernel(loweredActuals, callContext);
      callContext.stmtCtx.finalizeAndPop();
      builder.restoreInsertionPoint(insPt);
      return std::nullopt;
    }
    // Function case: generate call inside hlfir.elemental
    mlir::Type elementType =
        hlfir::getFortranElementType(*callContext.resultType);
    // Get result length parameters.
    llvm::SmallVector<mlir::Value> typeParams;
    if (mlir::isa<fir::CharacterType>(elementType) ||
        fir::isRecordWithTypeParameters(elementType)) {
      auto charType = mlir::dyn_cast<fir::CharacterType>(elementType);
      if (charType && charType.hasConstantLen())
        typeParams.push_back(builder.createIntegerConstant(
            loc, builder.getIndexType(), charType.getLen()));
      else if (charType)
        typeParams.push_back(impl().computeDynamicCharacterResultLength(
            loweredActuals, callContext));
      else
        TODO(
            loc,
            "compute elemental PDT function result length parameters in HLFIR");
    }
    auto genKernel = [&](mlir::Location l, fir::FirOpBuilder &b,
                         mlir::ValueRange oneBasedIndices) -> hlfir::Entity {
      callContext.stmtCtx.pushScope();
      for (auto &preparedActual : loweredActuals)
        if (preparedActual)
          preparedActual->setElementalIndices(oneBasedIndices);
      auto res = *impl().genElementalKernel(loweredActuals, callContext);
      callContext.stmtCtx.finalizeAndPop();
      // Note that an hlfir.destroy is not emitted for the result since it
      // is still used by the hlfir.yield_element that also marks its last
      // use.
      return res;
    };
    mlir::Value polymorphicMold;
    if (fir::isPolymorphicType(*callContext.resultType))
      polymorphicMold =
          impl().getPolymorphicResultMold(loweredActuals, callContext);
    mlir::Value elemental =
        hlfir::genElementalOp(loc, builder, elementType, shape, typeParams,
                              genKernel, !mustBeOrdered, polymorphicMold);
    // If the function result requires finalization, then it has to be done
    // for the array result of the elemental call. We have to communicate
    // this via the DestroyOp's attribute.
    bool mustFinalizeExpr = impl().resultMayRequireFinalization(callContext);
    fir::FirOpBuilder *bldr = &builder;
    callContext.stmtCtx.attachCleanup([=]() {
      bldr->create<hlfir::DestroyOp>(loc, elemental, mustFinalizeExpr);
    });
    return hlfir::EntityWithAttributes{elemental};
  }

private:
  ElementalCallBuilderImpl &impl() {
    return *static_cast<ElementalCallBuilderImpl *>(this);
  }
};

class ElementalUserCallBuilder
    : public ElementalCallBuilder<ElementalUserCallBuilder> {
public:
  ElementalUserCallBuilder(Fortran::lower::CallerInterface &caller,
                           mlir::FunctionType callSiteType)
      : caller{caller}, callSiteType{callSiteType} {}
  std::optional<hlfir::Entity>
  genElementalKernel(Fortran::lower::PreparedActualArguments &loweredActuals,
                     CallContext &callContext) {
    return genUserCall(loweredActuals, caller, callSiteType, callContext);
  }

  bool argMayBeModifiedByCall(unsigned argIdx) const {
    assert(argIdx < caller.getPassedArguments().size() && "bad argument index");
    return caller.getPassedArguments()[argIdx].mayBeModifiedByCall();
  }

  bool canLoadActualArgumentBeforeLoop(unsigned argIdx) const {
    using PassBy = Fortran::lower::CallerInterface::PassEntityBy;
    const auto &passedArgs{caller.getPassedArguments()};
    assert(argIdx < passedArgs.size() && "bad argument index");
    // If the actual argument does not need to be passed via an address,
    // or will be passed in the address of a temporary copy, it can be loaded
    // before the elemental loop nest.
    const auto &arg{passedArgs[argIdx]};
    return arg.passBy == PassBy::Value ||
           arg.passBy == PassBy::BaseAddressValueAttribute;
  }

  mlir::Value computeDynamicCharacterResultLength(
      Fortran::lower::PreparedActualArguments &loweredActuals,
      CallContext &callContext) {
    fir::FirOpBuilder &builder = callContext.getBuilder();
    mlir::Location loc = callContext.loc;
    auto &converter = callContext.converter;
    mlir::Type idxTy = builder.getIndexType();
    llvm::SmallVector<CallCleanUp> callCleanUps;

    prepareUserCallArguments(loweredActuals, caller, callSiteType, callContext,
                             callCleanUps);

    callContext.symMap.pushScope();

    // Map prepared argument to dummy symbol to be able to lower spec expr.
    for (const auto &arg : caller.getPassedArguments()) {
      const Fortran::semantics::Symbol *sym = caller.getDummySymbol(arg);
      assert(sym && "expect symbol for dummy argument");
      auto input = caller.getInput(arg);
      fir::ExtendedValue exv = Fortran::lower::translateToExtendedValue(
          loc, builder, hlfir::Entity{input}, callContext.stmtCtx);
      fir::FortranVariableOpInterface variableIface = hlfir::genDeclare(
          loc, builder, exv, "dummy.tmp", fir::FortranVariableFlagsAttr{});
      callContext.symMap.addVariableDefinition(*sym, variableIface);
    }

    auto lowerSpecExpr = [&](const auto &expr) -> mlir::Value {
      mlir::Value convertExpr = builder.createConvert(
          loc, idxTy,
          fir::getBase(converter.genExprValue(expr, callContext.stmtCtx)));
      return fir::factory::genMaxWithZero(builder, loc, convertExpr);
    };

    llvm::SmallVector<mlir::Value> lengths;
    caller.walkResultLengths(
        [&](const Fortran::lower::SomeExpr &e, bool isAssumedSizeExtent) {
          assert(!isAssumedSizeExtent && "result cannot be assumed-size");
          lengths.emplace_back(lowerSpecExpr(e));
        });
    callContext.symMap.popScope();
    assert(lengths.size() == 1 && "expect 1 length parameter for the result");
    return lengths[0];
  }

  mlir::Value getPolymorphicResultMold(
      Fortran::lower::PreparedActualArguments &loweredActuals,
      CallContext &callContext) {
    fir::emitFatalError(callContext.loc,
                        "elemental function call with polymorphic result");
    return {};
  }

  bool resultMayRequireFinalization(CallContext &callContext) const {
    std::optional<Fortran::evaluate::DynamicType> retTy =
        caller.getCallDescription().proc().GetType();
    if (!retTy)
      return false;

    if (retTy->IsPolymorphic() || retTy->IsUnlimitedPolymorphic())
      fir::emitFatalError(
          callContext.loc,
          "elemental function call with [unlimited-]polymorphic result");

    if (retTy->category() == Fortran::common::TypeCategory::Derived) {
      const Fortran::semantics::DerivedTypeSpec &typeSpec =
          retTy->GetDerivedTypeSpec();
      return Fortran::semantics::IsFinalizable(typeSpec);
    }

    return false;
  }

private:
  Fortran::lower::CallerInterface &caller;
  mlir::FunctionType callSiteType;
};

class ElementalIntrinsicCallBuilder
    : public ElementalCallBuilder<ElementalIntrinsicCallBuilder> {
public:
  ElementalIntrinsicCallBuilder(
      const Fortran::evaluate::SpecificIntrinsic *intrinsic,
      const fir::IntrinsicHandlerEntry &intrinsicEntry, bool isFunction)
      : intrinsic{intrinsic}, intrinsicEntry{intrinsicEntry},
        isFunction{isFunction} {}
  std::optional<hlfir::Entity>
  genElementalKernel(Fortran::lower::PreparedActualArguments &loweredActuals,
                     CallContext &callContext) {
    return genHLFIRIntrinsicRefCore(loweredActuals, intrinsic, intrinsicEntry,
                                    callContext);
  }
  // Elemental intrinsic functions cannot modify their arguments.
  bool argMayBeModifiedByCall(int) const { return !isFunction; }
  bool canLoadActualArgumentBeforeLoop(int) const {
    // Elemental intrinsic functions never need the actual addresses
    // of their arguments.
    return isFunction;
  }

  mlir::Value computeDynamicCharacterResultLength(
      Fortran::lower::PreparedActualArguments &loweredActuals,
      CallContext &callContext) {
    if (intrinsic)
      if (intrinsic->name == "adjustr" || intrinsic->name == "adjustl" ||
          intrinsic->name == "merge")
        return loweredActuals[0].value().genCharLength(
            callContext.loc, callContext.getBuilder());
    // Character MIN/MAX is the min/max of the arguments length that are
    // present.
    TODO(callContext.loc,
         "compute elemental character min/max function result length in HLFIR");
  }

  mlir::Value getPolymorphicResultMold(
      Fortran::lower::PreparedActualArguments &loweredActuals,
      CallContext &callContext) {
    if (!intrinsic)
      return {};

    if (intrinsic->name == "merge") {
      // MERGE seems to be the only elemental function that can produce
      // polymorphic result. The MERGE's result is polymorphic iff
      // both TSOURCE and FSOURCE are polymorphic, and they also must have
      // the same declared and dynamic types. So any of them can be used
      // for the mold.
      assert(!loweredActuals.empty());
      return loweredActuals.front()->getPolymorphicMold(callContext.loc);
    }

    return {};
  }

  bool resultMayRequireFinalization(
      [[maybe_unused]] CallContext &callContext) const {
    // FIXME: need access to the CallerInterface's return type
    // to check if the result may need finalization (e.g. the result
    // of MERGE).
    return false;
  }

private:
  const Fortran::evaluate::SpecificIntrinsic *intrinsic;
  fir::IntrinsicHandlerEntry intrinsicEntry;
  const bool isFunction;
};
} // namespace

static std::optional<mlir::Value>
genIsPresentIfArgMaybeAbsent(mlir::Location loc, hlfir::Entity actual,
                             const Fortran::lower::SomeExpr &expr,
                             CallContext &callContext,
                             bool passAsAllocatableOrPointer) {
  if (!Fortran::evaluate::MayBePassedAsAbsentOptional(expr))
    return std::nullopt;
  fir::FirOpBuilder &builder = callContext.getBuilder();
  if (!passAsAllocatableOrPointer &&
      Fortran::evaluate::IsAllocatableOrPointerObject(expr)) {
    // Passing Allocatable/Pointer to non-pointer/non-allocatable OPTIONAL.
    // Fortran 2018 15.5.2.12 point 1: If unallocated/disassociated, it is
    // as if the argument was absent. The main care here is to not do a
    // copy-in/copy-out because the temp address, even though pointing to a
    // null size storage, would not be a nullptr and therefore the argument
    // would not be considered absent on the callee side. Note: if the
    // allocatable/pointer is also optional, it cannot be absent as per
    // 15.5.2.12 point 7. and 8. We rely on this to un-conditionally read
    // the allocatable/pointer descriptor here.
    mlir::Value addr = genVariableRawAddress(loc, builder, actual);
    return builder.genIsNotNullAddr(loc, addr);
  }
  // TODO: what if passing allocatable target to optional intent(in) pointer?
  // May fall into the category above if the allocatable is not optional.

  // Passing an optional to an optional.
  return builder.create<fir::IsPresentOp>(loc, builder.getI1Type(), actual)
      .getResult();
}

// Lower a reference to an elemental intrinsic procedure with array arguments
// and custom optional handling
static std::optional<hlfir::EntityWithAttributes>
genCustomElementalIntrinsicRef(
    const Fortran::evaluate::SpecificIntrinsic *intrinsic,
    CallContext &callContext) {
  assert(callContext.isElementalProcWithArrayArgs() &&
         "Use genCustomIntrinsicRef for scalar calls");
  mlir::Location loc = callContext.loc;
  auto &converter = callContext.converter;
  Fortran::lower::PreparedActualArguments operands;
  assert(intrinsic && Fortran::lower::intrinsicRequiresCustomOptionalHandling(
                          callContext.procRef, *intrinsic, converter));

  // callback for optional arguments
  auto prepareOptionalArg = [&](const Fortran::lower::SomeExpr &expr) {
    hlfir::EntityWithAttributes actual = Fortran::lower::convertExprToHLFIR(
        loc, converter, expr, callContext.symMap, callContext.stmtCtx);
    std::optional<mlir::Value> isPresent =
        genIsPresentIfArgMaybeAbsent(loc, actual, expr, callContext,
                                     /*passAsAllocatableOrPointer=*/false);
    operands.emplace_back(
        Fortran::lower::PreparedActualArgument{actual, isPresent});
  };

  // callback for non-optional arguments
  auto prepareOtherArg = [&](const Fortran::lower::SomeExpr &expr,
                             fir::LowerIntrinsicArgAs lowerAs) {
    hlfir::EntityWithAttributes actual = Fortran::lower::convertExprToHLFIR(
        loc, converter, expr, callContext.symMap, callContext.stmtCtx);
    operands.emplace_back(Fortran::lower::PreparedActualArgument{
        actual, /*isPresent=*/std::nullopt});
  };

  Fortran::lower::prepareCustomIntrinsicArgument(
      callContext.procRef, *intrinsic, callContext.resultType,
      prepareOptionalArg, prepareOtherArg, converter);

  std::optional<fir::IntrinsicHandlerEntry> intrinsicEntry =
      fir::lookupIntrinsicHandler(callContext.getBuilder(),
                                  callContext.getProcedureName(),
                                  callContext.resultType);
  assert(intrinsicEntry.has_value() &&
         "intrinsic with custom handling for OPTIONAL arguments must have "
         "lowering entries");
  // All of the custom intrinsic elementals with custom handling are pure
  // functions
  return ElementalIntrinsicCallBuilder{intrinsic, *intrinsicEntry,
                                       /*isFunction=*/true}
      .genElementalCall(operands, /*isImpure=*/false, callContext);
}

// Lower a reference to an intrinsic procedure with custom optional handling
static std::optional<hlfir::EntityWithAttributes>
genCustomIntrinsicRef(const Fortran::evaluate::SpecificIntrinsic *intrinsic,
                      CallContext &callContext) {
  assert(!callContext.isElementalProcWithArrayArgs() &&
         "Needs to be run through ElementalIntrinsicCallBuilder first");
  mlir::Location loc = callContext.loc;
  fir::FirOpBuilder &builder = callContext.getBuilder();
  auto &converter = callContext.converter;
  auto &stmtCtx = callContext.stmtCtx;
  assert(intrinsic && Fortran::lower::intrinsicRequiresCustomOptionalHandling(
                          callContext.procRef, *intrinsic, converter));
  Fortran::lower::PreparedActualArguments loweredActuals;

  // callback for optional arguments
  auto prepareOptionalArg = [&](const Fortran::lower::SomeExpr &expr) {
    hlfir::EntityWithAttributes actual = Fortran::lower::convertExprToHLFIR(
        loc, converter, expr, callContext.symMap, callContext.stmtCtx);
    mlir::Value isPresent =
        genIsPresentIfArgMaybeAbsent(loc, actual, expr, callContext,
                                     /*passAsAllocatableOrPointer*/ false)
            .value();
    loweredActuals.emplace_back(
        Fortran::lower::PreparedActualArgument{actual, {isPresent}});
  };

  // callback for non-optional arguments
  auto prepareOtherArg = [&](const Fortran::lower::SomeExpr &expr,
                             fir::LowerIntrinsicArgAs lowerAs) {
    auto getActualFortranElementType = [&]() -> mlir::Type {
      return hlfir::getFortranElementType(converter.genType(expr));
    };
    hlfir::EntityWithAttributes actual = Fortran::lower::convertExprToHLFIR(
        loc, converter, expr, callContext.symMap, callContext.stmtCtx);
    std::optional<fir::ExtendedValue> exv;
    switch (lowerAs) {
    case fir::LowerIntrinsicArgAs::Value:
      exv = Fortran::lower::convertToValue(loc, converter, actual, stmtCtx);
      break;
    case fir::LowerIntrinsicArgAs::Addr:
      exv = Fortran::lower::convertToAddress(loc, converter, actual, stmtCtx,
                                             getActualFortranElementType());
      break;
    case fir::LowerIntrinsicArgAs::Box:
      exv = Fortran::lower::convertToBox(loc, converter, actual, stmtCtx,
                                         getActualFortranElementType());
      break;
    case fir::LowerIntrinsicArgAs::Inquired:
      exv = Fortran::lower::translateToExtendedValue(loc, builder, actual,
                                                     stmtCtx);
      break;
    }
    if (!exv)
      llvm_unreachable("bad switch");
    actual = extendedValueToHlfirEntity(loc, builder, exv.value(),
                                        "tmp.custom_intrinsic_arg");
    loweredActuals.emplace_back(Fortran::lower::PreparedActualArgument{
        actual, /*isPresent=*/std::nullopt});
  };

  Fortran::lower::prepareCustomIntrinsicArgument(
      callContext.procRef, *intrinsic, callContext.resultType,
      prepareOptionalArg, prepareOtherArg, converter);

  return genCustomIntrinsicRefCore(loweredActuals, intrinsic, callContext);
}

/// Lower an intrinsic procedure reference.
/// \p intrinsic is null if this is an intrinsic module procedure that must be
/// lowered as if it were an intrinsic module procedure (like C_LOC which is a
/// procedure from intrinsic module iso_c_binding). Otherwise, \p intrinsic
/// must not be null.

static std::optional<hlfir::EntityWithAttributes>
genIntrinsicRef(const Fortran::evaluate::SpecificIntrinsic *intrinsic,
                const fir::IntrinsicHandlerEntry &intrinsicEntry,
                CallContext &callContext) {
  mlir::Location loc = callContext.loc;
  Fortran::lower::PreparedActualArguments loweredActuals;
  const fir::IntrinsicArgumentLoweringRules *argLowering =
      intrinsicEntry.getArgumentLoweringRules();
  for (const auto &arg : llvm::enumerate(callContext.procRef.arguments())) {

    if (!arg.value()) {
      // Absent optional.
      loweredActuals.push_back(std::nullopt);
      continue;
    }
    auto *expr =
        Fortran::evaluate::UnwrapExpr<Fortran::lower::SomeExpr>(arg.value());
    if (!expr) {
      // TYPE(*) dummy. They are only allowed as argument of a few intrinsics
      // that do not take optional arguments: see Fortran 2018 standard C710.
      const Fortran::evaluate::Symbol *assumedTypeSym =
          arg.value()->GetAssumedTypeDummy();
      if (!assumedTypeSym)
        fir::emitFatalError(loc,
                            "expected assumed-type symbol as actual argument");
      std::optional<fir::FortranVariableOpInterface> var =
          callContext.symMap.lookupVariableDefinition(*assumedTypeSym);
      if (!var)
        fir::emitFatalError(loc, "assumed-type symbol was not lowered");
      assert(
          (!argLowering ||
           !fir::lowerIntrinsicArgumentAs(*argLowering, arg.index())
                .handleDynamicOptional) &&
          "TYPE(*) are not expected to appear as optional intrinsic arguments");
      loweredActuals.push_back(Fortran::lower::PreparedActualArgument{
          hlfir::Entity{*var}, /*isPresent=*/std::nullopt});
      continue;
    }
    // arguments of bitwise comparison functions may not have nsw flag
    // even if -fno-wrapv is enabled
    mlir::arith::IntegerOverflowFlags iofBackup{};
    auto isBitwiseComparison = [](const std::string intrinsicName) -> bool {
      if (intrinsicName == "bge" || intrinsicName == "bgt" ||
          intrinsicName == "ble" || intrinsicName == "blt")
        return true;
      return false;
    };
    if (isBitwiseComparison(callContext.getProcedureName())) {
      iofBackup = callContext.getBuilder().getIntegerOverflowFlags();
      callContext.getBuilder().setIntegerOverflowFlags(
          mlir::arith::IntegerOverflowFlags::none);
    }
    auto loweredActual = Fortran::lower::convertExprToHLFIR(
        loc, callContext.converter, *expr, callContext.symMap,
        callContext.stmtCtx);
    if (isBitwiseComparison(callContext.getProcedureName()))
      callContext.getBuilder().setIntegerOverflowFlags(iofBackup);

    std::optional<mlir::Value> isPresent;
    if (argLowering) {
      fir::ArgLoweringRule argRules =
          fir::lowerIntrinsicArgumentAs(*argLowering, arg.index());
      if (argRules.handleDynamicOptional)
        isPresent =
            genIsPresentIfArgMaybeAbsent(loc, loweredActual, *expr, callContext,
                                         /*passAsAllocatableOrPointer=*/false);
    }
    loweredActuals.push_back(
        Fortran::lower::PreparedActualArgument{loweredActual, isPresent});
  }

  if (callContext.isElementalProcWithArrayArgs()) {
    // All intrinsic elemental functions are pure.
    const bool isFunction = callContext.resultType.has_value();
    return ElementalIntrinsicCallBuilder{intrinsic, intrinsicEntry, isFunction}
        .genElementalCall(loweredActuals, /*isImpure=*/!isFunction,
                          callContext);
  }
  std::optional<hlfir::EntityWithAttributes> result = genHLFIRIntrinsicRefCore(
      loweredActuals, intrinsic, intrinsicEntry, callContext);
  if (result && mlir::isa<hlfir::ExprType>(result->getType())) {
    fir::FirOpBuilder *bldr = &callContext.getBuilder();
    callContext.stmtCtx.attachCleanup(
        [=]() { bldr->create<hlfir::DestroyOp>(loc, *result); });
  }
  return result;
}

static std::optional<hlfir::EntityWithAttributes>
genIntrinsicRef(const Fortran::evaluate::SpecificIntrinsic *intrinsic,
                CallContext &callContext) {
  mlir::Location loc = callContext.loc;
  auto &converter = callContext.converter;
  if (intrinsic && Fortran::lower::intrinsicRequiresCustomOptionalHandling(
                       callContext.procRef, *intrinsic, converter)) {
    if (callContext.isElementalProcWithArrayArgs())
      return genCustomElementalIntrinsicRef(intrinsic, callContext);
    return genCustomIntrinsicRef(intrinsic, callContext);
  }
  std::optional<fir::IntrinsicHandlerEntry> intrinsicEntry =
      fir::lookupIntrinsicHandler(callContext.getBuilder(),
                                  callContext.getProcedureName(),
                                  callContext.resultType);
  if (!intrinsicEntry)
    fir::crashOnMissingIntrinsic(loc, callContext.getProcedureName());
  return genIntrinsicRef(intrinsic, *intrinsicEntry, callContext);
}

/// Main entry point to lower procedure references, regardless of what they are.
static std::optional<hlfir::EntityWithAttributes>
genProcedureRef(CallContext &callContext) {
  mlir::Location loc = callContext.loc;
  fir::FirOpBuilder &builder = callContext.getBuilder();
  if (auto *intrinsic = callContext.procRef.proc().GetSpecificIntrinsic())
    return genIntrinsicRef(intrinsic, callContext);
  // Intercept non BIND(C) module procedure reference that have lowering
  // handlers defined for there name. Otherwise, lower them as user
  // procedure calls and expect the implementation to be part of
  // runtime libraries with the proper name mangling.
  if (Fortran::lower::isIntrinsicModuleProcRef(callContext.procRef) &&
      !callContext.isBindcCall())
    if (std::optional<fir::IntrinsicHandlerEntry> intrinsicEntry =
            fir::lookupIntrinsicHandler(builder, callContext.getProcedureName(),
                                        callContext.resultType))
      return genIntrinsicRef(nullptr, *intrinsicEntry, callContext);

  if (callContext.isStatementFunctionCall())
    return genStmtFunctionRef(loc, callContext.converter, callContext.symMap,
                              callContext.stmtCtx, callContext.procRef);

  Fortran::lower::CallerInterface caller(callContext.procRef,
                                         callContext.converter);
  mlir::FunctionType callSiteType = caller.genFunctionType();
  const bool isElemental = callContext.isElementalProcWithArrayArgs();
  Fortran::lower::PreparedActualArguments loweredActuals;
  // Lower the actual arguments
  for (const Fortran::lower::CallInterface<
           Fortran::lower::CallerInterface>::PassedEntity &arg :
       caller.getPassedArguments())
    if (const auto *actual = arg.entity) {
      const auto *expr = actual->UnwrapExpr();
      if (!expr) {
        // TYPE(*) actual argument.
        const Fortran::evaluate::Symbol *assumedTypeSym =
            actual->GetAssumedTypeDummy();
        if (!assumedTypeSym)
          fir::emitFatalError(
              loc, "expected assumed-type symbol as actual argument");
        std::optional<fir::FortranVariableOpInterface> var =
            callContext.symMap.lookupVariableDefinition(*assumedTypeSym);
        if (!var)
          fir::emitFatalError(loc, "assumed-type symbol was not lowered");
        hlfir::Entity actual{*var};
        std::optional<mlir::Value> isPresent;
        if (arg.isOptional()) {
          // Passing an optional TYPE(*) to an optional TYPE(*). Note that
          // TYPE(*) cannot be ALLOCATABLE/POINTER (C709) so there is no
          // need to cover the case of passing an ALLOCATABLE/POINTER to an
          // OPTIONAL.
          isPresent =
              builder.create<fir::IsPresentOp>(loc, builder.getI1Type(), actual)
                  .getResult();
        }
        loweredActuals.push_back(Fortran::lower::PreparedActualArgument{
            hlfir::Entity{*var}, isPresent});
        continue;
      }

      if (Fortran::evaluate::UnwrapExpr<Fortran::evaluate::NullPointer>(
              *expr)) {
        if ((arg.passBy !=
             Fortran::lower::CallerInterface::PassEntityBy::MutableBox) &&
            (arg.passBy !=
             Fortran::lower::CallerInterface::PassEntityBy::BoxProcRef)) {
          assert(
              arg.isOptional() &&
              "NULL must be passed only to pointer, allocatable, or OPTIONAL");
          // Trying to lower NULL() outside of any context would lead to
          // trouble. NULL() here is equivalent to not providing the
          // actual argument.
          loweredActuals.emplace_back(std::nullopt);
          continue;
        }
      }

      if (isElemental && !arg.hasValueAttribute() &&
          Fortran::evaluate::IsVariable(*expr) &&
          Fortran::evaluate::HasVectorSubscript(*expr)) {
        // Vector subscripted arguments are copied in calls, except in elemental
        // calls without VALUE attribute where Fortran 2018 15.5.2.4 point 21
        // does not apply and the address of each element must be passed.
        hlfir::ElementalAddrOp elementalAddr =
            Fortran::lower::convertVectorSubscriptedExprToElementalAddr(
                loc, callContext.converter, *expr, callContext.symMap,
                callContext.stmtCtx);
        loweredActuals.emplace_back(
            Fortran::lower::PreparedActualArgument{elementalAddr});
        continue;
      }

      auto loweredActual = Fortran::lower::convertExprToHLFIR(
          loc, callContext.converter, *expr, callContext.symMap,
          callContext.stmtCtx);
      std::optional<mlir::Value> isPresent;
      if (arg.isOptional())
        isPresent = genIsPresentIfArgMaybeAbsent(
            loc, loweredActual, *expr, callContext,
            arg.passBy ==
                Fortran::lower::CallerInterface::PassEntityBy::MutableBox);

      loweredActuals.emplace_back(
          Fortran::lower::PreparedActualArgument{loweredActual, isPresent});
    } else {
      // Optional dummy argument for which there is no actual argument.
      loweredActuals.emplace_back(std::nullopt);
    }
  if (isElemental) {
    bool isImpure = false;
    if (const Fortran::semantics::Symbol *procSym =
            callContext.procRef.proc().GetSymbol())
      isImpure = !Fortran::semantics::IsPureProcedure(*procSym);
    return ElementalUserCallBuilder{caller, callSiteType}.genElementalCall(
        loweredActuals, isImpure, callContext);
  }
  return genUserCall(loweredActuals, caller, callSiteType, callContext);
}

hlfir::Entity Fortran::lower::PreparedActualArgument::getActual(
    mlir::Location loc, fir::FirOpBuilder &builder) const {
  if (auto *actualEntity = std::get_if<hlfir::Entity>(&actual)) {
    if (oneBasedElementalIndices)
      return hlfir::getElementAt(loc, builder, *actualEntity,
                                 *oneBasedElementalIndices);
    return *actualEntity;
  }
  assert(oneBasedElementalIndices && "expect elemental context");
  hlfir::ElementalAddrOp elementalAddr =
      std::get<hlfir::ElementalAddrOp>(actual);
  mlir::IRMapping mapper;
  auto alwaysFalse = [](hlfir::ElementalOp) -> bool { return false; };
  mlir::Value addr = hlfir::inlineElementalOp(
      loc, builder, elementalAddr, *oneBasedElementalIndices, mapper,
      /*mustRecursivelyInline=*/alwaysFalse);
  assert(elementalAddr.getCleanup().empty() && "no clean-up expected");
  elementalAddr.erase();
  return hlfir::Entity{addr};
}

bool Fortran::lower::isIntrinsicModuleProcRef(
    const Fortran::evaluate::ProcedureRef &procRef) {
  const Fortran::semantics::Symbol *symbol = procRef.proc().GetSymbol();
  if (!symbol)
    return false;
  const Fortran::semantics::Symbol *module =
      symbol->GetUltimate().owner().GetSymbol();
  return module && module->attrs().test(Fortran::semantics::Attr::INTRINSIC);
}

static bool isInWhereMaskedExpression(fir::FirOpBuilder &builder) {
  // The MASK of the outer WHERE is not masked itself.
  mlir::Operation *op = builder.getRegion().getParentOp();
  return op && op->getParentOfType<hlfir::WhereOp>();
}

std::optional<hlfir::EntityWithAttributes> Fortran::lower::convertCallToHLFIR(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const evaluate::ProcedureRef &procRef, std::optional<mlir::Type> resultType,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx) {
  auto &builder = converter.getFirOpBuilder();
  if (resultType && !procRef.IsElemental() &&
      isInWhereMaskedExpression(builder) &&
      !builder.getRegion().getParentOfType<hlfir::ExactlyOnceOp>()) {
    // Non elemental calls inside a where-assignment-stmt must be executed
    // exactly once without mask control. Lower them in a special region so that
    // this can be enforced whenscheduling forall/where expression evaluations.
    Fortran::lower::StatementContext localStmtCtx;
    mlir::Type bogusType = builder.getIndexType();
    auto exactlyOnce = builder.create<hlfir::ExactlyOnceOp>(loc, bogusType);
    mlir::Block *block = builder.createBlock(&exactlyOnce.getBody());
    builder.setInsertionPointToStart(block);
    CallContext callContext(procRef, resultType, loc, converter, symMap,
                            localStmtCtx);
    std::optional<hlfir::EntityWithAttributes> res =
        genProcedureRef(callContext);
    assert(res.has_value() && "must be a function");
    auto yield = builder.create<hlfir::YieldOp>(loc, *res);
    Fortran::lower::genCleanUpInRegionIfAny(loc, builder, yield.getCleanup(),
                                            localStmtCtx);
    builder.setInsertionPointAfter(exactlyOnce);
    exactlyOnce->getResult(0).setType(res->getType());
    if (hlfir::isFortranValue(exactlyOnce.getResult()))
      return hlfir::EntityWithAttributes{exactlyOnce.getResult()};
    // Create hlfir.declare for the result to satisfy
    // hlfir::EntityWithAttributes requirements.
    auto [exv, cleanup] = hlfir::translateToExtendedValue(
        loc, builder, hlfir::Entity{exactlyOnce});
    assert(!cleanup && "resut is a variable");
    return hlfir::genDeclare(loc, builder, exv, ".func.pointer.result",
                             fir::FortranVariableFlagsAttr{});
  }
  CallContext callContext(procRef, resultType, loc, converter, symMap, stmtCtx);
  return genProcedureRef(callContext);
}

void Fortran::lower::convertUserDefinedAssignmentToHLFIR(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const evaluate::ProcedureRef &procRef, hlfir::Entity lhs, hlfir::Entity rhs,
    Fortran::lower::SymMap &symMap) {
  Fortran::lower::StatementContext definedAssignmentContext;
  CallContext callContext(procRef, /*resultType=*/std::nullopt, loc, converter,
                          symMap, definedAssignmentContext);
  Fortran::lower::CallerInterface caller(procRef, converter);
  mlir::FunctionType callSiteType = caller.genFunctionType();
  PreparedActualArgument preparedLhs{lhs, /*isPresent=*/std::nullopt};
  PreparedActualArgument preparedRhs{rhs, /*isPresent=*/std::nullopt};
  PreparedActualArguments loweredActuals{preparedLhs, preparedRhs};
  genUserCall(loweredActuals, caller, callSiteType, callContext);
  return;
}
