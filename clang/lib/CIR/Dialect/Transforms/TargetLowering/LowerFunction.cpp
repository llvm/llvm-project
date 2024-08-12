//===--- LowerFunction.cpp - Lower CIR Function Code ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file partially mimics clang/lib/CodeGen/CodeGenFunction.cpp. The queries
// are adapted to operate on the CIR dialect, however.
//
//===----------------------------------------------------------------------===//

#include "LowerFunction.h"
#include "CIRToCIRArgMapping.h"
#include "LowerCall.h"
#include "LowerFunctionInfo.h"
#include "LowerModule.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "clang/CIR/ABIArgInfo.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/MissingFeatures.h"
#include "clang/CIR/TypeEvaluationKind.h"
#include "llvm/Support/ErrorHandling.h"

using ABIArgInfo = ::cir::ABIArgInfo;

namespace mlir {
namespace cir {

namespace {

Value buildAddressAtOffset(LowerFunction &LF, Value addr,
                           const ABIArgInfo &info) {
  if (unsigned offset = info.getDirectOffset()) {
    llvm_unreachable("NYI");
  }
  return addr;
}

/// Given a struct pointer that we are accessing some number of bytes out of it,
/// try to gep into the struct to get at its inner goodness.  Dive as deep as
/// possible without entering an element with an in-memory size smaller than
/// DstSize.
Value enterStructPointerForCoercedAccess(Value SrcPtr, StructType SrcSTy,
                                         uint64_t DstSize, LowerFunction &CGF) {
  // We can't dive into a zero-element struct.
  if (SrcSTy.getNumElements() == 0)
    llvm_unreachable("NYI");

  Type FirstElt = SrcSTy.getMembers()[0];

  // If the first elt is at least as large as what we're looking for, or if the
  // first element is the same size as the whole struct, we can enter it. The
  // comparison must be made on the store size and not the alloca size. Using
  // the alloca size may overstate the size of the load.
  uint64_t FirstEltSize = CGF.LM.getDataLayout().getTypeStoreSize(FirstElt);
  if (FirstEltSize < DstSize &&
      FirstEltSize < CGF.LM.getDataLayout().getTypeStoreSize(SrcSTy))
    return SrcPtr;

  llvm_unreachable("NYI");
}

/// Create a store to \param Dst from \param Src where the source and
/// destination may have different types.
///
/// This safely handles the case when the src type is larger than the
/// destination type; the upper bits of the src will be lost.
void createCoercedStore(Value Src, Value Dst, bool DstIsVolatile,
                        LowerFunction &CGF) {
  Type SrcTy = Src.getType();
  Type DstTy = Dst.getType();
  if (SrcTy == DstTy) {
    llvm_unreachable("NYI");
  }

  // FIXME(cir): We need a better way to handle datalayout queries.
  assert(isa<IntType>(SrcTy));
  llvm::TypeSize SrcSize = CGF.LM.getDataLayout().getTypeAllocSize(SrcTy);

  if (StructType DstSTy = dyn_cast<StructType>(DstTy)) {
    Dst = enterStructPointerForCoercedAccess(Dst, DstSTy,
                                             SrcSize.getFixedValue(), CGF);
    assert(isa<PointerType>(Dst.getType()));
    DstTy = cast<PointerType>(Dst.getType()).getPointee();
  }

  PointerType SrcPtrTy = dyn_cast<PointerType>(SrcTy);
  PointerType DstPtrTy = dyn_cast<PointerType>(DstTy);
  // TODO(cir): Implement address space.
  if (SrcPtrTy && DstPtrTy && !::cir::MissingFeatures::addressSpace()) {
    llvm_unreachable("NYI");
  }

  // If the source and destination are integer or pointer types, just do an
  // extension or truncation to the desired type.
  if ((isa<IntegerType>(SrcTy) || isa<PointerType>(SrcTy)) &&
      (isa<IntegerType>(DstTy) || isa<PointerType>(DstTy))) {
    llvm_unreachable("NYI");
  }

  llvm::TypeSize DstSize = CGF.LM.getDataLayout().getTypeAllocSize(DstTy);

  // If store is legal, just bitcast the src pointer.
  assert(!::cir::MissingFeatures::vectorType());
  if (SrcSize.getFixedValue() <= DstSize.getFixedValue()) {
    // Dst = Dst.withElementType(SrcTy);
    CGF.buildAggregateStore(Src, Dst, DstIsVolatile);
  } else {
    llvm_unreachable("NYI");
  }
}

// FIXME(cir): Create a custom rewriter class to abstract this away.
Value createBitcast(Value Src, Type Ty, LowerFunction &LF) {
  return LF.getRewriter().create<CastOp>(Src.getLoc(), Ty, CastKind::bitcast,
                                         Src);
}

/// Coerces a \param Src value to a value of type \param Ty.
///
/// This safely handles the case when the src type is smaller than the
/// destination type; in this situation the values of bits which not present in
/// the src are undefined.
///
/// NOTE(cir): This method has partial parity with CGCall's CreateCoercedLoad.
/// Unlike the original codegen, this function does not emit a coerced load
/// since CIR's type checker wouldn't allow it. Instead, it casts the existing
/// ABI-agnostic value to it's ABI-aware counterpart. Nevertheless, we should
/// try to follow the same logic as the original codegen for correctness.
Value createCoercedValue(Value Src, Type Ty, LowerFunction &CGF) {
  Type SrcTy = Src.getType();

  // If SrcTy and Ty are the same, just reuse the exising load.
  if (SrcTy == Ty)
    return Src;

  // If it is the special boolean case, simply bitcast it.
  if ((isa<BoolType>(SrcTy) && isa<IntType>(Ty)) ||
      (isa<IntType>(SrcTy) && isa<BoolType>(Ty)))
    return createBitcast(Src, Ty, CGF);

  llvm::TypeSize DstSize = CGF.LM.getDataLayout().getTypeAllocSize(Ty);

  if (auto SrcSTy = dyn_cast<StructType>(SrcTy)) {
    Src = enterStructPointerForCoercedAccess(Src, SrcSTy,
                                             DstSize.getFixedValue(), CGF);
    SrcTy = Src.getType();
  }

  llvm::TypeSize SrcSize = CGF.LM.getDataLayout().getTypeAllocSize(SrcTy);

  // If the source and destination are integer or pointer types, just do an
  // extension or truncation to the desired type.
  if ((isa<IntType>(Ty) || isa<PointerType>(Ty)) &&
      (isa<IntType>(SrcTy) || isa<PointerType>(SrcTy))) {
    llvm_unreachable("NYI");
  }

  // If load is legal, just bitcast the src pointer.
  if (!SrcSize.isScalable() && !DstSize.isScalable() &&
      SrcSize.getFixedValue() >= DstSize.getFixedValue()) {
    // Generally SrcSize is never greater than DstSize, since this means we are
    // losing bits. However, this can happen in cases where the structure has
    // additional padding, for example due to a user specified alignment.
    //
    // FIXME: Assert that we aren't truncating non-padding bits when have access
    // to that information.
    // Src = Src.withElementType();
    return CGF.buildAggregateBitcast(Src, Ty);
  }

  llvm_unreachable("NYI");
}

Value emitAddressAtOffset(LowerFunction &LF, Value addr,
                          const ABIArgInfo &info) {
  if (unsigned offset = info.getDirectOffset()) {
    llvm_unreachable("NYI");
  }
  return addr;
}

/// After the calling convention is lowered, an ABI-agnostic type might have to
/// be loaded back to its ABI-aware couterpart so it may be returned. If they
/// differ, we have to do a coerced load. A coerced load, which means to load a
/// type to another despite that they represent the same value. The simplest
/// cases can be solved with a mere bitcast.
///
/// This partially replaces CreateCoercedLoad from the original codegen.
/// However, instead of emitting the load, it emits a cast.
///
/// FIXME(cir): Improve parity with the original codegen.
Value castReturnValue(Value Src, Type Ty, LowerFunction &LF) {
  Type SrcTy = Src.getType();

  // If SrcTy and Ty are the same, nothing to do.
  if (SrcTy == Ty)
    return Src;

  // If is the special boolean case, simply bitcast it.
  if (isa<BoolType>(SrcTy) && isa<IntType>(Ty))
    return createBitcast(Src, Ty, LF);

  llvm::TypeSize DstSize = LF.LM.getDataLayout().getTypeAllocSize(Ty);

  // FIXME(cir): Do we need the EnterStructPointerForCoercedAccess routine here?

  llvm::TypeSize SrcSize = LF.LM.getDataLayout().getTypeAllocSize(SrcTy);

  if ((isa<IntType>(Ty) || isa<PointerType>(Ty)) &&
      (isa<IntType>(SrcTy) || isa<PointerType>(SrcTy))) {
    llvm_unreachable("NYI");
  }

  // If load is legal, just bitcast the src pointer.
  if (!SrcSize.isScalable() && !DstSize.isScalable() &&
      SrcSize.getFixedValue() >= DstSize.getFixedValue()) {
    // Generally SrcSize is never greater than DstSize, since this means we are
    // losing bits. However, this can happen in cases where the structure has
    // additional padding, for example due to a user specified alignment.
    //
    // FIXME: Assert that we aren't truncating non-padding bits when have access
    // to that information.
    return LF.getRewriter().create<CastOp>(Src.getLoc(), Ty, CastKind::bitcast,
                                           Src);
  }

  llvm_unreachable("NYI");
}

} // namespace

// FIXME(cir): Pass SrcFn and NewFn around instead of having then as attributes.
LowerFunction::LowerFunction(LowerModule &LM, PatternRewriter &rewriter,
                             FuncOp srcFn, FuncOp newFn)
    : Target(LM.getTarget()), rewriter(rewriter), SrcFn(srcFn), NewFn(newFn),
      LM(LM) {}

LowerFunction::LowerFunction(LowerModule &LM, PatternRewriter &rewriter,
                             FuncOp srcFn, CallOp callOp)
    : Target(LM.getTarget()), rewriter(rewriter), SrcFn(srcFn), callOp(callOp),
      LM(LM) {}

/// This method has partial parity with CodeGenFunction::EmitFunctionProlog from
/// the original codegen. However, it focuses on the ABI-specific details. On
/// top of that, it is also responsible for rewriting the original function.
LogicalResult
LowerFunction::buildFunctionProlog(const LowerFunctionInfo &FI, FuncOp Fn,
                                   MutableArrayRef<BlockArgument> Args) {
  // NOTE(cir): Skipping naked and implicit-return-zero functions here. These
  // are dealt with in CIRGen.

  CIRToCIRArgMapping IRFunctionArgs(LM.getContext(), FI);
  assert(Fn.getNumArguments() == IRFunctionArgs.totalIRArgs());

  // If we're using inalloca, all the memory arguments are GEPs off of the last
  // parameter, which is a pointer to the complete memory area.
  assert(!::cir::MissingFeatures::inallocaArgs());

  // Name the struct return parameter.
  assert(!::cir::MissingFeatures::sretArgs());

  // Track if we received the parameter as a pointer (indirect, byval, or
  // inalloca). If already have a pointer, EmitParmDecl doesn't need to copy it
  // into a local alloca for us.
  SmallVector<Value, 8> ArgVals;
  ArgVals.reserve(Args.size());

  // Create a pointer value for every parameter declaration. This usually
  // entails copying one or more LLVM IR arguments into an alloca. Don't push
  // any cleanups or do anything that might unwind. We do that separately, so
  // we can push the cleanups in the correct order for the ABI.
  assert(FI.arg_size() == Args.size());
  unsigned ArgNo = 0;
  LowerFunctionInfo::const_arg_iterator info_it = FI.arg_begin();
  for (MutableArrayRef<BlockArgument>::const_iterator i = Args.begin(),
                                                      e = Args.end();
       i != e; ++i, ++info_it, ++ArgNo) {
    const Value Arg = *i;
    const ABIArgInfo &ArgI = info_it->info;

    bool isPromoted = ::cir::MissingFeatures::varDeclIsKNRPromoted();
    // We are converting from ABIArgInfo type to VarDecl type directly, unless
    // the parameter is promoted. In this case we convert to
    // CGFunctionInfo::ArgInfo type with subsequent argument demotion.
    Type Ty = {};
    if (isPromoted)
      llvm_unreachable("NYI");
    else
      Ty = Arg.getType();
    assert(!::cir::MissingFeatures::evaluationKind());

    unsigned FirstIRArg, NumIRArgs;
    std::tie(FirstIRArg, NumIRArgs) = IRFunctionArgs.getIRArgs(ArgNo);

    switch (ArgI.getKind()) {
    case ABIArgInfo::Extend:
    case ABIArgInfo::Direct: {
      auto AI = Fn.getArgument(FirstIRArg);
      Type LTy = Arg.getType();

      // Prepare parameter attributes. So far, only attributes for pointer
      // parameters are prepared. See
      // http://llvm.org/docs/LangRef.html#paramattrs.
      if (ArgI.getDirectOffset() == 0 && isa<PointerType>(LTy) &&
          isa<PointerType>(ArgI.getCoerceToType())) {
        llvm_unreachable("NYI");
      }

      // Prepare the argument value. If we have the trivial case, handle it
      // with no muss and fuss.
      if (!isa<StructType>(ArgI.getCoerceToType()) &&
          ArgI.getCoerceToType() == Ty && ArgI.getDirectOffset() == 0) {
        assert(NumIRArgs == 1);

        // LLVM expects swifterror parameters to be used in very restricted
        // ways. Copy the value into a less-restricted temporary.
        Value V = AI;
        if (::cir::MissingFeatures::extParamInfo()) {
          llvm_unreachable("NYI");
        }

        // Ensure the argument is the correct type.
        if (V.getType() != ArgI.getCoerceToType())
          llvm_unreachable("NYI");

        if (isPromoted)
          llvm_unreachable("NYI");

        ArgVals.push_back(V);

        // NOTE(cir): Here we have a trivial case, which means we can just
        // replace all uses of the original argument with the new one.
        Value oldArg = SrcFn.getArgument(ArgNo);
        Value newArg = Fn.getArgument(FirstIRArg);
        rewriter.replaceAllUsesWith(oldArg, newArg);

        break;
      }

      assert(!::cir::MissingFeatures::vectorType());

      // Allocate original argument to be "uncoerced".
      // FIXME(cir): We should have a alloca op builder that does not required
      // the pointer type to be explicitly passed.
      // FIXME(cir): Get the original name of the argument, as well as the
      // proper alignment for the given type being allocated.
      auto Alloca = rewriter.create<AllocaOp>(
          Fn.getLoc(), rewriter.getType<PointerType>(Ty), Ty,
          /*name=*/StringRef(""),
          /*alignment=*/rewriter.getI64IntegerAttr(4));

      Value Ptr = buildAddressAtOffset(*this, Alloca.getResult(), ArgI);

      // Fast-isel and the optimizer generally like scalar values better than
      // FCAs, so we flatten them if this is safe to do for this argument.
      StructType STy = dyn_cast<StructType>(ArgI.getCoerceToType());
      if (ArgI.isDirect() && ArgI.getCanBeFlattened() && STy &&
          STy.getNumElements() > 1) {
        llvm_unreachable("NYI");
      } else {
        // Simple case, just do a coerced store of the argument into the alloca.
        assert(NumIRArgs == 1);
        Value AI = Fn.getArgument(FirstIRArg);
        // TODO(cir): Set argument name in the new function.
        createCoercedStore(AI, Ptr, /*DstIsVolatile=*/false, *this);
      }

      // Match to what EmitParamDecl is expecting for this type.
      if (::cir::MissingFeatures::evaluationKind()) {
        llvm_unreachable("NYI");
      } else {
        // FIXME(cir): Should we have an ParamValue abstraction like in the
        // original codegen?
        ArgVals.push_back(Alloca);
      }

      // NOTE(cir): Once we have uncoerced the argument, we should be able to
      // RAUW the original argument alloca with the new one. This assumes that
      // the argument is used only to be stored in a alloca.
      Value arg = SrcFn.getArgument(ArgNo);
      assert(arg.hasOneUse());
      for (auto *firstStore : arg.getUsers()) {
        assert(isa<StoreOp>(firstStore));
        auto argAlloca = cast<StoreOp>(firstStore).getAddr();
        rewriter.replaceAllUsesWith(argAlloca, Alloca);
        rewriter.eraseOp(firstStore);
        rewriter.eraseOp(argAlloca.getDefiningOp());
      }

      break;
    }
    default:
      llvm_unreachable("Unhandled ABIArgInfo::Kind");
    }
  }

  if (getTarget().getCXXABI().areArgsDestroyedLeftToRightInCallee()) {
    llvm_unreachable("NYI");
  } else {
    // FIXME(cir): In the original codegen, EmitParamDecl is called here. It
    // is likely that said function considers ABI details during emission, so
    // we migth have to add a counter part here. Currently, it is not needed.
  }

  return success();
}

LogicalResult LowerFunction::buildFunctionEpilog(const LowerFunctionInfo &FI) {
  // NOTE(cir): no-return, naked, and no result functions should be handled in
  // CIRGen.

  Value RV = {};
  Type RetTy = FI.getReturnType();
  const ABIArgInfo &RetAI = FI.getReturnInfo();

  switch (RetAI.getKind()) {

  case ABIArgInfo::Ignore:
    break;

  case ABIArgInfo::Extend:
  case ABIArgInfo::Direct:
    // FIXME(cir): Should we call ConvertType(RetTy) here?
    if (RetAI.getCoerceToType() == RetTy && RetAI.getDirectOffset() == 0) {
      // The internal return value temp always will have pointer-to-return-type
      // type, just do a load.

      // If there is a dominating store to ReturnValue, we can elide
      // the load, zap the store, and usually zap the alloca.
      // NOTE(cir): This seems like a premature optimization case, so I'm
      // skipping it.
      if (::cir::MissingFeatures::returnValueDominatingStoreOptmiization()) {
        llvm_unreachable("NYI");
      }
      // Otherwise, we have to do a simple load.
      else {
        // NOTE(cir): Nothing to do here. The codegen already emitted this load
        // for us and there is no casting necessary to conform to the ABI. The
        // zero-extension is enforced by the return value's attribute. Just
        // early exit.
        return success();
      }
    } else {
      // NOTE(cir): Unlike the original codegen, CIR may have multiple return
      // statements in the function body. We have to handle this here.
      mlir::PatternRewriter::InsertionGuard guard(rewriter);
      NewFn->walk([&](ReturnOp returnOp) {
        rewriter.setInsertionPoint(returnOp);

        // TODO(cir): I'm not sure if we need this offset here or in CIRGen.
        // Perhaps both? For now I'm just ignoring it.
        // Value V = emitAddressAtOffset(*this, getResultAlloca(returnOp),
        // RetAI);

        RV = castReturnValue(returnOp->getOperand(0), RetAI.getCoerceToType(),
                             *this);
        rewriter.replaceOpWithNewOp<ReturnOp>(returnOp, RV);
      });
    }

    // TODO(cir): Should AutoreleaseResult be handled here?
    break;

  default:
    llvm_unreachable("Unhandled ABIArgInfo::Kind");
  }

  return success();
}

/// Generate code for a function based on the ABI-specific information.
///
/// This method has partial parity with CodeGenFunction::GenerateCode, but it
/// focuses on the ABI-specific details. So a lot of codegen stuff is removed.
LogicalResult LowerFunction::generateCode(FuncOp oldFn, FuncOp newFn,
                                          const LowerFunctionInfo &FnInfo) {
  assert(newFn && "generating code for null Function");
  auto Args = oldFn.getArguments();

  // Emit the ABI-specific function prologue.
  assert(newFn.empty() && "Function already has a body");
  rewriter.setInsertionPointToEnd(newFn.addEntryBlock());
  if (buildFunctionProlog(FnInfo, newFn, oldFn.getArguments()).failed())
    return failure();

  // Ensure that old ABI-agnostic arguments uses were replaced.
  const auto hasNoUses = [](Value val) { return val.getUses().empty(); };
  assert(std::all_of(Args.begin(), Args.end(), hasNoUses) && "Missing RAUW?");

  // Migrate function body to new ABI-aware function.
  assert(oldFn.getBody().hasOneBlock() &&
         "Multiple blocks in original function not supported");

  // Move old function body to new function.
  // FIXME(cir): The merge below is not very good: will not work if SrcFn has
  // multiple blocks and it mixes the new and old prologues.
  rewriter.mergeBlocks(&oldFn.getBody().front(), &newFn.getBody().front(),
                       newFn.getArguments());

  // FIXME(cir): What about saving parameters for corotines? Should we do
  // something about it in this pass? If the change with the calling
  // convention, we might have to handle this here.

  // Emit the standard function epilogue.
  if (buildFunctionEpilog(FnInfo).failed())
    return failure();

  return success();
}

void LowerFunction::buildAggregateStore(Value Val, Value Dest,
                                        bool DestIsVolatile) {
  // In LLVM codegen:
  // Function to store a first-class aggregate into memory. We prefer to
  // store the elements rather than the aggregate to be more friendly to
  // fast-isel.
  assert(mlir::isa<PointerType>(Dest.getType()) && "Storing in a non-pointer!");
  (void)DestIsVolatile;

  // Circumvent CIR's type checking.
  Type pointeeTy = mlir::cast<PointerType>(Dest.getType()).getPointee();
  if (Val.getType() != pointeeTy) {
    // NOTE(cir):  We only bitcast and store if the types have the same size.
    assert((LM.getDataLayout().getTypeSizeInBits(Val.getType()) ==
            LM.getDataLayout().getTypeSizeInBits(pointeeTy)) &&
           "Incompatible types");
    auto loc = Val.getLoc();
    Val = rewriter.create<CastOp>(loc, pointeeTy, CastKind::bitcast, Val);
  }

  rewriter.create<StoreOp>(Val.getLoc(), Val, Dest);
}

Value LowerFunction::buildAggregateBitcast(Value Val, Type DestTy) {
  return rewriter.create<CastOp>(Val.getLoc(), DestTy, CastKind::bitcast, Val);
}

/// Rewrite a call operation to abide to the ABI calling convention.
///
/// FIXME(cir): This method has partial parity to CodeGenFunction's
/// EmitCallEpxr method defined in CGExpr.cpp. This could likely be
/// removed in favor of a more direct approach.
LogicalResult LowerFunction::rewriteCallOp(CallOp op,
                                           ReturnValueSlot retValSlot) {

  // TODO(cir): Check if BlockCall, CXXMemberCall, CUDAKernelCall, or
  // CXXOperatorMember require special handling here. These should be handled
  // in CIRGen, unless there is call conv or ABI-specific stuff to be handled,
  // them we should do it here.

  // TODO(cir): Also check if Builtin and CXXPeseudoDtor need special handling
  // here. These should be handled in CIRGen, unless there is call conv or
  // ABI-specific stuff to be handled, them we should do it here.

  // NOTE(cir): There is no direct way to fetch the function type from the
  // CallOp, so we fetch it from the source function. This assumes the
  // function definition has not yet been lowered.
  assert(SrcFn && "No source function");
  auto fnType = SrcFn.getFunctionType();

  // Rewrite the call operation to abide to the ABI calling convention.
  auto Ret = rewriteCallOp(fnType, SrcFn, op, retValSlot);

  // Replace the original call result with the new one.
  if (Ret)
    rewriter.replaceAllUsesWith(op.getResult(), Ret);

  // Erase original ABI-agnostic call.
  rewriter.eraseOp(op);
  return success();
}

/// Rewrite a call operation to abide to the ABI calling convention.
///
/// FIXME(cir): This method has partial parity to CodeGenFunction's EmitCall
/// method defined in CGExpr.cpp. This could likely be removed in favor of a
/// more direct approach since most of the code here is exclusively CodeGen.
Value LowerFunction::rewriteCallOp(FuncType calleeTy, FuncOp origCallee,
                                   CallOp callOp, ReturnValueSlot retValSlot,
                                   Value Chain) {
  // NOTE(cir): Skip a bunch of function pointer stuff and AST declaration
  // asserts. Also skip sanitizers, as these should likely be handled at
  // CIRGen.
  CallArgList Args;
  if (Chain)
    llvm_unreachable("NYI");

  // NOTE(cir): Call args were already emitted in CIRGen. Skip the evaluation
  // order done in CIRGen and just fetch the exiting arguments here.
  Args = callOp.getArgOperands();

  const LowerFunctionInfo &FnInfo = LM.getTypes().arrangeFreeFunctionCall(
      callOp.getArgOperands(), calleeTy, /*chainCall=*/false);

  // C99 6.5.2.2p6:
  //   If the expression that denotes the called function has a type
  //   that does not include a prototype, [the default argument
  //   promotions are performed]. If the number of arguments does not
  //   equal the number of parameters, the behavior is undefined. If
  //   the function is defined with a type that includes a prototype,
  //   and either the prototype ends with an ellipsis (, ...) or the
  //   types of the arguments after promotion are not compatible with
  //   the types of the parameters, the behavior is undefined. If the
  //   function is defined with a type that does not include a
  //   prototype, and the types of the arguments after promotion are
  //   not compatible with those of the parameters after promotion,
  //   the behavior is undefined [except in some trivial cases].
  // That is, in the general case, we should assume that a call
  // through an unprototyped function type works like a *non-variadic*
  // call.  The way we make this work is to cast to the exact type
  // of the promoted arguments.
  //
  // Chain calls use this same code path to add the invisible chain parameter
  // to the function type.
  if (origCallee.getNoProto() || Chain) {
    llvm_unreachable("NYI");
  }

  assert(!::cir::MissingFeatures::CUDA());

  // TODO(cir): LLVM IR has the concept of "CallBase", which is a base class
  // for all types of calls. Perhaps we should have a CIR interface to mimic
  // this class.
  CallOp CallOrInvoke = {};
  Value CallResult =
      rewriteCallOp(FnInfo, origCallee, callOp, retValSlot, Args, CallOrInvoke,
                    /*isMustTail=*/false, callOp.getLoc());

  // NOTE(cir): Skipping debug stuff here.

  return CallResult;
}

// NOTE(cir): This method has partial parity to CodeGenFunction's EmitCall
// method in CGCall.cpp. When incrementing it, use the original codegen as a
// reference: add ABI-specific stuff and skip codegen stuff.
Value LowerFunction::rewriteCallOp(const LowerFunctionInfo &CallInfo,
                                   FuncOp Callee, CallOp Caller,
                                   ReturnValueSlot ReturnValue,
                                   CallArgList &CallArgs, CallOp CallOrInvoke,
                                   bool isMustTail, Location loc) {
  // FIXME: We no longer need the types from CallArgs; lift up and simplify.

  // Handle struct-return functions by passing a pointer to the
  // location that we would like to return into.
  Type RetTy = CallInfo.getReturnType(); // ABI-agnostic type.
  const ::cir::ABIArgInfo &RetAI = CallInfo.getReturnInfo();

  FuncType IRFuncTy = LM.getTypes().getFunctionType(CallInfo);

  // NOTE(cir): Some target/ABI related checks happen here. I'm skipping them
  // under the assumption that they are handled in CIRGen.

  // 1. Set up the arguments.

  // If we're using inalloca, insert the allocation after the stack save.
  // FIXME: Do this earlier rather than hacking it in here!
  if (StructType ArgStruct = CallInfo.getArgStruct()) {
    llvm_unreachable("NYI");
  }

  CIRToCIRArgMapping IRFunctionArgs(LM.getContext(), CallInfo);
  SmallVector<Value, 16> IRCallArgs(IRFunctionArgs.totalIRArgs());

  // If the call returns a temporary with struct return, create a temporary
  // alloca to hold the result, unless one is given to us.
  if (RetAI.isIndirect() || RetAI.isCoerceAndExpand() || RetAI.isInAlloca()) {
    llvm_unreachable("NYI");
  }

  assert(!::cir::MissingFeatures::swift());

  // NOTE(cir): Skipping lifetime markers here.

  // Translate all of the arguments as necessary to match the IR lowering.
  assert(CallInfo.arg_size() == CallArgs.size() &&
         "Mismatch between function signature & arguments.");
  unsigned ArgNo = 0;
  LowerFunctionInfo::const_arg_iterator info_it = CallInfo.arg_begin();
  for (auto I = CallArgs.begin(), E = CallArgs.end(); I != E;
       ++I, ++info_it, ++ArgNo) {
    const ABIArgInfo &ArgInfo = info_it->info;

    if (IRFunctionArgs.hasPaddingArg(ArgNo))
      llvm_unreachable("NYI");

    unsigned FirstIRArg, NumIRArgs;
    std::tie(FirstIRArg, NumIRArgs) = IRFunctionArgs.getIRArgs(ArgNo);

    switch (ArgInfo.getKind()) {
    case ABIArgInfo::Extend:
    case ABIArgInfo::Direct: {

      if (isa<BoolType>(info_it->type)) {
        IRCallArgs[FirstIRArg] = *I;
        break;
      }

      if (!isa<StructType>(ArgInfo.getCoerceToType()) &&
          ArgInfo.getCoerceToType() == info_it->type &&
          ArgInfo.getDirectOffset() == 0) {
        assert(NumIRArgs == 1);
        Value V;
        if (!isa<StructType>(I->getType())) {
          V = *I;
        } else {
          llvm_unreachable("NYI");
        }

        if (::cir::MissingFeatures::extParamInfo()) {
          llvm_unreachable("NYI");
        }

        if (ArgInfo.getCoerceToType() != V.getType() &&
            isa<IntType>(V.getType()))
          llvm_unreachable("NYI");

        if (FirstIRArg < IRFuncTy.getNumInputs() &&
            V.getType() != IRFuncTy.getInput(FirstIRArg))
          llvm_unreachable("NYI");

        if (::cir::MissingFeatures::undef())
          llvm_unreachable("NYI");
        IRCallArgs[FirstIRArg] = V;
        break;
      }

      // FIXME: Avoid the conversion through memory if possible.
      Value Src = {};
      if (!isa<StructType>(I->getType())) {
        llvm_unreachable("NYI");
      } else {
        // NOTE(cir): I'm leaving L/RValue stuff for CIRGen to handle.
        Src = *I;
      }

      // If the value is offst in memory, apply the offset now.
      // FIXME(cir): Is this offset already handled in CIRGen?
      Src = emitAddressAtOffset(*this, Src, ArgInfo);

      // Fast-isel and the optimizer generally like scalar values better than
      // FCAs, so we flatten them if this is safe to do for this argument.
      StructType STy = dyn_cast<StructType>(ArgInfo.getCoerceToType());
      if (STy && ArgInfo.isDirect() && ArgInfo.getCanBeFlattened()) {
        llvm_unreachable("NYI");
      } else {
        // In the simple case, just pass the coerced loaded value.
        assert(NumIRArgs == 1);
        Value Load = createCoercedValue(Src, ArgInfo.getCoerceToType(), *this);

        // FIXME(cir): We should probably handle CMSE non-secure calls here

        // since they are a ARM-specific feature.
        if (::cir::MissingFeatures::undef())
          llvm_unreachable("NYI");
        IRCallArgs[FirstIRArg] = Load;
      }

      break;
    }
    default:
      llvm::outs() << "Missing ABIArgInfo::Kind: " << ArgInfo.getKind() << "\n";
      llvm_unreachable("NYI");
    }
  }

  // 2. Prepare the function pointer.
  // NOTE(cir): This is not needed for CIR.

  // 3. Perform the actual call.

  // NOTE(cir): CIRGen handle when to "deactive" cleanups. We also skip some
  // debugging stuff here.

  // Update the largest vector width if any arguments have vector types.
  assert(!::cir::MissingFeatures::vectorType());

  // Compute the calling convention and attributes.

  // FIXME(cir): Skipping call attributes for now. Not sure if we have to do
  // this at all since we already do it for the function definition.

  // FIXME(cir): Implement the required procedures for strictfp function and
  // fast-math.

  // FIXME(cir): Add missing call-site attributes here if they are
  // ABI/target-specific, otherwise, do it in CIRGen.

  // NOTE(cir): Deciding whether to use Call or Invoke is done in CIRGen.

  // Rewrite the actual call operation.
  // TODO(cir): Handle other types of CIR calls (e.g. cir.try_call).
  // NOTE(cir): We don't know if the callee was already lowered, so we only
  // fetch the name from the callee, while the return type is fetch from the
  // lowering types manager.
  CallOp newCallOp = rewriter.create<CallOp>(
      loc, Caller.getCalleeAttr(), IRFuncTy.getReturnType(), IRCallArgs);
  auto extraAttrs =
      rewriter.getAttr<ExtraFuncAttributesAttr>(rewriter.getDictionaryAttr({}));
  newCallOp->setAttr("extra_attrs", extraAttrs);

  assert(!::cir::MissingFeatures::vectorType());

  // NOTE(cir): Skipping some ObjC, tail-call, debug, and attribute stuff
  // here.

  // 4. Finish the call.

  // NOTE(cir): Skipping no-return, isMustTail, swift error handling, and
  // writebacks here. These should be handled in CIRGen, I think.

  // Convert return value from ABI-agnostic to ABI-aware.
  Value Ret = [&] {
    // NOTE(cir): CIRGen already handled the emission of the return value. We
    // need only to handle the ABI-specific to ABI-agnostic cast here.
    switch (RetAI.getKind()) {

    case ::cir::ABIArgInfo::Ignore:
      // If we are ignoring an argument that had a result, make sure to
      // construct the appropriate return value for our caller.
      return getUndefRValue(RetTy);

    case ABIArgInfo::Extend:
    case ABIArgInfo::Direct: {
      Type RetIRTy = RetTy;
      if (RetAI.getCoerceToType() == RetIRTy && RetAI.getDirectOffset() == 0) {
        switch (getEvaluationKind(RetTy)) {
        case ::cir::TypeEvaluationKind::TEK_Scalar: {
          // If the argument doesn't match, perform a bitcast to coerce it.
          // This can happen due to trivial type mismatches. NOTE(cir):
          // Perhaps this section should handle CIR's boolean case.
          Value V = newCallOp.getResult();
          if (V.getType() != RetIRTy)
            llvm_unreachable("NYI");
          return V;
        }
        default:
          llvm_unreachable("NYI");
        }
      }

      // If coercing a fixed vector from a scalable vector for ABI
      // compatibility, and the types match, use the llvm.vector.extract
      // intrinsic to perform the conversion.
      if (::cir::MissingFeatures::vectorType()) {
        llvm_unreachable("NYI");
      }

      // FIXME(cir): Use return value slot here.
      Value RetVal = callOp.getResult();
      // TODO(cir): Check for volatile return values.

      // NOTE(cir): If the function returns, there should always be a valid
      // return value present. Instead of setting the return value here, we
      // should have the ReturnValueSlot object set it beforehand.
      if (!RetVal) {
        RetVal = callOp.getResult();
        // TODO(cir): Check for volatile return values.
      }

      // An empty record can overlap other data (if declared with
      // no_unique_address); omit the store for such types - as there is no
      // actual data to store.
      if (dyn_cast<StructType>(RetTy) &&
          cast<StructType>(RetTy).getNumElements() != 0) {
        // NOTE(cir): I'm assuming we don't need to change any offsets here.
        // Value StorePtr = emitAddressAtOffset(*this, RetVal, RetAI);
        RetVal =
            createCoercedValue(newCallOp.getResult(), RetVal.getType(), *this);
      }

      // NOTE(cir): No need to convert from a temp to an RValue. This is
      // done in CIRGen
      return RetVal;
    }
    default:
      llvm::errs() << "Unhandled ABIArgInfo kind: " << RetAI.getKind() << "\n";
      llvm_unreachable("NYI");
    }
  }();

  // NOTE(cir): Skipping Emissions, lifetime markers, and dtors here that
  // should be handled in CIRGen.

  return Ret;
}

// NOTE(cir): This method has partial parity to CodeGenFunction's
// GetUndefRValue defined in CGExpr.cpp.
Value LowerFunction::getUndefRValue(Type Ty) {
  if (isa<VoidType>(Ty))
    return nullptr;

  llvm::outs() << "Missing undef handler for value type: " << Ty << "\n";
  llvm_unreachable("NYI");
}

::cir::TypeEvaluationKind LowerFunction::getEvaluationKind(Type type) {
  // FIXME(cir): Implement type classes for CIR types.
  if (isa<StructType>(type))
    return ::cir::TypeEvaluationKind::TEK_Aggregate;
  if (isa<BoolType, IntType, SingleType, DoubleType>(type))
    return ::cir::TypeEvaluationKind::TEK_Scalar;
  llvm_unreachable("NYI");
}

} // namespace cir
} // namespace mlir
