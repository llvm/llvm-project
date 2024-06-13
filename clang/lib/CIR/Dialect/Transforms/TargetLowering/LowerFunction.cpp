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
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "clang/CIR/ABIArgInfo.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/Support/ErrorHandling.h"

using ABIArgInfo = ::cir::ABIArgInfo;

namespace mlir {
namespace cir {

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
    llvm_unreachable("NYI");
  }

  if (getTarget().getCXXABI().areArgsDestroyedLeftToRightInCallee()) {
    llvm_unreachable("NYI");
  } else {
    // FIXME(cir): In the original codegen, EmitParamDecl is called here. It is
    // likely that said function considers ABI details during emission, so we
    // migth have to add a counter part here. Currently, it is not needed.
  }

  return success();
}

LogicalResult LowerFunction::buildFunctionEpilog(const LowerFunctionInfo &FI) {
  const ABIArgInfo &RetAI = FI.getReturnInfo();

  switch (RetAI.getKind()) {

  case ABIArgInfo::Ignore:
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

/// Rewrite a call operation to abide to the ABI calling convention.
///
/// FIXME(cir): This method has partial parity to CodeGenFunction's
/// EmitCallEpxr method defined in CGExpr.cpp. This could likely be
/// removed in favor of a more direct approach.
LogicalResult LowerFunction::rewriteCallOp(CallOp op,
                                           ReturnValueSlot retValSlot) {

  // TODO(cir): Check if BlockCall, CXXMemberCall, CUDAKernelCall, or
  // CXXOperatorMember require special handling here. These should be handled in
  // CIRGen, unless there is call conv or ABI-specific stuff to be handled, them
  // we should do it here.

  // TODO(cir): Also check if Builtin and CXXPeseudoDtor need special handling
  // here. These should be handled in CIRGen, unless there is call conv or
  // ABI-specific stuff to be handled, them we should do it here.

  // NOTE(cir): There is no direct way to fetch the function type from the
  // CallOp, so we fetch it from the source function. This assumes the function
  // definition has not yet been lowered.
  assert(SrcFn && "No source function");
  auto fnType = SrcFn.getFunctionType();

  // Rewrite the call operation to abide to the ABI calling convention.
  auto Ret = rewriteCallOp(fnType, SrcFn, op, retValSlot);

  // Replace the original call result with the new one.
  if (Ret)
    rewriter.replaceAllUsesWith(op.getResult(), Ret);

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

  // TODO(cir): LLVM IR has the concept of "CallBase", which is a base class for
  // all types of calls. Perhaps we should have a CIR interface to mimic this
  // class.
  CallOp CallOrInvoke = {};
  Value CallResult = {};
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
    llvm_unreachable("NYI");
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
  CallOp callOp = rewriter.create<CallOp>(loc, Caller.getCalleeAttr(),
                                          IRFuncTy.getReturnType(), IRCallArgs);
  callOp.setExtraAttrsAttr(Caller.getExtraAttrs());

  assert(!::cir::MissingFeatures::vectorType());

  // NOTE(cir): Skipping some ObjC, tail-call, debug, and attribute stuff here.

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
    default:
      llvm::errs() << "Unhandled ABIArgInfo kind: " << RetAI.getKind() << "\n";
      llvm_unreachable("NYI");
    }
  }();

  // NOTE(cir): Skipping Emissions, lifetime markers, and dtors here that should
  // be handled in CIRGen.

  return Ret;
}

// NOTE(cir): This method has partial parity to CodeGenFunction's GetUndefRValue
// defined in CGExpr.cpp.
Value LowerFunction::getUndefRValue(Type Ty) {
  if (Ty.isa<VoidType>())
    return nullptr;

  llvm::outs() << "Missing undef handler for value type: " << Ty << "\n";
  llvm_unreachable("NYI");
}

} // namespace cir
} // namespace mlir
