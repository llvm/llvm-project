//===----- CGCUDARuntime.cpp - Interface to CUDA Runtimes -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides an abstract class for CUDA code generation.  Concrete
// subclasses of this implement code generation for specific CUDA
// runtime libraries.
//
//===----------------------------------------------------------------------===//

#include "CGCUDARuntime.h"
#include "CGCall.h"
#include "CodeGenFunction.h"
#include "clang/AST/ExprCXX.h"

using namespace clang;
using namespace CodeGen;

CGCUDARuntime::~CGCUDARuntime() {}

static llvm::Value *emitGetParamBuf(CodeGenFunction &CGF,
                                    const CUDAKernelCallExpr *E) {
  auto *GetParamBuf = CGF.getContext().getcudaGetParameterBufferDecl();
  const FunctionProtoType *GetParamBufProto =
      GetParamBuf->getType()->getAs<FunctionProtoType>();

  DeclRefExpr *DRE = DeclRefExpr::Create(
      CGF.getContext(), {}, {}, GetParamBuf,
      /*RefersToEnclosingVariableOrCapture=*/false, GetParamBuf->getNameInfo(),
      GetParamBuf->getType(), VK_PRValue);
  auto *ImpCast = ImplicitCastExpr::Create(
      CGF.getContext(), CGF.getContext().getPointerType(GetParamBuf->getType()),
      CK_FunctionToPointerDecay, DRE, nullptr, VK_PRValue, FPOptionsOverride());

  CGCallee Callee = CGF.EmitCallee(ImpCast);
  CallArgList Args;
  // Use 64B alignment.
  Args.add(RValue::get(CGF.CGM.getSize(CharUnits::fromQuantity(64))),
           CGF.getContext().getSizeType());
  // Calculate parameter sizes.
  const PointerType *PT = E->getCallee()->getType()->getAs<PointerType>();
  const FunctionProtoType *FTP =
      PT->getPointeeType()->getAs<FunctionProtoType>();
  CharUnits Offset = CharUnits::Zero();
  for (auto ArgTy : FTP->getParamTypes()) {
    auto TInfo = CGF.CGM.getContext().getTypeInfoInChars(ArgTy);
    Offset = Offset.alignTo(TInfo.Align) + TInfo.Width;
  }
  Args.add(RValue::get(CGF.CGM.getSize(Offset)),
           CGF.getContext().getSizeType());
  const CGFunctionInfo &CallInfo = CGF.CGM.getTypes().arrangeFreeFunctionCall(
      Args, GetParamBufProto, /*ChainCall=*/false);
  auto Ret = CGF.EmitCall(CallInfo, Callee, /*ReturnValue=*/{}, Args);

  return Ret.getScalarVal();
}

RValue CGCUDARuntime::EmitCUDADeviceKernelCallExpr(
    CodeGenFunction &CGF, const CUDAKernelCallExpr *E,
    ReturnValueSlot ReturnValue, llvm::CallBase **CallOrInvoke) {
  assert(CGM.getContext().getcudaLaunchDeviceDecl() ==
         E->getConfig()->getDirectCallee());

  llvm::BasicBlock *ConfigOKBlock = CGF.createBasicBlock("dkcall.configok");
  llvm::BasicBlock *ContBlock = CGF.createBasicBlock("dkcall.end");

  llvm::Value *Config = emitGetParamBuf(CGF, E);
  CGF.Builder.CreateCondBr(
      CGF.Builder.CreateICmpNE(Config,
                               llvm::Constant::getNullValue(Config->getType())),
      ConfigOKBlock, ContBlock);

  CodeGenFunction::ConditionalEvaluation eval(CGF);

  eval.begin(CGF);
  CGF.EmitBlock(ConfigOKBlock);

  QualType KernelCalleeFuncTy =
      E->getCallee()->getType()->getAs<PointerType>()->getPointeeType();
  CGCallee KernelCallee = CGF.EmitCallee(E->getCallee());
  // Emit kernel arguments.
  CallArgList KernelCallArgs;
  CGF.EmitCallArgs(KernelCallArgs,
                   KernelCalleeFuncTy->getAs<FunctionProtoType>(),
                   E->arguments(), E->getDirectCallee());
  // Copy emitted kernel arguments into that parameter buffer.
  RawAddress CfgBase(Config, CGM.Int8Ty,
                     /*Alignment=*/CharUnits::fromQuantity(64));
  CharUnits Offset = CharUnits::Zero();
  for (auto &Arg : KernelCallArgs) {
    auto TInfo = CGM.getContext().getTypeInfoInChars(Arg.getType());
    Offset = Offset.alignTo(TInfo.Align);
    Address Addr =
        CGF.Builder.CreateConstInBoundsGEP(CfgBase, Offset.getQuantity());
    Arg.copyInto(CGF, Addr);
    Offset += TInfo.Width;
  }
  // Make `cudaLaunchDevice` call, i.e. E->getConfig().
  const CallExpr *LaunchCall = E->getConfig();
  QualType LaunchCalleeFuncTy = LaunchCall->getCallee()
                                    ->getType()
                                    ->getAs<PointerType>()
                                    ->getPointeeType();
  CGCallee LaunchCallee = CGF.EmitCallee(LaunchCall->getCallee());
  CallArgList LaunchCallArgs;
  CGF.EmitCallArgs(LaunchCallArgs,
                   LaunchCalleeFuncTy->getAs<FunctionProtoType>(),
                   LaunchCall->arguments(), LaunchCall->getDirectCallee());
  // Replace func and paramterbuffer arguments.
  LaunchCallArgs[0] = CallArg(RValue::get(KernelCallee.getFunctionPointer()),
                              CGM.getContext().VoidPtrTy);
  LaunchCallArgs[1] = CallArg(RValue::get(Config), CGM.getContext().VoidPtrTy);
  const CGFunctionInfo &LaunchCallInfo = CGM.getTypes().arrangeFreeFunctionCall(
      LaunchCallArgs, LaunchCalleeFuncTy->getAs<FunctionProtoType>(),
      /*ChainCall=*/false);
  CGF.EmitCall(LaunchCallInfo, LaunchCallee, ReturnValue, LaunchCallArgs,
               CallOrInvoke,
               /*IsMustTail=*/false, E->getExprLoc());
  CGF.EmitBranch(ContBlock);

  CGF.EmitBlock(ContBlock);
  eval.end(CGF);

  return RValue::get(nullptr);
}

RValue CGCUDARuntime::EmitCUDAKernelCallExpr(CodeGenFunction &CGF,
                                             const CUDAKernelCallExpr *E,
                                             ReturnValueSlot ReturnValue,
                                             llvm::CallBase **CallOrInvoke) {
  llvm::BasicBlock *ConfigOKBlock = CGF.createBasicBlock("kcall.configok");
  llvm::BasicBlock *ContBlock = CGF.createBasicBlock("kcall.end");

  CodeGenFunction::ConditionalEvaluation eval(CGF);
  CGF.EmitBranchOnBoolExpr(E->getConfig(), ContBlock, ConfigOKBlock,
                           /*TrueCount=*/0);

  eval.begin(CGF);
  CGF.EmitBlock(ConfigOKBlock);
  CGF.EmitSimpleCallExpr(E, ReturnValue, CallOrInvoke);
  CGF.EmitBranch(ContBlock);

  CGF.EmitBlock(ContBlock);
  eval.end(CGF);

  return RValue::get(nullptr);
}
