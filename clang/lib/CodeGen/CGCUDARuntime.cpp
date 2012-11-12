//===----- CGCUDARuntime.cpp - Interface to CUDA Runtimes -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This provides an abstract class for CUDA code generation.  Concrete
// subclasses of this implement code generation for specific CUDA
// runtime libraries.
//
//===----------------------------------------------------------------------===//

#include "CGCUDARuntime.h"
#include "clang/AST/Decl.h"
#include "clang/AST/ExprCXX.h"
#include "CGCall.h"
#include "CodeGenFunction.h"

using namespace clang;
using namespace CodeGen;

CGCUDARuntime::~CGCUDARuntime() {}

RValue CGCUDARuntime::EmitCUDAKernelCallExpr(CodeGenFunction &CGF,
                                             const CUDAKernelCallExpr *E,
                                             ReturnValueSlot ReturnValue) {
  llvm::BasicBlock *ConfigOKBlock = CGF.createBasicBlock("kcall.configok");
  llvm::BasicBlock *ContBlock = CGF.createBasicBlock("kcall.end");

  CodeGenFunction::ConditionalEvaluation eval(CGF);
  CGF.EmitBranchOnBoolExpr(E->getConfig(), ContBlock, ConfigOKBlock);

  eval.begin(CGF);
  CGF.EmitBlock(ConfigOKBlock);

  const Decl *TargetDecl = 0;
  if (const ImplicitCastExpr *CE = dyn_cast<ImplicitCastExpr>(E->getCallee())) {
    if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(CE->getSubExpr())) {
      TargetDecl = DRE->getDecl();
    }
  }

  llvm::Value *Callee = CGF.EmitScalarExpr(E->getCallee());
  CGF.EmitCall(E->getCallee()->getType(), Callee, ReturnValue,
               E->arg_begin(), E->arg_end(), TargetDecl);
  CGF.EmitBranch(ContBlock);

  CGF.EmitBlock(ContBlock);
  eval.end(CGF);

  return RValue::get(0);
}
