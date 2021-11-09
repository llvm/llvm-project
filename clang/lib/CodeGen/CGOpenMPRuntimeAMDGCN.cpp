//===-- CGOpenMPRuntimeAMDGCN.cpp - Interface to OpenMP AMDGCN Runtimes --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides a class for OpenMP runtime code generation specialized to
// AMDGCN targets from generalized CGOpenMPRuntimeGPU class.
//
//===----------------------------------------------------------------------===//

#include "CGOpenMPRuntimeAMDGCN.h"
#include "CGOpenMPRuntimeGPU.h"
#include "CodeGenFunction.h"
#include "clang/AST/Attr.h"
#include "clang/AST/DeclOpenMP.h"
#include "clang/AST/StmtOpenMP.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/Cuda.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Frontend/OpenMP/OMPGridValues.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"

using namespace clang;
using namespace CodeGen;
using namespace llvm::omp;

CGOpenMPRuntimeAMDGCN::CGOpenMPRuntimeAMDGCN(CodeGenModule &CGM)
    : CGOpenMPRuntimeGPU(CGM) {
  if (!CGM.getLangOpts().OpenMPIsDevice)
    llvm_unreachable("OpenMP AMDGCN can only handle device code.");
}

llvm::Value *CGOpenMPRuntimeAMDGCN::getGPUWarpSize(CodeGenFunction &CGF) {
  CGBuilderTy &Bld = CGF.Builder;
  // return constant compile-time target-specific warp size
  unsigned WarpSize = CGF.getTarget().getGridValue().GV_Warp_Size;
  return Bld.getInt32(WarpSize);
}

llvm::Value *CGOpenMPRuntimeAMDGCN::getGPUThreadID(CodeGenFunction &CGF) {
  CGBuilderTy &Bld = CGF.Builder;
  llvm::Function *F =
      CGF.CGM.getIntrinsic(llvm::Intrinsic::amdgcn_workitem_id_x);
  return Bld.CreateCall(F, llvm::None, "nvptx_tid");
}
