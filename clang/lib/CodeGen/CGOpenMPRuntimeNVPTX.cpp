//===---- CGOpenMPRuntimeNVPTX.cpp - Interface to OpenMP NVPTX Runtimes ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides a class for OpenMP runtime code generation specialized to NVPTX
// targets from generalized CGOpenMPRuntimeGPU class.
//
//===----------------------------------------------------------------------===//

#include "CGOpenMPRuntimeNVPTX.h"
#include "CGOpenMPRuntimeGPU.h"
#include "CodeGenFunction.h"
#include "clang/AST/Attr.h"
#include "clang/AST/DeclOpenMP.h"
#include "clang/AST/StmtOpenMP.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/Cuda.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/IntrinsicsNVPTX.h"

using namespace clang;
using namespace CodeGen;
using namespace llvm::omp;

CGOpenMPRuntimeNVPTX::CGOpenMPRuntimeNVPTX(CodeGenModule &CGM)
    : CGOpenMPRuntimeGPU(CGM) {
  if (!CGM.getLangOpts().OpenMPIsDevice)
    llvm_unreachable("OpenMP NVPTX can only handle device code.");
}

llvm::Value *CGOpenMPRuntimeNVPTX::getGPUWarpSize(CodeGenFunction &CGF) {
  return CGF.EmitRuntimeCall(
      llvm::Intrinsic::getDeclaration(
          &CGF.CGM.getModule(), llvm::Intrinsic::nvvm_read_ptx_sreg_warpsize),
      "nvptx_warp_size");
}

llvm::Value *CGOpenMPRuntimeNVPTX::getGPUThreadID(CodeGenFunction &CGF) {
  CGBuilderTy &Bld = CGF.Builder;
  llvm::Function *F;
  F = llvm::Intrinsic::getDeclaration(
      &CGF.CGM.getModule(), llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x);
  return Bld.CreateCall(F, llvm::None, "nvptx_tid");
}
