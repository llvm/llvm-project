//===--- CGOpenMPRuntimeAMDGCN.h - Interface to OpenMP AMDGCN Runtimes ---===//
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

#ifndef LLVM_CLANG_LIB_CODEGEN_CGOPENMPRUNTIMEAMDGCN_H
#define LLVM_CLANG_LIB_CODEGEN_CGOPENMPRUNTIMEAMDGCN_H

#include "CGOpenMPRuntime.h"
#include "CGOpenMPRuntimeGPU.h"
#include "CodeGenFunction.h"
#include "clang/AST/StmtOpenMP.h"

namespace clang {
namespace CodeGen {

class CGOpenMPRuntimeAMDGCN final : public CGOpenMPRuntimeGPU {

public:
  explicit CGOpenMPRuntimeAMDGCN(CodeGenModule &CGM);

  /// Get the GPU warp size.
  llvm::Value *getGPUWarpSize(CodeGenFunction &CGF) override;

  /// Get the id of the current thread on the GPU.
  llvm::Value *getGPUThreadID(CodeGenFunction &CGF) override;

  /// Get the maximum number of threads in a block of the GPU.
  llvm::Value *getGPUNumThreads(CodeGenFunction &CGF) override;
};

} // namespace CodeGen
} // namespace clang

#endif // LLVM_CLANG_LIB_CODEGEN_CGOPENMPRUNTIMEAMDGCN_H
