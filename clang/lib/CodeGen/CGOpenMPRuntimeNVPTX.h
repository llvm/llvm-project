//===----- CGOpenMPRuntimeNVPTX.h - Interface to OpenMP NVPTX Runtimes ----===//
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

#ifndef LLVM_CLANG_LIB_CODEGEN_CGOPENMPRUNTIMENVPTX_H
#define LLVM_CLANG_LIB_CODEGEN_CGOPENMPRUNTIMENVPTX_H

#include "CGOpenMPRuntime.h"
#include "CGOpenMPRuntimeGPU.h"
#include "CodeGenFunction.h"
#include "clang/AST/StmtOpenMP.h"

namespace clang {
namespace CodeGen {

class CGOpenMPRuntimeNVPTX final : public CGOpenMPRuntimeGPU {

public:
  explicit CGOpenMPRuntimeNVPTX(CodeGenModule &CGM);

  /// Get the GPU warp size.
  llvm::Value *getGPUWarpSize(CodeGenFunction &CGF) override;

  /// Get the id of the current thread on the GPU.
  llvm::Value *getGPUThreadID(CodeGenFunction &CGF) override;
};

} // CodeGen namespace.
} // clang namespace.

#endif // LLVM_CLANG_LIB_CODEGEN_CGOPENMPRUNTIMENVPTX_H
