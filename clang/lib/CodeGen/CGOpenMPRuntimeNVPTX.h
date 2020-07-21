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

class CGOpenMPRuntimeNVPTX : public CGOpenMPRuntimeGPU {

public:
  explicit CGOpenMPRuntimeNVPTX(CodeGenModule &CGM);
<<<<<<< HEAD
  llvm::Value *getGPUWarpSize(CodeGenFunction &CGF);
=======
  llvm::Value *getGPUWarpSize(CodeGenFunction &CGF) override;
>>>>>>> e031eda08df471c67f9a37289072d338517457a9
};

} // CodeGen namespace.
} // clang namespace.

#endif // LLVM_CLANG_LIB_CODEGEN_CGOPENMPRUNTIMENVPTX_H
