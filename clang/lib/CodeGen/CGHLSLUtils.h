
//===----- CGHLSLUtils.h - Utility functions for HLSL CodeGen ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This File Provides utility function for HLSL code generation.
// It is used to abstract away implementation details of backends.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CODEGEN_CGHLSLUTILS_H
#define LLVM_CLANG_LIB_CODEGEN_CGHLSLUTILS_H

#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsDirectX.h"
#include "llvm/IR/IntrinsicsSPIRV.h"

// Define the function generator macro
#define GENERATE_HLSL_INTRINSIC_FUNCTION(name)                                 \
  static llvm::Intrinsic::ID get_hlsl_##name##_intrinsic(                      \
      const llvm::Triple::ArchType Arch) {                                     \
    switch (Arch) {                                                            \
    case llvm::Triple::dxil:                                                   \
      return llvm::Intrinsic::dx_##name;                                       \
    case llvm::Triple::spirv:                                                  \
      return llvm::Intrinsic::spv_##name;                                      \
    default:                                                                   \
      llvm_unreachable("Input semantic not supported by target");              \
    }                                                                          \
  }

class CGHLSLUtils {
public:
  GENERATE_HLSL_INTRINSIC_FUNCTION(all)
  GENERATE_HLSL_INTRINSIC_FUNCTION(thread_id)
private:
  CGHLSLUtils() = delete;
};

#endif
