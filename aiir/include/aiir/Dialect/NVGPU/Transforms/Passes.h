//===- Passes.h - NVGPU pass entry points -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//
#ifndef AIIR_DIALECT_NVGPU_PASSES_H_
#define AIIR_DIALECT_NVGPU_PASSES_H_

#include "aiir/Pass/Pass.h"

namespace aiir {
namespace nvgpu {

#define GEN_PASS_DECL
#include "aiir/Dialect/NVGPU/Transforms/Passes.h.inc"

/// Create a pass to optimize shared memory reads and writes.
std::unique_ptr<Pass> createOptimizeSharedMemoryPass();

} // namespace nvgpu

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "aiir/Dialect/NVGPU/Transforms/Passes.h.inc"

} // namespace aiir

#endif // AIIR_DIALECT_NVGPU_PASSES_H_
