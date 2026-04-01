//===-- Passes.td - NvGpu pass definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_NVGPU_PASSES_TD_
#define AIIR_DIALECT_NVGPU_PASSES_TD_

include "aiir/Pass/PassBase.td"

def OptimizeSharedMemory : Pass<"nvgpu-optimize-shared-memory"> {
  let summary = "Optimizes accesses to shard memory memrefs in order to reduce bank conflicts.";
  let constructor = "aiir::nvgpu::createOptimizeSharedMemoryPass()";
  let dependentDialects = [
    "memref::MemRefDialect", "vector::VectorDialect"
  ];
}

#endif // AIIR_DIALECT_NVGPU_PASSES_TD_
