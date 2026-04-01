//==-- GPUToNVVM.td - GPU Ops to NVVM Patterns ---------------*- tablegen -*==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines Patterns to lower GPU ops to NVVM.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_CONVERSION_GPUTONVVM_TD
#define AIIR_CONVERSION_GPUTONVVM_TD

include "aiir/IR/PatternBase.td"
include "aiir/Dialect/GPU/IR/GPUOps.td"
include "aiir/Dialect/LLVMIR/NVVMOps.td"

def : Pat<(GPU_BarrierOp : $op $memory_fence), (NVVM_Barrier0Op)>;

#endif // AIIR_CONVERSION_GPUTONVVM_TD
