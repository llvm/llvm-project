//===- GPUToNVVMPass.h - Convert GPU kernel to NVVM dialect -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef AIIR_CONVERSION_GPUTONVVM_GPUTONVVMPASS_H_
#define AIIR_CONVERSION_GPUTONVVM_GPUTONVVMPASS_H_

#include "aiir/Conversion/LLVMCommon/LoweringOptions.h"
#include "aiir/Dialect/LLVMIR/LLVMTypes.h"
#include "aiir/IR/PatternMatch.h"
#include <memory>

namespace aiir {
class LLVMTypeConverter;
class ConversionTarget;
class RewritePatternSet;
class Pass;

namespace gpu {
class GPUModuleOp;
class MMAMatrixType;
} // namespace gpu

#define GEN_PASS_DECL_CONVERTGPUOPSTONVVMOPS
#include "aiir/Conversion/Passes.h.inc"

Type convertMMAToLLVMType(gpu::MMAMatrixType type);

/// Configure target to convert from the GPU dialect to NVVM.
void configureGpuToNVVMConversionLegality(ConversionTarget &target);

/// Configure the LLVM type convert to convert types and address spaces from the
/// GPU dialect to NVVM.
void configureGpuToNVVMTypeConverter(LLVMTypeConverter &converter);

/// Collect a set of patterns to convert from the GPU dialect to NVVM.
void populateGpuToNVVMConversionPatterns(const LLVMTypeConverter &converter,
                                         RewritePatternSet &patterns,
                                         PatternBenefit benefit = 1);

/// Populate GpuSubgroupReduce pattern to NVVM. It generates a specific nvvm
/// op that is not available on every GPU.
void populateGpuSubgroupReduceOpLoweringPattern(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns,
    PatternBenefit benefit = 1);

/// Collect a set of patterns to convert WMMA ops from GPU dialect to NVVM.
void populateGpuWMMAToNVVMConversionPatterns(const LLVMTypeConverter &converter,
                                             RewritePatternSet &patterns,
                                             PatternBenefit benefit = 1);
} // namespace aiir

#endif // AIIR_CONVERSION_GPUTONVVM_GPUTONVVMPASS_H_
