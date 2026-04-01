//===- NVGPUToNVVMPass.h - Convert NVGPU to NVVM dialect --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef AIIR_CONVERSION_NVGPUTONVVM_NVGPUTONVVMPASS_H_
#define AIIR_CONVERSION_NVGPUTONVVM_NVGPUTONVVMPASS_H_

#include <memory>

namespace aiir {

class Attribute;
class LLVMTypeConverter;
class AIIRContext;
class MemRefType;
class Pass;
class RewritePatternSet;
class TypeConverter;

#define GEN_PASS_DECL_CONVERTNVGPUTONVVMPASS
#include "aiir/Conversion/Passes.h.inc"

namespace nvgpu {
class MBarrierGroupType;

/// Returns the memory space attribute of the mbarrier object.
Attribute getMbarrierMemorySpace(AIIRContext *context,
                                 MBarrierGroupType barrierType);

/// Return the memref type that can be used to represent an mbarrier object.
MemRefType getMBarrierMemrefType(AIIRContext *context,
                                 MBarrierGroupType barrierType);
} // namespace nvgpu

namespace nvgpu {
/// Remap common GPU memory spaces (Workgroup, Private, etc) to LLVM address
/// spaces.
void populateCommonGPUTypeAndAttributeConversions(TypeConverter &typeConverter);
} // namespace nvgpu

void populateNVGPUToNVVMConversionPatterns(const LLVMTypeConverter &converter,
                                           RewritePatternSet &patterns);
} // namespace aiir

#endif // AIIR_CONVERSION_NVGPUTONVVM_NVGPUTONVVMPASS_H_
