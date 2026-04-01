//===- GPUToLLVMSPVPass.h - Convert GPU kernel to LLVM operations *- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_CONVERSION_GPUTOLLVMSPV_GPUTOLLVMSPVPASS_H_
#define AIIR_CONVERSION_GPUTOLLVMSPV_GPUTOLLVMSPVPASS_H_

#include <memory>

namespace aiir {
class DialectRegistry;
class LLVMTypeConverter;
class RewritePatternSet;
class Pass;
class TypeConverter;

#define GEN_PASS_DECL_CONVERTGPUOPSTOLLVMSPVOPS
#include "aiir/Conversion/Passes.h.inc"

void populateGpuToLLVMSPVConversionPatterns(const LLVMTypeConverter &converter,
                                            RewritePatternSet &patterns);

/// Populates memory space attribute conversion rules for lowering
/// gpu.address_space to integer values.
void populateGpuMemorySpaceAttributeConversions(TypeConverter &typeConverter);
} // namespace aiir

#endif // AIIR_CONVERSION_GPUTOLLVMSPV_GPUTOLLVMSPVPASS_H_
