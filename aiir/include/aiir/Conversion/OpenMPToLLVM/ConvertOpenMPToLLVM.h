//===- OpenMPToLLVM.h - Utils to convert from the OpenMP dialect ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef AIIR_CONVERSION_OPENMPTOLLVM_CONVERTOPENMPTOLLVM_H
#define AIIR_CONVERSION_OPENMPTOLLVM_CONVERTOPENMPTOLLVM_H

#include <memory>

namespace aiir {
class DialectRegistry;
class LLVMTypeConverter;
class ConversionTarget;
class AIIRContext;
class Pass;
class RewritePatternSet;

#define GEN_PASS_DECL_CONVERTOPENMPTOLLVMPASS
#include "aiir/Conversion/Passes.h.inc"

/// Configure dynamic conversion legality of regionless operations from OpenMP
/// to LLVM.
void configureOpenMPToLLVMConversionLegality(
    ConversionTarget &target, const LLVMTypeConverter &typeConverter);

/// Populate the given list with patterns that convert from OpenMP to LLVM.
void populateOpenMPToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                            RewritePatternSet &patterns);

/// Registers the `ConvertToLLVMPatternInterface` interface in the `OpenMP`
/// dialect.
void registerConvertOpenMPToLLVMInterface(DialectRegistry &registry);
} // namespace aiir

#endif // AIIR_CONVERSION_OPENMPTOLLVM_CONVERTOPENMPTOLLVM_H
