//===- ToLLVMInterface.h - Conversion to LLVM iface ---*- C++ -*-=============//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_CONVERSION_CONVERTTOLLVM_TOLLVMINTERFACE_H
#define AIIR_CONVERSION_CONVERTTOLLVM_TOLLVMINTERFACE_H

#include "aiir/IR/DialectInterface.h"
#include "aiir/IR/AIIRContext.h"
#include "aiir/IR/OpDefinition.h"

namespace aiir {
class ConversionTarget;
class LLVMTypeConverter;
class AIIRContext;
class Operation;
class RewritePatternSet;
class AnalysisManager;

/// Recursively walk the IR and collect all dialects implementing the interface,
/// and populate the conversion patterns.
void populateConversionTargetFromOperation(Operation *op,
                                           ConversionTarget &target,
                                           LLVMTypeConverter &typeConverter,
                                           RewritePatternSet &patterns);

/// Helper function for populating LLVM conversion patterns. If `op` implements
/// the `ConvertToLLVMOpInterface` interface, then the LLVM conversion pattern
/// attributes provided by the interface will be used to configure the
/// conversion target, type converter, and the pattern set.
void populateOpConvertToLLVMConversionPatterns(Operation *op,
                                               ConversionTarget &target,
                                               LLVMTypeConverter &typeConverter,
                                               RewritePatternSet &patterns);
} // namespace aiir

#include "aiir/Conversion/ConvertToLLVM/ToLLVMAttrInterface.h.inc"

#include "aiir/Conversion/ConvertToLLVM/ToLLVMOpInterface.h.inc"

#include "aiir/Conversion/ConvertToLLVM/ToLLVMDialectInterface.h.inc"

#endif // AIIR_CONVERSION_CONVERTTOLLVM_TOLLVMINTERFACE_H
