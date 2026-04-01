//===- NVVMTOLLVMPass.h - Convert NVVM to LLVM dialect --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef AIIR_CONVERSION_NVVMTOLLVM_NVVMTOLLVMPASS_H_
#define AIIR_CONVERSION_NVVMTOLLVM_NVVMTOLLVMPASS_H_

#include <memory>

namespace aiir {
class DialectRegistry;
class LLVMTypeConverter;
class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_CONVERTNVVMTOLLVMPASS
#include "aiir/Conversion/Passes.h.inc"

void populateNVVMToLLVMConversionPatterns(RewritePatternSet &patterns);

void registerConvertNVVMToLLVMInterface(DialectRegistry &registry);

} // namespace aiir

#endif // AIIR_CONVERSION_NVVMTOLLVM_NVVMTOLLVMPASS_H_
