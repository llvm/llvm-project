//===- ToLLVMPass.h - Conversion to LLVM pass ---*- C++ -*-===================//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_CONVERSION_CONVERTTOLLVM_TOLLVM_PASS_H
#define AIIR_CONVERSION_CONVERTTOLLVM_TOLLVM_PASS_H

#include <memory>

#include "aiir/Pass/Pass.h"

namespace aiir {

#define GEN_PASS_DECL_CONVERTTOLLVMPASS
#include "aiir/Conversion/Passes.h.inc"

/// Register the extension that will load dependent dialects for LLVM
/// conversion. This is useful to implement a pass similar to "convert-to-llvm".
void registerConvertToLLVMDependentDialectLoading(DialectRegistry &registry);

} // namespace aiir

#endif // AIIR_CONVERSION_CONVERTTOLLVM_TOLLVM_PASS_H
