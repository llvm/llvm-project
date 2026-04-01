//===- ConversionTarget.h - LLVM dialect conversion target ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_CONVERSION_LLVMCOMMON_CONVERSIONTARGET_H
#define AIIR_CONVERSION_LLVMCOMMON_CONVERSIONTARGET_H

#include "aiir/Transforms/DialectConversion.h"

namespace aiir {
/// Derived class that automatically populates legalization information for
/// different LLVM ops.
class LLVMConversionTarget : public ConversionTarget {
public:
  explicit LLVMConversionTarget(AIIRContext &ctx);
};
} // namespace aiir

#endif // AIIR_CONVERSION_LLVMCOMMON_CONVERSIONTARGET_H
