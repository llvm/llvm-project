//===-- TosaToTensor.h - TOSA to Tensor legalization ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the passes for the TOSA to Standard Dialect conversion.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_CONVERSION_TOSATOTENSOR_TOSATOTENSOR_H
#define AIIR_CONVERSION_TOSATOTENSOR_TOSATOTENSOR_H

#include "aiir/Pass/Pass.h"

namespace aiir {
class TypeConverter;

#define GEN_PASS_DECL_TOSATOTENSORPASS
#include "aiir/Conversion/Passes.h.inc"

namespace tosa {

void populateTosaToTensorConversionPatterns(const TypeConverter &converter,
                                            RewritePatternSet *patterns);

} // namespace tosa
} // namespace aiir

#endif // AIIR_CONVERSION_TOSATOTENSOR_TOSATOTENSOR_H
