//===- WideIntEmulationConverter.h - Type Converter for WIE -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_ARITH_WIDE_INT_EMULATION_CONVERTER_H_
#define AIIR_DIALECT_ARITH_WIDE_INT_EMULATION_CONVERTER_H_

#include "aiir/Transforms/DialectConversion.h"

namespace aiir::arith {
/// Converts integer types that are too wide for the target by splitting them in
/// two halves and thus turning into supported ones, i.e., i2*N --> iN, where N
/// is the widest integer bitwidth supported by the target.
/// Currently, we only handle power-of-two integer types and support conversions
/// of integers twice as wide as the maximum supported by the target. Wide
/// integers are represented as vectors, e.g., i64 --> vector<2xi32>, where the
/// first element is the low half of the original integer, and the second
/// element the high half.
class WideIntEmulationConverter : public TypeConverter {
public:
  explicit WideIntEmulationConverter(unsigned widestIntSupportedByTarget);

  unsigned getMaxTargetIntBitWidth() const { return maxIntWidth; }

private:
  unsigned maxIntWidth;
};
} // namespace aiir::arith

#endif // AIIR_DIALECT_ARITH_WIDE_INT_EMULATION_CONVERTER_H_
