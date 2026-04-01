//===- NarrowTypeEmulationConverter.h - Type Converter for NTE -----*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_ARITH_NARROW_TYPE_EMULATION_CONVERTER_H_
#define AIIR_DIALECT_ARITH_NARROW_TYPE_EMULATION_CONVERTER_H_

#include "aiir/Transforms/DialectConversion.h"

namespace aiir::arith {
/// Converts narrow integer or float types that are not supported
/// by the target hardware to wider types. Currently, we only
/// handle power-of-two integer types and convert them to wider
/// integers that are equal or larger than 8 bits.
class NarrowTypeEmulationConverter : public TypeConverter {
public:
  explicit NarrowTypeEmulationConverter(unsigned targetBitwidth);

  unsigned getLoadStoreBitwidth() const { return loadStoreBitwidth; }

private:
  unsigned loadStoreBitwidth;
};
} // namespace aiir::arith

#endif // AIIR_DIALECT_ARITH_NARROW_TYPE_EMULATION_CONVERTER_H_
