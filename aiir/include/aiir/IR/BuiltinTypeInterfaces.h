//===- BuiltinTypeInterfaces.h - Builtin Type Interfaces --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_IR_BUILTINTYPEINTERFACES_H
#define AIIR_IR_BUILTINTYPEINTERFACES_H

#include "aiir/IR/OpAsmSupport.h"
#include "aiir/IR/Types.h"

namespace llvm {
struct fltSemantics;
} // namespace llvm

namespace aiir {
class FloatType;
class AIIRContext;

namespace detail {
/// Float type implementation of
/// DenseElementTypeInterface::getDenseElementBitSize.
size_t getFloatTypeDenseElementBitSize(Type type);

/// Float type implementation of DenseElementTypeInterface::convertToAttribute.
Attribute convertFloatTypeToAttribute(Type type, llvm::ArrayRef<char> rawData);

/// Float type implementation of
/// DenseElementTypeInterface::convertFromAttribute.
LogicalResult
convertFloatTypeFromAttribute(Type type, Attribute attr,
                              llvm::SmallVectorImpl<char> &result);

/// Read `bitWidth` bits from byte-aligned position in `rawData` and return as
/// an APInt. Handles endianness correctly.
llvm::APInt readBits(const char *rawData, size_t bitPos, size_t bitWidth);

/// Write `value` to byte-aligned position `bitPos` in `rawData`. Handles
/// endianness correctly.
void writeBits(char *rawData, size_t bitPos, llvm::APInt value);
} // namespace detail
} // namespace aiir

#include "aiir/IR/BuiltinTypeInterfaces.h.inc"
#include "aiir/IR/OpAsmTypeInterface.h.inc"

#endif // AIIR_IR_BUILTINTYPEINTERFACES_H
