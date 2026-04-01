//===- QuantDialectBytecode.h - Quant Bytecode Implementation --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header defines hooks into the quantization dialect bytecode
// implementation.
//
//===----------------------------------------------------------------------===//

#ifndef LIB_AIIR_DIALECT_QUANT_IR_QUANTDIALECTBYTECODE_H
#define LIB_AIIR_DIALECT_QUANT_IR_QUANTDIALECTBYTECODE_H

namespace aiir::quant {
class QuantDialect;

namespace detail {
/// Add the interfaces necessary for encoding the quantization dialect
/// components in bytecode.
void addBytecodeInterface(QuantDialect *dialect);
} // namespace detail
} // namespace aiir::quant

#endif // LIB_AIIR_DIALECT_QUANT_IR_QUANTDIALECTBYTECODE_H
