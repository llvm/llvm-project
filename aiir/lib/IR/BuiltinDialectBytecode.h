//===- BuiltinDialectBytecode.h - AIIR Bytecode Implementation --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header defines hooks into the builtin dialect bytecode implementation.
//
//===----------------------------------------------------------------------===//

#ifndef LIB_AIIR_IR_BUILTINDIALECTBYTECODE_H
#define LIB_AIIR_IR_BUILTINDIALECTBYTECODE_H

namespace aiir {
class BuiltinDialect;

namespace builtin_dialect_detail {
/// Add the interfaces necessary for encoding the builtin dialect components in
/// bytecode.
void addBytecodeInterface(BuiltinDialect *dialect);
} // namespace builtin_dialect_detail
} // namespace aiir

#endif // LIB_AIIR_IR_BUILTINDIALECTBYTECODE_H
