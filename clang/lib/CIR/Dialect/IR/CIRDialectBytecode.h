//===- CIRDialectBytecode.h - CIR Bytecode Implementation -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header defines hooks into the CIR dialect bytecode implementation.
//
//===----------------------------------------------------------------------===//

#ifndef LIB_CLANG_CIR_DIALECT_IR_CIRDIALECTBYTECODE_H
#define LIB_CLANG_CIR_DIALECT_IR_CIRDIALECTBYTECODE_H

namespace cir {
class CIRDialect;

namespace detail {
/// Add the interfaces necessary for encoding the CIR dialect components in
/// bytecode.
void addBytecodeInterface(CIRDialect *dialect);
} // namespace detail
} // namespace cir

#endif // LIB_CLANG_CIR_DIALECT_IR_CIRDIALECTBYTECODE_H
