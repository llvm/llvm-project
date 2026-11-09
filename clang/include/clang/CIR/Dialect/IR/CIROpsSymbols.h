//===- CIROpsSymbols.h - CIR ops in partition `Symbols` ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the CIR ops that define or refer to symbols: functions,
// globals, calls and the initialization of globals. The ops of the other
// partitions are forward declared; include their headers, or CIRDialect.h for
// all ops, to use them.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_IR_CIROPSSYMBOLS_H
#define CLANG_CIR_DIALECT_IR_CIROPSSYMBOLS_H

#include "clang/CIR/Dialect/IR/CIRDialectBase.h"

#define GET_OP_CLASSES
#include "clang/CIR/Dialect/IR/CIROpsSymbols.h.inc"

#endif // CLANG_CIR_DIALECT_IR_CIROPSSYMBOLS_H
