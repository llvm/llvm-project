//===- CIROpsArith.h - CIR ops in partition `Arith` -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the CIR arithmetic ops: unary and binary arithmetic,
// comparisons, shifts and bit manipulation. The ops of the other partitions
// are forward declared; include their headers, or CIRDialect.h for all ops,
// to use them.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_IR_CIROPSARITH_H
#define CLANG_CIR_DIALECT_IR_CIROPSARITH_H

#include "clang/CIR/Dialect/IR/CIRDialectBase.h"

#define GET_OP_CLASSES
#include "clang/CIR/Dialect/IR/CIROpsArith.h.inc"

#endif // CLANG_CIR_DIALECT_IR_CIROPSARITH_H
