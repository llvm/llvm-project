//===- CIROpsCore.h - CIR ops in partition `Core` ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the core CIR ops: casts, constants, memory accesses and
// record member accesses. The ops of the other partitions are forward
// declared; include their headers, or CIRDialect.h for all ops, to use them.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_IR_CIROPSCORE_H
#define CLANG_CIR_DIALECT_IR_CIROPSCORE_H

#include "clang/CIR/Dialect/IR/CIRDialectBase.h"

#define GET_OP_CLASSES
#include "clang/CIR/Dialect/IR/CIROpsCore.h.inc"

namespace cir {
/// The alloca that defines \p addr, looking through casts that preserve the
/// underlying storage.  Null when the chain does not end at an alloca.
///
/// Use the result only to inspect the allocation, such as its alignment or
/// its allocated type.  Addressing has to keep using \p addr, whose type and
/// address space may differ from the alloca's own result.
AllocaOp getUnderlyingAlloca(mlir::Value addr);
} // namespace cir

#endif // CLANG_CIR_DIALECT_IR_CIROPSCORE_H
