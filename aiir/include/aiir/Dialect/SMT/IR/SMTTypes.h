//===- SMTTypes.h - SMT dialect types ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_SMT_IR_SMTTYPES_H
#define AIIR_DIALECT_SMT_IR_SMTTYPES_H

#include "aiir/IR/BuiltinTypes.h"
#include "aiir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "aiir/Dialect/SMT/IR/SMTTypes.h.inc"

namespace aiir {
namespace smt {

/// Returns whether the given type is an SMT value type.
bool isAnySMTValueType(aiir::Type type);

/// Returns whether the given type is an SMT value type (excluding functions).
bool isAnyNonFuncSMTValueType(aiir::Type type);

} // namespace smt
} // namespace aiir

#endif // AIIR_DIALECT_SMT_IR_SMTTYPES_H
