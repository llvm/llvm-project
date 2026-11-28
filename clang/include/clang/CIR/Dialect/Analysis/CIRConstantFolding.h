//===- CIRConstantFolding.h - Fold CIR values without rewriting -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares queries that fold a CIR value to a constant by running
// the operation folders speculatively, leaving the IR unchanged, for analyses
// that must not rewrite what they inspect.
//
// MLIR's folding drivers (OperationFolder, the greedy rewriter) materialize
// what they fold in the IR, and its dataflow constant propagation solves a
// whole function at once; these queries fold only the value asked about.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_ANALYSIS_CIRCONSTANTFOLDING_H
#define CLANG_CIR_DIALECT_ANALYSIS_CIRCONSTANTFOLDING_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Value.h"
#include <optional>

namespace cir {

/// Fold `value` to a constant attribute, or return null. Operands fold
/// recursively, and region operations such as `cir.ternary` fold to what the
/// regions they may enter yield.
mlir::Attribute foldToConstant(mlir::Value value);

/// Fold a `!cir.bool` value to its constant value, if it has one.
std::optional<bool> tryFoldConstantBool(mlir::Value value);

} // namespace cir

#endif // CLANG_CIR_DIALECT_ANALYSIS_CIRCONSTANTFOLDING_H
