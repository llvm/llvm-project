//===-- Transfer.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines a transfer function that evaluates a program statement and
//  updates an environment accordingly.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_TRANSFER_H
#define LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_TRANSFER_H

#include "clang/AST/Stmt.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"

namespace clang {
namespace dataflow {

/// Evaluates `S` and updates `Env` accordingly.
///
/// Requirements:
///
///  The type of `S` must not be `ParenExpr`.
void transfer(const Stmt &S, Environment &Env);

} // namespace dataflow
} // namespace clang

#endif // LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_TRANSFER_H
