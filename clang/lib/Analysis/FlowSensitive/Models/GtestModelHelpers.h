//===-- GtestModelHelpers.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines helpers for handling gtest constructs in dataflow models.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_ANALYSIS_FLOWSENSITIVE_MODELS_GTESTMODELHELPERS_H
#define CLANG_ANALYSIS_FLOWSENSITIVE_MODELS_GTESTMODELHELPERS_H

#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/StorageLocation.h"
#include <cassert>

namespace clang {
namespace dataflow {
namespace gtest {
void transferAssertionResultExpectationOperatorBoolCall(
    const CXXMemberCallExpr *Expr, Environment &Env,
    llvm::function_ref<StorageLocation &(RecordStorageLocation &)> GetOk);

clang::ast_matchers::StatementMatcher
isAssertionResultExpectationOperatorBoolCall();
} // namespace gtest
} // namespace dataflow
} // namespace clang

#endif // CLANG_ANALYSIS_FLOWSENSITIVE_MODELS_GTESTMODELHELPERS_H
