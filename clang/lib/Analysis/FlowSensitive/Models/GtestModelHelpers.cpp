//===-- GtestModelHelpers.cpp -----------------------------------*- C++ -*-===//
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

#include "GtestModelHelpers.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/FlowSensitive/StorageLocation.h"
#include "llvm/ADT/STLFunctionalExtras.h"

using namespace clang::dataflow::gtest;
using namespace clang::dataflow;
using namespace clang;

void clang::dataflow::gtest::transferAssertionResultExpectationOperatorBoolCall(
    const CXXMemberCallExpr *Expr, Environment &Env,
    llvm::function_ref<StorageLocation &(RecordStorageLocation &)> GetOk) {
  auto *RecordLoc = getImplicitObjectLocation(*Expr, Env);
  if (RecordLoc == nullptr)
    return;
  RecordStorageLocation *AssertionResultLoc = nullptr;
  StorageLocation *ExpectedResultLoc = nullptr;
  for (auto [Field, ChildLoc] : RecordLoc->children()) {
    if (Field->getName() == "assertion_result")
      AssertionResultLoc = dyn_cast_or_null<RecordStorageLocation>(ChildLoc);
    else if (Field->getName() == "expected_result")
      ExpectedResultLoc = ChildLoc;
  }
  if (AssertionResultLoc == nullptr || ExpectedResultLoc == nullptr)
    return;
  BoolValue *SuccessVal = Env.get<BoolValue>(GetOk(*AssertionResultLoc));
  BoolValue *ExpectedVal = Env.get<BoolValue>(*ExpectedResultLoc);
  if (SuccessVal == nullptr || ExpectedVal == nullptr)
    return;
  auto &A = Env.arena();
  auto &Res = Env.makeAtomicBoolValue();
  Env.assume(A.makeEquals(Res.formula(), A.makeEquals(SuccessVal->formula(),
                                                      ExpectedVal->formula())));
  Env.setValue(*Expr, Res);
}

clang::ast_matchers::StatementMatcher
clang::dataflow::gtest::isAssertionResultExpectationOperatorBoolCall() {
  using namespace clang::ast_matchers;
  return cxxMemberCallExpr(
      on(expr(unless(cxxThisExpr()))),
      callee(cxxMethodDecl(
          hasName("operator bool"),
          ofClass(hasName("testing::internal::AssertionResultExpectation")))));
}
