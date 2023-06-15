//===- unittests/Analysis/FlowSensitive/DebugSupportTest.cpp --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/FlowSensitive/DebugSupport.h"
#include "TestingSupport.h"
#include "clang/Analysis/FlowSensitive/Formula.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/raw_ostream.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace {

using namespace clang;
using namespace dataflow;

using test::ConstraintContext;
using testing::StrEq;

TEST(BoolValueDebugStringTest, AtomicBoolean) {
  ConstraintContext Ctx;
  auto B = Ctx.atom();

  auto Expected = "V0";
  EXPECT_THAT(llvm::to_string(*B), StrEq(Expected));
}

TEST(BoolValueDebugStringTest, Negation) {
  ConstraintContext Ctx;
  auto B = Ctx.neg(Ctx.atom());

  auto Expected = "!V0";
  EXPECT_THAT(llvm::to_string(*B), StrEq(Expected));
}

TEST(BoolValueDebugStringTest, Conjunction) {
  ConstraintContext Ctx;
  auto B = Ctx.conj(Ctx.atom(), Ctx.atom());

  auto Expected = "(V0 & V1)";
  EXPECT_THAT(llvm::to_string(*B), StrEq(Expected));
}

TEST(BoolValueDebugStringTest, Disjunction) {
  ConstraintContext Ctx;
  auto B = Ctx.disj(Ctx.atom(), Ctx.atom());

  auto Expected = "(V0 | V1)";
  EXPECT_THAT(llvm::to_string(*B), StrEq(Expected));
}

TEST(BoolValueDebugStringTest, Implication) {
  ConstraintContext Ctx;
  auto B = Ctx.impl(Ctx.atom(), Ctx.atom());

  auto Expected = "(V0 => V1)";
  EXPECT_THAT(llvm::to_string(*B), StrEq(Expected));
}

TEST(BoolValueDebugStringTest, Iff) {
  ConstraintContext Ctx;
  auto B = Ctx.iff(Ctx.atom(), Ctx.atom());

  auto Expected = "(V0 = V1)";
  EXPECT_THAT(llvm::to_string(*B), StrEq(Expected));
}

TEST(BoolValueDebugStringTest, Xor) {
  ConstraintContext Ctx;
  auto B0 = Ctx.atom();
  auto B1 = Ctx.atom();
  auto B = Ctx.disj(Ctx.conj(B0, Ctx.neg(B1)), Ctx.conj(Ctx.neg(B0), B1));

  auto Expected = "((V0 & !V1) | (!V0 & V1))";
  EXPECT_THAT(llvm::to_string(*B), StrEq(Expected));
}

TEST(BoolValueDebugStringTest, NestedBoolean) {
  ConstraintContext Ctx;
  auto B = Ctx.conj(
      Ctx.atom(),
      Ctx.disj(Ctx.atom(),
               Ctx.conj(Ctx.atom(), Ctx.disj(Ctx.atom(), Ctx.atom()))));

  auto Expected = "(V0 & (V1 | (V2 & (V3 | V4))))";
  EXPECT_THAT(llvm::to_string(*B), StrEq(Expected));
}

TEST(BoolValueDebugStringTest, ComplexBooleanWithSomeNames) {
  ConstraintContext Ctx;
  auto True = Ctx.atom();
  auto False = Ctx.atom();
  Formula::AtomNames Names;
  Names[True->getAtom()] = "true";
  Names[False->getAtom()] = "false";
  auto B = Ctx.disj(Ctx.conj(False, Ctx.atom()), Ctx.disj(True, Ctx.atom()));

  auto Expected = R"(((false & V2) | (true | V3)))";
  std::string Actual;
  llvm::raw_string_ostream OS(Actual);
  B->print(OS, &Names);
  EXPECT_THAT(Actual, StrEq(Expected));
}

} // namespace
