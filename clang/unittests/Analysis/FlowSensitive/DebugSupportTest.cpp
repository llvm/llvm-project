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

TEST(BoolValueDebugStringTest, Literal) {
  ConstraintContext Ctx;
  EXPECT_EQ("true", llvm::to_string(*Ctx.literal(true)));
  EXPECT_EQ("false", llvm::to_string(*Ctx.literal(false)));
}

TEST(BoolValueDebugStringTest, Negation) {
  ConstraintContext Ctx;
  auto B = Ctx.neg(Ctx.atom());

  auto Expected = "!V0";
  EXPECT_THAT(llvm::to_string(*B), StrEq(Expected));
}

TEST(BoolValueDebugStringTest, Conjunction) {
  ConstraintContext Ctx;
  auto *V0 = Ctx.atom();
  auto *V1 = Ctx.atom();
  EXPECT_EQ("(V0 & V1)", llvm::to_string(*Ctx.conj(V0, V1)));
}

TEST(BoolValueDebugStringTest, Disjunction) {
  ConstraintContext Ctx;
  auto *V0 = Ctx.atom();
  auto *V1 = Ctx.atom();
  EXPECT_EQ("(V0 | V1)", llvm::to_string(*Ctx.disj(V0, V1)));
}

TEST(BoolValueDebugStringTest, Implication) {
  ConstraintContext Ctx;
  auto *V0 = Ctx.atom();
  auto *V1 = Ctx.atom();
  EXPECT_EQ("(V0 => V1)", llvm::to_string(*Ctx.impl(V0, V1)));
}

TEST(BoolValueDebugStringTest, Iff) {
  ConstraintContext Ctx;
  auto *V0 = Ctx.atom();
  auto *V1 = Ctx.atom();
  EXPECT_EQ("(V0 = V1)", llvm::to_string(*Ctx.iff(V0, V1)));
}

TEST(BoolValueDebugStringTest, Xor) {
  ConstraintContext Ctx;
  auto V0 = Ctx.atom();
  auto V1 = Ctx.atom();
  auto B = Ctx.disj(Ctx.conj(V0, Ctx.neg(V1)), Ctx.conj(Ctx.neg(V0), V1));

  auto Expected = "((V0 & !V1) | (!V0 & V1))";
  EXPECT_THAT(llvm::to_string(*B), StrEq(Expected));
}

TEST(BoolValueDebugStringTest, NestedBoolean) {
  ConstraintContext Ctx;
  auto V0 = Ctx.atom();
  auto V1 = Ctx.atom();
  auto V2 = Ctx.atom();
  auto V3 = Ctx.atom();
  auto V4 = Ctx.atom();
  auto B = Ctx.conj(V0, Ctx.disj(V1, Ctx.conj(V2, Ctx.disj(V3, V4))));

  auto Expected = "(V0 & (V1 | (V2 & (V3 | V4))))";
  EXPECT_THAT(llvm::to_string(*B), StrEq(Expected));
}

TEST(BoolValueDebugStringTest, ComplexBooleanWithSomeNames) {
  ConstraintContext Ctx;
  auto X = Ctx.atom();
  auto Y = Ctx.atom();
  Formula::AtomNames Names;
  Names[X->getAtom()] = "X";
  Names[Y->getAtom()] = "Y";
  auto B = Ctx.disj(Ctx.conj(Y, Ctx.atom()), Ctx.disj(X, Ctx.atom()));

  auto Expected = R"(((Y & V2) | (X | V3)))";
  std::string Actual;
  llvm::raw_string_ostream OS(Actual);
  B->print(OS, &Names);
  EXPECT_THAT(Actual, StrEq(Expected));
}

} // namespace
