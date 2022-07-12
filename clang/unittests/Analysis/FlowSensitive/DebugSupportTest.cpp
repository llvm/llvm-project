//===- unittests/Analysis/FlowSensitive/DebugSupportTest.cpp --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/FlowSensitive/DebugSupport.h"
#include "TestingSupport.h"
#include "clang/Analysis/FlowSensitive/Value.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace {

using namespace clang;
using namespace dataflow;

using test::ConstraintContext;
using testing::StrEq;

TEST(BoolValueDebugStringTest, AtomicBoolean) {
  // B0
  ConstraintContext Ctx;
  auto B = Ctx.atom();

  auto Expected = R"(B0)";
  debugString(*B);
  EXPECT_THAT(debugString(*B), StrEq(Expected));
}

TEST(BoolValueDebugStringTest, Negation) {
  // !B0
  ConstraintContext Ctx;
  auto B = Ctx.neg(Ctx.atom());

  auto Expected = R"((not
    B0))";
  EXPECT_THAT(debugString(*B), StrEq(Expected));
}

TEST(BoolValueDebugStringTest, Conjunction) {
  // B0 ^ B1
  ConstraintContext Ctx;
  auto B = Ctx.conj(Ctx.atom(), Ctx.atom());

  auto Expected = R"((and
    B0
    B1))";
  EXPECT_THAT(debugString(*B), StrEq(Expected));
}

TEST(BoolValueDebugStringTest, Disjunction) {
  // B0 v B1
  ConstraintContext Ctx;
  auto B = Ctx.disj(Ctx.atom(), Ctx.atom());

  auto Expected = R"((or
    B0
    B1))";
  EXPECT_THAT(debugString(*B), StrEq(Expected));
}

TEST(BoolValueDebugStringTest, Implication) {
  // B0 => B1, implemented as !B0 v B1
  ConstraintContext Ctx;
  auto B = Ctx.disj(Ctx.neg(Ctx.atom()), Ctx.atom());

  auto Expected = R"((or
    (not
        B0)
    B1))";
  EXPECT_THAT(debugString(*B), StrEq(Expected));
}

TEST(BoolValueDebugStringTest, Iff) {
  // B0 <=> B1, implemented as (!B0 v B1) ^ (B0 v !B1)
  ConstraintContext Ctx;
  auto B0 = Ctx.atom();
  auto B1 = Ctx.atom();
  auto B = Ctx.conj(Ctx.disj(Ctx.neg(B0), B1), Ctx.disj(B0, Ctx.neg(B1)));

  auto Expected = R"((and
    (or
        (not
            B0)
        B1)
    (or
        B0
        (not
            B1))))";
  EXPECT_THAT(debugString(*B), StrEq(Expected));
}

TEST(BoolValueDebugStringTest, Xor) {
  // (B0 ^ !B1) V (!B0 ^ B1)
  ConstraintContext Ctx;
  auto B0 = Ctx.atom();
  auto B1 = Ctx.atom();
  auto B = Ctx.disj(Ctx.conj(B0, Ctx.neg(B1)), Ctx.conj(Ctx.neg(B0), B1));

  auto Expected = R"((or
    (and
        B0
        (not
            B1))
    (and
        (not
            B0)
        B1)))";
  EXPECT_THAT(debugString(*B), StrEq(Expected));
}

TEST(BoolValueDebugStringTest, NestedBoolean) {
  // B0 ^ (B1 v (B2 ^ (B3 v B4)))
  ConstraintContext Ctx;
  auto B = Ctx.conj(
      Ctx.atom(),
      Ctx.disj(Ctx.atom(),
               Ctx.conj(Ctx.atom(), Ctx.disj(Ctx.atom(), Ctx.atom()))));

  auto Expected = R"((and
    B0
    (or
        B1
        (and
            B2
            (or
                B3
                B4)))))";
  EXPECT_THAT(debugString(*B), StrEq(Expected));
}

TEST(BoolValueDebugStringTest, AtomicBooleanWithName) {
  // True
  ConstraintContext Ctx;
  auto True = cast<AtomicBoolValue>(Ctx.atom());
  auto B = True;

  auto Expected = R"(True)";
  EXPECT_THAT(debugString(*B, {{True, "True"}}), StrEq(Expected));
}

TEST(BoolValueDebugStringTest, ComplexBooleanWithNames) {
  // (Cond ^ Then ^ !Else) v (!Cond ^ !Then ^ Else)
  ConstraintContext Ctx;
  auto Cond = cast<AtomicBoolValue>(Ctx.atom());
  auto Then = cast<AtomicBoolValue>(Ctx.atom());
  auto Else = cast<AtomicBoolValue>(Ctx.atom());
  auto B = Ctx.disj(Ctx.conj(Cond, Ctx.conj(Then, Ctx.neg(Else))),
                    Ctx.conj(Ctx.neg(Cond), Ctx.conj(Ctx.neg(Then), Else)));

  auto Expected = R"((or
    (and
        Cond
        (and
            Then
            (not
                Else)))
    (and
        (not
            Cond)
        (and
            (not
                Then)
            Else))))";
  EXPECT_THAT(debugString(*B, {{Cond, "Cond"}, {Then, "Then"}, {Else, "Else"}}),
              StrEq(Expected));
}

TEST(BoolValueDebugStringTest, ComplexBooleanWithSomeNames) {
  // (False && B0) v (True v B1)
  ConstraintContext Ctx;
  auto True = cast<AtomicBoolValue>(Ctx.atom());
  auto False = cast<AtomicBoolValue>(Ctx.atom());
  auto B = Ctx.disj(Ctx.conj(False, Ctx.atom()), Ctx.disj(True, Ctx.atom()));

  auto Expected = R"((or
    (and
        False
        B0)
    (or
        True
        B1)))";
  EXPECT_THAT(debugString(*B, {{True, "True"}, {False, "False"}}),
              StrEq(Expected));
}

} // namespace
