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
#include "clang/Analysis/FlowSensitive/WatchedLiteralsSolver.h"
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
  // B0 => B1
  ConstraintContext Ctx;
  auto B = Ctx.impl(Ctx.atom(), Ctx.atom());

  auto Expected = R"((=>
    B0
    B1))";
  EXPECT_THAT(debugString(*B), StrEq(Expected));
}

TEST(BoolValueDebugStringTest, Iff) {
  // B0 <=> B1
  ConstraintContext Ctx;
  auto B = Ctx.iff(Ctx.atom(), Ctx.atom());

  auto Expected = R"((=
    B0
    B1))";
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

TEST(ConstraintSetDebugStringTest, Empty) {
  llvm::DenseSet<BoolValue *> Constraints;
  EXPECT_THAT(debugString(Constraints), StrEq(""));
}

TEST(ConstraintSetDebugStringTest, Simple) {
  ConstraintContext Ctx;
  llvm::DenseSet<BoolValue *> Constraints;
  auto X = cast<AtomicBoolValue>(Ctx.atom());
  auto Y = cast<AtomicBoolValue>(Ctx.atom());
  Constraints.insert(Ctx.disj(X, Y));
  Constraints.insert(Ctx.disj(X, Ctx.neg(Y)));

  auto Expected = R"((or
    X
    (not
        Y))
(or
    X
    Y)
)";
  EXPECT_THAT(debugString(Constraints, {{X, "X"}, {Y, "Y"}}),
              StrEq(Expected));
}

Solver::Result CheckSAT(std::vector<BoolValue *> Constraints) {
  llvm::DenseSet<BoolValue *> ConstraintsSet(Constraints.begin(),
                                             Constraints.end());
  return WatchedLiteralsSolver().solve(std::move(ConstraintsSet));
}

TEST(SATCheckDebugStringTest, AtomicBoolean) {
  // B0
  ConstraintContext Ctx;
  std::vector<BoolValue *> Constraints({Ctx.atom()});
  auto Result = CheckSAT(Constraints);

  auto Expected = R"(
Constraints
------------
B0
------------
Satisfiable.

B0 = True
)";
  EXPECT_THAT(debugString(Constraints, Result), StrEq(Expected));
}

TEST(SATCheckDebugStringTest, AtomicBooleanAndNegation) {
  // B0, !B0
  ConstraintContext Ctx;
  auto B0 = Ctx.atom();
  std::vector<BoolValue *> Constraints({B0, Ctx.neg(B0)});
  auto Result = CheckSAT(Constraints);

  auto Expected = R"(
Constraints
------------
B0

(not
    B0)
------------
Unsatisfiable.

)";
  EXPECT_THAT(debugString(Constraints, Result), StrEq(Expected));
}

TEST(SATCheckDebugStringTest, MultipleAtomicBooleans) {
  // B0, B1
  ConstraintContext Ctx;
  std::vector<BoolValue *> Constraints({Ctx.atom(), Ctx.atom()});
  auto Result = CheckSAT(Constraints);

  auto Expected = R"(
Constraints
------------
B0

B1
------------
Satisfiable.

B0 = True
B1 = True
)";
  EXPECT_THAT(debugString(Constraints, Result), StrEq(Expected));
}

TEST(SATCheckDebugStringTest, Implication) {
  // B0, B0 => B1
  ConstraintContext Ctx;
  auto B0 = Ctx.atom();
  auto B1 = Ctx.atom();
  auto Impl = Ctx.disj(Ctx.neg(B0), B1);
  std::vector<BoolValue *> Constraints({B0, Impl});
  auto Result = CheckSAT(Constraints);

  auto Expected = R"(
Constraints
------------
B0

(or
    (not
        B0)
    B1)
------------
Satisfiable.

B0 = True
B1 = True
)";
  EXPECT_THAT(debugString(Constraints, Result), StrEq(Expected));
}

TEST(SATCheckDebugStringTest, Iff) {
  // B0, B0 <=> B1
  ConstraintContext Ctx;
  auto B0 = Ctx.atom();
  auto B1 = Ctx.atom();
  auto Iff = Ctx.conj(Ctx.disj(Ctx.neg(B0), B1), Ctx.disj(B0, Ctx.neg(B1)));
  std::vector<BoolValue *> Constraints({B0, Iff});
  auto Result = CheckSAT(Constraints);

  auto Expected = R"(
Constraints
------------
B0

(and
    (or
        (not
            B0)
        B1)
    (or
        B0
        (not
            B1)))
------------
Satisfiable.

B0 = True
B1 = True
)";
  EXPECT_THAT(debugString(Constraints, Result), StrEq(Expected));
}

TEST(SATCheckDebugStringTest, Xor) {
  // B0, XOR(B0, B1)
  ConstraintContext Ctx;
  auto B0 = Ctx.atom();
  auto B1 = Ctx.atom();
  auto XOR = Ctx.disj(Ctx.conj(B0, Ctx.neg(B1)), Ctx.conj(Ctx.neg(B0), B1));
  std::vector<BoolValue *> Constraints({B0, XOR});
  auto Result = CheckSAT(Constraints);

  auto Expected = R"(
Constraints
------------
B0

(or
    (and
        B0
        (not
            B1))
    (and
        (not
            B0)
        B1))
------------
Satisfiable.

B0 = True
B1 = False
)";
  EXPECT_THAT(debugString(Constraints, Result), StrEq(Expected));
}

TEST(SATCheckDebugStringTest, ComplexBooleanWithNames) {
  // Cond, (Cond ^ Then ^ !Else) v (!Cond ^ !Then ^ Else)
  ConstraintContext Ctx;
  auto Cond = cast<AtomicBoolValue>(Ctx.atom());
  auto Then = cast<AtomicBoolValue>(Ctx.atom());
  auto Else = cast<AtomicBoolValue>(Ctx.atom());
  auto B = Ctx.disj(Ctx.conj(Cond, Ctx.conj(Then, Ctx.neg(Else))),
                    Ctx.conj(Ctx.neg(Cond), Ctx.conj(Ctx.neg(Then), Else)));
  std::vector<BoolValue *> Constraints({Cond, B});
  auto Result = CheckSAT(Constraints);

  auto Expected = R"(
Constraints
------------
Cond

(or
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
            Else)))
------------
Satisfiable.

Cond = True
Else = False
Then = True
)";
  EXPECT_THAT(debugString(Constraints, Result,
                          {{Cond, "Cond"}, {Then, "Then"}, {Else, "Else"}}),
              StrEq(Expected));
}

TEST(SATCheckDebugStringTest, ComplexBooleanWithLongNames) {
  // ThisIntEqZero, !IntsAreEq, ThisIntEqZero ^ OtherIntEqZero => IntsAreEq
  ConstraintContext Ctx;
  auto IntsAreEq = cast<AtomicBoolValue>(Ctx.atom());
  auto ThisIntEqZero = cast<AtomicBoolValue>(Ctx.atom());
  auto OtherIntEqZero = cast<AtomicBoolValue>(Ctx.atom());
  auto BothZeroImpliesEQ =
      Ctx.disj(Ctx.neg(Ctx.conj(ThisIntEqZero, OtherIntEqZero)), IntsAreEq);
  std::vector<BoolValue *> Constraints(
      {ThisIntEqZero, Ctx.neg(IntsAreEq), BothZeroImpliesEQ});
  auto Result = CheckSAT(Constraints);

  auto Expected = R"(
Constraints
------------
ThisIntEqZero

(not
    IntsAreEq)

(or
    (not
        (and
            ThisIntEqZero
            OtherIntEqZero))
    IntsAreEq)
------------
Satisfiable.

IntsAreEq      = False
OtherIntEqZero = False
ThisIntEqZero  = True
)";
  EXPECT_THAT(debugString(Constraints, Result,
                          {{IntsAreEq, "IntsAreEq"},
                           {ThisIntEqZero, "ThisIntEqZero"},
                           {OtherIntEqZero, "OtherIntEqZero"}}),
              StrEq(Expected));
}
} // namespace
