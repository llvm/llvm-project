//===- unittests/Analysis/FlowSensitive/SimplifyConstraintsTest.cpp -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/FlowSensitive/SimplifyConstraints.h"
#include "TestingSupport.h"
#include "clang/Analysis/FlowSensitive/Arena.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace {

using namespace clang;
using namespace dataflow;

using ::testing::ElementsAre;
using ::testing::IsEmpty;

class SimplifyConstraintsTest : public ::testing::Test {
protected:
  llvm::SetVector<const Formula *> parse(StringRef Lines) {
    std::vector<const Formula *> formulas = test::parseFormulas(A, Lines);
    llvm::SetVector<const Formula *> Constraints(formulas.begin(),
                                                 formulas.end());
    return Constraints;
  }

  llvm::SetVector<const Formula *> simplify(StringRef Lines,
                                            SimplifyConstraintsInfo &Info) {
    llvm::SetVector<const Formula *> Constraints = parse(Lines);
    simplifyConstraints(Constraints, A, &Info);
    return Constraints;
  }

  Arena A;
};

void printConstraints(const llvm::SetVector<const Formula *> &Constraints,
                      raw_ostream &OS) {
  if (Constraints.empty()) {
    OS << "empty";
    return;
  }
  for (const auto *Constraint : Constraints) {
    Constraint->print(OS);
    OS << "\n";
  }
}

std::string
constraintsToString(const llvm::SetVector<const Formula *> &Constraints) {
  std::string Str;
  llvm::raw_string_ostream OS(Str);
  printConstraints(Constraints, OS);
  return Str;
}

MATCHER_P(EqualsConstraints, Constraints,
          "constraints are: " + constraintsToString(Constraints)) {
  if (arg == Constraints)
    return true;

  if (result_listener->stream()) {
    llvm::raw_os_ostream OS(*result_listener->stream());
    OS << "constraints are: ";
    printConstraints(arg, OS);
  }
  return false;
}

TEST_F(SimplifyConstraintsTest, TriviallySatisfiable) {
  SimplifyConstraintsInfo Info;
  EXPECT_THAT(simplify(R"(
     V0
  )",
                       Info),
              EqualsConstraints(parse("")));
  EXPECT_THAT(Info.EquivalentAtoms, IsEmpty());
  EXPECT_THAT(Info.TrueAtoms, ElementsAre(Atom(0)));
  EXPECT_THAT(Info.FalseAtoms, IsEmpty());
}

TEST_F(SimplifyConstraintsTest, SimpleContradiction) {
  SimplifyConstraintsInfo Info;
  EXPECT_THAT(simplify(R"(
     V0
     !V0
  )",
                       Info),
              EqualsConstraints(parse("false")));
  EXPECT_THAT(Info.EquivalentAtoms, IsEmpty());
  EXPECT_THAT(Info.TrueAtoms, IsEmpty());
  EXPECT_THAT(Info.FalseAtoms, IsEmpty());
}

TEST_F(SimplifyConstraintsTest, ContradictionThroughEquivalence) {
  SimplifyConstraintsInfo Info;
  EXPECT_THAT(simplify(R"(
     (V0 = V1)
     V0
     !V1
  )",
                       Info),
              EqualsConstraints(parse("false")));
  EXPECT_THAT(Info.EquivalentAtoms, IsEmpty());
  EXPECT_THAT(Info.TrueAtoms, IsEmpty());
  EXPECT_THAT(Info.FalseAtoms, IsEmpty());
}

TEST_F(SimplifyConstraintsTest, EquivalenceChain) {
  SimplifyConstraintsInfo Info;
  EXPECT_THAT(simplify(R"(
     (V0 | V3)
     (V1 = V2)
     (V2 = V3)
  )",
                       Info),
              EqualsConstraints(parse("(V0 | V1)")));
  EXPECT_THAT(Info.EquivalentAtoms,
              ElementsAre(ElementsAre(Atom(1), Atom(2), Atom(3))));
  EXPECT_THAT(Info.TrueAtoms, IsEmpty());
  EXPECT_THAT(Info.FalseAtoms, IsEmpty());
}

TEST_F(SimplifyConstraintsTest, TrueAndFalseAtomsSimplifyOtherExpressions) {
  SimplifyConstraintsInfo Info;
  EXPECT_THAT(simplify(R"(
    V0
    !V1
    (V0 & (V2 => V3))
    (V1 | (V4 => V5))
  )",
                       Info),
              EqualsConstraints(parse(R"(
    (V2 => V3)
    (V4 => V5)
  )")));
  EXPECT_THAT(Info.EquivalentAtoms, IsEmpty());
  EXPECT_THAT(Info.TrueAtoms, ElementsAre(Atom(0)));
  EXPECT_THAT(Info.FalseAtoms, ElementsAre(Atom(1)));
}

TEST_F(SimplifyConstraintsTest, TrueAtomUnlocksEquivalenceChain) {
  SimplifyConstraintsInfo Info;
  EXPECT_THAT(simplify(R"(
     V0
     (V0 & (V1 = V2))
     (V0 & (V2 = V3))
  )",
                       Info),
              EqualsConstraints(parse("")));
  EXPECT_THAT(Info.EquivalentAtoms,
              ElementsAre(ElementsAre(Atom(1), Atom(2), Atom(3))));
  EXPECT_THAT(Info.TrueAtoms, ElementsAre(Atom(0)));
  EXPECT_THAT(Info.FalseAtoms, IsEmpty());
}

TEST_F(SimplifyConstraintsTest, TopLevelAndSplitIntoMultipleConstraints) {
  SimplifyConstraintsInfo Info;
  EXPECT_THAT(simplify(R"(
     ((V0 => V1) & (V2 => V3))
  )",
                       Info),
              EqualsConstraints(parse(R"(
    (V0 => V1)
    (V2 => V3)
  )")));
  EXPECT_THAT(Info.EquivalentAtoms, IsEmpty());
  EXPECT_THAT(Info.TrueAtoms, IsEmpty());
  EXPECT_THAT(Info.FalseAtoms, IsEmpty());
}

} // namespace
