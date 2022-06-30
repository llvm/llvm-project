//===- unittests/Analysis/FlowSensitive/MatchSwitchTest.cpp ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines a simplistic version of Constant Propagation as an example
//  of a forward, monotonic dataflow analysis. The analysis tracks all
//  variables in the scope, but lacks escape analysis.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/FlowSensitive/MatchSwitch.h"
#include "TestingSupport.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/FlowSensitive/DataflowAnalysis.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/DataflowLattice.h"
#include "clang/Analysis/FlowSensitive/MapLattice.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Testing/Support/Annotations.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <utility>

using namespace clang;
using namespace dataflow;

namespace {
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

class BooleanLattice {
public:
  BooleanLattice() : Value(false) {}
  explicit BooleanLattice(bool B) : Value(B) {}

  static BooleanLattice bottom() { return BooleanLattice(false); }

  static BooleanLattice top() { return BooleanLattice(true); }

  LatticeJoinEffect join(BooleanLattice Other) {
    auto Prev = Value;
    Value = Value || Other.Value;
    return Prev == Value ? LatticeJoinEffect::Unchanged
                         : LatticeJoinEffect::Changed;
  }

  friend bool operator==(BooleanLattice LHS, BooleanLattice RHS) {
    return LHS.Value == RHS.Value;
  }

  friend std::ostream &operator<<(std::ostream &Os, const BooleanLattice &B) {
    Os << B.Value;
    return Os;
  }

  bool value() const { return Value; }

private:
  bool Value;
};
} // namespace

MATCHER_P(Holds, m,
          ((negation ? "doesn't hold" : "holds") +
           llvm::StringRef(" a lattice element that ") +
           ::testing::DescribeMatcher<BooleanLattice>(m, negation))
              .str()) {
  return ExplainMatchResult(m, arg.Lattice, result_listener);
}

void TransferSetTrue(const DeclRefExpr *,
                     const ast_matchers::MatchFinder::MatchResult &,
                     TransferState<BooleanLattice> &State) {
  State.Lattice = BooleanLattice(true);
}

void TransferSetFalse(const Stmt *,
                      const ast_matchers::MatchFinder::MatchResult &,
                      TransferState<BooleanLattice> &State) {
  State.Lattice = BooleanLattice(false);
}

class TestAnalysis : public DataflowAnalysis<TestAnalysis, BooleanLattice> {
  MatchSwitch<TransferState<BooleanLattice>> TransferSwitch;

public:
  explicit TestAnalysis(ASTContext &Context)
      : DataflowAnalysis<TestAnalysis, BooleanLattice>(Context) {
    using namespace ast_matchers;
    TransferSwitch =
        MatchSwitchBuilder<TransferState<BooleanLattice>>()
            .CaseOf<DeclRefExpr>(declRefExpr(to(varDecl(hasName("X")))),
                                 TransferSetTrue)
            .CaseOf<Stmt>(callExpr(callee(functionDecl(hasName("Foo")))),
                          TransferSetFalse)
            .Build();
  }

  static BooleanLattice initialElement() { return BooleanLattice::bottom(); }

  void transfer(const Stmt *S, BooleanLattice &L, Environment &Env) {
    TransferState<BooleanLattice> State(L, Env);
    TransferSwitch(*S, getASTContext(), State);
  }
};

class MatchSwitchTest : public ::testing::Test {
protected:
  template <typename Matcher>
  void RunDataflow(llvm::StringRef Code, Matcher Expectations) {
    ASSERT_THAT_ERROR(
        test::checkDataflow<TestAnalysis>(
            Code, "fun",
            [](ASTContext &C, Environment &) { return TestAnalysis(C); },
            [&Expectations](
                llvm::ArrayRef<std::pair<
                    std::string, DataflowAnalysisState<TestAnalysis::Lattice>>>
                    Results,
                ASTContext &) { EXPECT_THAT(Results, Expectations); },
            {"-fsyntax-only", "-std=c++17"}),
        llvm::Succeeded());
  }
};

TEST_F(MatchSwitchTest, JustX) {
  std::string Code = R"(
    void fun() {
      int X = 1;
      (void)X;
      // [[p]]
    }
  )";
  RunDataflow(Code,
              UnorderedElementsAre(Pair("p", Holds(BooleanLattice(true)))));
}

TEST_F(MatchSwitchTest, JustFoo) {
  std::string Code = R"(
    void Foo();
    void fun() {
      Foo();
      // [[p]]
    }
  )";
  RunDataflow(Code,
              UnorderedElementsAre(Pair("p", Holds(BooleanLattice(false)))));
}

TEST_F(MatchSwitchTest, XThenFoo) {
  std::string Code = R"(
    void Foo();
    void fun() {
      int X = 1;
      (void)X;
      Foo();
      // [[p]]
    }
  )";
  RunDataflow(Code,
              UnorderedElementsAre(Pair("p", Holds(BooleanLattice(false)))));
}

TEST_F(MatchSwitchTest, FooThenX) {
  std::string Code = R"(
    void Foo();
    void fun() {
      Foo();
      int X = 1;
      (void)X;
      // [[p]]
    }
  )";
  RunDataflow(Code,
              UnorderedElementsAre(Pair("p", Holds(BooleanLattice(true)))));
}

TEST_F(MatchSwitchTest, Neither) {
  std::string Code = R"(
    void Bar();
    void fun(bool b) {
      Bar();
      // [[p]]
    }
  )";
  RunDataflow(Code,
              UnorderedElementsAre(Pair("p", Holds(BooleanLattice(false)))));
}

TEST_F(MatchSwitchTest, ReturnNonVoid) {
  using namespace ast_matchers;

  auto Unit =
      tooling::buildASTFromCode("void f() { int x = 42; }", "input.cc",
                                std::make_shared<PCHContainerOperations>());
  auto &Context = Unit->getASTContext();
  const auto *S =
      selectFirst<FunctionDecl>(
          "f",
          match(functionDecl(isDefinition(), hasName("f")).bind("f"), Context))
          ->getBody();

  MatchSwitch<const int, std::vector<int>> Switch =
      MatchSwitchBuilder<const int, std::vector<int>>()
          .CaseOf<Stmt>(stmt(),
                        [](const Stmt *, const MatchFinder::MatchResult &,
                           const int &State) -> std::vector<int> {
                          return {1, State, 3};
                        })
          .Build();
  std::vector<int> Actual = Switch(*S, Context, 7);
  std::vector<int> Expected{1, 7, 3};
  EXPECT_EQ(Actual, Expected);
}
