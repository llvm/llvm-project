//===- unittests/Analysis/FlowSensitive/TypeErasedDataflowAnalysisTest.cpp ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Decl.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/FlowSensitive/DataflowAnalysis.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/DataflowLattice.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/StringRef.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cassert>
#include <memory>
#include <vector>

using namespace clang;
using namespace dataflow;

template <typename AnalysisT>
class AnalysisCallback : public ast_matchers::MatchFinder::MatchCallback {
public:
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override {
    assert(BlockStates.empty());

    const auto *Func = Result.Nodes.getNodeAs<FunctionDecl>("func");
    assert(Func != nullptr);

    Stmt *Body = Func->getBody();
    assert(Body != nullptr);

    // FIXME: Consider providing a utility that returns a `CFG::BuildOptions`
    // which is a good default for most clients or a utility that directly
    // builds the `CFG` using default `CFG::BuildOptions`.
    CFG::BuildOptions Options;
    Options.AddImplicitDtors = true;
    Options.AddTemporaryDtors = true;
    Options.setAllAlwaysAdd();

    std::unique_ptr<CFG> Cfg =
        CFG::buildCFG(nullptr, Body, Result.Context, Options);
    assert(Cfg != nullptr);

    AnalysisT Analysis(*Result.Context);
    Environment Env;
    BlockStates = runDataflowAnalysis(*Cfg, Analysis, Env);
  }

  std::vector<
      llvm::Optional<DataflowAnalysisState<typename AnalysisT::Lattice>>>
      BlockStates;
};

template <typename AnalysisT>
std::vector<llvm::Optional<DataflowAnalysisState<typename AnalysisT::Lattice>>>
runAnalysis(llvm::StringRef Code) {
  std::unique_ptr<ASTUnit> AST =
      tooling::buildASTFromCodeWithArgs(Code, {"-std=c++11"});

  AnalysisCallback<AnalysisT> Callback;
  ast_matchers::MatchFinder Finder;
  Finder.addMatcher(
      ast_matchers::functionDecl(ast_matchers::hasName("target")).bind("func"),
      &Callback);
  Finder.matchAST(AST->getASTContext());

  return Callback.BlockStates;
}

class NoopLattice {
public:
  bool operator==(const NoopLattice &) const { return true; }

  LatticeJoinEffect join(const NoopLattice &) {
    return LatticeJoinEffect::Unchanged;
  }
};

class NoopAnalysis : public DataflowAnalysis<NoopAnalysis, NoopLattice> {
public:
  NoopAnalysis(ASTContext &Context)
      : DataflowAnalysis<NoopAnalysis, NoopLattice>(Context) {}

  static NoopLattice initialElement() { return {}; }

  NoopLattice transfer(const Stmt *S, const NoopLattice &E, Environment &Env) {
    return {};
  }
};

TEST(DataflowAnalysisTest, NoopAnalysis) {
  auto BlockStates = runAnalysis<NoopAnalysis>(R"(
    void target() {}
  )");
  EXPECT_EQ(BlockStates.size(), 2u);
  EXPECT_TRUE(BlockStates[0].hasValue());
  EXPECT_TRUE(BlockStates[1].hasValue());
}

struct NonConvergingLattice {
  int State;

  bool operator==(const NonConvergingLattice &Other) const {
    return State == Other.State;
  }

  LatticeJoinEffect join(const NonConvergingLattice &Other) {
    if (Other.State == 0)
      return LatticeJoinEffect::Unchanged;
    State += Other.State;
    return LatticeJoinEffect::Changed;
  }
};

class NonConvergingAnalysis
    : public DataflowAnalysis<NonConvergingAnalysis, NonConvergingLattice> {
public:
  explicit NonConvergingAnalysis(ASTContext &Context)
      : DataflowAnalysis<NonConvergingAnalysis, NonConvergingLattice>(Context) {
  }

  static NonConvergingLattice initialElement() { return {0}; }

  NonConvergingLattice transfer(const Stmt *S, const NonConvergingLattice &E,
                                Environment &Env) {
    return {E.State + 1};
  }
};

TEST(DataflowAnalysisTest, NonConvergingAnalysis) {
  auto BlockStates = runAnalysis<NonConvergingAnalysis>(R"(
    void target() {
      while(true) {}
    }
  )");
  EXPECT_EQ(BlockStates.size(), 4u);
  EXPECT_FALSE(BlockStates[0].hasValue());
  EXPECT_TRUE(BlockStates[1].hasValue());
  EXPECT_TRUE(BlockStates[2].hasValue());
  EXPECT_TRUE(BlockStates[3].hasValue());
}
