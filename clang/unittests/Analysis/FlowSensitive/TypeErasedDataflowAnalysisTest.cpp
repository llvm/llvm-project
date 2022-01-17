//===- unittests/Analysis/FlowSensitive/TypeErasedDataflowAnalysisTest.cpp ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NoopAnalysis.h"
#include "TestingSupport.h"
#include "clang/AST/Decl.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/FlowSensitive/DataflowAnalysis.h"
#include "clang/Analysis/FlowSensitive/DataflowAnalysisContext.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/DataflowLattice.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cassert>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace clang;
using namespace dataflow;
using ::testing::IsEmpty;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

template <typename AnalysisT>
class AnalysisCallback : public ast_matchers::MatchFinder::MatchCallback {
public:
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override {
    assert(BlockStates.empty());

    const auto *Func = Result.Nodes.getNodeAs<FunctionDecl>("func");
    assert(Func != nullptr);

    Stmt *Body = Func->getBody();
    assert(Body != nullptr);

    auto CFCtx = llvm::cantFail(
        ControlFlowContext::build(nullptr, Body, Result.Context));

    AnalysisT Analysis(*Result.Context);
    DataflowAnalysisContext DACtx;
    Environment Env(DACtx);
    BlockStates = runDataflowAnalysis(CFCtx, Analysis, Env);
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

  void transfer(const Stmt *S, NonConvergingLattice &E, Environment &Env) {
    ++E.State;
  }
};

TEST(DataflowAnalysisTest, NonConvergingAnalysis) {
  auto BlockStates = runAnalysis<NonConvergingAnalysis>(R"(
    void target() {
      while(true) {}
    }
  )");
  EXPECT_EQ(BlockStates.size(), 4u);
  EXPECT_TRUE(BlockStates[0].hasValue());
  EXPECT_TRUE(BlockStates[1].hasValue());
  EXPECT_TRUE(BlockStates[2].hasValue());
  EXPECT_TRUE(BlockStates[3].hasValue());
}

struct FunctionCallLattice {
  llvm::SmallSet<std::string, 8> CalledFunctions;

  bool operator==(const FunctionCallLattice &Other) const {
    return CalledFunctions == Other.CalledFunctions;
  }

  LatticeJoinEffect join(const FunctionCallLattice &Other) {
    if (Other.CalledFunctions.empty())
      return LatticeJoinEffect::Unchanged;
    const size_t size_before = CalledFunctions.size();
    CalledFunctions.insert(Other.CalledFunctions.begin(),
                           Other.CalledFunctions.end());
    return CalledFunctions.size() == size_before ? LatticeJoinEffect::Unchanged
                                                 : LatticeJoinEffect::Changed;
  }
};

std::ostream &operator<<(std::ostream &OS, const FunctionCallLattice &L) {
  std::string S;
  llvm::raw_string_ostream ROS(S);
  llvm::interleaveComma(L.CalledFunctions, ROS);
  return OS << "{" << S << "}";
}

class FunctionCallAnalysis
    : public DataflowAnalysis<FunctionCallAnalysis, FunctionCallLattice> {
public:
  explicit FunctionCallAnalysis(ASTContext &Context)
      : DataflowAnalysis<FunctionCallAnalysis, FunctionCallLattice>(Context) {}

  static FunctionCallLattice initialElement() { return {}; }

  void transfer(const Stmt *S, FunctionCallLattice &E, Environment &Env) {
    if (auto *C = dyn_cast<CallExpr>(S)) {
      if (auto *F = dyn_cast<FunctionDecl>(C->getCalleeDecl())) {
        E.CalledFunctions.insert(F->getNameInfo().getAsString());
      }
    }
  }
};

class NoreturnDestructorTest : public ::testing::Test {
protected:
  template <typename Matcher>
  void runDataflow(llvm::StringRef Code, Matcher Expectations) {
    tooling::FileContentMappings FilesContents;
    FilesContents.push_back(std::make_pair<std::string, std::string>(
        "noreturn_destructor_test_defs.h", R"(
      int foo();

      class Fatal {
       public:
        ~Fatal() __attribute__((noreturn));
        int bar();
        int baz();
      };

      class NonFatal {
       public:
        ~NonFatal();
        int bar();
      };
    )"));

    test::checkDataflow<FunctionCallAnalysis>(
        Code, "target",
        [](ASTContext &C, Environment &) { return FunctionCallAnalysis(C); },
        [&Expectations](
            llvm::ArrayRef<std::pair<
                std::string, DataflowAnalysisState<FunctionCallLattice>>>
                Results,
            ASTContext &) { EXPECT_THAT(Results, Expectations); },
        {"-fsyntax-only", "-std=c++17"}, FilesContents);
  }
};

MATCHER_P(HoldsFunctionCallLattice, m,
          ((negation ? "doesn't hold" : "holds") +
           llvm::StringRef(" a lattice element that ") +
           ::testing::DescribeMatcher<FunctionCallLattice>(m, negation))
              .str()) {
  return ExplainMatchResult(m, arg.Lattice, result_listener);
}

MATCHER_P(HasCalledFunctions, m, "") {
  return ExplainMatchResult(m, arg.CalledFunctions, result_listener);
}

TEST_F(NoreturnDestructorTest, ConditionalOperatorBothBranchesReturn) {
  std::string Code = R"(
    #include "noreturn_destructor_test_defs.h"

    void target(bool b) {
      int value = b ? foo() : NonFatal().bar();
      (void)0;
      // [[p]]
    }
  )";
  runDataflow(Code, UnorderedElementsAre(
                        Pair("p", HoldsFunctionCallLattice(HasCalledFunctions(
                                      UnorderedElementsAre("foo", "bar"))))));
}

TEST_F(NoreturnDestructorTest, ConditionalOperatorLeftBranchReturns) {
  std::string Code = R"(
    #include "noreturn_destructor_test_defs.h"

    void target(bool b) {
      int value = b ? foo() : Fatal().bar();
      (void)0;
      // [[p]]
    }
  )";
  runDataflow(Code, UnorderedElementsAre(
                        Pair("p", HoldsFunctionCallLattice(HasCalledFunctions(
                                      UnorderedElementsAre("foo"))))));
}

TEST_F(NoreturnDestructorTest, ConditionalOperatorRightBranchReturns) {
  std::string Code = R"(
    #include "noreturn_destructor_test_defs.h"

    void target(bool b) {
      int value = b ? Fatal().bar() : foo();
      (void)0;
      // [[p]]
    }
  )";
  runDataflow(Code, UnorderedElementsAre(
                        Pair("p", HoldsFunctionCallLattice(HasCalledFunctions(
                                      UnorderedElementsAre("foo"))))));
}

TEST_F(NoreturnDestructorTest, ConditionalOperatorNestedBranchesDoNotReturn) {
  std::string Code = R"(
    #include "noreturn_destructor_test_defs.h"

    void target(bool b1, bool b2) {
      int value = b1 ? foo() : (b2 ? Fatal().bar() : Fatal().baz());
      (void)0;
      // [[p]]
    }
  )";
  runDataflow(Code, IsEmpty());
  // FIXME: Called functions at point `p` should contain "foo".
}

TEST_F(NoreturnDestructorTest, ConditionalOperatorNestedBranchReturns) {
  std::string Code = R"(
    #include "noreturn_destructor_test_defs.h"

    void target(bool b1, bool b2) {
      int value = b1 ? Fatal().bar() : (b2 ? Fatal().baz() : foo());
      (void)0;
      // [[p]]
    }
  )";
  runDataflow(Code, UnorderedElementsAre(
                        Pair("p", HoldsFunctionCallLattice(HasCalledFunctions(
                                      UnorderedElementsAre("baz", "foo"))))));
  // FIXME: Called functions at point `p` should contain only "foo".
}

} // namespace
