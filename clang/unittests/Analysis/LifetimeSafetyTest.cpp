//===- LifetimeSafetyTest.cpp - Lifetime Safety Tests -*---------- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Analyses/LifetimeSafety.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Testing/TestAST.h"
#include "llvm/ADT/StringMap.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <optional>
#include <vector>

namespace clang::lifetimes::internal {
namespace {

using namespace ast_matchers;
using ::testing::UnorderedElementsAreArray;

// A helper class to run the full lifetime analysis on a piece of code
// and provide an interface for querying the results.
class LifetimeTestRunner {
public:
  LifetimeTestRunner(llvm::StringRef Code) {
    std::string FullCode = R"(
      #define POINT(name) void("__lifetime_test_point_" #name)
      struct MyObj { ~MyObj() {} int i; };
    )";
    FullCode += Code.str();

    AST = std::make_unique<clang::TestAST>(FullCode);
    ASTCtx = &AST->context();

    // Find the target function using AST matchers.
    auto MatchResult =
        match(functionDecl(hasName("target")).bind("target"), *ASTCtx);
    auto *FD = selectFirst<FunctionDecl>("target", MatchResult);
    if (!FD) {
      ADD_FAILURE() << "Test case must have a function named 'target'";
      return;
    }
    AnalysisCtx = std::make_unique<AnalysisDeclContext>(nullptr, FD);
    AnalysisCtx->getCFGBuildOptions().setAllAlwaysAdd();

    // Run the main analysis.
    Analysis = std::make_unique<LifetimeSafetyAnalysis>(*AnalysisCtx);
    Analysis->run();

    AnnotationToPointMap = Analysis->getTestPoints();
  }

  LifetimeSafetyAnalysis &getAnalysis() { return *Analysis; }
  ASTContext &getASTContext() { return *ASTCtx; }

  ProgramPoint getProgramPoint(llvm::StringRef Annotation) {
    auto It = AnnotationToPointMap.find(Annotation);
    if (It == AnnotationToPointMap.end()) {
      ADD_FAILURE() << "Annotation '" << Annotation << "' not found.";
      return nullptr;
    }
    return It->second;
  }

private:
  std::unique_ptr<TestAST> AST;
  ASTContext *ASTCtx = nullptr;
  std::unique_ptr<AnalysisDeclContext> AnalysisCtx;
  std::unique_ptr<LifetimeSafetyAnalysis> Analysis;
  llvm::StringMap<ProgramPoint> AnnotationToPointMap;
};

// A convenience wrapper that uses the LifetimeSafetyAnalysis public API.
class LifetimeTestHelper {
public:
  LifetimeTestHelper(LifetimeTestRunner &Runner)
      : Runner(Runner), Analysis(Runner.getAnalysis()) {}

  std::optional<OriginID> getOriginForDecl(llvm::StringRef VarName) {
    auto *VD = findDecl<ValueDecl>(VarName);
    if (!VD)
      return std::nullopt;
    auto OID = Analysis.getOriginIDForDecl(VD);
    if (!OID)
      ADD_FAILURE() << "Origin for '" << VarName << "' not found.";
    return OID;
  }

  std::optional<LoanID> getLoanForVar(llvm::StringRef VarName) {
    auto *VD = findDecl<VarDecl>(VarName);
    if (!VD)
      return std::nullopt;
    std::vector<LoanID> LID = Analysis.getLoanIDForVar(VD);
    if (LID.empty()) {
      ADD_FAILURE() << "Loan for '" << VarName << "' not found.";
      return std::nullopt;
    }
    // TODO: Support retrieving more than one loans to a var.
    if (LID.size() > 1) {
      ADD_FAILURE() << "More than 1 loans found for '" << VarName;
      return std::nullopt;
    }
    return LID[0];
  }

  std::optional<LoanSet> getLoansAtPoint(OriginID OID,
                                         llvm::StringRef Annotation) {
    ProgramPoint PP = Runner.getProgramPoint(Annotation);
    if (!PP)
      return std::nullopt;
    return Analysis.getLoansAtPoint(OID, PP);
  }

private:
  template <typename DeclT> DeclT *findDecl(llvm::StringRef Name) {
    auto &Ctx = Runner.getASTContext();
    auto Results = match(valueDecl(hasName(Name)).bind("v"), Ctx);
    if (Results.empty()) {
      ADD_FAILURE() << "Declaration '" << Name << "' not found in AST.";
      return nullptr;
    }
    return const_cast<DeclT *>(selectFirst<DeclT>("v", Results));
  }

  LifetimeTestRunner &Runner;
  LifetimeSafetyAnalysis &Analysis;
};

// ========================================================================= //
//                         GTest Matchers & Fixture
// ========================================================================= //

// It holds the name of the origin variable and a reference to the helper.
class OriginInfo {
public:
  OriginInfo(llvm::StringRef OriginVar, LifetimeTestHelper &Helper)
      : OriginVar(OriginVar), Helper(Helper) {}
  llvm::StringRef OriginVar;
  LifetimeTestHelper &Helper;
};

/// Matcher to verify the set of loans held by an origin at a specific
/// program point.
///
/// This matcher is intended to be used with an \c OriginInfo object.
///
/// \param LoanVars A vector of strings, where each string is the name of a
/// variable expected to be the source of a loan.
/// \param Annotation A string identifying the program point (created with
/// POINT()) where the check should be performed.
MATCHER_P2(HasLoansToImpl, LoanVars, Annotation, "") {
  const OriginInfo &Info = arg;
  std::optional<OriginID> OIDOpt = Info.Helper.getOriginForDecl(Info.OriginVar);
  if (!OIDOpt) {
    *result_listener << "could not find origin for '" << Info.OriginVar.str()
                     << "'";
    return false;
  }

  std::optional<LoanSet> ActualLoansSetOpt =
      Info.Helper.getLoansAtPoint(*OIDOpt, Annotation);
  if (!ActualLoansSetOpt) {
    *result_listener << "could not get a valid loan set at point '"
                     << Annotation << "'";
    return false;
  }
  std::vector<LoanID> ActualLoans(ActualLoansSetOpt->begin(),
                                  ActualLoansSetOpt->end());

  std::vector<LoanID> ExpectedLoans;
  for (const auto &LoanVar : LoanVars) {
    std::optional<LoanID> ExpectedLIDOpt = Info.Helper.getLoanForVar(LoanVar);
    if (!ExpectedLIDOpt) {
      *result_listener << "could not find loan for var '" << LoanVar << "'";
      return false;
    }
    ExpectedLoans.push_back(*ExpectedLIDOpt);
  }

  return ExplainMatchResult(UnorderedElementsAreArray(ExpectedLoans),
                            ActualLoans, result_listener);
}

// Base test fixture to manage the runner and helper.
class LifetimeAnalysisTest : public ::testing::Test {
protected:
  void SetupTest(llvm::StringRef Code) {
    Runner = std::make_unique<LifetimeTestRunner>(Code);
    Helper = std::make_unique<LifetimeTestHelper>(*Runner);
  }

  OriginInfo Origin(llvm::StringRef OriginVar) {
    return OriginInfo(OriginVar, *Helper);
  }

  // Factory function that hides the std::vector creation.
  auto HasLoansTo(std::initializer_list<std::string> LoanVars,
                  const char *Annotation) {
    return HasLoansToImpl(std::vector<std::string>(LoanVars), Annotation);
  }

  std::unique_ptr<LifetimeTestRunner> Runner;
  std::unique_ptr<LifetimeTestHelper> Helper;
};

// ========================================================================= //
//                                 TESTS
// ========================================================================= //

TEST_F(LifetimeAnalysisTest, SimpleLoanAndOrigin) {
  SetupTest(R"(
    void target() {
      int x;
      int* p = &x;
      POINT(p1);
    }
  )");
  EXPECT_THAT(Origin("p"), HasLoansTo({"x"}, "p1"));
}

TEST_F(LifetimeAnalysisTest, OverwriteOrigin) {
  SetupTest(R"(
    void target() {
      MyObj s1, s2;

      MyObj* p = &s1;
      POINT(after_s1);

      p = &s2;
      POINT(after_s2);
    }
  )");
  EXPECT_THAT(Origin("p"), HasLoansTo({"s1"}, "after_s1"));
  EXPECT_THAT(Origin("p"), HasLoansTo({"s2"}, "after_s2"));
}

TEST_F(LifetimeAnalysisTest, ConditionalLoan) {
  SetupTest(R"(
    void target(bool cond) {
      int a, b;
      int *p = nullptr;
      if (cond) {
        p = &a;
        POINT(after_then);
      } else {
        p = &b;
        POINT(after_else);
      }
      POINT(after_if);
    }
  )");
  EXPECT_THAT(Origin("p"), HasLoansTo({"a"}, "after_then"));
  EXPECT_THAT(Origin("p"), HasLoansTo({"b"}, "after_else"));
  EXPECT_THAT(Origin("p"), HasLoansTo({"a", "b"}, "after_if"));
}

TEST_F(LifetimeAnalysisTest, PointerChain) {
  SetupTest(R"(
    void target() {
      MyObj y;
      MyObj* ptr1 = &y;
      POINT(p1);

      MyObj* ptr2 = ptr1;
      POINT(p2);

      ptr2 = ptr1;
      POINT(p3);

      ptr2 = ptr2; // Self assignment
      POINT(p4);
    }
  )");
  EXPECT_THAT(Origin("ptr1"), HasLoansTo({"y"}, "p1"));
  EXPECT_THAT(Origin("ptr2"), HasLoansTo({"y"}, "p2"));
  EXPECT_THAT(Origin("ptr2"), HasLoansTo({"y"}, "p3"));
  EXPECT_THAT(Origin("ptr2"), HasLoansTo({"y"}, "p4"));
}

TEST_F(LifetimeAnalysisTest, ReassignToNull) {
  SetupTest(R"(
    void target() {
      MyObj s1;
      MyObj* p = &s1;
      POINT(before_null);
      p = nullptr;
      POINT(after_null);
    }
  )");
  EXPECT_THAT(Origin("p"), HasLoansTo({"s1"}, "before_null"));
  // After assigning to null, the origin for `p` should have no loans.
  EXPECT_THAT(Origin("p"), HasLoansTo({}, "after_null"));
}

TEST_F(LifetimeAnalysisTest, ReassignInIf) {
  SetupTest(R"(
    void target(bool condition) {
      MyObj s1, s2;
      MyObj* p = &s1;
      POINT(before_if);
      if (condition) {
        p = &s2;
        POINT(after_reassign);
      }
      POINT(after_if);
    }
  )");
  EXPECT_THAT(Origin("p"), HasLoansTo({"s1"}, "before_if"));
  EXPECT_THAT(Origin("p"), HasLoansTo({"s2"}, "after_reassign"));
  EXPECT_THAT(Origin("p"), HasLoansTo({"s1", "s2"}, "after_if"));
}

TEST_F(LifetimeAnalysisTest, AssignInSwitch) {
  SetupTest(R"(
    void target(int mode) {
      MyObj s1, s2, s3;
      MyObj* p = nullptr;
      switch (mode) {
        case 1:
          p = &s1;
          POINT(case1);
          break;
        case 2:
          p = &s2;
          POINT(case2);
          break;
        default:
          p = &s3;
          POINT(case3);
          break;
      }
      POINT(after_switch);
    }
  )");
  EXPECT_THAT(Origin("p"), HasLoansTo({"s1"}, "case1"));
  EXPECT_THAT(Origin("p"), HasLoansTo({"s2"}, "case2"));
  EXPECT_THAT(Origin("p"), HasLoansTo({"s3"}, "case3"));
  EXPECT_THAT(Origin("p"), HasLoansTo({"s1", "s2", "s3"}, "after_switch"));
}

TEST_F(LifetimeAnalysisTest, LoanInLoop) {
  SetupTest(R"(
    void target(bool condition) {
      MyObj* p = nullptr;
      while (condition) {
        MyObj inner;
        p = &inner;
        POINT(in_loop);
      }
      POINT(after_loop);
    }
  )");
  EXPECT_THAT(Origin("p"), HasLoansTo({"inner"}, "in_loop"));
  EXPECT_THAT(Origin("p"), HasLoansTo({"inner"}, "after_loop"));
}

TEST_F(LifetimeAnalysisTest, LoopWithBreak) {
  SetupTest(R"(
    void target(int count) {
      MyObj s1;
      MyObj s2;
      MyObj* p = &s1;
      POINT(before_loop);
      for (int i = 0; i < count; ++i) {
        if (i == 5) {
          p = &s2;
          POINT(inside_if);
          break;
        }
        POINT(after_if);
      }
      POINT(after_loop);
    }
  )");
  EXPECT_THAT(Origin("p"), HasLoansTo({"s1"}, "before_loop"));
  EXPECT_THAT(Origin("p"), HasLoansTo({"s2"}, "inside_if"));
  // At the join point after if, s2 cannot make it to p without the if.
  EXPECT_THAT(Origin("p"), HasLoansTo({"s1"}, "after_if"));
  // At the join point after the loop, p could hold a loan to s1 (if the loop
  // completed normally) or to s2 (if the loop was broken).
  EXPECT_THAT(Origin("p"), HasLoansTo({"s1", "s2"}, "after_loop"));
}

TEST_F(LifetimeAnalysisTest, PointersInACycle) {
  SetupTest(R"(
    void target(bool condition) {
      MyObj v1, v2, v3;
      MyObj *p1 = &v1, *p2 = &v2, *p3 = &v3;

      POINT(before_while);
      while (condition) {
        MyObj* temp = p1;
        p1 = p2;
        p2 = p3;
        p3 = temp;
      }
      POINT(after_loop);
    }
  )");
  EXPECT_THAT(Origin("p1"), HasLoansTo({"v1"}, "before_while"));
  EXPECT_THAT(Origin("p2"), HasLoansTo({"v2"}, "before_while"));
  EXPECT_THAT(Origin("p3"), HasLoansTo({"v3"}, "before_while"));

  // At the fixed point after the loop, all pointers could point to any of
  // the three variables.
  EXPECT_THAT(Origin("p1"), HasLoansTo({"v1", "v2", "v3"}, "after_loop"));
  EXPECT_THAT(Origin("p2"), HasLoansTo({"v1", "v2", "v3"}, "after_loop"));
  EXPECT_THAT(Origin("p3"), HasLoansTo({"v1", "v2", "v3"}, "after_loop"));
  EXPECT_THAT(Origin("temp"), HasLoansTo({"v1", "v2", "v3"}, "after_loop"));
}

TEST_F(LifetimeAnalysisTest, NestedScopes) {
  SetupTest(R"(
    void target() {
      MyObj* p = nullptr;
      {
        MyObj outer;
        p = &outer;
        POINT(before_inner_scope);
        {
          MyObj inner;
          p = &inner;
          POINT(inside_inner_scope);
        } // inner expires
        POINT(after_inner_scope);
      } // outer expires
    }
  )");
  EXPECT_THAT(Origin("p"), HasLoansTo({"outer"}, "before_inner_scope"));
  EXPECT_THAT(Origin("p"), HasLoansTo({"inner"}, "inside_inner_scope"));
  EXPECT_THAT(Origin("p"), HasLoansTo({"inner"}, "after_inner_scope"));
}

} // anonymous namespace
} // namespace clang::lifetimes::internal
