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
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <optional>
#include <vector>

namespace clang::lifetimes::internal {
namespace {

using namespace ast_matchers;
using ::testing::SizeIs;
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

    Inputs = TestInputs(FullCode);
    Inputs.Language = TestLanguage::Lang_CXX20;
    AST = std::make_unique<clang::TestAST>(Inputs);
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
    CFG::BuildOptions &BuildOptions = AnalysisCtx->getCFGBuildOptions();
    BuildOptions.setAllAlwaysAdd();
    BuildOptions.AddImplicitDtors = true;
    BuildOptions.AddTemporaryDtors = true;

    // Run the main analysis.
    Analysis = std::make_unique<LifetimeSafetyAnalysis>(*AnalysisCtx, nullptr);
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
  TestInputs Inputs;
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

  std::vector<LoanID> getLoansForVar(llvm::StringRef VarName) {
    auto *VD = findDecl<VarDecl>(VarName);
    if (!VD) {
      ADD_FAILURE() << "No VarDecl found for '" << VarName << "'";
      return {};
    }
    std::vector<LoanID> LID = Analysis.getLoanIDForVar(VD);
    if (LID.empty()) {
      ADD_FAILURE() << "Loan for '" << VarName << "' not found.";
      return {};
    }
    return LID;
  }

  std::optional<LoanSet> getLoansAtPoint(OriginID OID,
                                         llvm::StringRef Annotation) {
    ProgramPoint PP = Runner.getProgramPoint(Annotation);
    if (!PP)
      return std::nullopt;
    return Analysis.getLoansAtPoint(OID, PP);
  }

  std::optional<std::vector<LoanID>>
  getExpiredLoansAtPoint(llvm::StringRef Annotation) {
    ProgramPoint PP = Runner.getProgramPoint(Annotation);
    if (!PP)
      return std::nullopt;
    return Analysis.getExpiredLoansAtPoint(PP);
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

// A helper class to represent a set of loans, identified by variable names.
class LoanSetInfo {
public:
  LoanSetInfo(const std::vector<std::string> &Vars, LifetimeTestHelper &H)
      : LoanVars(Vars), Helper(H) {}
  std::vector<std::string> LoanVars;
  LifetimeTestHelper &Helper;
};

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
    std::vector<LoanID> ExpectedLIDs = Info.Helper.getLoansForVar(LoanVar);
    if (ExpectedLIDs.empty()) {
      *result_listener << "could not find loan for var '" << LoanVar << "'";
      return false;
    }
    ExpectedLoans.insert(ExpectedLoans.end(), ExpectedLIDs.begin(),
                         ExpectedLIDs.end());
  }

  return ExplainMatchResult(UnorderedElementsAreArray(ExpectedLoans),
                            ActualLoans, result_listener);
}

/// Matcher to verify that the complete set of expired loans at a program point
/// matches the expected loan set.
MATCHER_P(AreExpiredAt, Annotation, "") {
  const LoanSetInfo &Info = arg;
  auto &Helper = Info.Helper;

  auto ActualExpiredSetOpt = Helper.getExpiredLoansAtPoint(Annotation);
  if (!ActualExpiredSetOpt) {
    *result_listener << "could not get a valid expired loan set at point '"
                     << Annotation << "'";
    return false;
  }
  std::vector<LoanID> ActualExpiredLoans = *ActualExpiredSetOpt;
  std::vector<LoanID> ExpectedExpiredLoans;
  for (const auto &VarName : Info.LoanVars) {
    auto LoanIDs = Helper.getLoansForVar(VarName);
    if (LoanIDs.empty()) {
      *result_listener << "could not find a loan for variable '" << VarName
                       << "'";
      return false;
    }
    ExpectedExpiredLoans.insert(ExpectedExpiredLoans.end(), LoanIDs.begin(),
                                LoanIDs.end());
  }
  return ExplainMatchResult(UnorderedElementsAreArray(ExpectedExpiredLoans),
                            ActualExpiredLoans, result_listener);
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

  /// Factory function that hides the std::vector creation.
  LoanSetInfo LoansTo(std::initializer_list<std::string> LoanVars) {
    return LoanSetInfo({LoanVars}, *Helper);
  }

  /// A convenience helper for asserting that no loans are expired.
  LoanSetInfo NoLoans() { return LoansTo({}); }

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
        POINT(start_loop);
        MyObj inner;
        p = &inner;
        POINT(end_loop);
      }
      POINT(after_loop);
    }
  )");
  EXPECT_THAT(Origin("p"), HasLoansTo({"inner"}, "start_loop"));
  EXPECT_THAT(LoansTo({"inner"}), AreExpiredAt("start_loop"));

  EXPECT_THAT(Origin("p"), HasLoansTo({"inner"}, "end_loop"));
  EXPECT_THAT(NoLoans(), AreExpiredAt("end_loop"));

  EXPECT_THAT(Origin("p"), HasLoansTo({"inner"}, "after_loop"));
  EXPECT_THAT(LoansTo({"inner"}), AreExpiredAt("after_loop"));
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

TEST_F(LifetimeAnalysisTest, PointersAndExpirationInACycle) {
  SetupTest(R"(
    void target(bool condition) {
      MyObj v1, v2;
      MyObj *p1 = &v1, *p2 = &v2;

      POINT(before_while);
      while (condition) {
        POINT(in_loop_before_temp);
        MyObj temp;
        p1 = &temp;
        POINT(in_loop_after_temp);

        MyObj* q = p1;
        p1 = p2;
        p2 = q;
      }
      POINT(after_loop);
    }
  )");
  EXPECT_THAT(Origin("p1"), HasLoansTo({"v1"}, "before_while"));
  EXPECT_THAT(Origin("p2"), HasLoansTo({"v2"}, "before_while"));
  EXPECT_THAT(NoLoans(), AreExpiredAt("before_while"));

  EXPECT_THAT(Origin("p1"),
              HasLoansTo({"v1", "v2", "temp"}, "in_loop_before_temp"));
  EXPECT_THAT(Origin("p2"), HasLoansTo({"v2", "temp"}, "in_loop_before_temp"));
  EXPECT_THAT(LoansTo({"temp"}), AreExpiredAt("in_loop_before_temp"));

  EXPECT_THAT(Origin("p1"), HasLoansTo({"temp"}, "in_loop_after_temp"));
  EXPECT_THAT(Origin("p2"), HasLoansTo({"v2", "temp"}, "in_loop_after_temp"));
  EXPECT_THAT(NoLoans(), AreExpiredAt("in_loop_after_temp"));

  EXPECT_THAT(Origin("p1"), HasLoansTo({"v1", "v2", "temp"}, "after_loop"));
  EXPECT_THAT(Origin("p2"), HasLoansTo({"v2", "temp"}, "after_loop"));
  EXPECT_THAT(LoansTo({"temp"}), AreExpiredAt("after_loop"));
}

TEST_F(LifetimeAnalysisTest, InfiniteLoopPrunesEdges) {
  SetupTest(R"(
    void target(MyObj out) {
      MyObj *p = &out;
      POINT(before_loop);

      for (;;) {
        POINT(begin);
        MyObj in;
        p = &in;
        POINT(end);
      }
    }
  )");
  EXPECT_THAT(Origin("p"), HasLoansTo({"out"}, "before_loop"));
  EXPECT_THAT(Origin("p"), HasLoansTo({"in", "out"}, "begin"));
  EXPECT_THAT(Origin("p"), HasLoansTo({"in"}, "end"));
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

TEST_F(LifetimeAnalysisTest, SimpleExpiry) {
  SetupTest(R"(
    void target() {
      MyObj* p = nullptr;
      {
        MyObj s;
        p = &s;
        POINT(before_expiry);
      } // s goes out of scope here
      POINT(after_expiry);
    }
  )");
  EXPECT_THAT(NoLoans(), AreExpiredAt("before_expiry"));
  EXPECT_THAT(LoansTo({"s"}), AreExpiredAt("after_expiry"));
}

TEST_F(LifetimeAnalysisTest, NestedExpiry) {
  SetupTest(R"(
    void target() {
      MyObj s1;
      MyObj* p = &s1;
      POINT(before_inner);
      {
        MyObj s2;
        p = &s2;
        POINT(in_inner);
      } // s2 expires
      POINT(after_inner);
    }
  )");
  EXPECT_THAT(NoLoans(), AreExpiredAt("before_inner"));
  EXPECT_THAT(NoLoans(), AreExpiredAt("in_inner"));
  EXPECT_THAT(LoansTo({"s2"}), AreExpiredAt("after_inner"));
}

TEST_F(LifetimeAnalysisTest, ConditionalExpiry) {
  SetupTest(R"(
    void target(bool cond) {
      MyObj s1;
      MyObj* p = &s1;
      POINT(before_if);
      if (cond) {
        MyObj s2;
        p = &s2;
        POINT(then_block);
      } // s2 expires here
      POINT(after_if);
    }
  )");
  EXPECT_THAT(NoLoans(), AreExpiredAt("before_if"));
  EXPECT_THAT(NoLoans(), AreExpiredAt("then_block"));
  EXPECT_THAT(LoansTo({"s2"}), AreExpiredAt("after_if"));
}

TEST_F(LifetimeAnalysisTest, LoopExpiry) {
  SetupTest(R"(
    void target() {
      MyObj *p = nullptr;
      for (int i = 0; i < 2; ++i) {
        POINT(start_loop);
        MyObj s;
        p = &s;
        POINT(end_loop);
      } // s expires here on each iteration
      POINT(after_loop);
    }
  )");
  EXPECT_THAT(LoansTo({"s"}), AreExpiredAt("start_loop"));
  EXPECT_THAT(NoLoans(), AreExpiredAt("end_loop"));
  EXPECT_THAT(LoansTo({"s"}), AreExpiredAt("after_loop"));
}

TEST_F(LifetimeAnalysisTest, MultipleExpiredLoans) {
  SetupTest(R"(
    void target() {
      MyObj *p1, *p2, *p3;
      {
        MyObj s1;
        p1 = &s1;
        POINT(p1);
      } // s1 expires
      POINT(p2);
      {
        MyObj s2;
        p2 = &s2;
        MyObj s3;
        p3 = &s3;
        POINT(p3);
      } // s2, s3 expire
      POINT(p4);
    }
  )");
  EXPECT_THAT(NoLoans(), AreExpiredAt("p1"));
  EXPECT_THAT(LoansTo({"s1"}), AreExpiredAt("p2"));
  EXPECT_THAT(LoansTo({"s1"}), AreExpiredAt("p3"));
  EXPECT_THAT(LoansTo({"s1", "s2", "s3"}), AreExpiredAt("p4"));
}

TEST_F(LifetimeAnalysisTest, GotoJumpsOutOfScope) {
  SetupTest(R"(
    void target(bool cond) {
      MyObj *p = nullptr;
      {
        MyObj s;
        p = &s;
        POINT(before_goto);
        if (cond) {
          goto end;
        }
      } // `s` expires here on the path that doesn't jump
      POINT(after_scope);
    end:
      POINT(after_goto);
    }
  )");
  EXPECT_THAT(NoLoans(), AreExpiredAt("before_goto"));
  EXPECT_THAT(LoansTo({"s"}), AreExpiredAt("after_scope"));
  EXPECT_THAT(LoansTo({"s"}), AreExpiredAt("after_goto"));
}

TEST_F(LifetimeAnalysisTest, ContinueInLoop) {
  SetupTest(R"(
    void target(int count) {
      MyObj *p = nullptr;
      MyObj outer;
      p = &outer;
      POINT(before_loop);

      for (int i = 0; i < count; ++i) {
        if (i % 2 == 0) {
          MyObj s_even;
          p = &s_even;
          POINT(in_even_iter);
          continue;
        }
        MyObj s_odd;
        p = &s_odd;
        POINT(in_odd_iter);
      }
      POINT(after_loop);
    }
  )");
  EXPECT_THAT(NoLoans(), AreExpiredAt("before_loop"));
  EXPECT_THAT(LoansTo({"s_odd"}), AreExpiredAt("in_even_iter"));
  EXPECT_THAT(LoansTo({"s_even"}), AreExpiredAt("in_odd_iter"));
  EXPECT_THAT(LoansTo({"s_even", "s_odd"}), AreExpiredAt("after_loop"));
}

TEST_F(LifetimeAnalysisTest, ReassignedPointerThenOriginalExpires) {
  SetupTest(R"(
    void target() {
      MyObj* p = nullptr;
      {
        MyObj s1;
        p = &s1;
        POINT(p_has_s1);
        {
          MyObj s2;
          p = &s2;
          POINT(p_has_s2);
        }
        POINT(p_after_s2_expires);
      } // s1 expires here.
      POINT(p_after_s1_expires);
    }
  )");
  EXPECT_THAT(NoLoans(), AreExpiredAt("p_has_s1"));
  EXPECT_THAT(NoLoans(), AreExpiredAt("p_has_s2"));
  EXPECT_THAT(LoansTo({"s2"}), AreExpiredAt("p_after_s2_expires"));
  EXPECT_THAT(LoansTo({"s1", "s2"}), AreExpiredAt("p_after_s1_expires"));
}

TEST_F(LifetimeAnalysisTest, NoDuplicateLoansForImplicitCastToConst) {
  SetupTest(R"(
    void target() {
      MyObj a;
      const MyObj* p = &a;
      const MyObj* q = &a;
      POINT(at_end);
    }
  )");
  EXPECT_THAT(Helper->getLoansForVar("a"), SizeIs(2));
}

} // anonymous namespace
} // namespace clang::lifetimes::internal
