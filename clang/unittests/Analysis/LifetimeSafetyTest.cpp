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

      struct [[gsl::Pointer()]] View { 
        View(const MyObj&);
        View();
      };
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
  AnalysisDeclContext &getAnalysisContext() { return *AnalysisCtx; }

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
      ADD_FAILURE() << "Failed to find VarDecl for '" << VarName << "'";
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

  std::optional<std::vector<std::pair<OriginID, LivenessKind>>>
  getLiveOriginsAtPoint(llvm::StringRef Annotation) {
    ProgramPoint PP = Runner.getProgramPoint(Annotation);
    if (!PP)
      return std::nullopt;
    return Analysis.getLiveOriginsAtPoint(PP);
  }

private:
  template <typename DeclT> DeclT *findDecl(llvm::StringRef Name) {
    auto &Ctx = Runner.getASTContext();
    const auto *TargetFunc = Runner.getAnalysisContext().getDecl();
    auto Results =
        match(valueDecl(hasName(Name),
                        hasAncestor(functionDecl(equalsNode(TargetFunc))))
                  .bind("v"),
              Ctx);
    if (Results.empty()) {
      ADD_FAILURE() << "Declaration '" << Name << "' not found in AST.";
      return nullptr;
    }
    if (Results.size() > 1) {
      ADD_FAILURE() << "Multiple declarations found for '" << Name << "'";
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

// A helper class to represent a set of origins, identified by variable names.
class OriginsInfo {
public:
  OriginsInfo(const std::vector<std::string> &Vars, LifetimeTestHelper &H)
      : OriginVars(Vars), Helper(H) {}
  std::vector<std::string> OriginVars;
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
  std::sort(ExpectedLoans.begin(), ExpectedLoans.end());
  std::sort(ActualLoans.begin(), ActualLoans.end());
  if (ExpectedLoans != ActualLoans) {
    *result_listener << "Expected: {";
    for (const auto &LoanID : ExpectedLoans) {
      *result_listener << LoanID.Value << ", ";
    }
    *result_listener << "} Actual: {";
    for (const auto &LoanID : ActualLoans) {
      *result_listener << LoanID.Value << ", ";
    }
    *result_listener << "}";
    return false;
  }

  return ExplainMatchResult(UnorderedElementsAreArray(ExpectedLoans),
                            ActualLoans, result_listener);
}

enum class LivenessKindFilter { Maybe, Must, All };

/// Matcher to verify the complete set of live origins at a program point.
MATCHER_P2(AreLiveAtImpl, Annotation, ConfFilter, "") {
  const OriginsInfo &Info = arg;
  auto &Helper = Info.Helper;
  auto ActualLiveSetOpt = Helper.getLiveOriginsAtPoint(Annotation);
  if (!ActualLiveSetOpt) {
    *result_listener << "could not get a valid live origin set at point '"
                     << Annotation << "'";
    return false;
  }
  std::vector<OriginID> ActualLiveOrigins;
  for (const auto [OID, ActualConfidence] : ActualLiveSetOpt.value()) {
    if (ConfFilter == LivenessKindFilter::All)
      ActualLiveOrigins.push_back(OID);
    if (ActualConfidence == LivenessKind::Maybe &&
        ConfFilter == LivenessKindFilter::Maybe)
      ActualLiveOrigins.push_back(OID);
    if (ActualConfidence == LivenessKind::Must &&
        ConfFilter == LivenessKindFilter::Must)
      ActualLiveOrigins.push_back(OID);
  }

  std::vector<OriginID> ExpectedLiveOrigins;
  for (const auto &VarName : Info.OriginVars) {
    auto OriginIDOpt = Helper.getOriginForDecl(VarName);
    if (!OriginIDOpt) {
      *result_listener << "could not find an origin for variable '" << VarName
                       << "'";
      return false;
    }
    ExpectedLiveOrigins.push_back(*OriginIDOpt);
  }
  std::sort(ExpectedLiveOrigins.begin(), ExpectedLiveOrigins.end());
  std::sort(ActualLiveOrigins.begin(), ActualLiveOrigins.end());
  if (ExpectedLiveOrigins != ActualLiveOrigins) {
    *result_listener << "Expected: {";
    for (const auto &OriginID : ExpectedLiveOrigins) {
      *result_listener << OriginID.Value << ", ";
    }
    *result_listener << "} Actual: {";
    for (const auto &OriginID : ActualLiveOrigins) {
      *result_listener << OriginID.Value << ", ";
    }
    *result_listener << "}";
    return false;
  }
  return true;
}

MATCHER_P(MustBeLiveAt, Annotation, "") {
  return ExplainMatchResult(AreLiveAtImpl(Annotation, LivenessKindFilter::Must),
                            arg, result_listener);
}

MATCHER_P(MaybeLiveAt, Annotation, "") {
  return ExplainMatchResult(
      AreLiveAtImpl(Annotation, LivenessKindFilter::Maybe), arg,
      result_listener);
}

MATCHER_P(AreLiveAt, Annotation, "") {
  return ExplainMatchResult(AreLiveAtImpl(Annotation, LivenessKindFilter::All),
                            arg, result_listener);
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
  OriginsInfo Origins(std::initializer_list<std::string> OriginVars) {
    return OriginsInfo({OriginVars}, *Helper);
  }

  OriginsInfo NoOrigins() { return Origins({}); }

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

  EXPECT_THAT(Origin("p1"),
              HasLoansTo({"v1", "v2", "temp"}, "in_loop_before_temp"));
  EXPECT_THAT(Origin("p2"), HasLoansTo({"v2", "temp"}, "in_loop_before_temp"));

  EXPECT_THAT(Origin("p1"), HasLoansTo({"temp"}, "in_loop_after_temp"));
  EXPECT_THAT(Origin("p2"), HasLoansTo({"v2", "temp"}, "in_loop_after_temp"));

  EXPECT_THAT(Origin("p1"), HasLoansTo({"v1", "v2", "temp"}, "after_loop"));
  EXPECT_THAT(Origin("p2"), HasLoansTo({"v2", "temp"}, "after_loop"));
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

TEST_F(LifetimeAnalysisTest, GslPointerSimpleLoan) {
  SetupTest(R"(
    void target() {
      MyObj a;
      View x = a;
      POINT(p1);
    }
  )");
  EXPECT_THAT(Origin("x"), HasLoansTo({"a"}, "p1"));
}

TEST_F(LifetimeAnalysisTest, GslPointerConstructFromOwner) {
  SetupTest(R"(
    void target() {
      MyObj al, bl, cl, dl, el, fl;
      View a = View(al);
      View b = View{bl};
      View c = View(View(View(cl)));
      View d = View{View(View(dl))};
      View e = View{View{View{el}}};
      View f = {fl};
      POINT(p1);
    }
  )");
  EXPECT_THAT(Origin("a"), HasLoansTo({"al"}, "p1"));
  EXPECT_THAT(Origin("b"), HasLoansTo({"bl"}, "p1"));
  EXPECT_THAT(Origin("c"), HasLoansTo({"cl"}, "p1"));
  EXPECT_THAT(Origin("d"), HasLoansTo({"dl"}, "p1"));
  EXPECT_THAT(Origin("e"), HasLoansTo({"el"}, "p1"));
  EXPECT_THAT(Origin("f"), HasLoansTo({"fl"}, "p1"));
}

TEST_F(LifetimeAnalysisTest, GslPointerConstructFromView) {
  SetupTest(R"(
    void target() {
      MyObj a;
      View x = View(a);
      View y = View{x};
      View z = View(View(View(y)));
      View p = View{View(View(x))};
      View q = {x};
      POINT(p1);
    }
  )");
  EXPECT_THAT(Origin("x"), HasLoansTo({"a"}, "p1"));
  EXPECT_THAT(Origin("y"), HasLoansTo({"a"}, "p1"));
  EXPECT_THAT(Origin("z"), HasLoansTo({"a"}, "p1"));
  EXPECT_THAT(Origin("p"), HasLoansTo({"a"}, "p1"));
  EXPECT_THAT(Origin("q"), HasLoansTo({"a"}, "p1"));
}

// FIXME: Handle loans in ternary operator!
TEST_F(LifetimeAnalysisTest, GslPointerInConditionalOperator) {
  SetupTest(R"(
    void target(bool cond) {
      MyObj a, b;
      View v = cond ? a : b;
      POINT(p1);
    }
  )");
  EXPECT_THAT(Origin("v"), HasLoansTo({}, "p1"));
}

// FIXME: Handle temporaries.
TEST_F(LifetimeAnalysisTest, ViewFromTemporary) {
  SetupTest(R"(
    MyObj temporary();
    void target() {
      View v = temporary();
      POINT(p1);
    }
  )");
  EXPECT_THAT(Origin("v"), HasLoansTo({}, "p1"));
}

TEST_F(LifetimeAnalysisTest, GslPointerWithConstAndAuto) {
  SetupTest(R"(
    void target() {
      MyObj a;
      const View v1 = a;
      auto v2 = v1;
      const auto& v3 = v2;
      POINT(p1);
    }
  )");
  EXPECT_THAT(Origin("v1"), HasLoansTo({"a"}, "p1"));
  EXPECT_THAT(Origin("v2"), HasLoansTo({"a"}, "p1"));
  EXPECT_THAT(Origin("v3"), HasLoansTo({"a"}, "p1"));
}

TEST_F(LifetimeAnalysisTest, GslPointerPropagation) {
  SetupTest(R"(
    void target() {
      MyObj a;
      View x = a;
      POINT(p1);

      View y = x; // Propagation via copy-construction
      POINT(p2);

      View z;
      z = x;       // Propagation via copy-assignment
      POINT(p3);
    }
  )");

  EXPECT_THAT(Origin("x"), HasLoansTo({"a"}, "p1"));
  EXPECT_THAT(Origin("y"), HasLoansTo({"a"}, "p2"));
  EXPECT_THAT(Origin("z"), HasLoansTo({"a"}, "p3"));
}

TEST_F(LifetimeAnalysisTest, GslPointerReassignment) {
  SetupTest(R"(
    void target() {
      MyObj safe;
      View v;
      v = safe;
      POINT(p1);
      {
        MyObj unsafe;
        v = unsafe;
        POINT(p2);
      } // `unsafe` expires here.
      POINT(p3);
    }
  )");

  EXPECT_THAT(Origin("v"), HasLoansTo({"safe"}, "p1"));
  EXPECT_THAT(Origin("v"), HasLoansTo({"unsafe"}, "p2"));
  EXPECT_THAT(Origin("v"), HasLoansTo({"unsafe"}, "p3"));
}

TEST_F(LifetimeAnalysisTest, GslPointerConversionOperator) {
  SetupTest(R"(
    struct String;

    struct [[gsl::Pointer()]] StringView {
      StringView() = default;
    };

    struct String {
      ~String() {}
      operator StringView() const;
    };

    void target() {
      String xl, yl;
      StringView x = xl;
      StringView y;
      y = yl;
      POINT(p1);
    }
  )");
  EXPECT_THAT(Origin("x"), HasLoansTo({"xl"}, "p1"));
  EXPECT_THAT(Origin("y"), HasLoansTo({"yl"}, "p1"));
}

TEST_F(LifetimeAnalysisTest, LifetimeboundSimple) {
  SetupTest(R"(
    View Identity(View v [[clang::lifetimebound]]);
    void target() {
      MyObj a, b;
      View v1 = a;
      POINT(p1);

      View v2 = Identity(v1);
      View v3 = Identity(b);
      POINT(p2);
    }
  )");
  EXPECT_THAT(Origin("v1"), HasLoansTo({"a"}, "p1"));
  // The origin of v2 should now contain the loan to 'o' from v1.
  EXPECT_THAT(Origin("v2"), HasLoansTo({"a"}, "p2"));
  EXPECT_THAT(Origin("v3"), HasLoansTo({"b"}, "p2"));
}

TEST_F(LifetimeAnalysisTest, LifetimeboundMemberFunction) {
  SetupTest(R"(
    struct [[gsl::Pointer()]] MyView {
      MyView(const MyObj& o) {}
      MyView pass() [[clang::lifetimebound]] { return *this; }
    };
    void target() {
      MyObj o;
      MyView v1 = o;
      POINT(p1);
      MyView v2 = v1.pass();
      POINT(p2);
    }
  )");
  EXPECT_THAT(Origin("v1"), HasLoansTo({"o"}, "p1"));
  // The call v1.pass() is bound to 'v1'. The origin of v2 should get the loans
  // from v1.
  EXPECT_THAT(Origin("v2"), HasLoansTo({"o"}, "p2"));
}

TEST_F(LifetimeAnalysisTest, LifetimeboundMultipleArgs) {
  SetupTest(R"(
    View Choose(bool cond, View a [[clang::lifetimebound]], View b [[clang::lifetimebound]]);
    void target() {
      MyObj o1, o2;
      View v1 = o1;
      View v2 = o2;
      POINT(p1);

      View v3 = Choose(true, v1, v2);
      POINT(p2);
    }
  )");
  EXPECT_THAT(Origin("v1"), HasLoansTo({"o1"}, "p1"));
  EXPECT_THAT(Origin("v2"), HasLoansTo({"o2"}, "p2"));
  // v3 should have loans from both v1 and v2, demonstrating the union of
  // loans.
  EXPECT_THAT(Origin("v3"), HasLoansTo({"o1", "o2"}, "p2"));
}

TEST_F(LifetimeAnalysisTest, LifetimeboundMixedArgs) {
  SetupTest(R"(
    View Choose(bool cond, View a [[clang::lifetimebound]], View b);
    void target() {
      MyObj o1, o2;
      View v1 = o1;
      View v2 = o2;
      POINT(p1);

      View v3 = Choose(true, v1, v2);
      POINT(p2);
    }
  )");
  EXPECT_THAT(Origin("v1"), HasLoansTo({"o1"}, "p1"));
  EXPECT_THAT(Origin("v2"), HasLoansTo({"o2"}, "p1"));
  // v3 should only have loans from v1, as v2 is not lifetimebound.
  EXPECT_THAT(Origin("v3"), HasLoansTo({"o1"}, "p2"));
}

TEST_F(LifetimeAnalysisTest, LifetimeboundChainOfViews) {
  SetupTest(R"(
    View Identity(View v [[clang::lifetimebound]]);
    View DoubleIdentity(View v [[clang::lifetimebound]]);

    void target() {
      MyObj obj;
      View v1 = obj;
      POINT(p1);
      View v2 = DoubleIdentity(Identity(v1));
      POINT(p2);
    }
  )");
  EXPECT_THAT(Origin("v1"), HasLoansTo({"obj"}, "p1"));
  // v2 should inherit the loan from v1 through the chain of calls.
  EXPECT_THAT(Origin("v2"), HasLoansTo({"obj"}, "p2"));
}

TEST_F(LifetimeAnalysisTest, LifetimeboundRawPointerParameter) {
  SetupTest(R"(
    View ViewFromPtr(const MyObj* p [[clang::lifetimebound]]);
    MyObj* PtrFromPtr(const MyObj* p [[clang::lifetimebound]]);
    MyObj* PtrFromView(View v [[clang::lifetimebound]]);

    void target() {
      MyObj a;
      View v = ViewFromPtr(&a);
      POINT(p1);

      MyObj b;
      MyObj* ptr1 = PtrFromPtr(&b);
      MyObj* ptr2 = PtrFromPtr(PtrFromPtr(PtrFromPtr(ptr1)));
      POINT(p2);

      MyObj c;
      View v2 = ViewFromPtr(PtrFromView(c));
      POINT(p3);
    }
  )");
  EXPECT_THAT(Origin("v"), HasLoansTo({"a"}, "p1"));
  EXPECT_THAT(Origin("ptr1"), HasLoansTo({"b"}, "p2"));
  EXPECT_THAT(Origin("ptr2"), HasLoansTo({"b"}, "p2"));
  EXPECT_THAT(Origin("v2"), HasLoansTo({"c"}, "p3"));
}

// FIXME: This can be controversial and may be revisited in the future.
TEST_F(LifetimeAnalysisTest, LifetimeboundConstRefViewParameter) {
  SetupTest(R"(
    View Identity(const View& v [[clang::lifetimebound]]);
    void target() {
      MyObj o;
      View v1 = o;
      View v2 = Identity(v1);
      POINT(p1);
    }
  )");
  EXPECT_THAT(Origin("v2"), HasLoansTo({"o"}, "p1"));
}

TEST_F(LifetimeAnalysisTest, LifetimeboundConstRefObjParam) {
  SetupTest(R"(
    View Identity(const MyObj& o [[clang::lifetimebound]]);
    void target() {
      MyObj a;
      View v1 = Identity(a);
      POINT(p1);
    }
  )");
  EXPECT_THAT(Origin("v1"), HasLoansTo({"a"}, "p1"));
}

TEST_F(LifetimeAnalysisTest, LifetimeboundReturnReference) {
  SetupTest(R"(
    const MyObj& Identity(View v [[clang::lifetimebound]]);
    void target() {
      MyObj a;
      View v1 = a;
      POINT(p1);

      View v2 = Identity(v1);
      
      const MyObj& b = Identity(v1);
      View v3 = Identity(b);
      POINT(p2);

      MyObj c;
      View v4 = Identity(c);
      POINT(p3);
    }
  )");
  EXPECT_THAT(Origin("v1"), HasLoansTo({"a"}, "p1"));
  EXPECT_THAT(Origin("v2"), HasLoansTo({"a"}, "p2"));

  // FIXME: Handle reference types. 'v3' should have loan to 'a' instead of 'b'.
  EXPECT_THAT(Origin("v3"), HasLoansTo({"b"}, "p2"));

  EXPECT_THAT(Origin("v4"), HasLoansTo({"c"}, "p3"));
}

TEST_F(LifetimeAnalysisTest, LifetimeboundTemplateFunction) {
  SetupTest(R"(
    template <typename T>
    const T& Identity(T&& v [[clang::lifetimebound]]);
    void target() {
      MyObj a;
      View v1 = Identity(a);
      POINT(p1);

      View v2 = Identity(v1);
      const View& v3 = Identity(v1);
      POINT(p2);
    }
  )");
  EXPECT_THAT(Origin("v1"), HasLoansTo({"a"}, "p1"));
  EXPECT_THAT(Origin("v2"), HasLoansTo({"a"}, "p2"));
  EXPECT_THAT(Origin("v3"), HasLoansTo({"a"}, "p2"));
}

TEST_F(LifetimeAnalysisTest, LifetimeboundTemplateClass) {
  SetupTest(R"(
    template<typename T>
    struct [[gsl::Pointer()]] MyTemplateView {
      MyTemplateView(const T& o) {}
      MyTemplateView pass() [[clang::lifetimebound]] { return *this; }
    };
    void target() {
      MyObj o;
      MyTemplateView<MyObj> v1 = o;
      POINT(p1);
      MyTemplateView<MyObj> v2 = v1.pass();
      POINT(p2);
    }
  )");
  EXPECT_THAT(Origin("v1"), HasLoansTo({"o"}, "p1"));
  EXPECT_THAT(Origin("v2"), HasLoansTo({"o"}, "p2"));
}

TEST_F(LifetimeAnalysisTest, LifetimeboundConversionOperator) {
  SetupTest(R"(
    struct MyOwner {
      MyObj o;
      operator View() const [[clang::lifetimebound]];
    };

    void target() {
      MyOwner owner;
      View v = owner;
      POINT(p1);
    }
  )");
  EXPECT_THAT(Origin("v"), HasLoansTo({"owner"}, "p1"));
}

TEST_F(LifetimeAnalysisTest, LivenessDeadPointer) {
  SetupTest(R"(
    void target() {
      POINT(p1);
      MyObj s;
      MyObj* p = &s;
      POINT(p2);
    }
  )");
  EXPECT_THAT(NoOrigins(), AreLiveAt("p2"));
  EXPECT_THAT(NoOrigins(), AreLiveAt("p1"));
}

TEST_F(LifetimeAnalysisTest, LivenessSimpleReturn) {
  SetupTest(R"(
    MyObj* target() {
      MyObj s;
      MyObj* p = &s;
      POINT(p1);
      return p;
    }
  )");
  EXPECT_THAT(Origins({"p"}), MustBeLiveAt("p1"));
}

TEST_F(LifetimeAnalysisTest, LivenessKilledByReassignment) {
  SetupTest(R"(
    MyObj* target() {
      MyObj s1, s2;
      MyObj* p = &s1;
      POINT(p1);
      p = &s2;
      POINT(p2);
      return p;
    }
  )");
  EXPECT_THAT(Origins({"p"}), MustBeLiveAt("p2"));
  EXPECT_THAT(NoOrigins(), AreLiveAt("p1"));
}

TEST_F(LifetimeAnalysisTest, LivenessAcrossBranches) {
  SetupTest(R"(
    MyObj* target(bool c) {
      MyObj x, y;
      MyObj* p = nullptr;
      POINT(p1);
      if (c) {
        p = &x;
        POINT(p2);
      } else {
        p = &y;
        POINT(p3);
      }
      return p;
    }
  )");
  EXPECT_THAT(Origins({"p"}), MustBeLiveAt("p2"));
  EXPECT_THAT(Origins({"p"}), MustBeLiveAt("p3"));
  // Before the `if`, the value of `p` (`nullptr`) is always overwritten before.
  EXPECT_THAT(NoOrigins(), AreLiveAt("p1"));
}

TEST_F(LifetimeAnalysisTest, LivenessInLoop) {
  SetupTest(R"(
    MyObj* target(bool c) {
      MyObj s1, s2;
      MyObj* p = &s1;
      MyObj* q = &s2;
      POINT(p1);
      while(c) {
        POINT(p2);

        p = q;
        POINT(p3);
      }
      POINT(p4);
      return p;
    }
  )");

  EXPECT_THAT(Origins({"p"}), MustBeLiveAt("p4"));
  EXPECT_THAT(NoOrigins(), MaybeLiveAt("p4"));

  EXPECT_THAT(Origins({"p", "q"}), MaybeLiveAt("p3"));

  EXPECT_THAT(Origins({"q"}), MustBeLiveAt("p2"));
  EXPECT_THAT(NoOrigins(), MaybeLiveAt("p2"));

  EXPECT_THAT(Origins({"p", "q"}), MaybeLiveAt("p1"));
}

TEST_F(LifetimeAnalysisTest, LivenessInLoopAndIf) {
  // See https://github.com/llvm/llvm-project/issues/156959.
  SetupTest(R"(
    void target(bool cond) {
      MyObj b;
      while (cond) {
        POINT(p1);

        MyObj a;
        View p = b;

        POINT(p2);

        if (cond) {
          POINT(p3);
          p = a;
        }
        POINT(p4);
        (void)p;
        POINT(p5);
      }
    }
  )");
  EXPECT_THAT(NoOrigins(), AreLiveAt("p5"));
  EXPECT_THAT(Origins({"p"}), MustBeLiveAt("p4"));
  EXPECT_THAT(NoOrigins(), AreLiveAt("p3"));
  EXPECT_THAT(Origins({"p"}), MaybeLiveAt("p2"));
  EXPECT_THAT(NoOrigins(), AreLiveAt("p1"));
}

TEST_F(LifetimeAnalysisTest, LivenessInLoopAndIf2) {
  SetupTest(R"(
    void target(MyObj safe, bool condition) {
      MyObj* p = &safe;
      MyObj* q = &safe;
      POINT(p1);

      while (condition) {
        POINT(p2);
        MyObj x;
        p = &x;

        POINT(p3);

        if (condition) {
          q = p;
          POINT(p4);
        }
        
        POINT(p5);
        (void)*p;
        (void)*q;
        POINT(p6);
      }
    }
  )");
  EXPECT_THAT(Origins({"q"}), MaybeLiveAt("p6"));
  EXPECT_THAT(NoOrigins(), MustBeLiveAt("p6"));

  EXPECT_THAT(Origins({"p", "q"}), MustBeLiveAt("p5"));

  EXPECT_THAT(Origins({"p", "q"}), MustBeLiveAt("p4"));

  EXPECT_THAT(Origins({"p"}), MustBeLiveAt("p3"));
  EXPECT_THAT(Origins({"q"}), MaybeLiveAt("p3"));

  EXPECT_THAT(Origins({"q"}), MaybeLiveAt("p2"));
  EXPECT_THAT(NoOrigins(), MustBeLiveAt("p2"));

  EXPECT_THAT(Origins({"q"}), MaybeLiveAt("p1"));
  EXPECT_THAT(NoOrigins(), MustBeLiveAt("p1"));
}

TEST_F(LifetimeAnalysisTest, LivenessOutsideLoop) {
  SetupTest(R"(
    void target(MyObj safe) {
      MyObj* p = &safe;
      for (int i = 0; i < 1; ++i) {
        MyObj s;
        p = &s;
        POINT(p1);
      }
      POINT(p2);
      (void)*p;
    }
  )");
  EXPECT_THAT(Origins({"p"}), MustBeLiveAt("p2"));
  EXPECT_THAT(Origins({"p"}), MaybeLiveAt("p1"));
}

} // anonymous namespace
} // namespace clang::lifetimes::internal
