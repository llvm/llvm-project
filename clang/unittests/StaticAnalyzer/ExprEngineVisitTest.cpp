//===- ExprEngineVisitTest.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CheckerRegistration.h"
#include "clang/AST/Stmt.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SVals.h"
#include "llvm/Support/Casting.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace ento;

namespace {

void emitErrorReport(CheckerContext &C, const BugType &Bug,
                     const std::string &Desc) {
  if (ExplodedNode *Node = C.generateNonFatalErrorNode(C.getState())) {
    auto Report = std::make_unique<PathSensitiveBugReport>(Bug, Desc, Node);
    C.emitReport(std::move(Report));
  }
}

inline std::string getMemRegionName(const SVal &Val) {
  if (auto MemVal = llvm::dyn_cast<loc::MemRegionVal>(Val))
    return MemVal->getRegion()->getDescriptiveName(false);
  if (auto ComVal = llvm::dyn_cast<nonloc::LazyCompoundVal>(Val))
    return ComVal->getRegion()->getDescriptiveName(false);
  return "";
}

#define CREATE_EXPR_ENGINE_CHECKER(CHECKER_NAME, CALLBACK, STMT_TYPE,          \
                                   BUG_NAME)                                   \
  class CHECKER_NAME : public Checker<check::CALLBACK<STMT_TYPE>> {            \
  public:                                                                      \
    void check##CALLBACK(const STMT_TYPE *ASM, CheckerContext &C) const {      \
      emitErrorReport(C, Bug, "check" #CALLBACK "<" #STMT_TYPE ">");           \
    }                                                                          \
                                                                               \
  private:                                                                     \
    const BugType Bug{this, BUG_NAME};                                         \
  };

CREATE_EXPR_ENGINE_CHECKER(ExprEngineVisitPreChecker, PreStmt, GCCAsmStmt,
                           "GCCAsmStmtBug")
CREATE_EXPR_ENGINE_CHECKER(ExprEngineVisitPostChecker, PostStmt, GCCAsmStmt,
                           "GCCAsmStmtBug")

class MemAccessChecker : public Checker<check::Location, check::Bind> {
public:
  void checkLocation(const SVal &Loc, bool IsLoad, const Stmt *S,
                     CheckerContext &C) const {
    emitErrorReport(C, Bug, "checkLocation: Loc = " + getMemRegionName(Loc));
  }

  void checkBind(SVal Loc, SVal Val, const Stmt *S, CheckerContext &C) const {
    emitErrorReport(C, Bug, "checkBind: Loc = " + getMemRegionName(Loc));
  }

private:
  const BugType Bug{this, "MemAccess"};
};

void addExprEngineVisitPreChecker(AnalysisASTConsumer &AnalysisConsumer,
                                  AnalyzerOptions &AnOpts) {
  AnOpts.CheckersAndPackages = {{"ExprEngineVisitPreChecker", true}};
  AnalysisConsumer.AddCheckerRegistrationFn([](CheckerRegistry &Registry) {
    Registry.addChecker<ExprEngineVisitPreChecker>("ExprEngineVisitPreChecker",
                                                   "Desc", "DocsURI");
  });
}

void addExprEngineVisitPostChecker(AnalysisASTConsumer &AnalysisConsumer,
                                   AnalyzerOptions &AnOpts) {
  AnOpts.CheckersAndPackages = {{"ExprEngineVisitPostChecker", true}};
  AnalysisConsumer.AddCheckerRegistrationFn([](CheckerRegistry &Registry) {
    Registry.addChecker<ExprEngineVisitPostChecker>(
        "ExprEngineVisitPostChecker", "Desc", "DocsURI");
  });
}

void addMemAccessChecker(AnalysisASTConsumer &AnalysisConsumer,
                         AnalyzerOptions &AnOpts) {
  AnOpts.CheckersAndPackages = {{"MemAccessChecker", true}};
  AnalysisConsumer.AddCheckerRegistrationFn([](CheckerRegistry &Registry) {
    Registry.addChecker<MemAccessChecker>("MemAccessChecker", "Desc",
                                          "DocsURI");
  });
}

TEST(ExprEngineVisitTest, checkPreStmtGCCAsmStmt) {
  std::string Diags;
  EXPECT_TRUE(runCheckerOnCode<addExprEngineVisitPreChecker>(R"(
    void top() {
      asm("");
    }
  )",
                                                             Diags));
  EXPECT_EQ(Diags, "ExprEngineVisitPreChecker: checkPreStmt<GCCAsmStmt>\n");
}

TEST(ExprEngineVisitTest, checkPostStmtGCCAsmStmt) {
  std::string Diags;
  EXPECT_TRUE(runCheckerOnCode<addExprEngineVisitPostChecker>(R"(
    void top() {
      asm("");
    }
  )",
                                                              Diags));
  EXPECT_EQ(Diags, "ExprEngineVisitPostChecker: checkPostStmt<GCCAsmStmt>\n");
}

TEST(ExprEngineVisitTest, checkLocationAndBind) {
  std::string Diags;
  EXPECT_TRUE(runCheckerOnCode<addMemAccessChecker>(R"(
    class MyClass{
    public:
      int Value;
    };
    extern MyClass MyClassWrite, MyClassRead; 
    void top() {
      MyClassWrite = MyClassRead;
    }
  )",
                                                    Diags));

  std::string RHSMsg = "checkLocation: Loc = MyClassRead";
  std::string LHSMsg = "checkBind: Loc = MyClassWrite";
  EXPECT_NE(Diags.find(RHSMsg), std::string::npos);
  EXPECT_NE(Diags.find(LHSMsg), std::string::npos);
}

} // namespace
