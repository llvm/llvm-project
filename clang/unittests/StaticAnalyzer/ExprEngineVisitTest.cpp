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

class ExprEngineVisitPreChecker : public Checker<check::PreStmt<GCCAsmStmt>> {
public:
  void checkPreStmt(const GCCAsmStmt *ASM, CheckerContext &C) const {
    emitErrorReport(C, Bug, "PreStmt<GCCAsmStmt>");
  }

private:
  const BugType Bug{this, "GCCAsmStmtBug"};
};

class ExprEngineVisitPostChecker : public Checker<check::PostStmt<GCCAsmStmt>> {
public:
  void checkPostStmt(const GCCAsmStmt *ASM, CheckerContext &C) const {
    emitErrorReport(C, Bug, "PostStmt<GCCAsmStmt>");
  }

private:
  const BugType Bug{this, "GCCAsmStmtBug"};
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

TEST(ExprEngineVisitTest, checkPreStmtGCCAsmStmt) {
  std::string Diags;
  EXPECT_TRUE(runCheckerOnCode<addExprEngineVisitPreChecker>(R"(
    void top() {
      asm("");
    }
  )",
                                                             Diags));
  EXPECT_EQ(Diags, "ExprEngineVisitPreChecker: PreStmt<GCCAsmStmt>\n");
}

TEST(ExprEngineVisitTest, checkPostStmtGCCAsmStmt) {
  std::string Diags;
  EXPECT_TRUE(runCheckerOnCode<addExprEngineVisitPostChecker>(R"(
    void top() {
      asm("");
    }
  )",
                                                              Diags));
  EXPECT_EQ(Diags, "ExprEngineVisitPostChecker: PostStmt<GCCAsmStmt>\n");
}

} // namespace
