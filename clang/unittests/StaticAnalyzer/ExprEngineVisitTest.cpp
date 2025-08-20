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
    emitErrorReport(C, Bug,
                    "checkLocation: Loc = " + dumpToString(Loc) +
                        ", Stmt = " + S->getStmtClassName());
  }

  void checkBind(SVal Loc, SVal Val, const Stmt *S, bool AtDeclInit,
                 CheckerContext &C) const {
    emitErrorReport(C, Bug,
                    "checkBind: Loc = " + dumpToString(Loc) +
                        ", Val = " + dumpToString(Val) +
                        ", Stmt = " + S->getStmtClassName() +
                        ", AtDeclInit = " + (AtDeclInit ? "true" : "false"));
  }

private:
  const BugType Bug{this, "MemAccess"};

  std::string dumpToString(SVal V) const {
    std::string StrBuf;
    llvm::raw_string_ostream StrStream{StrBuf};
    V.dumpToStream(StrStream);
    return StrBuf;
  }
};

void addExprEngineVisitPreChecker(AnalysisASTConsumer &AnalysisConsumer,
                                  AnalyzerOptions &AnOpts) {
  AnOpts.CheckersAndPackages = {{"ExprEngineVisitPreChecker", true}};
  AnalysisConsumer.AddCheckerRegistrationFn([](CheckerRegistry &Registry) {
    Registry.addChecker<ExprEngineVisitPreChecker>("ExprEngineVisitPreChecker",
                                                   "MockDescription");
  });
}

void addExprEngineVisitPostChecker(AnalysisASTConsumer &AnalysisConsumer,
                                   AnalyzerOptions &AnOpts) {
  AnOpts.CheckersAndPackages = {{"ExprEngineVisitPostChecker", true}};
  AnalysisConsumer.AddCheckerRegistrationFn([](CheckerRegistry &Registry) {
    Registry.addChecker<ExprEngineVisitPostChecker>(
        "ExprEngineVisitPostChecker", "MockDescription");
  });
}

void addMemAccessChecker(AnalysisASTConsumer &AnalysisConsumer,
                         AnalyzerOptions &AnOpts) {
  AnOpts.CheckersAndPackages = {{"MemAccessChecker", true}};
  AnalysisConsumer.AddCheckerRegistrationFn([](CheckerRegistry &Registry) {
    Registry.addChecker<MemAccessChecker>("MemAccessChecker",
                                          "MockDescription");
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

  std::string LocMsg = "checkLocation: Loc = lazyCompoundVal{0x0,MyClassRead}, "
                       "Stmt = ImplicitCastExpr";
  std::string BindMsg =
      "checkBind: Loc = &MyClassWrite, Val = lazyCompoundVal{0x0,MyClassRead}, "
      "Stmt = CXXOperatorCallExpr, AtDeclInit = false";
  std::size_t LocPos = Diags.find(LocMsg);
  std::size_t BindPos = Diags.find(BindMsg);
  EXPECT_NE(LocPos, std::string::npos);
  EXPECT_NE(BindPos, std::string::npos);
  // Check order: first checkLocation is called, then checkBind.
  // In the diagnosis, however, the messages appear in reverse order.
  EXPECT_TRUE(LocPos > BindPos);
}

TEST(ExprEngineVisitTest, checkLocationAndBindInitialization) {
  std::string Diags;
  EXPECT_TRUE(runCheckerOnCode<addMemAccessChecker>(R"(
    class MyClass{
    public:
      int Value;
    };
    void top(MyClass param) {
      MyClass MyClassWrite = param;
    }
  )",
                                                    Diags));

  EXPECT_TRUE(StringRef(Diags).contains("AtDeclInit = true"));
}

} // namespace
