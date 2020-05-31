//===- unittests/StaticAnalyzer/RegisterCustomCheckersTest.cpp ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CheckerRegistration.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/AnalysisManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Frontend/AnalysisConsumer.h"
#include "clang/StaticAnalyzer/Frontend/CheckerRegistry.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"
#include <memory>

namespace clang {
namespace ento {
namespace {

//===----------------------------------------------------------------------===//
// Just a minimal test for how checker registration works with statically
// linked, non TableGen generated checkers.
//===----------------------------------------------------------------------===//

class CustomChecker : public Checker<check::ASTCodeBody> {
public:
  void checkASTCodeBody(const Decl *D, AnalysisManager &Mgr,
                        BugReporter &BR) const {
    BR.EmitBasicReport(D, this, "Custom diagnostic", categories::LogicError,
                       "Custom diagnostic description",
                       PathDiagnosticLocation(D, Mgr.getSourceManager()), {});
  }
};

void addCustomChecker(AnalysisASTConsumer &AnalysisConsumer,
                      AnalyzerOptions &AnOpts) {
  AnOpts.CheckersAndPackages = {{"custom.CustomChecker", true}};
  AnalysisConsumer.AddCheckerRegistrationFn([](CheckerRegistry &Registry) {
    Registry.addChecker<CustomChecker>("custom.CustomChecker", "Description",
                                       "");
  });
}

TEST(RegisterCustomCheckers, RegisterChecker) {
  std::string Diags;
  EXPECT_TRUE(runCheckerOnCode<addCustomChecker>("void f() {;}", Diags));
  EXPECT_EQ(Diags, "custom.CustomChecker:Custom diagnostic description\n");
}

//===----------------------------------------------------------------------===//
// Pretty much the same.
//===----------------------------------------------------------------------===//

class LocIncDecChecker : public Checker<check::Location> {
public:
  void checkLocation(SVal Loc, bool IsLoad, const Stmt *S,
                     CheckerContext &C) const {
    const auto *UnaryOp = dyn_cast<UnaryOperator>(S);
    if (UnaryOp && !IsLoad) {
      EXPECT_FALSE(UnaryOp->isIncrementOp());
    }
  }
};

void addLocIncDecChecker(AnalysisASTConsumer &AnalysisConsumer,
                         AnalyzerOptions &AnOpts) {
  AnOpts.CheckersAndPackages = {{"test.LocIncDecChecker", true}};
  AnalysisConsumer.AddCheckerRegistrationFn([](CheckerRegistry &Registry) {
    Registry.addChecker<CustomChecker>("test.LocIncDecChecker", "Description",
                                       "");
  });
}

TEST(RegisterCustomCheckers, CheckLocationIncDec) {
  EXPECT_TRUE(
      runCheckerOnCode<addLocIncDecChecker>("void f() { int *p; (*p)++; }"));
}

//===----------------------------------------------------------------------===//
// Unsatisfied checker dependency
//===----------------------------------------------------------------------===//

class CheckerRegistrationOrderPrinter
    : public Checker<check::PreStmt<DeclStmt>> {
  std::unique_ptr<BuiltinBug> BT =
      std::make_unique<BuiltinBug>(this, "Registration order");

public:
  void checkPreStmt(const DeclStmt *DS, CheckerContext &C) const {
    ExplodedNode *N = nullptr;
    N = C.generateErrorNode();
    llvm::SmallString<200> Buf;
    llvm::raw_svector_ostream OS(Buf);
    C.getAnalysisManager()
        .getCheckerManager()
        ->getCheckerRegistry()
        .printEnabledCheckerList(OS);
    // Strip a newline off.
    auto R =
        std::make_unique<PathSensitiveBugReport>(*BT, OS.str().drop_back(1), N);
    C.emitReport(std::move(R));
  }
};

void registerCheckerRegistrationOrderPrinter(CheckerManager &mgr) {
  mgr.registerChecker<CheckerRegistrationOrderPrinter>();
}

bool shouldRegisterCheckerRegistrationOrderPrinter(const CheckerManager &mgr) {
  return true;
}

void addCheckerRegistrationOrderPrinter(CheckerRegistry &Registry) {
  Registry.addChecker(registerCheckerRegistrationOrderPrinter,
                      shouldRegisterCheckerRegistrationOrderPrinter,
                      "custom.RegistrationOrder", "Description", "", false);
}

#define UNITTEST_CHECKER(CHECKER_NAME, DIAG_MSG)                               \
  class CHECKER_NAME : public Checker<check::PreStmt<DeclStmt>> {              \
    std::unique_ptr<BuiltinBug> BT =                                           \
        std::make_unique<BuiltinBug>(this, DIAG_MSG);                          \
                                                                               \
  public:                                                                      \
    void checkPreStmt(const DeclStmt *DS, CheckerContext &C) const {}          \
  };                                                                           \
                                                                               \
  void register##CHECKER_NAME(CheckerManager &mgr) {                           \
    mgr.registerChecker<CHECKER_NAME>();                                       \
  }                                                                            \
                                                                               \
  bool shouldRegister##CHECKER_NAME(const CheckerManager &mgr) {               \
    return true;                                                               \
  }                                                                            \
  void add##CHECKER_NAME(CheckerRegistry &Registry) {                          \
    Registry.addChecker(register##CHECKER_NAME, shouldRegister##CHECKER_NAME,  \
                        "custom." #CHECKER_NAME, "Description", "", false);    \
  }

UNITTEST_CHECKER(StrongDep, "Strong")
UNITTEST_CHECKER(Dep, "Dep")

bool shouldRegisterStrongFALSE(const CheckerManager &mgr) {
  return false;
}


void addDep(AnalysisASTConsumer &AnalysisConsumer,
                  AnalyzerOptions &AnOpts) {
  AnOpts.CheckersAndPackages = {{"custom.Dep", true},
                                {"custom.RegistrationOrder", true}};
  AnalysisConsumer.AddCheckerRegistrationFn([](CheckerRegistry &Registry) {
    Registry.addChecker(registerStrongDep, shouldRegisterStrongFALSE,
                        "custom.Strong", "Description", "", false);
    addStrongDep(Registry);
    addDep(Registry);
    addCheckerRegistrationOrderPrinter(Registry);
    Registry.addDependency("custom.Dep", "custom.Strong");
  });
}

TEST(RegisterDeps, UnsatisfiedDependency) {
  std::string Diags;
  EXPECT_TRUE(runCheckerOnCode<addDep>("void f() {int i;}", Diags));
  EXPECT_EQ(Diags, "custom.RegistrationOrder:custom.RegistrationOrder\n");
}
} // namespace
} // namespace ento
} // namespace clang
