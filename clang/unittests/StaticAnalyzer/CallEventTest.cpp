//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "CheckerRegistration.h"
#include "clang/Basic/LLVM.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/BugReporter/CommonBugCategories.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/AnalysisManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExplodedGraph.h"
#include "clang/StaticAnalyzer/Frontend/AnalysisConsumer.h"
#include "clang/StaticAnalyzer/Frontend/CheckerRegistry.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"

namespace clang {
namespace ento {
namespace {

void reportBug(const CheckerBase *Checker, const CallEvent &Call,
               CheckerContext &C, StringRef WarningMsg) {
  C.getBugReporter().EmitBasicReport(
      nullptr, Checker, "", categories::LogicError, WarningMsg,
      PathDiagnosticLocation(Call.getOriginExpr(), C.getSourceManager(),
                             C.getLocationContext()),
      {});
}

class CXXDeallocatorChecker : public Checker<check::PreCall> {
public:
  void checkPreCall(const CallEvent &Call, CheckerContext &C) const {
    const auto *DC = dyn_cast<CXXDeallocatorCall>(&Call);
    if (!DC) {
      return;
    }

    SmallString<100> WarningBuf;
    llvm::raw_svector_ostream WarningOS(WarningBuf);
    WarningOS << "NumArgs: " << DC->getNumArgs();

    reportBug(this, *DC, C, WarningBuf);
  }
};

void addCXXDeallocatorChecker(AnalysisASTConsumer &AnalysisConsumer,
                              AnalyzerOptions &AnOpts) {
  AnOpts.CheckersAndPackages = {{"test.CXXDeallocator", true}};
  AnalysisConsumer.AddCheckerRegistrationFn([](CheckerRegistry &Registry) {
    Registry.addChecker<CXXDeallocatorChecker>("test.CXXDeallocator",
                                               "Description", "");
  });
}

// TODO: What we should really be testing here is all the different varieties
// of delete operators, and wether the retrieval of their arguments works as
// intended. At the time of writing this file, CXXDeallocatorCall doesn't pick
// up on much of those due to the AST not containing CXXDeleteExpr for most of
// the standard/custom deletes.
TEST(CXXDeallocatorCall, SimpleDestructor) {
  std::string Diags;
  EXPECT_TRUE(runCheckerOnCode<addCXXDeallocatorChecker>(R"(
    struct A {};

    void f() {
      A *a = new A;
      delete a;
    }
  )",
                                                         Diags));
  EXPECT_EQ(Diags, "test.CXXDeallocator: NumArgs: 2\n");
}

} // namespace
} // namespace ento
} // namespace clang
