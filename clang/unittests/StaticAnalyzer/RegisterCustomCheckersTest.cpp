//===- unittests/StaticAnalyzer/RegisterCustomCheckersTest.cpp ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/CompilerInstance.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/AnalysisManager.h"
#include "clang/StaticAnalyzer/Frontend/AnalysisConsumer.h"
#include "clang/StaticAnalyzer/Frontend/CheckerRegistry.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"

namespace clang {
namespace ento {
namespace {

template <typename CheckerT>
class TestAction : public ASTFrontendAction {
  class DiagConsumer : public PathDiagnosticConsumer {
    llvm::raw_ostream &Output;

  public:
    DiagConsumer(llvm::raw_ostream &Output) : Output(Output) {}
    void FlushDiagnosticsImpl(std::vector<const PathDiagnostic *> &Diags,
                              FilesMade *filesMade) override {
      for (const auto *PD : Diags)
        Output << PD->getCheckerName() << ":" << PD->getShortDescription();
    }

    StringRef getName() const override { return "Test"; }
  };

  llvm::raw_ostream &DiagsOutput;

public:
  TestAction(llvm::raw_ostream &DiagsOutput) : DiagsOutput(DiagsOutput) {}

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &Compiler,
                                                 StringRef File) override {
    std::unique_ptr<AnalysisASTConsumer> AnalysisConsumer =
        CreateAnalysisConsumer(Compiler);
    AnalysisConsumer->AddDiagnosticConsumer(new DiagConsumer(DiagsOutput));
    Compiler.getAnalyzerOpts()->CheckersAndPackages = {
        {"custom.CustomChecker", true}};
    AnalysisConsumer->AddCheckerRegistrationFn([](CheckerRegistry &Registry) {
      Registry.addChecker<CheckerT>("custom.CustomChecker", "Description", "");
    });
    return std::move(AnalysisConsumer);
  }
};

template <typename CheckerT>
bool runCheckerOnCode(const std::string &Code, std::string &Diags) {
  llvm::raw_string_ostream OS(Diags);
  return tooling::runToolOnCode(std::make_unique<TestAction<CheckerT>>(OS),
                                Code);
}
template <typename CheckerT>
bool runCheckerOnCode(const std::string &Code) {
  std::string Diags;
  return runCheckerOnCode<CheckerT>(Code, Diags);
}


class CustomChecker : public Checker<check::ASTCodeBody> {
public:
  void checkASTCodeBody(const Decl *D, AnalysisManager &Mgr,
                        BugReporter &BR) const {
    BR.EmitBasicReport(D, this, "Custom diagnostic", categories::LogicError,
                       "Custom diagnostic description",
                       PathDiagnosticLocation(D, Mgr.getSourceManager()), {});
  }
};

TEST(RegisterCustomCheckers, RegisterChecker) {
  std::string Diags;
  EXPECT_TRUE(runCheckerOnCode<CustomChecker>("void f() {;}", Diags));
  EXPECT_EQ(Diags, "custom.CustomChecker:Custom diagnostic description");
}

class LocIncDecChecker : public Checker<check::Location> {
public:
  void checkLocation(SVal Loc, bool IsLoad, const Stmt *S,
                     CheckerContext &C) const {
    auto UnaryOp = dyn_cast<UnaryOperator>(S);
    if (UnaryOp && !IsLoad) {
      EXPECT_FALSE(UnaryOp->isIncrementOp());
    }
  }
};

TEST(RegisterCustomCheckers, CheckLocationIncDec) {
  EXPECT_TRUE(
      runCheckerOnCode<LocIncDecChecker>("void f() { int *p; (*p)++; }"));
}

}
}
}
