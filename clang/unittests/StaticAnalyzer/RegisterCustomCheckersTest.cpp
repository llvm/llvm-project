//===- unittests/StaticAnalyzer/RegisterCustomCheckersTest.cpp ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

class CustomChecker : public Checker<check::ASTCodeBody> {
public:
  void checkASTCodeBody(const Decl *D, AnalysisManager &Mgr,
                        BugReporter &BR) const {
    BR.EmitBasicReport(D, this, "Custom diagnostic", categories::LogicError,
                       "Custom diagnostic description",
                       PathDiagnosticLocation(D, Mgr.getSourceManager()), {});
  }
};

class TestAction : public ASTFrontendAction {
  class DiagConsumer : public PathDiagnosticConsumer {
    llvm::raw_ostream &Output;

  public:
    DiagConsumer(llvm::raw_ostream &Output) : Output(Output) {}
    void FlushDiagnosticsImpl(std::vector<const PathDiagnostic *> &Diags,
                              FilesMade *filesMade) override {
      for (const auto *PD : Diags)
        Output << PD->getCheckName() << ":" << PD->getShortDescription();
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
    Compiler.getAnalyzerOpts()->CheckersControlList = {
        {"custom.CustomChecker", true}};
    AnalysisConsumer->AddCheckerRegistrationFn([](CheckerRegistry &Registry) {
      Registry.addChecker<CustomChecker>("custom.CustomChecker", "Description",
                                         "");
    });
    return std::move(AnalysisConsumer);
  }
};


TEST(RegisterCustomCheckers, RegisterChecker) {
  std::string Diags;
  {
    llvm::raw_string_ostream OS(Diags);
    EXPECT_TRUE(tooling::runToolOnCode(new TestAction(OS), "void f() {;}"));
  }
  EXPECT_EQ(Diags, "custom.CustomChecker:Custom diagnostic description");
}

}
}
}
