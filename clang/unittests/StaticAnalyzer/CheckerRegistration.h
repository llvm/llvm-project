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

class DiagConsumer : public PathDiagnosticConsumer {
  llvm::raw_ostream &Output;

public:
  DiagConsumer(llvm::raw_ostream &Output) : Output(Output) {}
  void FlushDiagnosticsImpl(std::vector<const PathDiagnostic *> &Diags,
                            FilesMade *filesMade) override {
    for (const auto *PD : Diags)
      Output << PD->getCheckerName() << ":" << PD->getShortDescription() << '\n';
  }

  StringRef getName() const override { return "Test"; }
};

using AddCheckerFn = void(AnalysisASTConsumer &AnalysisConsumer,
                          AnalyzerOptions &AnOpts);

template <AddCheckerFn Fn1, AddCheckerFn Fn2, AddCheckerFn... Fns>
void addChecker(AnalysisASTConsumer &AnalysisConsumer,
                AnalyzerOptions &AnOpts) {
  Fn1(AnalysisConsumer, AnOpts);
  addChecker<Fn2, Fns...>(AnalysisConsumer, AnOpts);
}

template <AddCheckerFn Fn1>
void addChecker(AnalysisASTConsumer &AnalysisConsumer,
                AnalyzerOptions &AnOpts) {
  Fn1(AnalysisConsumer, AnOpts);
}

template <AddCheckerFn... Fns>
class TestAction : public ASTFrontendAction {
  llvm::raw_ostream &DiagsOutput;

public:
  TestAction(llvm::raw_ostream &DiagsOutput) : DiagsOutput(DiagsOutput) {}

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &Compiler,
                                                 StringRef File) override {
    std::unique_ptr<AnalysisASTConsumer> AnalysisConsumer =
        CreateAnalysisConsumer(Compiler);
    AnalysisConsumer->AddDiagnosticConsumer(new DiagConsumer(DiagsOutput));
    addChecker<Fns...>(*AnalysisConsumer, *Compiler.getAnalyzerOpts());
    return std::move(AnalysisConsumer);
  }
};

inline SmallString<80> getCurrentTestNameAsFileName() {
  const ::testing::TestInfo *Info =
      ::testing::UnitTest::GetInstance()->current_test_info();

  SmallString<80> FileName;
  (Twine{Info->name()} + ".cc").toVector(FileName);
  return FileName;
}

template <AddCheckerFn... Fns>
bool runCheckerOnCode(const std::string &Code, std::string &Diags) {
  const SmallVectorImpl<char> &FileName = getCurrentTestNameAsFileName();
  llvm::raw_string_ostream OS(Diags);
  return tooling::runToolOnCode(std::make_unique<TestAction<Fns...>>(OS), Code,
                                FileName);
}

template <AddCheckerFn... Fns>
bool runCheckerOnCode(const std::string &Code) {
  std::string Diags;
  return runCheckerOnCode<Fns...>(Code, Diags);
}

template <AddCheckerFn... Fns>
bool runCheckerOnCodeWithArgs(const std::string &Code,
                              const std::vector<std::string> &Args,
                              std::string &Diags) {
  const SmallVectorImpl<char> &FileName = getCurrentTestNameAsFileName();
  llvm::raw_string_ostream OS(Diags);
  return tooling::runToolOnCodeWithArgs(
      std::make_unique<TestAction<Fns...>>(OS), Code, Args, FileName);
}

template <AddCheckerFn... Fns>
bool runCheckerOnCodeWithArgs(const std::string &Code,
                              const std::vector<std::string> &Args) {
  std::string Diags;
  return runCheckerOnCodeWithArgs<Fns...>(Code, Args, Diags);
}

} // namespace ento
} // namespace clang
