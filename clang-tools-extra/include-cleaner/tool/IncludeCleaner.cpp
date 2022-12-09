//===--- IncludeCleaner.cpp - standalone tool for include analysis --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AnalysisInternal.h"
#include "clang-include-cleaner/Record.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace include_cleaner {
namespace {
namespace cl = llvm::cl;

llvm::StringRef Overview = llvm::StringLiteral(R"(
clang-include-cleaner analyzes the #include directives in source code.

It suggests removing headers that the code is not using.
It suggests inserting headers that the code relies on, but does not include.
These changes make the file more self-contained and (at scale) make the codebase
easier to reason about and modify.

The tool operates on *working* source code. This means it can suggest including
headers that are only indirectly included, but cannot suggest those that are
missing entirely. (clang-include-fixer can do this).
)")
                               .trim();

cl::OptionCategory IncludeCleaner("clang-include-cleaner");

cl::opt<std::string> HTMLReportPath{
    "html",
    cl::desc("Specify an output filename for an HTML report. "
             "This describes both recommendations and reasons for changes."),
    cl::cat(IncludeCleaner),
};

class HTMLReportAction : public clang::ASTFrontendAction {
  RecordedAST AST;
  RecordedPP PP;
  PragmaIncludes PI;

  void ExecuteAction() override {
    auto &P = getCompilerInstance().getPreprocessor();
    P.addPPCallbacks(PP.record(P));
    PI.record(getCompilerInstance());
    ASTFrontendAction::ExecuteAction();
  }

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef File) override {
    return AST.record();
  }

  void EndSourceFile() override {
    std::error_code EC;
    llvm::raw_fd_ostream OS(HTMLReportPath, EC);
    if (EC) {
      llvm::errs() << "Unable to write HTML report to " << HTMLReportPath
                   << ": " << EC.message() << "\n";
      exit(1);
    }
    writeHTMLReport(AST.Ctx->getSourceManager().getMainFileID(), AST.Roots,
                    PP.MacroReferences, *AST.Ctx, &PI, OS);
  }
};

} // namespace
} // namespace include_cleaner
} // namespace clang

int main(int argc, const char **argv) {
  using namespace clang::include_cleaner;

  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  auto OptionsParser =
      clang::tooling::CommonOptionsParser::create(argc, argv, IncludeCleaner);
  if (!OptionsParser) {
    llvm::errs() << toString(OptionsParser.takeError());
    return 1;
  }

  std::unique_ptr<clang::tooling::FrontendActionFactory> Factory;
  if (HTMLReportPath.getNumOccurrences()) {
    if (OptionsParser->getSourcePathList().size() != 1) {
      llvm::errs() << "-" << HTMLReportPath.ArgStr
                   << " requires a single input file";
      return 1;
    }
    Factory = clang::tooling::newFrontendActionFactory<HTMLReportAction>();
  } else {
    llvm::errs() << "Unimplemented\n";
    return 1;
  }

  return clang::tooling::ClangTool(OptionsParser->getCompilations(),
                                   OptionsParser->getSourcePathList())
      .run(Factory.get());
}
