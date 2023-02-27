//===--- IncludeCleaner.cpp - standalone tool for include analysis --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AnalysisInternal.h"
#include "clang-include-cleaner/Analysis.h"
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

enum class PrintStyle { Changes, Final };
cl::opt<PrintStyle> Print{
    "print",
    cl::values(
        clEnumValN(PrintStyle::Changes, "changes", "Print symbolic changes"),
        clEnumValN(PrintStyle::Final, "", "Print final code")),
    cl::ValueOptional,
    cl::init(PrintStyle::Final),
    cl::desc("Print the list of headers to insert and remove"),
    cl::cat(IncludeCleaner),
};

cl::opt<bool> Edit{
    "edit",
    cl::desc("Apply edits to analyzed source files"),
    cl::cat(IncludeCleaner),
};

cl::opt<bool> Insert{
    "insert",
    cl::desc("Allow header insertions"),
    cl::init(true),
    cl::cat(IncludeCleaner),
};
cl::opt<bool> Remove{
    "remove",
    cl::desc("Allow header removals"),
    cl::init(true),
    cl::cat(IncludeCleaner),
};

std::atomic<unsigned> Errors = ATOMIC_VAR_INIT(0);

format::FormatStyle getStyle(llvm::StringRef Filename) {
  auto S = format::getStyle(format::DefaultFormatStyle, Filename,
                            format::DefaultFallbackStyle);
  if (!S || !S->isCpp()) {
    consumeError(S.takeError());
    return format::getLLVMStyle();
  }
  return std::move(*S);
}

class Action : public clang::ASTFrontendAction {
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
    if (!HTMLReportPath.empty())
      writeHTML();

    const auto &SM = getCompilerInstance().getSourceManager();
    auto &HS = getCompilerInstance().getPreprocessor().getHeaderSearchInfo();
    llvm::StringRef Path =
        SM.getFileEntryForID(SM.getMainFileID())->tryGetRealPathName();
    assert(!Path.empty() && "Main file path not known?");
    llvm::StringRef Code = SM.getBufferData(SM.getMainFileID());

    auto Results =
        analyze(AST.Roots, PP.MacroReferences, PP.Includes, &PI, SM, HS);
    if (!Insert)
      Results.Missing.clear();
    if (!Remove)
      Results.Unused.clear();
    std::string Final = fixIncludes(Results, Code, getStyle(Path));

    if (Print.getNumOccurrences()) {
      switch (Print) {
      case PrintStyle::Changes:
        for (const Include *I : Results.Unused)
          llvm::outs() << "- " << I->quote() << " @Line:" << I->Line << "\n";
        for (const auto &I : Results.Missing)
          llvm::outs() << "+ " << I << "\n";
        break;
      case PrintStyle::Final:
        llvm::outs() << Final;
        break;
      }
    }

    if (Edit) {
      if (auto Err = llvm::writeToOutput(
              Path, [&](llvm::raw_ostream &OS) -> llvm::Error {
                OS << Final;
                return llvm::Error::success();
              })) {
        llvm::errs() << "Failed to apply edits to " << Path << ": "
                     << toString(std::move(Err)) << "\n";
        ++Errors;
      }
    }
  }

  void writeHTML() {
    std::error_code EC;
    llvm::raw_fd_ostream OS(HTMLReportPath, EC);
    if (EC) {
      llvm::errs() << "Unable to write HTML report to " << HTMLReportPath
                   << ": " << EC.message() << "\n";
      ++Errors;
      return;
    }
    writeHTMLReport(
        AST.Ctx->getSourceManager().getMainFileID(), PP.Includes, AST.Roots,
        PP.MacroReferences, *AST.Ctx,
        getCompilerInstance().getPreprocessor().getHeaderSearchInfo(), &PI, OS);
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

  if (OptionsParser->getSourcePathList().size() != 1) {
    std::vector<cl::Option *> IncompatibleFlags = {&HTMLReportPath, &Print};
    for (const auto *Flag : IncompatibleFlags) {
      if (Flag->getNumOccurrences()) {
        llvm::errs() << "-" << Flag->ArgStr << " requires a single input file";
        return 1;
      }
    }
  }
  auto Factory = clang::tooling::newFrontendActionFactory<Action>();
  return clang::tooling::ClangTool(OptionsParser->getCompilations(),
                                   OptionsParser->getSourcePathList())
             .run(Factory.get()) ||
         Errors != 0;
}
