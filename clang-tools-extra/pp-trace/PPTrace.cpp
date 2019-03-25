//===--- tools/pp-trace/PPTrace.cpp - Clang preprocessor tracer -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements pp-trace, a tool for displaying a textual trace
// of the Clang preprocessor activity.  It's based on a derivation of the
// PPCallbacks class, that once registerd with Clang, receives callback calls
// to its virtual members, and outputs the information passed to the callbacks
// in a high-level YAML format.
//
// The pp-trace tool also serves as the basis for a test of the PPCallbacks
// mechanism.
//
// The pp-trace tool supports the following general command line format:
//
//    pp-trace [options] file... [-- compiler options]
//
// Basically you put the pp-trace options first, then the source file or files,
// and then -- followed by any options you want to pass to the compiler.
//
//===----------------------------------------------------------------------===//

#include "PPCallbacksTracker.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Driver/Options.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/Execution.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/GlobPattern.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/WithColor.h"
#include <string>
#include <vector>

using namespace llvm;

namespace clang {
namespace pp_trace {

static cl::OptionCategory Cat("pp-trace options");

static cl::opt<std::string> Callbacks(
    "callbacks", cl::init("*"),
    cl::desc("Comma-separated list of globs describing the list of callbacks "
             "to output. Globs are processed in order of appearance. Globs "
             "with the '-' prefix remove callbacks from the set. e.g. "
             "'*,-Macro*'."),
    cl::cat(Cat));

static cl::opt<std::string> OutputFileName(
    "output", cl::init("-"),
    cl::desc("Output trace to the given file name or '-' for stdout."),
    cl::cat(Cat));

LLVM_ATTRIBUTE_NORETURN static void error(Twine Message) {
  WithColor::error() << Message << '\n';
  exit(1);
}

namespace {

class PPTraceAction : public ASTFrontendAction {
public:
  PPTraceAction(const FilterType &Filters, raw_ostream &OS)
      : Filters(Filters), OS(OS) {}

protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override {
    Preprocessor &PP = CI.getPreprocessor();
    PP.addPPCallbacks(
        make_unique<PPCallbacksTracker>(Filters, CallbackCalls, PP));
    return make_unique<ASTConsumer>();
  }

  void EndSourceFileAction() override {
    OS << "---\n";
    for (const CallbackCall &Callback : CallbackCalls) {
      OS << "- Callback: " << Callback.Name << "\n";
      for (const Argument &Arg : Callback.Arguments)
        OS << "  " << Arg.Name << ": " << Arg.Value << "\n";
    }
    OS << "...\n";

    CallbackCalls.clear();
  }

private:
  const FilterType &Filters;
  raw_ostream &OS;
  std::vector<CallbackCall> CallbackCalls;
};

class PPTraceFrontendActionFactory : public tooling::FrontendActionFactory {
public:
  PPTraceFrontendActionFactory(const FilterType &Filters, raw_ostream &OS)
      : Filters(Filters), OS(OS) {}

  PPTraceAction *create() override { return new PPTraceAction(Filters, OS); }

private:
  const FilterType &Filters;
  raw_ostream &OS;
};
} // namespace
} // namespace pp_trace
} // namespace clang

int main(int argc, const char **argv) {
  using namespace clang::pp_trace;

  InitLLVM X(argc, argv);
  auto Exec =
      clang::tooling::createExecutorFromCommandLineArgs(argc, argv, Cat);
  if (!Exec)
    error(toString(Exec.takeError()));

  // Parse the IgnoreCallbacks list into strings.
  SmallVector<StringRef, 32> Patterns;
  FilterType Filters;
  StringRef(Callbacks).split(Patterns, ",",
                             /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (StringRef Pattern : Patterns) {
    Pattern = Pattern.trim();
    bool Enabled = !Pattern.consume_front("-");
    if (Expected<GlobPattern> Pat = GlobPattern::create(Pattern))
      Filters.emplace_back(std::move(*Pat), Enabled);
    else
      error(toString(Pat.takeError()));
  }

  std::error_code EC;
  llvm::ToolOutputFile Out(OutputFileName, EC, llvm::sys::fs::F_Text);
  if (EC)
    error(EC.message());

  if (Error Err = Exec->get()->execute(
      make_unique<PPTraceFrontendActionFactory>(Filters, Out.os())))
    error(toString(std::move(Err)));
  Out.keep();
  return 0;
}
