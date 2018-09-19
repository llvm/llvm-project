//===--- tools/pp-trace/PPTrace.cpp - Clang preprocessor tracer -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
//    pp-trace [pp-trace options] (source file) [compiler options]
//
// Basically you put the pp-trace options first, then the source file or files,
// and then any options you want to pass to the compiler.
//
// These are the pp-trace options:
//
//    -ignore (callback list)     Don't display output for a comma-separated
//                                list of callbacks, i.e.:
//                                  -ignore "FileChanged,InclusionDirective"
//
//    -output (file)              Output trace to the given file in a YAML
//                                format, e.g.:
//
//                                  ---
//                                  - Callback: Name
//                                    Argument1: Value1
//                                    Argument2: Value2
//                                  (etc.)
//                                  ...
//
// Future Directions:
//
// 1. Add option opposite to "-ignore" that specifys a comma-separated option
// list of callbacs.  Perhaps "-only" or "-exclusive".
//
//===----------------------------------------------------------------------===//

#include "PPCallbacksTracker.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Driver/Options.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"
#include <algorithm>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using namespace clang;
using namespace clang::tooling;
using namespace llvm;

// Options:

// Collect the source files.
static cl::list<std::string> SourcePaths(cl::Positional,
                                         cl::desc("<source0> [... <sourceN>]"),
                                         cl::OneOrMore);

// Option to specify a list or one or more callback names to ignore.
static cl::opt<std::string> IgnoreCallbacks(
    "ignore", cl::init(""),
    cl::desc("Ignore callbacks, i.e. \"Callback1, Callback2...\"."));

// Option to specify the trace output file name.
static cl::opt<std::string> OutputFileName(
    "output", cl::init(""),
    cl::desc("Output trace to the given file name or '-' for stdout."));

// Collect all other arguments, which will be passed to the front end.
static cl::list<std::string>
    CC1Arguments(cl::ConsumeAfter,
                 cl::desc("<arguments to be passed to front end>..."));

// Frontend action stuff:

namespace {
// Consumer is responsible for setting up the callbacks.
class PPTraceConsumer : public ASTConsumer {
public:
  PPTraceConsumer(SmallSet<std::string, 4> &Ignore,
                  std::vector<CallbackCall> &CallbackCalls, Preprocessor &PP) {
    // PP takes ownership.
    PP.addPPCallbacks(llvm::make_unique<PPCallbacksTracker>(Ignore,
                                                            CallbackCalls, PP));
  }
};

class PPTraceAction : public SyntaxOnlyAction {
public:
  PPTraceAction(SmallSet<std::string, 4> &Ignore,
                std::vector<CallbackCall> &CallbackCalls)
      : Ignore(Ignore), CallbackCalls(CallbackCalls) {}

protected:
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(CompilerInstance &CI, StringRef InFile) override {
    return llvm::make_unique<PPTraceConsumer>(Ignore, CallbackCalls,
                                              CI.getPreprocessor());
  }

private:
  SmallSet<std::string, 4> &Ignore;
  std::vector<CallbackCall> &CallbackCalls;
};

class PPTraceFrontendActionFactory : public FrontendActionFactory {
public:
  PPTraceFrontendActionFactory(SmallSet<std::string, 4> &Ignore,
                               std::vector<CallbackCall> &CallbackCalls)
      : Ignore(Ignore), CallbackCalls(CallbackCalls) {}

  PPTraceAction *create() override {
    return new PPTraceAction(Ignore, CallbackCalls);
  }

private:
  SmallSet<std::string, 4> &Ignore;
  std::vector<CallbackCall> &CallbackCalls;
};
} // namespace

// Output the trace given its data structure and a stream.
static int outputPPTrace(std::vector<CallbackCall> &CallbackCalls,
                         llvm::raw_ostream &OS) {
  // Mark start of document.
  OS << "---\n";

  for (std::vector<CallbackCall>::const_iterator I = CallbackCalls.begin(),
                                                 E = CallbackCalls.end();
       I != E; ++I) {
    const CallbackCall &Callback = *I;
    OS << "- Callback: " << Callback.Name << "\n";

    for (auto AI = Callback.Arguments.begin(), AE = Callback.Arguments.end();
         AI != AE; ++AI) {
      const Argument &Arg = *AI;
      OS << "  " << Arg.Name << ": " << Arg.Value << "\n";
    }
  }

  // Mark end of document.
  OS << "...\n";

  return 0;
}

// Program entry point.
int main(int Argc, const char **Argv) {

  // Parse command line.
  cl::ParseCommandLineOptions(Argc, Argv, "pp-trace.\n");

  // Parse the IgnoreCallbacks list into strings.
  SmallVector<StringRef, 32> IgnoreCallbacksStrings;
  StringRef(IgnoreCallbacks).split(IgnoreCallbacksStrings, ",",
                                   /*MaxSplit=*/ -1, /*KeepEmpty=*/false);
  SmallSet<std::string, 4> Ignore;
  for (SmallVector<StringRef, 32>::iterator I = IgnoreCallbacksStrings.begin(),
                                            E = IgnoreCallbacksStrings.end();
       I != E; ++I)
    Ignore.insert(*I);

  // Create the compilation database.
  SmallString<256> PathBuf;
  sys::fs::current_path(PathBuf);
  std::unique_ptr<CompilationDatabase> Compilations;
  Compilations.reset(
      new FixedCompilationDatabase(Twine(PathBuf), CC1Arguments));

  // Store the callback trace information here.
  std::vector<CallbackCall> CallbackCalls;

  // Create the tool and run the compilation.
  ClangTool Tool(*Compilations, SourcePaths);
  PPTraceFrontendActionFactory Factory(Ignore, CallbackCalls);
  int HadErrors = Tool.run(&Factory);

  // If we had errors, exit early.
  if (HadErrors)
    return HadErrors;

  // Do the output.
  if (!OutputFileName.size()) {
    HadErrors = outputPPTrace(CallbackCalls, llvm::outs());
  } else {
    // Set up output file.
    std::error_code EC;
    llvm::ToolOutputFile Out(OutputFileName, EC, llvm::sys::fs::F_Text);
    if (EC) {
      llvm::errs() << "pp-trace: error creating " << OutputFileName << ":"
                   << EC.message() << "\n";
      return 1;
    }

    HadErrors = outputPPTrace(CallbackCalls, Out.os());

    // Tell ToolOutputFile that we want to keep the file.
    if (HadErrors == 0)
      Out.keep();
  }

  return HadErrors;
}
