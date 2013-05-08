//===-- cpp11-migrate/Cpp11Migrate.cpp - Main file C++11 migration tool ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file implements the C++11 feature migration tool main function
/// and transformation framework.
///
/// See user documentation for usage instructions.
///
//===----------------------------------------------------------------------===//

#include "Core/Transforms.h"
#include "Core/Transform.h"
#include "LoopConvert/LoopConvert.h"
#include "UseNullptr/UseNullptr.h"
#include "UseAuto/UseAuto.h"
#include "AddOverride/AddOverride.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/Signals.h"

namespace cl = llvm::cl;
using namespace clang::tooling;

static cl::opt<RiskLevel> MaxRiskLevel(
    "risk", cl::desc("Select a maximum risk level:"),
    cl::values(clEnumValN(RL_Safe, "safe", "Only safe transformations"),
               clEnumValN(RL_Reasonable, "reasonable",
                          "Enable transformations that might change "
                          "semantics (default)"),
               clEnumValN(RL_Risky, "risky",
                          "Enable transformations that are likely to "
                          "change semantics"),
               clEnumValEnd),
    cl::init(RL_Reasonable));

static cl::opt<bool> FinalSyntaxCheck(
    "final-syntax-check",
    cl::desc("Check for correct syntax after applying transformations"),
    cl::init(false));

static cl::opt<bool>
SummaryMode("summary", cl::desc("Print transform summary"),
            cl::init(false));

// TODO: Remove cl::Hidden when functionality for acknowledging include/exclude
// options are implemented in the tool.
static cl::opt<std::string>
IncludePaths("include", cl::Hidden,
             cl::desc("Comma seperated list of paths to consider to be "
                      "transformed"));
static cl::opt<std::string>
ExcludePaths("exclude", cl::Hidden,
             cl::desc("Comma seperated list of paths that can not "
                      "be transformed"));
static cl::opt<std::string>
IncludeFromFile("include-from", cl::Hidden, cl::value_desc("filename"),
                cl::desc("File containing a list of paths to consider to "
                         "be transformed"));
static cl::opt<std::string>
ExcludeFromFile("exclude-from", cl::Hidden, cl::value_desc("filename"),
                cl::desc("File containing a list of paths that can not be "
                         "transforms"));

class EndSyntaxArgumentsAdjuster : public ArgumentsAdjuster {
  CommandLineArguments Adjust(const CommandLineArguments &Args) {
    CommandLineArguments AdjustedArgs = Args;
    AdjustedArgs.push_back("-fsyntax-only");
    AdjustedArgs.push_back("-std=c++11");
    return AdjustedArgs;
  }
};

int main(int argc, const char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal();
  Transforms TransformManager;

  TransformManager.registerTransform(
      "loop-convert", "Make use of range-based for loops where possible",
      &ConstructTransform<LoopConvertTransform>);
  TransformManager.registerTransform(
      "use-nullptr", "Make use of nullptr keyword where possible",
      &ConstructTransform<UseNullptrTransform>);
  TransformManager.registerTransform(
      "use-auto", "Use of 'auto' type specifier",
      &ConstructTransform<UseAutoTransform>);
  TransformManager.registerTransform(
      "add-override", "Make use of override specifier where possible",
      &ConstructTransform<AddOverrideTransform>);
  // Add more transform options here.

  // This causes options to be parsed.
  CommonOptionsParser OptionsParser(argc, argv);

  TransformManager.createSelectedTransforms();

  if (TransformManager.begin() == TransformManager.end()) {
    llvm::errs() << "No selected transforms\n";
    return 1;
  }

  FileContentsByPath FileStates1, FileStates2,
      *InputFileStates = &FileStates1, *OutputFileStates = &FileStates2;

  // Apply transforms.
  for (Transforms::const_iterator I = TransformManager.begin(),
                                  E = TransformManager.end();
       I != E; ++I) {
    if ((*I)->apply(*InputFileStates, MaxRiskLevel,
                    OptionsParser.getCompilations(),
                    OptionsParser.getSourcePathList(), *OutputFileStates) !=
        0) {
      // FIXME: Improve ClangTool to not abort if just one file fails.
      return 1;
    }
    if (SummaryMode) {
      llvm::outs() << "Transform: " << (*I)->getName()
                   << " - Accepted: "
                   << (*I)->getAcceptedChanges();
      if ((*I)->getChangesNotMade()) {
         llvm::outs() << " - Rejected: "
                      << (*I)->getRejectedChanges()
                      << " - Deferred: "
                      << (*I)->getDeferredChanges();
      }
      llvm::outs() << "\n";
    }
    std::swap(InputFileStates, OutputFileStates);
    OutputFileStates->clear();
  }

  // Final state of files is pointed at by InputFileStates.

  if (FinalSyntaxCheck) {
    ClangTool EndSyntaxTool(OptionsParser.getCompilations(),
                            OptionsParser.getSourcePathList());

    // Add c++11 support to clang.
    EndSyntaxTool.setArgumentsAdjuster(new EndSyntaxArgumentsAdjuster);

    for (FileContentsByPath::const_iterator I = InputFileStates->begin(),
                                            E = InputFileStates->end();
         I != E; ++I) {
      EndSyntaxTool.mapVirtualFile(I->first, I->second);
    }

    if (EndSyntaxTool.run(newFrontendActionFactory<clang::SyntaxOnlyAction>())
        != 0) {
      return 1;
    }
  }

  // Write results to file.
  for (FileContentsByPath::const_iterator I = InputFileStates->begin(),
                                          E = InputFileStates->end();
       I != E; ++I) {
    std::string ErrorInfo;
    llvm::raw_fd_ostream FileStream(I->first.c_str(), ErrorInfo,
                                    llvm::raw_fd_ostream::F_Binary);
    FileStream << I->second;
  }

  return 0;
}
