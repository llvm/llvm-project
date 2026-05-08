//===- SSAFAnalyzer.cpp - SSAF Analyzer Tool ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements the SSAF analyzer tool that runs whole-program
//  analyses over an LUSummary and writes the resulting WPASuite to an
//  output file.
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysisFramework/Core/EntityLinker/LUSummary.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/AnalysisDriver.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/AnalysisName.h"
#include "clang/ScalableStaticAnalysisFramework/SSAFForceLinker.h" // IWYU pragma: keep
#include "clang/ScalableStaticAnalysisFramework/Tool/Utils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include <memory>
#include <string>

using namespace llvm;
using namespace clang::ssaf;

namespace {

//===----------------------------------------------------------------------===//
// Command-Line Options
//===----------------------------------------------------------------------===//

cl::OptionCategory SsafAnalyzerCategory("clang-ssaf-analyzer options");

cl::opt<std::string> InputPath(cl::Positional, cl::desc("<input file>"),
                               cl::Required, cl::cat(SsafAnalyzerCategory));

cl::opt<std::string> OutputPath("o", cl::desc("Output file path"),
                                cl::value_desc("path"), cl::Required,
                                cl::cat(SsafAnalyzerCategory));

cl::list<std::string> AnalysisNames("a", cl::desc("Analysis name to run"),
                                    cl::value_desc("name"),
                                    cl::cat(SsafAnalyzerCategory));

cl::alias AnalysisNamesAlias("analysis", cl::aliasopt(AnalysisNames),
                             cl::desc("Alias for -a"));

cl::list<std::string> LoadPlugins("load",
                                  cl::desc("Load a plugin shared library"),
                                  cl::value_desc("path"),
                                  cl::cat(SsafAnalyzerCategory));

cl::alias LoadPluginsAlias("l", cl::aliasopt(LoadPlugins),
                           cl::desc("Alias for --load"));

//===----------------------------------------------------------------------===//
// Input Validation
//===----------------------------------------------------------------------===//

struct AnalyzerInput {
  FormatFile InputFile;
  FormatFile OutputFile;
  llvm::SmallVector<AnalysisName> Names;
};

AnalyzerInput validate() {
  AnalyzerInput AI;

  // Validate the input path.
  AI.InputFile = FormatFile::fromInputPath(InputPath);

  // Validate the output path.
  AI.OutputFile = FormatFile::fromOutputPath(OutputPath);

  // Build and validate analysis names.
  for (const auto &Name : AnalysisNames) {
    if (Name.empty()) {
      fail("analysis name must not be empty");
    }
    AI.Names.push_back(AnalysisName(Name));
  }

  return AI;
}

//===----------------------------------------------------------------------===//
// Analysis Pipeline
//===----------------------------------------------------------------------===//

void analyze(const AnalyzerInput &AI) {
  // Read the LUSummary.
  auto ExpectedLU = AI.InputFile.Format->readLUSummary(AI.InputFile.Path);
  if (!ExpectedLU) {
    fail(ExpectedLU.takeError());
  }

  // Run analyses. If specific names were given, run only those;
  // otherwise run all registered analyses.
  AnalysisDriver Driver(std::make_unique<LUSummary>(std::move(*ExpectedLU)));
  auto ExpectedSuite =
      AI.Names.empty() ? std::move(Driver).run() : Driver.run(AI.Names);
  if (!ExpectedSuite) {
    fail(ExpectedSuite.takeError());
  }

  // Write the WPASuite.
  if (auto Err = AI.OutputFile.Format->writeWPASuite(*ExpectedSuite,
                                                     AI.OutputFile.Path)) {
    fail(std::move(Err));
  }
}

} // namespace

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

int main(int argc, const char **argv) {
  llvm::StringRef ToolHeading = "SSAF Analyzer";

  InitLLVM X(argc, argv);
  initTool(argc, argv, "0.1", SsafAnalyzerCategory, ToolHeading);

  loadPlugins(LoadPlugins);

  AnalyzerInput AI = validate();
  analyze(AI);

  return 0;
}
