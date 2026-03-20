//===- SSAFAnalyzer.cpp - SSAF Analyzer Tool ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements the SSAF analyzer tool that runs whole-program analyses
//  over an LUSummary and writes the resulting WPASuite.
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysisFramework/Core/EntityLinker/LUSummary.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Serialization/SerializationFormat.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/AnalysisDriver.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/AnalysisName.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/WPASuite.h"
#include "clang/ScalableStaticAnalysisFramework/SSAFForceLinker.h" // IWYU pragma: keep
#include "clang/ScalableStaticAnalysisFramework/Tool/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <string>
#include <system_error>
#include <vector>

using namespace llvm;
using namespace clang::ssaf;

namespace fs = llvm::sys::fs;
namespace path = llvm::sys::path;

namespace {

//===----------------------------------------------------------------------===//
// Command-Line Options
//===----------------------------------------------------------------------===//

cl::OptionCategory SsafAnalyzerCategory("clang-ssaf-analyzer options");

cl::list<std::string> LoadPlugins("load",
                                  cl::desc("Load a plugin shared library"),
                                  cl::value_desc("path"),
                                  cl::cat(SsafAnalyzerCategory));

cl::opt<std::string> LUSummaryPath(cl::Positional, cl::desc("<lu-summary>"),
                                   cl::cat(SsafAnalyzerCategory));

cl::opt<std::string> OutputPath("o", cl::desc("Output WPASuite path"),
                                cl::value_desc("path"), cl::Required,
                                cl::cat(SsafAnalyzerCategory));

cl::list<std::string>
    AnalysisNameStrs("analysis",
                     cl::desc("Name of an analysis to run (may be repeated; "
                              "default: run all registered analyses)"),
                     cl::value_desc("name"), cl::cat(SsafAnalyzerCategory));

cl::opt<bool> Verbose("verbose", cl::desc("Enable verbose output"),
                      cl::init(false), cl::cat(SsafAnalyzerCategory));

cl::opt<bool> Time("time", cl::desc("Enable timing"), cl::init(false),
                   cl::cat(SsafAnalyzerCategory));

//===----------------------------------------------------------------------===//
// Error Messages
//===----------------------------------------------------------------------===//

namespace LocalErrorMessages {

constexpr const char *RunningAnalysis = "Running analysis '{0}'";

} // namespace LocalErrorMessages

//===----------------------------------------------------------------------===//
// Diagnostic Utilities
//===----------------------------------------------------------------------===//

constexpr unsigned IndentationWidth = 2;

template <typename... Ts>
void info(unsigned IndentationLevel, const char *Fmt, Ts &&...Args) {
  if (Verbose) {
    llvm::WithColor::note()
        << std::string(IndentationLevel * IndentationWidth, ' ') << "- "
        << llvm::formatv(Fmt, std::forward<Ts>(Args)...) << "\n";
  }
}

//===----------------------------------------------------------------------===//
// Data Structures
//===----------------------------------------------------------------------===//

struct AnalyzerInput {
  SummaryFile LUSummaryFile;
  SummaryFile WPAOutputFile;
  std::vector<AnalysisName> Names; // Empty means run all registered analyses.
};

//===----------------------------------------------------------------------===//
// Pipeline
//===----------------------------------------------------------------------===//

AnalyzerInput validate(llvm::TimerGroup &TG) {
  llvm::Timer TValidate("validate", "Validate Input", TG);
  llvm::TimeRegion _(Time ? &TValidate : nullptr);

  AnalyzerInput AI;

  // Validate the LUSummary input path.
  {
    if (LUSummaryPath.empty()) {
      fail("no LUSummary file specified");
    }

    llvm::SmallString<256> RealInputPath;
    if (std::error_code EC =
            fs::real_path(LUSummaryPath, RealInputPath, /*expand_tilde=*/true))
      fail(ErrorMessages::CannotValidateSummary, LUSummaryPath, EC.message());

    AI.LUSummaryFile = SummaryFile::fromPath(RealInputPath);
  }

  info(2, "Validated LUSummary input path '{0}'.", AI.LUSummaryFile.Path);

  // Validate the WPASuite output path.
  {
    llvm::StringRef ParentDir = path::parent_path(OutputPath);
    llvm::StringRef DirToCheck = ParentDir.empty() ? "." : ParentDir;

    if (!fs::exists(DirToCheck)) {
      fail(ErrorMessages::CannotValidateSummary, OutputPath,
           ErrorMessages::OutputDirectoryMissing);
    }

    // The output file does not exist yet, so real_path cannot be called on it
    // directly. Resolve the parent directory first, then append the filename.
    llvm::SmallString<256> RealParentDir;
    if (std::error_code EC = fs::real_path(DirToCheck, RealParentDir))
      fail(ErrorMessages::CannotValidateSummary, OutputPath, EC.message());

    llvm::SmallString<256> RealOutputPath = RealParentDir;
    path::append(RealOutputPath, path::filename(OutputPath));

    AI.WPAOutputFile = SummaryFile::fromPath(RealOutputPath);
  }

  info(2, "Validated WPASuite output path '{0}'.", AI.WPAOutputFile.Path);

  // Convert analysis name strings to AnalysisName objects.
  for (const auto &Name : AnalysisNameStrs)
    AI.Names.emplace_back(Name);

  if (AI.Names.empty())
    info(2, "No analyses specified; all registered analyses will be run.");
  else
    info(2, "Running {0} named {1}.", AI.Names.size(),
         AI.Names.size() == 1 ? "analysis" : "analyses");

  return AI;
}

void analyze(const AnalyzerInput &AI, llvm::TimerGroup &TG) {
  llvm::Timer TRead("read", "Read LUSummary", TG);
  llvm::Timer TRun("run", "Run Analyses", TG);
  llvm::Timer TWrite("write", "Write WPASuite", TG);

  // Read the LUSummary.
  std::unique_ptr<LUSummary> LU;
  {
    info(2, "Reading LUSummary from '{0}'.", AI.LUSummaryFile.Path);
    llvm::TimeRegion _(Time ? &TRead : nullptr);

    auto ExpectedLU =
        AI.LUSummaryFile.Format->readLUSummary(AI.LUSummaryFile.Path);
    if (!ExpectedLU)
      fail(ExpectedLU.takeError());

    LU = std::make_unique<LUSummary>(std::move(*ExpectedLU));
  }

  // Run analyses.
  WPASuite Suite;
  {
    info(2, "Running analyses.");
    llvm::TimeRegion _(Time ? &TRun : nullptr);

    AnalysisDriver Driver(std::move(LU));

    llvm::Expected<WPASuite> ExpectedSuite =
        AI.Names.empty() ? std::move(Driver).run() : Driver.run(AI.Names);
    if (!ExpectedSuite)
      fail(ExpectedSuite.takeError());

    Suite = std::move(*ExpectedSuite);
  }

  // Write the WPASuite.
  {
    info(2, "Writing WPASuite to '{0}'.", AI.WPAOutputFile.Path);
    llvm::TimeRegion _(Time ? &TWrite : nullptr);

    if (auto Err = AI.WPAOutputFile.Format->writeWPASuite(
            Suite, AI.WPAOutputFile.Path))
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

  llvm::TimerGroup AnalyzerTimers(getToolName(), ToolHeading);

  {
    info(0, "Analysis started.");

    AnalyzerInput AI;

    {
      info(1, "Validating input.");
      AI = validate(AnalyzerTimers);
    }

    {
      info(1, "Running analyses.");
      analyze(AI, AnalyzerTimers);
    }

    info(0, "Analysis finished.");
  }

  return 0;
}
