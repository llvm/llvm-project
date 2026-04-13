//===- SSAFLinker.cpp - SSAF Linker ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements the SSAF entity linker tool that performs entity
//  linking across multiple TU summaries using the EntityLinker framework.
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysisFramework/Core/EntityLinker/EntityLinker.h"
#include "clang/ScalableStaticAnalysisFramework/Core/EntityLinker/TUSummaryEncoding.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/BuildNamespace.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Support/ErrorBuilder.h"
#include "clang/ScalableStaticAnalysisFramework/SSAFForceLinker.h" // IWYU pragma: keep
#include "clang/ScalableStaticAnalysisFramework/Tool/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <string>

using namespace llvm;
using namespace clang::ssaf;

namespace path = llvm::sys::path;

namespace {

//===----------------------------------------------------------------------===//
// Command-Line Options
//===----------------------------------------------------------------------===//

cl::OptionCategory SsafLinkerCategory("clang-ssaf-linker options");

cl::list<std::string> InputPaths(cl::Positional, cl::desc("<input files>"),
                                 cl::OneOrMore, cl::cat(SsafLinkerCategory));

cl::opt<std::string> OutputPath("o", cl::desc("Output file path"),
                                cl::value_desc("path"), cl::Required,
                                cl::cat(SsafLinkerCategory));

cl::opt<bool> Verbose("verbose", cl::desc("Enable verbose output"),
                      cl::init(false), cl::cat(SsafLinkerCategory));

cl::opt<bool> Time("time", cl::desc("Enable timing"), cl::init(false),
                   cl::cat(SsafLinkerCategory));

//===----------------------------------------------------------------------===//
// Error Messages
//===----------------------------------------------------------------------===//

namespace LocalErrorMessages {

constexpr const char *LinkingSummary = "Linking summary '{0}'";

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

struct LinkerInput {
  std::vector<FormatFile> InputFiles;
  FormatFile OutputFile;
  std::string LinkUnitName;
};

//===----------------------------------------------------------------------===//
// Pipeline
//===----------------------------------------------------------------------===//

LinkerInput validate(llvm::TimerGroup &TG) {
  llvm::Timer TValidate("validate", "Validate Input", TG);
  LinkerInput LI;

  {
    llvm::TimeRegion _(Time ? &TValidate : nullptr);

    LI.OutputFile = FormatFile::fromOutputPath(OutputPath);
    LI.LinkUnitName = path::stem(LI.OutputFile.Path).str();
  }

  info(2, "Validated output summary path '{0}'.", LI.OutputFile.Path);

  {
    llvm::TimeRegion _(Time ? &TValidate : nullptr);
    for (const auto &InputPath : InputPaths) {
      LI.InputFiles.push_back(FormatFile::fromInputPath(InputPath));
    }
  }

  info(2, "Validated {0} input summary paths.", LI.InputFiles.size());

  return LI;
}

void link(const LinkerInput &LI, llvm::TimerGroup &TG) {
  info(2, "Constructing linker.");

  EntityLinker EL(NestedBuildNamespace(
      BuildNamespace(BuildNamespaceKind::LinkUnit, LI.LinkUnitName)));

  llvm::Timer TRead("read", "Read Summaries", TG);
  llvm::Timer TLink("link", "Link Summaries", TG);
  llvm::Timer TWrite("write", "Write Summary", TG);

  info(2, "Linking summaries.");

  for (auto [Index, InputFile] : llvm::enumerate(LI.InputFiles)) {
    std::unique_ptr<TUSummaryEncoding> Summary;

    {
      info(3, "[{0}/{1}] Reading '{2}'.", (Index + 1), LI.InputFiles.size(),
           InputFile.Path);

      llvm::TimeRegion _(Time ? &TRead : nullptr);

      auto ExpectedSummaryEncoding =
          InputFile.Format->readTUSummaryEncoding(InputFile.Path);
      if (!ExpectedSummaryEncoding) {
        fail(ExpectedSummaryEncoding.takeError());
      }

      Summary = std::make_unique<TUSummaryEncoding>(
          std::move(*ExpectedSummaryEncoding));
    }

    {
      info(3, "[{0}/{1}] Linking '{2}'.", (Index + 1), LI.InputFiles.size(),
           InputFile.Path);

      llvm::TimeRegion _(Time ? &TLink : nullptr);

      if (auto Err = EL.link(std::move(Summary))) {
        fail(ErrorBuilder::wrap(std::move(Err))
                 .context(LocalErrorMessages::LinkingSummary, InputFile.Path)
                 .build());
      }
    }
  }

  {
    info(2, "Writing output summary to '{0}'.", LI.OutputFile.Path);

    llvm::TimeRegion _(Time ? &TWrite : nullptr);

    auto Output = std::move(EL).getOutput();
    if (auto Err = LI.OutputFile.Format->writeLUSummaryEncoding(
            Output, LI.OutputFile.Path)) {
      fail(std::move(Err));
    }
  }
}

} // namespace

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

int main(int argc, const char **argv) {
  llvm::StringRef ToolHeading = "SSAF Linker";

  InitLLVM X(argc, argv);
  initTool(argc, argv, "0.1", SsafLinkerCategory, ToolHeading);

  llvm::TimerGroup LinkerTimers(getToolName(), ToolHeading);
  LinkerInput LI;

  {
    info(0, "Linking started.");

    {
      info(1, "Validating input.");
      LI = validate(LinkerTimers);
    }

    {
      info(1, "Linking input.");
      link(LI, LinkerTimers);
    }

    info(0, "Linking finished.");
  }

  return 0;
}
