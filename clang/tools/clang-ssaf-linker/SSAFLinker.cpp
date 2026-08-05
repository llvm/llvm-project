//===- SSAFLinker.cpp - SSAF Linker ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements the SSAF entity linker tool. Its default behavior
//  is to link N TU summaries into one LU summary via the EntityLinker
//  framework. It also provides the `static-library` subcommand for
//  bundling TU summaries into a StaticLibrary.
//
//===----------------------------------------------------------------------===//

#include "StaticLibraryCreateCLI.h"

#include "clang/ScalableStaticAnalysis/Core/EntityLinker/EntityLinker.h"
#include "clang/ScalableStaticAnalysis/Core/EntityLinker/TUSummaryEncoding.h"
#include "clang/ScalableStaticAnalysis/Core/Model/BuildNamespace.h"
#include "clang/ScalableStaticAnalysis/Core/Support/ErrorBuilder.h"
#include "clang/ScalableStaticAnalysis/SSAFForceLinker.h" // IWYU pragma: keep
#include "clang/ScalableStaticAnalysis/Tool/Utils.h"
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

// The `static-library` subcommand groups all StaticLibrary operations.
cl::SubCommand StaticLibraryCmd("static-library",
                                "Operations on StaticLibraries");

// Top-level (default) `link` action positionals.
cl::list<std::string> InputPaths(cl::Positional, cl::desc("<input files>"),
                                 cl::OneOrMore, cl::cat(SsafLinkerCategory));

cl::opt<std::string> OutputPath("o", cl::desc("Output file path"),
                                cl::value_desc("path"), cl::Required,
                                cl::cat(SsafLinkerCategory));

// --verbose and --time apply to every subcommand.
cl::opt<bool> Verbose("verbose", cl::desc("Enable verbose output"),
                      cl::init(false), cl::cat(SsafLinkerCategory),
                      cl::sub(cl::SubCommand::getTopLevel()),
                      cl::sub(StaticLibraryCmd));

cl::opt<bool> Time("time", cl::desc("Enable timing"), cl::init(false),
                   cl::cat(SsafLinkerCategory),
                   cl::sub(cl::SubCommand::getTopLevel()),
                   cl::sub(StaticLibraryCmd));

// The `static-library` subcommand's verb positional. Declared BEFORE
// StaticLibraryInputs so cl-lib binds argv[0] under the subcommand to the
// verb rather than to the greedy input list.
cl::opt<std::string> StaticLibraryVerb(cl::Positional, cl::Required,
                                       cl::sub(StaticLibraryCmd),
                                       cl::desc("<verb>"),
                                       cl::value_desc("create"),
                                       cl::cat(SsafLinkerCategory));

// The `static-library` subcommand's action-specific positional input
// list. Currently consumed by `static-library create`; if future verbs
// need different input shapes they'll declare their own positionals.
cl::list<std::string> StaticLibraryInputs(cl::Positional,
                                          cl::sub(StaticLibraryCmd),
                                          cl::desc("<TU summary files>"),
                                          cl::cat(SsafLinkerCategory));

cl::opt<std::string> StaticLibraryOutput("o", cl::Required,
                                         cl::sub(StaticLibraryCmd),
                                         cl::desc("Output file path"),
                                         cl::value_desc("path"),
                                         cl::cat(SsafLinkerCategory));

cl::opt<std::string> StaticLibraryNamespace(
    "namespace", cl::sub(StaticLibraryCmd),
    cl::desc("Namespace name for the StaticLibrary (defaults to output "
             "file stem)"),
    cl::value_desc("name"), cl::cat(SsafLinkerCategory));

cl::opt<std::string> StaticLibraryTriple(
    "target-triple", cl::sub(StaticLibraryCmd),
    cl::desc("Target triple (defaults to inputs' triple; must match all "
             "inputs when set)"),
    cl::value_desc("triple"), cl::cat(SsafLinkerCategory));

//===----------------------------------------------------------------------===//
// StaticLibrary Verbs
//===----------------------------------------------------------------------===//

// Verb strings for the `static-library` subcommand. Kept in sync with
// UnknownStaticLibraryVerb below.
constexpr const char *StaticLibraryCreateVerb = "create";

//===----------------------------------------------------------------------===//
// Error Messages
//===----------------------------------------------------------------------===//

namespace LocalErrorMessages {

constexpr const char *LinkingSummary = "Linking summary '{0}'";

constexpr const char *UnknownStaticLibraryVerb =
    "unknown static-library verb '{0}': expected 'create'";

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
// link action
//===----------------------------------------------------------------------===//

struct LinkerInput {
  std::vector<FormatFile> InputFiles;
  FormatFile OutputFile;
  std::string LinkUnitName;
};

LinkerInput validateLinkInput(llvm::TimerGroup &TG) {
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

void runLink(llvm::TimerGroup &TG) {
  info(0, "Linking started.");

  LinkerInput LI;
  {
    info(1, "Validating input.");
    LI = validateLinkInput(TG);
  }

  info(1, "Linking input.");
  info(2, "Constructing linker.");

  // TODO: The linker currently uses a hardcoded target triple. Architecture
  // tracking in the linker will be handled properly in a separate PR.
  EntityLinker EL(llvm::Triple("arm64-apple-macosx"),
                  NestedBuildNamespace(BuildNamespace(
                      BuildNamespaceKind::LinkUnit, LI.LinkUnitName)));

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

    auto Output = std::move(EL).takeOutput();
    if (auto Err = LI.OutputFile.Format->writeLUSummaryEncoding(
            Output, LI.OutputFile.Path)) {
      fail(std::move(Err));
    }
  }

  info(0, "Linking finished.");
}

//===----------------------------------------------------------------------===//
// static-library subcommand dispatch
//===----------------------------------------------------------------------===//

void runStaticLibrary(llvm::TimerGroup &TG) {
  if (StaticLibraryVerb == StaticLibraryCreateVerb) {
    StaticLibraryCreateCLI::Config Cfg;
    Cfg.InputPaths = StaticLibraryInputs;
    Cfg.OutputPath = StaticLibraryOutput;
    Cfg.Namespace = StaticLibraryNamespace;
    Cfg.TargetTriple = StaticLibraryTriple;
    Cfg.Verbose = Verbose;
    Cfg.Time = Time;

    StaticLibraryCreateCLI SLC;
    SLC.run(TG, Cfg);
    return;
  }
  fail(LocalErrorMessages::UnknownStaticLibraryVerb,
       StaticLibraryVerb.getValue());
}

} // namespace

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

int main(int argc, const char **argv) {
  llvm::StringRef ToolHeading = "SSAF Linker";

  InitLLVM X(argc, argv);
  initTool(argc, argv, "0.1", SsafLinkerCategory, ToolHeading);

  llvm::TimerGroup Timers(getToolName(), ToolHeading);

  if (StaticLibraryCmd) {
    runStaticLibrary(Timers);
  } else {
    // Default (no subcommand): run the linker pipeline.
    runLink(Timers);
  }

  return 0;
}
