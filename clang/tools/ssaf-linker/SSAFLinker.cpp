//===--- tools/ssaf-linker/SSAFLinker.cpp - SSAF Linker -------------------===//
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

#include "clang/Analysis/Scalable/EntityLinker/EntityLinker.h"
#include "clang/Analysis/Scalable/EntityLinker/TUSummaryEncoding.h"
#include "clang/Analysis/Scalable/Model/BuildNamespace.h"
#include "clang/Analysis/Scalable/Serialization/JSONFormat.h"
#include "clang/Analysis/Scalable/Serialization/SerializationFormatRegistry.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <string>

using namespace llvm;
using namespace clang::ssaf;

namespace fs = llvm::sys::fs;
namespace path = llvm::sys::path;

namespace {

//===----------------------------------------------------------------------===//
// Command-Line Options
//===----------------------------------------------------------------------===//

cl::OptionCategory SsafLinkerCategory("ssaf-linker options");

cl::list<std::string> InputPaths(cl::Positional, cl::desc("<input files>"),
                                 cl::OneOrMore, cl::cat(SsafLinkerCategory));

cl::opt<std::string> OutputPath("o", cl::desc("Output summary path"),
                                cl::value_desc("path"), cl::Required,
                                cl::cat(SsafLinkerCategory));

cl::alias OutputPathLong("output", cl::aliasopt(OutputPath));

cl::opt<bool> Verbose("verbose", cl::desc("Enable verbose output"),
                      cl::init(false), cl::cat(SsafLinkerCategory));

cl::opt<bool> Time("time", cl::desc("Enable timing"), cl::init(false),
                   cl::cat(SsafLinkerCategory));

//===----------------------------------------------------------------------===//
// Error Messages
//===----------------------------------------------------------------------===//

constexpr const char *ErrorCannotValidateSummary =
    "failed to validate summary '{0}': {1}.";

constexpr const char *ErrorOutputDirectoryMissing =
    "Parent directory does not exist.";

constexpr const char *ErrorOutputDirectoryNotWritable =
    "Parent directory is not writable.";

constexpr const char *ErrorExtensionNotSupplied = "Extension not supplied.";

constexpr const char *ErrorNoFormatForExtension =
    "Format not registered for extension '{0}'.";

constexpr const char *ErrorLoadFailed =
    "failed to load input summary '{0}': {1}.";

constexpr const char *ErrorLinkFailed =
    "failed to link input summary '{0}': {1}.";

constexpr const char *ErrorWriteFailed =
    "failed to write output summary '{0}': {1}.";

//===----------------------------------------------------------------------===//
// Diagnostic Utilities
//===----------------------------------------------------------------------===//

constexpr unsigned IndentationWidth = 2;

static llvm::StringRef ToolName;

static void PrintVersion(llvm::raw_ostream &OS) { OS << "ssaf-linker 0.1\n"; }

template <typename... Ts>
[[noreturn]] void Fail(const char *Fmt, Ts &&...Args) {
  llvm::WithColor::error(llvm::errs(), ToolName)
      << llvm::formatv(Fmt, std::forward<Ts>(Args)...) << "\n";
  llvm::sys::Process::Exit(1);
}

template <typename... Ts>
void Info(unsigned IndentationLevel, const char *Fmt, Ts &&...Args) {
  if (Verbose) {
    llvm::WithColor::note()
        << std::string(IndentationLevel * IndentationWidth, ' ') << "- "
        << llvm::formatv(Fmt, std::forward<Ts>(Args)...) << "\n";
  }
}

//===----------------------------------------------------------------------===//
// Format Registry
//===----------------------------------------------------------------------===//

SerializationFormat *GetFormatForExtension(llvm::StringRef Extension) {
  static llvm::SmallVector<
      std::pair<std::string, std::unique_ptr<SerializationFormat>>, 4>
      ExtensionFormatList;

  // Most recently used format is most likely to be reused again.
  auto ReversedList = llvm::reverse(ExtensionFormatList);
  auto It = llvm::find_if(ReversedList, [&](const auto &Entry) {
    return Entry.first == Extension;
  });
  if (It != ReversedList.end()) {
    return It->second.get();
  }

  // SerializationFormats are uppercase while file extensions are lowercase.
  std::string CapitalizedExtension = Extension.upper();
  auto Format = makeFormat(CapitalizedExtension);
  SerializationFormat *Result = Format.get();

  if (Result) {
    ExtensionFormatList.emplace_back(Extension, std::move(Format));
  }

  return Result;
}

//===----------------------------------------------------------------------===//
// Data Structures
//===----------------------------------------------------------------------===//

struct SummaryFile {
  std::string Path;
  SerializationFormat *Format = nullptr;

  static SummaryFile FromPath(llvm::StringRef Path) {
    llvm::StringRef Extension = path::extension(Path);
    if (Extension.empty()) {
      Fail(ErrorCannotValidateSummary, Path, ErrorExtensionNotSupplied);
    }
    Extension = Extension.drop_front();
    SerializationFormat *Format = GetFormatForExtension(Extension);
    if (!Format) {
      std::string Suffix = llvm::formatv(ErrorNoFormatForExtension, Extension);
      Fail(ErrorCannotValidateSummary, Path, Suffix);
    }
    return {Path.str(), Format};
  }
};

struct LinkerInput {
  std::vector<SummaryFile> InputFiles;
  SummaryFile OutputFile;
  std::string LinkUnitName;
};

//===----------------------------------------------------------------------===//
// Pipeline
//===----------------------------------------------------------------------===//

LinkerInput Validate(llvm::TimerGroup &TG) {
  llvm::Timer TValidate("validate", "Validate Input", TG);
  LinkerInput LI;

  {
    llvm::TimeRegion _(Time ? &TValidate : nullptr);
    llvm::StringRef ParentDir = path::parent_path(OutputPath);
    llvm::StringRef DirToCheck = ParentDir.empty() ? "." : ParentDir;

    if (!fs::exists(DirToCheck)) {
      Fail(ErrorCannotValidateSummary, OutputPath, ErrorOutputDirectoryMissing);
    }

    if (fs::access(DirToCheck, fs::AccessMode::Write)) {
      Fail(ErrorCannotValidateSummary, OutputPath,
           ErrorOutputDirectoryNotWritable);
    }

    LI.OutputFile = SummaryFile::FromPath(OutputPath);
    LI.LinkUnitName = path::stem(LI.OutputFile.Path).str();
  }

  Info(2, "Validated output summary path '{0}'.", LI.OutputFile.Path);

  {
    llvm::TimeRegion _(Time ? &TValidate : nullptr);
    for (const auto &InputPath : InputPaths) {
      llvm::SmallString<256> RealPath;
      std::error_code EC = fs::real_path(InputPath, RealPath, true);
      if (EC) {
        Fail(ErrorCannotValidateSummary, InputPath, EC.message());
      }
      LI.InputFiles.push_back(SummaryFile::FromPath(RealPath));
    }
  }

  Info(2, "Validated {0} input summary paths.", LI.InputFiles.size());

  return LI;
}

void Link(const LinkerInput &LI, llvm::TimerGroup &TG) {
  Info(2, "Constructing linker.");

  EntityLinker EL(NestedBuildNamespace(
      BuildNamespace(BuildNamespaceKind::LinkUnit, LI.LinkUnitName)));

  llvm::Timer TRead("read", "Read Summaries", TG);
  llvm::Timer TLink("link", "Link Summaries", TG);
  llvm::Timer TWrite("write", "Write Summary", TG);

  Info(2, "Linking summaries.");

  for (auto [Index, InputFile] : llvm::enumerate(LI.InputFiles)) {
    std::unique_ptr<TUSummaryEncoding> Summary;

    {
      Info(3, "[{0}/{1}] Reading '{2}'.", (Index + 1), LI.InputFiles.size(),
           InputFile.Path);

      llvm::TimeRegion _(Time ? &TRead : nullptr);

      auto ExpectedSummaryEncoding =
          InputFile.Format->readTUSummaryEncoding(InputFile.Path);
      if (!ExpectedSummaryEncoding) {
        Fail(ErrorLoadFailed, InputFile.Path,
             toString(ExpectedSummaryEncoding.takeError()));
      }

      Summary = std::make_unique<TUSummaryEncoding>(
          std::move(*ExpectedSummaryEncoding));
    }

    {
      Info(3, "[{0}/{1}] Linking '{2}'.", (Index + 1), LI.InputFiles.size(),
           InputFile.Path);

      llvm::TimeRegion _(Time ? &TLink : nullptr);

      if (auto Err = EL.link(std::move(Summary))) {
        Fail(ErrorLinkFailed, InputFile.Path, toString(std::move(Err)));
      }
    }
  }

  {
    Info(2, "Writing output summary to '{0}'.", LI.OutputFile.Path);

    llvm::TimeRegion _(Time ? &TWrite : nullptr);

    auto Output = std::move(EL).getOutput();
    if (auto Err = LI.OutputFile.Format->writeLUSummaryEncoding(
            Output, LI.OutputFile.Path)) {
      Fail(ErrorWriteFailed, LI.OutputFile.Path, toString(std::move(Err)));
    }
  }
}

} // namespace

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

int main(int argc, const char **argv) {
  InitLLVM X(argc, argv);
  ToolName = llvm::sys::path::filename(argv[0]);
  initializeJSONFormat();

  cl::HideUnrelatedOptions(SsafLinkerCategory);
  cl::SetVersionPrinter(PrintVersion);
  cl::ParseCommandLineOptions(argc, argv, "SSAF Linker\n");

  llvm::TimerGroup LinkerTimers("ssaf-linker", "SSAF Linker");
  LinkerInput LI;

  {
    Info(0, "Linking started.");

    {
      Info(1, "Validating input.");
      LI = Validate(LinkerTimers);
    }

    {
      Info(1, "Linking input.");
      Link(LI, LinkerTimers);
    }

    Info(0, "Linking finished.");
  }

  return 0;
}
