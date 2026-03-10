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
#include "clang/Analysis/Scalable/Support/ErrorBuilder.h"
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
#include <system_error>

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

cl::opt<bool> Verbose("verbose", cl::desc("Enable verbose output"),
                      cl::init(false), cl::cat(SsafLinkerCategory));

cl::opt<bool> Time("time", cl::desc("Enable timing"), cl::init(false),
                   cl::cat(SsafLinkerCategory));

//===----------------------------------------------------------------------===//
// Error Messages
//===----------------------------------------------------------------------===//

namespace ErrorMessages {

constexpr const char *CannotValidateSummary =
    "failed to validate summary '{0}': {1}";

constexpr const char *OutputDirectoryMissing =
    "Parent directory does not exist";

constexpr const char *OutputDirectoryNotWritable =
    "Parent directory is not writable";

constexpr const char *ExtensionNotSupplied = "Extension not supplied";

constexpr const char *NoFormatForExtension =
    "Format not registered for extension '{0}'";

constexpr const char *LinkingSummary = "Linking summary '{0}'";

} // namespace ErrorMessages

//===----------------------------------------------------------------------===//
// Diagnostic Utilities
//===----------------------------------------------------------------------===//

constexpr unsigned IndentationWidth = 2;

llvm::StringRef ToolName;

template <typename... Ts> [[noreturn]] void fail(const char *Msg) {
  llvm::WithColor::error(llvm::errs(), ToolName) << Msg << "\n";
  llvm::sys::Process::Exit(1);
}

template <typename... Ts>
[[noreturn]] void fail(const char *Fmt, Ts &&...Args) {
  std::string Message = llvm::formatv(Fmt, std::forward<Ts>(Args)...);
  fail(Message.data());
}

template <typename... Ts> [[noreturn]] void fail(llvm::Error Err) {
  fail(toString(std::move(Err)).data());
}

template <typename... Ts>
void info(unsigned IndentationLevel, const char *Fmt, Ts &&...Args) {
  if (Verbose) {
    llvm::WithColor::note()
        << std::string(IndentationLevel * IndentationWidth, ' ') << "- "
        << llvm::formatv(Fmt, std::forward<Ts>(Args)...) << "\n";
  }
}

//===----------------------------------------------------------------------===//
// Format Registry
//===----------------------------------------------------------------------===//

SerializationFormat *getFormatForExtension(llvm::StringRef Extension) {
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

  if (!isFormatRegistered(CapitalizedExtension)) {
    return nullptr;
  }

  auto Format = makeFormat(CapitalizedExtension);
  SerializationFormat *Result = Format.get();
  assert(Result);

  ExtensionFormatList.emplace_back(Extension, std::move(Format));

  return Result;
}

//===----------------------------------------------------------------------===//
// Data Structures
//===----------------------------------------------------------------------===//

struct SummaryFile {
  std::string Path;
  SerializationFormat *Format = nullptr;

  static SummaryFile fromPath(llvm::StringRef Path) {
    llvm::StringRef Extension = path::extension(Path);
    if (Extension.empty()) {
      fail(ErrorMessages::CannotValidateSummary, Path,
           ErrorMessages::ExtensionNotSupplied);
    }
    Extension = Extension.drop_front();
    SerializationFormat *Format = getFormatForExtension(Extension);
    if (!Format) {
      std::string BadExtension =
          llvm::formatv(ErrorMessages::NoFormatForExtension, Extension);
      fail(ErrorMessages::CannotValidateSummary, Path, BadExtension);
    }
    return {Path.str(), Format};
  }
};

struct LinkerInput {
  std::vector<SummaryFile> InputFiles;
  SummaryFile OutputFile;
  std::string LinkUnitName;
};

static void printVersion(llvm::raw_ostream &OS) { OS << ToolName << " 0.1\n"; }

//===----------------------------------------------------------------------===//
// Pipeline
//===----------------------------------------------------------------------===//

LinkerInput validate(llvm::TimerGroup &TG) {
  llvm::Timer TValidate("validate", "Validate Input", TG);
  LinkerInput LI;

  {
    llvm::TimeRegion _(Time ? &TValidate : nullptr);
    llvm::StringRef ParentDir = path::parent_path(OutputPath);
    llvm::StringRef DirToCheck = ParentDir.empty() ? "." : ParentDir;

    if (!fs::exists(DirToCheck)) {
      fail(ErrorMessages::CannotValidateSummary, OutputPath,
           ErrorMessages::OutputDirectoryMissing);
    }

    if (fs::access(DirToCheck, fs::AccessMode::Write)) {
      fail(ErrorMessages::CannotValidateSummary, OutputPath,
           ErrorMessages::OutputDirectoryNotWritable);
    }

    LI.OutputFile = SummaryFile::fromPath(OutputPath);
    LI.LinkUnitName = path::stem(LI.OutputFile.Path).str();
  }

  info(2, "Validated output summary path '{0}'.", LI.OutputFile.Path);

  {
    llvm::TimeRegion _(Time ? &TValidate : nullptr);
    for (const auto &InputPath : InputPaths) {
      llvm::SmallString<256> RealPath;
      std::error_code EC = fs::real_path(InputPath, RealPath, true);
      if (EC) {
        fail(ErrorMessages::CannotValidateSummary, InputPath, EC.message());
      }
      LI.InputFiles.push_back(SummaryFile::fromPath(RealPath));
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
                 .context(ErrorMessages::LinkingSummary, InputFile.Path)
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
  InitLLVM X(argc, argv);
  // path::stem strips the .exe extension on Windows so ToolName is consistent.
  ToolName = llvm::sys::path::stem(argv[0]);

  // Hide options unrelated to ssaf-linker from --help output.
  cl::HideUnrelatedOptions(SsafLinkerCategory);
  // Register a custom version printer for the --version flag.
  cl::SetVersionPrinter(printVersion);
  // Parse command-line arguments and exit with an error if they are invalid.
  cl::ParseCommandLineOptions(argc, argv, "SSAF Linker\n");

  initializeJSONFormat();

  llvm::TimerGroup LinkerTimers("ssaf-linker", "SSAF Linker");
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
