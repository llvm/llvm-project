//===- SSAFFormat.cpp - SSAF Format Tool ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements the SSAF format tool that validates and converts
//  TU and LU summaries between registered serialization formats.
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysisFramework/Core/EntityLinker/LUSummaryEncoding.h"
#include "clang/ScalableStaticAnalysisFramework/Core/EntityLinker/TUSummaryEncoding.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Serialization/JSONFormat.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Serialization/SerializationFormatRegistry.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <optional>
#include <string>
#include <system_error>

using namespace llvm;
using namespace clang::ssaf;

namespace {

namespace fs = llvm::sys::fs;
namespace path = llvm::sys::path;

//===----------------------------------------------------------------------===//
// Summary Type
//===----------------------------------------------------------------------===//

enum class SummaryType { TU, LU };

//===----------------------------------------------------------------------===//
// Command-Line Options
//===----------------------------------------------------------------------===//

cl::OptionCategory SsafFormatCategory("clang-ssaf-format options");

cl::list<std::string> LoadPlugins("load",
                                  cl::desc("Load a plugin shared library"),
                                  cl::value_desc("path"),
                                  cl::cat(SsafFormatCategory));

// --type and the input file are required for convert/validateInput operations
// but must be optional at the cl layer so that --list can be used standalone.
cl::opt<SummaryType> Type(
    "type", cl::desc("Summary type (required unless --list is given)"),
    cl::values(clEnumValN(SummaryType::TU, "tu", "Translation unit summary"),
               clEnumValN(SummaryType::LU, "lu", "Link unit summary")),
    cl::cat(SsafFormatCategory));

cl::opt<std::string> InputPath(cl::Positional, cl::desc("<input file>"),
                               cl::cat(SsafFormatCategory));

cl::opt<std::string> OutputPath("o", cl::desc("Output summary path"),
                                cl::value_desc("path"),
                                cl::cat(SsafFormatCategory));

cl::opt<bool> UseEncoding("encoding",
                          cl::desc("Read and write summary encodings rather "
                                   "than decoded summaries"),
                          cl::cat(SsafFormatCategory));

cl::opt<bool> ListFormats("list",
                          cl::desc("List registered serialization formats and "
                                   "analyses, then exit"),
                          cl::init(false), cl::cat(SsafFormatCategory));

llvm::StringRef ToolName;

void printVersion(llvm::raw_ostream &OS) { OS << ToolName << " 0.1\n"; }

//===----------------------------------------------------------------------===//
// Error Messages
//===----------------------------------------------------------------------===//

namespace ErrorMessages {

constexpr const char *FailedToLoadPlugin = "failed to load plugin '{0}': {1}";

constexpr const char *CannotValidateSummary =
    "failed to validate summary '{0}': {1}";

constexpr const char *ExtensionNotSupplied = "Extension not supplied";

constexpr const char *NoFormatForExtension =
    "Format not registered for extension '{0}'";

constexpr const char *OutputDirectoryMissing =
    "Parent directory does not exist";

constexpr const char *OutputFileAlreadyExists = "Output file already exists";

constexpr const char *InputOutputSamePath =
    "Input and Output resolve to the same path";

} // namespace ErrorMessages

//===----------------------------------------------------------------------===//
// Diagnostic Utilities
//===----------------------------------------------------------------------===//

[[noreturn]] void fail(const char *Msg) {
  llvm::WithColor::error(llvm::errs(), ToolName) << Msg << "\n";
  llvm::sys::Process::Exit(1);
}

template <typename... Ts>
[[noreturn]] void fail(const char *Fmt, Ts &&...Args) {
  std::string Message = llvm::formatv(Fmt, std::forward<Ts>(Args)...);
  fail(Message.data());
}

[[noreturn]] void fail(llvm::Error Err) {
  fail(toString(std::move(Err)).data());
}

//===----------------------------------------------------------------------===//
// Format Registry
//===----------------------------------------------------------------------===//

// FIXME: This will be revisited after we add support for registering formats
// with extensions.
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
// Format Listing
//===----------------------------------------------------------------------===//

constexpr size_t FormatIndent = 4;
constexpr size_t AnalysisIndent = 4;

struct AnalysisData {
  std::string Name;
  std::string Desc;
};

struct FormatData {
  std::string Name;
  std::string Desc;
  llvm::SmallVector<AnalysisData> Analyses;
};

struct PrintLayout {
  size_t FormatNumWidth;
  size_t MaxFormatNameWidth;
  size_t FormatNameCol;
  size_t AnalysisCol;
  size_t AnalysisNumWidth;
  size_t MaxAnalysisNameWidth;
};

llvm::SmallVector<FormatData> collectFormats() {
  llvm::SmallVector<FormatData> Formats;
  for (const auto &Entry : SerializationFormatRegistry::entries()) {
    FormatData FD;
    FD.Name = Entry.getName().str();
    FD.Desc = Entry.getDesc().str();
    auto Format = Entry.instantiate();
    Format->forEachRegisteredAnalysis(
        [&](llvm::StringRef Name, llvm::StringRef Desc) {
          FD.Analyses.push_back({Name.str(), Desc.str()});
        });
    Formats.push_back(std::move(FD));
  }
  return Formats;
}

void printAnalysis(const AnalysisData &AD, size_t AnalysisIndex,
                   size_t FormatIndex, const PrintLayout &Layout) {
  std::string AnalysisNum = std::to_string(FormatIndex + 1) + "." +
                            std::to_string(AnalysisIndex + 1) + ".";
  llvm::outs().indent(Layout.AnalysisCol)
      << llvm::right_justify(AnalysisNum, Layout.AnalysisNumWidth) << " "
      << llvm::left_justify(AD.Name, Layout.MaxAnalysisNameWidth) << "  "
      << AD.Desc << "\n";
}

void printAnalyses(const llvm::SmallVector<AnalysisData> &Analyses,
                   size_t FormatIndex, const PrintLayout &Layout) {
  if (Analyses.empty()) {
    llvm::outs().indent(Layout.FormatNameCol) << "Analyses: (none)\n";
    return;
  }

  llvm::outs().indent(Layout.FormatNameCol) << "Analyses:\n";

  for (size_t AnalysisIndex = 0; AnalysisIndex < Analyses.size();
       ++AnalysisIndex) {
    printAnalysis(Analyses[AnalysisIndex], AnalysisIndex, FormatIndex, Layout);
  }
}

void printFormat(const FormatData &FD, size_t FormatIndex,
                 const PrintLayout &Layout) {
  // Blank line before each format entry for readability.
  llvm::outs() << "\n";

  std::string FormatNum = std::to_string(FormatIndex + 1) + ".";
  llvm::outs().indent(FormatIndent)
      << llvm::right_justify(FormatNum, Layout.FormatNumWidth) << " "
      << llvm::left_justify(FD.Name, Layout.MaxFormatNameWidth) << "  "
      << FD.Desc << "\n";

  printAnalyses(FD.Analyses, FormatIndex, Layout);
}

void printFormats(const llvm::SmallVector<FormatData> &Formats,
                  const PrintLayout &Layout) {
  llvm::outs() << "Registered serialization formats:\n";
  for (size_t FormatIndex = 0; FormatIndex < Formats.size(); ++FormatIndex) {
    printFormat(Formats[FormatIndex], FormatIndex, Layout);
  }
}

PrintLayout computePrintLayout(const llvm::SmallVector<FormatData> &Formats) {
  size_t MaxFormatNameWidth = 0;
  size_t MaxAnalysisCount = 0;
  size_t MaxAnalysisNameWidth = 0;
  for (const auto &FD : Formats) {
    MaxFormatNameWidth = std::max(MaxFormatNameWidth, FD.Name.size());
    MaxAnalysisCount = std::max(MaxAnalysisCount, FD.Analyses.size());
    for (const auto &AD : FD.Analyses) {
      MaxAnalysisNameWidth = std::max(MaxAnalysisNameWidth, AD.Name.size());
    }
  }

  // Width of the widest format number string, e.g. "10." -> 3.
  size_t FormatNumWidth =
      std::to_string(Formats.size()).size() + 1; // +1 for '.'
  // Width of the widest analysis number string, e.g. "10.10." -> 6.
  size_t AnalysisNumWidth = std::to_string(Formats.size()).size() + 1 +
                            std::to_string(MaxAnalysisCount).size() + 1;

  // Where the format name starts (also where "Analyses:" is indented to).
  size_t FormatNameCol = FormatIndent + FormatNumWidth + 1;
  // Where the analysis number starts.
  size_t AnalysisCol = FormatNameCol + AnalysisIndent;

  return {
      FormatNumWidth, MaxFormatNameWidth, FormatNameCol,
      AnalysisCol,    AnalysisNumWidth,   MaxAnalysisNameWidth,
  };
}

void listFormats() {
  llvm::SmallVector<FormatData> Formats = collectFormats();
  if (Formats.empty()) {
    llvm::outs() << "No serialization formats registered.\n";
    return;
  }
  printFormats(Formats, computePrintLayout(Formats));
}

//===----------------------------------------------------------------------===//
// Plugin Loading
//===----------------------------------------------------------------------===//

void loadPlugins() {
  for (const auto &PluginPath : LoadPlugins) {
    std::string ErrMsg;
    if (llvm::sys::DynamicLibrary::LoadLibraryPermanently(PluginPath.c_str(),
                                                          &ErrMsg)) {
      fail(ErrorMessages::FailedToLoadPlugin, PluginPath, ErrMsg);
    }
  }
}

//===----------------------------------------------------------------------===//
// Input Validation
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
      std::string Msg =
          llvm::formatv(ErrorMessages::NoFormatForExtension, Extension);
      fail(ErrorMessages::CannotValidateSummary, Path, Msg);
    }

    return {Path.str(), Format};
  }
};

struct FormatInput {
  SummaryFile InputFile;
  std::optional<SummaryFile> OutputFile;
};

FormatInput validateInput() {
  assert(!ListFormats);

  FormatInput FI;

  // Validate Type explicitly since we don't want to specify it if --list is
  // provided.
  if (!Type.getNumOccurrences()) {
    fail("'--type' option is required");
  }

  // Validate the input path.
  {
    if (InputPath.empty()) {
      fail("no input file specified");
    }

    llvm::SmallString<256> RealInputPath;
    std::error_code EC =
        fs::real_path(InputPath, RealInputPath, /*expand_tilde=*/true);
    if (EC) {
      fail(ErrorMessages::CannotValidateSummary, InputPath, EC.message());
    }

    FI.InputFile = SummaryFile::fromPath(RealInputPath);
  }

  // Validate the output path.
  if (!OutputPath.empty()) {
    llvm::StringRef ParentDir = path::parent_path(OutputPath);
    llvm::StringRef DirToCheck = ParentDir.empty() ? "." : ParentDir;

    if (!fs::exists(DirToCheck)) {
      fail(ErrorMessages::CannotValidateSummary, OutputPath,
           ErrorMessages::OutputDirectoryMissing);
    }

    // Reconstruct the real output path from the real parent directory and the
    // output filename. The output file does not exist yet so real_path cannot
    // be called on the full output path directly.
    llvm::SmallString<256> RealParentDir;
    if (std::error_code EC = fs::real_path(DirToCheck, RealParentDir)) {
      fail(ErrorMessages::CannotValidateSummary, OutputPath, EC.message());
    }

    llvm::SmallString<256> RealOutputPath = RealParentDir;
    path::append(RealOutputPath, path::filename(OutputPath));

    if (RealOutputPath == FI.InputFile.Path) {
      fail(ErrorMessages::CannotValidateSummary, OutputPath,
           ErrorMessages::InputOutputSamePath);
    }

    if (fs::exists(RealOutputPath)) {
      fail(ErrorMessages::CannotValidateSummary, OutputPath,
           ErrorMessages::OutputFileAlreadyExists);
    }

    FI.OutputFile = SummaryFile::fromPath(RealOutputPath);
  }
  return FI;
}

//===----------------------------------------------------------------------===//
// Format Conversion
//===----------------------------------------------------------------------===//

template <typename ReadFn, typename WriteFn>
void run(const FormatInput &FI, ReadFn Read, WriteFn Write) {
  auto ExpectedResult = (FI.InputFile.Format->*Read)(FI.InputFile.Path);
  if (!ExpectedResult) {
    fail(ExpectedResult.takeError());
  }

  if (!FI.OutputFile) {
    return;
  }

  auto Err =
      (FI.OutputFile->Format->*Write)(*ExpectedResult, FI.OutputFile->Path);
  if (Err) {
    fail(std::move(Err));
  }
}

void convert(const FormatInput &FI) {
  switch (Type) {
  case SummaryType::TU:
    if (UseEncoding) {
      run(FI, &SerializationFormat::readTUSummaryEncoding,
          &SerializationFormat::writeTUSummaryEncoding);
    } else {
      run(FI, &SerializationFormat::readTUSummary,
          &SerializationFormat::writeTUSummary);
    }
    return;
  case SummaryType::LU:
    if (UseEncoding) {
      run(FI, &SerializationFormat::readLUSummaryEncoding,
          &SerializationFormat::writeLUSummaryEncoding);
    } else {
      run(FI, &SerializationFormat::readLUSummary,
          &SerializationFormat::writeLUSummary);
    }
    return;
  }

  llvm_unreachable("Unhandled SummaryType variant");
}

} // namespace

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

int main(int argc, const char **argv) {
  InitLLVM X(argc, argv);
  // path::stem strips the .exe extension on Windows so ToolName is consistent.
  ToolName = path::stem(argv[0]);

  cl::HideUnrelatedOptions(SsafFormatCategory);
  cl::SetVersionPrinter(printVersion);
  cl::ParseCommandLineOptions(argc, argv, "SSAF Format\n");

  loadPlugins();

  initializeJSONFormat();

  if (ListFormats) {
    listFormats();
  } else {
    FormatInput FI = validateInput();
    convert(FI);
  }

  return 0;
}
