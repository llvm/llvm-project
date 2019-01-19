//===--- ClangTidyOptions.cpp - clang-tidy ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ClangTidyOptions.h"
#include "ClangTidyModuleRegistry.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"
#include <utility>

#define DEBUG_TYPE "clang-tidy-options"

using clang::tidy::ClangTidyOptions;
using clang::tidy::FileFilter;
using OptionsSource = clang::tidy::ClangTidyOptionsProvider::OptionsSource;

LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(FileFilter)
LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(FileFilter::LineRange)

namespace llvm {
namespace yaml {

// Map std::pair<int, int> to a JSON array of size 2.
template <> struct SequenceTraits<FileFilter::LineRange> {
  static size_t size(IO &IO, FileFilter::LineRange &Range) {
    return Range.first == 0 ? 0 : Range.second == 0 ? 1 : 2;
  }
  static unsigned &element(IO &IO, FileFilter::LineRange &Range, size_t Index) {
    if (Index > 1)
      IO.setError("Too many elements in line range.");
    return Index == 0 ? Range.first : Range.second;
  }
};

template <> struct MappingTraits<FileFilter> {
  static void mapping(IO &IO, FileFilter &File) {
    IO.mapRequired("name", File.Name);
    IO.mapOptional("lines", File.LineRanges);
  }
  static StringRef validate(IO &io, FileFilter &File) {
    if (File.Name.empty())
      return "No file name specified";
    for (const FileFilter::LineRange &Range : File.LineRanges) {
      if (Range.first <= 0 || Range.second <= 0)
        return "Invalid line range";
    }
    return StringRef();
  }
};

template <> struct MappingTraits<ClangTidyOptions::StringPair> {
  static void mapping(IO &IO, ClangTidyOptions::StringPair &KeyValue) {
    IO.mapRequired("key", KeyValue.first);
    IO.mapRequired("value", KeyValue.second);
  }
};

struct NOptionMap {
  NOptionMap(IO &) {}
  NOptionMap(IO &, const ClangTidyOptions::OptionMap &OptionMap)
      : Options(OptionMap.begin(), OptionMap.end()) {}
  ClangTidyOptions::OptionMap denormalize(IO &) {
    ClangTidyOptions::OptionMap Map;
    for (const auto &KeyValue : Options)
      Map[KeyValue.first] = KeyValue.second;
    return Map;
  }
  std::vector<ClangTidyOptions::StringPair> Options;
};

template <> struct MappingTraits<ClangTidyOptions> {
  static void mapping(IO &IO, ClangTidyOptions &Options) {
    MappingNormalization<NOptionMap, ClangTidyOptions::OptionMap> NOpts(
        IO, Options.CheckOptions);
    bool Ignored = false;
    IO.mapOptional("Checks", Options.Checks);
    IO.mapOptional("WarningsAsErrors", Options.WarningsAsErrors);
    IO.mapOptional("HeaderFilterRegex", Options.HeaderFilterRegex);
    IO.mapOptional("AnalyzeTemporaryDtors", Ignored); // legacy compatibility
    IO.mapOptional("FormatStyle", Options.FormatStyle);
    IO.mapOptional("User", Options.User);
    IO.mapOptional("CheckOptions", NOpts->Options);
    IO.mapOptional("ExtraArgs", Options.ExtraArgs);
    IO.mapOptional("ExtraArgsBefore", Options.ExtraArgsBefore);
  }
};

} // namespace yaml
} // namespace llvm

namespace clang {
namespace tidy {

ClangTidyOptions ClangTidyOptions::getDefaults() {
  ClangTidyOptions Options;
  Options.Checks = "";
  Options.WarningsAsErrors = "";
  Options.HeaderFilterRegex = "";
  Options.SystemHeaders = false;
  Options.FormatStyle = "none";
  Options.User = llvm::None;
  for (ClangTidyModuleRegistry::iterator I = ClangTidyModuleRegistry::begin(),
                                         E = ClangTidyModuleRegistry::end();
       I != E; ++I)
    Options = Options.mergeWith(I->instantiate()->getModuleOptions());
  return Options;
}

template <typename T>
static void mergeVectors(Optional<T> &Dest, const Optional<T> &Src) {
  if (Src) {
    if (Dest)
      Dest->insert(Dest->end(), Src->begin(), Src->end());
    else
      Dest = Src;
  }
}

static void mergeCommaSeparatedLists(Optional<std::string> &Dest,
                                     const Optional<std::string> &Src) {
  if (Src)
    Dest = (Dest && !Dest->empty() ? *Dest + "," : "") + *Src;
}

template <typename T>
static void overrideValue(Optional<T> &Dest, const Optional<T> &Src) {
  if (Src)
    Dest = Src;
}

ClangTidyOptions
ClangTidyOptions::mergeWith(const ClangTidyOptions &Other) const {
  ClangTidyOptions Result = *this;

  mergeCommaSeparatedLists(Result.Checks, Other.Checks);
  mergeCommaSeparatedLists(Result.WarningsAsErrors, Other.WarningsAsErrors);
  overrideValue(Result.HeaderFilterRegex, Other.HeaderFilterRegex);
  overrideValue(Result.SystemHeaders, Other.SystemHeaders);
  overrideValue(Result.FormatStyle, Other.FormatStyle);
  overrideValue(Result.User, Other.User);
  mergeVectors(Result.ExtraArgs, Other.ExtraArgs);
  mergeVectors(Result.ExtraArgsBefore, Other.ExtraArgsBefore);

  for (const auto &KeyValue : Other.CheckOptions)
    Result.CheckOptions[KeyValue.first] = KeyValue.second;

  return Result;
}

const char ClangTidyOptionsProvider::OptionsSourceTypeDefaultBinary[] =
    "clang-tidy binary";
const char ClangTidyOptionsProvider::OptionsSourceTypeCheckCommandLineOption[] =
    "command-line option '-checks'";
const char
    ClangTidyOptionsProvider::OptionsSourceTypeConfigCommandLineOption[] =
        "command-line option '-config'";

ClangTidyOptions
ClangTidyOptionsProvider::getOptions(llvm::StringRef FileName) {
  ClangTidyOptions Result;
  for (const auto &Source : getRawOptions(FileName))
    Result = Result.mergeWith(Source.first);
  return Result;
}

std::vector<OptionsSource>
DefaultOptionsProvider::getRawOptions(llvm::StringRef FileName) {
  std::vector<OptionsSource> Result;
  Result.emplace_back(DefaultOptions, OptionsSourceTypeDefaultBinary);
  return Result;
}

ConfigOptionsProvider::ConfigOptionsProvider(
    const ClangTidyGlobalOptions &GlobalOptions,
    const ClangTidyOptions &DefaultOptions,
    const ClangTidyOptions &ConfigOptions,
    const ClangTidyOptions &OverrideOptions)
    : DefaultOptionsProvider(GlobalOptions, DefaultOptions),
      ConfigOptions(ConfigOptions), OverrideOptions(OverrideOptions) {}

std::vector<OptionsSource>
ConfigOptionsProvider::getRawOptions(llvm::StringRef FileName) {
  std::vector<OptionsSource> RawOptions =
      DefaultOptionsProvider::getRawOptions(FileName);
  RawOptions.emplace_back(ConfigOptions,
                          OptionsSourceTypeConfigCommandLineOption);
  RawOptions.emplace_back(OverrideOptions,
                          OptionsSourceTypeCheckCommandLineOption);
  return RawOptions;
}

FileOptionsProvider::FileOptionsProvider(
    const ClangTidyGlobalOptions &GlobalOptions,
    const ClangTidyOptions &DefaultOptions,
    const ClangTidyOptions &OverrideOptions,
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS)
    : DefaultOptionsProvider(GlobalOptions, DefaultOptions),
      OverrideOptions(OverrideOptions), FS(std::move(VFS)) {
  if (!FS)
    FS = llvm::vfs::getRealFileSystem();
  ConfigHandlers.emplace_back(".clang-tidy", parseConfiguration);
}

FileOptionsProvider::FileOptionsProvider(
    const ClangTidyGlobalOptions &GlobalOptions,
    const ClangTidyOptions &DefaultOptions,
    const ClangTidyOptions &OverrideOptions,
    const FileOptionsProvider::ConfigFileHandlers &ConfigHandlers)
    : DefaultOptionsProvider(GlobalOptions, DefaultOptions),
      OverrideOptions(OverrideOptions), ConfigHandlers(ConfigHandlers) {}

// FIXME: This method has some common logic with clang::format::getStyle().
// Consider pulling out common bits to a findParentFileWithName function or
// similar.
std::vector<OptionsSource>
FileOptionsProvider::getRawOptions(StringRef FileName) {
  LLVM_DEBUG(llvm::dbgs() << "Getting options for file " << FileName
                          << "...\n");
  assert(FS && "FS must be set.");

  llvm::SmallString<128> AbsoluteFilePath(FileName);

  if (FS->makeAbsolute(AbsoluteFilePath))
    return {};

  std::vector<OptionsSource> RawOptions =
      DefaultOptionsProvider::getRawOptions(AbsoluteFilePath.str());
  OptionsSource CommandLineOptions(OverrideOptions,
                                   OptionsSourceTypeCheckCommandLineOption);
  // Look for a suitable configuration file in all parent directories of the
  // file. Start with the immediate parent directory and move up.
  StringRef Path = llvm::sys::path::parent_path(AbsoluteFilePath.str());
  for (StringRef CurrentPath = Path; !CurrentPath.empty();
       CurrentPath = llvm::sys::path::parent_path(CurrentPath)) {
    llvm::Optional<OptionsSource> Result;

    auto Iter = CachedOptions.find(CurrentPath);
    if (Iter != CachedOptions.end())
      Result = Iter->second;

    if (!Result)
      Result = tryReadConfigFile(CurrentPath);

    if (Result) {
      // Store cached value for all intermediate directories.
      while (Path != CurrentPath) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Caching configuration for path " << Path << ".\n");
        CachedOptions[Path] = *Result;
        Path = llvm::sys::path::parent_path(Path);
      }
      CachedOptions[Path] = *Result;

      RawOptions.push_back(*Result);
      break;
    }
  }
  RawOptions.push_back(CommandLineOptions);
  return RawOptions;
}

llvm::Optional<OptionsSource>
FileOptionsProvider::tryReadConfigFile(StringRef Directory) {
  assert(!Directory.empty());

  if (!llvm::sys::fs::is_directory(Directory)) {
    llvm::errs() << "Error reading configuration from " << Directory
                 << ": directory doesn't exist.\n";
    return llvm::None;
  }

  for (const ConfigFileHandler &ConfigHandler : ConfigHandlers) {
    SmallString<128> ConfigFile(Directory);
    llvm::sys::path::append(ConfigFile, ConfigHandler.first);
    LLVM_DEBUG(llvm::dbgs() << "Trying " << ConfigFile << "...\n");

    bool IsFile = false;
    // Ignore errors from is_regular_file: we only need to know if we can read
    // the file or not.
    llvm::sys::fs::is_regular_file(Twine(ConfigFile), IsFile);
    if (!IsFile)
      continue;

    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> Text =
        llvm::MemoryBuffer::getFile(ConfigFile.c_str());
    if (std::error_code EC = Text.getError()) {
      llvm::errs() << "Can't read " << ConfigFile << ": " << EC.message()
                   << "\n";
      continue;
    }

    // Skip empty files, e.g. files opened for writing via shell output
    // redirection.
    if ((*Text)->getBuffer().empty())
      continue;
    llvm::ErrorOr<ClangTidyOptions> ParsedOptions =
        ConfigHandler.second((*Text)->getBuffer());
    if (!ParsedOptions) {
      if (ParsedOptions.getError())
        llvm::errs() << "Error parsing " << ConfigFile << ": "
                     << ParsedOptions.getError().message() << "\n";
      continue;
    }
    return OptionsSource(*ParsedOptions, ConfigFile.c_str());
  }
  return llvm::None;
}

/// \brief Parses -line-filter option and stores it to the \c Options.
std::error_code parseLineFilter(StringRef LineFilter,
                                clang::tidy::ClangTidyGlobalOptions &Options) {
  llvm::yaml::Input Input(LineFilter);
  Input >> Options.LineFilter;
  return Input.error();
}

llvm::ErrorOr<ClangTidyOptions> parseConfiguration(StringRef Config) {
  llvm::yaml::Input Input(Config);
  ClangTidyOptions Options;
  Input >> Options;
  if (Input.error())
    return Input.error();
  return Options;
}

std::string configurationAsText(const ClangTidyOptions &Options) {
  std::string Text;
  llvm::raw_string_ostream Stream(Text);
  llvm::yaml::Output Output(Stream);
  // We use the same mapping method for input and output, so we need a non-const
  // reference here.
  ClangTidyOptions NonConstValue = Options;
  Output << NonConstValue;
  return Stream.str();
}

} // namespace tidy
} // namespace clang
