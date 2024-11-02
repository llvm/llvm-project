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
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/YAMLTraits.h"
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
  static std::string validate(IO &Io, FileFilter &File) {
    if (File.Name.empty())
      return "No file name specified";
    for (const FileFilter::LineRange &Range : File.LineRanges) {
      if (Range.first <= 0 || Range.second <= 0)
        return "Invalid line range";
    }
    return "";
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
  NOptionMap(IO &, const ClangTidyOptions::OptionMap &OptionMap) {
    Options.reserve(OptionMap.size());
    for (const auto &KeyValue : OptionMap)
      Options.emplace_back(std::string(KeyValue.getKey()), KeyValue.getValue().Value);
  }
  ClangTidyOptions::OptionMap denormalize(IO &) {
    ClangTidyOptions::OptionMap Map;
    for (const auto &KeyValue : Options)
      Map[KeyValue.first] = ClangTidyOptions::ClangTidyValue(KeyValue.second);
    return Map;
  }
  std::vector<ClangTidyOptions::StringPair> Options;
};

template <>
void yamlize(IO &IO, ClangTidyOptions::OptionMap &Options, bool,
             EmptyContext &Ctx) {
  if (IO.outputting()) {
    IO.beginMapping();
    // Only output as a map
    for (auto &Key : Options) {
      bool UseDefault;
      void *SaveInfo;
      IO.preflightKey(Key.getKey().data(), true, false, UseDefault, SaveInfo);
      StringRef S = Key.getValue().Value;
      IO.scalarString(S, needsQuotes(S));
      IO.postflightKey(SaveInfo);
    }
    IO.endMapping();
  } else {
    // We need custom logic here to support the old method of specifying check
    // options using a list of maps containing key and value keys.
    Input &I = reinterpret_cast<Input &>(IO);
    if (isa<SequenceNode>(I.getCurrentNode())) {
      MappingNormalization<NOptionMap, ClangTidyOptions::OptionMap> NOpts(
          IO, Options);
      EmptyContext Ctx;
      yamlize(IO, NOpts->Options, true, Ctx);
    } else if (isa<MappingNode>(I.getCurrentNode())) {
      IO.beginMapping();
      for (StringRef Key : IO.keys()) {
        IO.mapRequired(Key.data(), Options[Key].Value);
      }
      IO.endMapping();
    } else {
      IO.setError("expected a sequence or map");
    }
  }
}

template <> struct MappingTraits<ClangTidyOptions> {
  static void mapping(IO &IO, ClangTidyOptions &Options) {
    bool Ignored = false;
    IO.mapOptional("Checks", Options.Checks);
    IO.mapOptional("WarningsAsErrors", Options.WarningsAsErrors);
    IO.mapOptional("HeaderFilterRegex", Options.HeaderFilterRegex);
    IO.mapOptional("AnalyzeTemporaryDtors", Ignored); // legacy compatibility
    IO.mapOptional("FormatStyle", Options.FormatStyle);
    IO.mapOptional("User", Options.User);
    IO.mapOptional("CheckOptions", Options.CheckOptions);
    IO.mapOptional("ExtraArgs", Options.ExtraArgs);
    IO.mapOptional("ExtraArgsBefore", Options.ExtraArgsBefore);
    IO.mapOptional("InheritParentConfig", Options.InheritParentConfig);
    IO.mapOptional("UseColor", Options.UseColor);
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
  Options.User = std::nullopt;
  for (const ClangTidyModuleRegistry::entry &Module :
       ClangTidyModuleRegistry::entries())
    Options.mergeWith(Module.instantiate()->getModuleOptions(), 0);
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

ClangTidyOptions &ClangTidyOptions::mergeWith(const ClangTidyOptions &Other,
                                              unsigned Order) {
  mergeCommaSeparatedLists(Checks, Other.Checks);
  mergeCommaSeparatedLists(WarningsAsErrors, Other.WarningsAsErrors);
  overrideValue(HeaderFilterRegex, Other.HeaderFilterRegex);
  overrideValue(SystemHeaders, Other.SystemHeaders);
  overrideValue(FormatStyle, Other.FormatStyle);
  overrideValue(User, Other.User);
  overrideValue(UseColor, Other.UseColor);
  mergeVectors(ExtraArgs, Other.ExtraArgs);
  mergeVectors(ExtraArgsBefore, Other.ExtraArgsBefore);

  for (const auto &KeyValue : Other.CheckOptions) {
    CheckOptions.insert_or_assign(
        KeyValue.getKey(),
        ClangTidyValue(KeyValue.getValue().Value,
                       KeyValue.getValue().Priority + Order));
  }
  return *this;
}

ClangTidyOptions ClangTidyOptions::merge(const ClangTidyOptions &Other,
                                         unsigned Order) const {
  ClangTidyOptions Result = *this;
  Result.mergeWith(Other, Order);
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
  unsigned Priority = 0;
  for (auto &Source : getRawOptions(FileName))
    Result.mergeWith(Source.first, ++Priority);
  return Result;
}

std::vector<OptionsSource>
DefaultOptionsProvider::getRawOptions(llvm::StringRef FileName) {
  std::vector<OptionsSource> Result;
  Result.emplace_back(DefaultOptions, OptionsSourceTypeDefaultBinary);
  return Result;
}

ConfigOptionsProvider::ConfigOptionsProvider(
    ClangTidyGlobalOptions GlobalOptions, ClangTidyOptions DefaultOptions,
    ClangTidyOptions ConfigOptions, ClangTidyOptions OverrideOptions,
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS)
    : FileOptionsBaseProvider(std::move(GlobalOptions),
                              std::move(DefaultOptions),
                              std::move(OverrideOptions), std::move(FS)),
      ConfigOptions(std::move(ConfigOptions)) {}

std::vector<OptionsSource>
ConfigOptionsProvider::getRawOptions(llvm::StringRef FileName) {
  std::vector<OptionsSource> RawOptions =
      DefaultOptionsProvider::getRawOptions(FileName);
  if (ConfigOptions.InheritParentConfig.value_or(false)) {
    LLVM_DEBUG(llvm::dbgs()
               << "Getting options for file " << FileName << "...\n");
    assert(FS && "FS must be set.");

    llvm::SmallString<128> AbsoluteFilePath(FileName);

    if (!FS->makeAbsolute(AbsoluteFilePath)) {
      addRawFileOptions(AbsoluteFilePath, RawOptions);
    }
  }
  RawOptions.emplace_back(ConfigOptions,
                          OptionsSourceTypeConfigCommandLineOption);
  RawOptions.emplace_back(OverrideOptions,
                          OptionsSourceTypeCheckCommandLineOption);
  return RawOptions;
}

FileOptionsBaseProvider::FileOptionsBaseProvider(
    ClangTidyGlobalOptions GlobalOptions, ClangTidyOptions DefaultOptions,
    ClangTidyOptions OverrideOptions,
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS)
    : DefaultOptionsProvider(std::move(GlobalOptions),
                             std::move(DefaultOptions)),
      OverrideOptions(std::move(OverrideOptions)), FS(std::move(VFS)) {
  if (!FS)
    FS = llvm::vfs::getRealFileSystem();
  ConfigHandlers.emplace_back(".clang-tidy", parseConfiguration);
}

FileOptionsBaseProvider::FileOptionsBaseProvider(
    ClangTidyGlobalOptions GlobalOptions, ClangTidyOptions DefaultOptions,
    ClangTidyOptions OverrideOptions,
    FileOptionsBaseProvider::ConfigFileHandlers ConfigHandlers)
    : DefaultOptionsProvider(std::move(GlobalOptions),
                             std::move(DefaultOptions)),
      OverrideOptions(std::move(OverrideOptions)),
      ConfigHandlers(std::move(ConfigHandlers)) {}

void FileOptionsBaseProvider::addRawFileOptions(
    llvm::StringRef AbsolutePath, std::vector<OptionsSource> &CurOptions) {
  auto CurSize = CurOptions.size();

  // Look for a suitable configuration file in all parent directories of the
  // file. Start with the immediate parent directory and move up.
  StringRef Path = llvm::sys::path::parent_path(AbsolutePath);
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
        if (!CachedOptions.count(Path))
          CachedOptions[Path] = *Result;
        Path = llvm::sys::path::parent_path(Path);
      }
      CachedOptions[Path] = *Result;

      CurOptions.push_back(*Result);
      if (!Result->first.InheritParentConfig.value_or(false))
        break;
    }
  }
  // Reverse order of file configs because closer configs should have higher
  // priority.
  std::reverse(CurOptions.begin() + CurSize, CurOptions.end());
}

FileOptionsProvider::FileOptionsProvider(
    ClangTidyGlobalOptions GlobalOptions, ClangTidyOptions DefaultOptions,
    ClangTidyOptions OverrideOptions,
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS)
    : FileOptionsBaseProvider(std::move(GlobalOptions),
                              std::move(DefaultOptions),
                              std::move(OverrideOptions), std::move(VFS)) {}

FileOptionsProvider::FileOptionsProvider(
    ClangTidyGlobalOptions GlobalOptions, ClangTidyOptions DefaultOptions,
    ClangTidyOptions OverrideOptions,
    FileOptionsBaseProvider::ConfigFileHandlers ConfigHandlers)
    : FileOptionsBaseProvider(
          std::move(GlobalOptions), std::move(DefaultOptions),
          std::move(OverrideOptions), std::move(ConfigHandlers)) {}

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
  addRawFileOptions(AbsoluteFilePath, RawOptions);
  OptionsSource CommandLineOptions(OverrideOptions,
                                   OptionsSourceTypeCheckCommandLineOption);

  RawOptions.push_back(CommandLineOptions);
  return RawOptions;
}

llvm::Optional<OptionsSource>
FileOptionsBaseProvider::tryReadConfigFile(StringRef Directory) {
  assert(!Directory.empty());

  llvm::ErrorOr<llvm::vfs::Status> DirectoryStatus = FS->status(Directory);

  if (!DirectoryStatus || !DirectoryStatus->isDirectory()) {
    llvm::errs() << "Error reading configuration from " << Directory
                 << ": directory doesn't exist.\n";
    return std::nullopt;
  }

  for (const ConfigFileHandler &ConfigHandler : ConfigHandlers) {
    SmallString<128> ConfigFile(Directory);
    llvm::sys::path::append(ConfigFile, ConfigHandler.first);
    LLVM_DEBUG(llvm::dbgs() << "Trying " << ConfigFile << "...\n");

    llvm::ErrorOr<llvm::vfs::Status> FileStatus = FS->status(ConfigFile);

    if (!FileStatus || !FileStatus->isRegularFile())
      continue;

    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> Text =
        FS->getBufferForFile(ConfigFile);
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
        ConfigHandler.second({(*Text)->getBuffer(), ConfigFile});
    if (!ParsedOptions) {
      if (ParsedOptions.getError())
        llvm::errs() << "Error parsing " << ConfigFile << ": "
                     << ParsedOptions.getError().message() << "\n";
      continue;
    }
    return OptionsSource(*ParsedOptions, std::string(ConfigFile));
  }
  return std::nullopt;
}

/// Parses -line-filter option and stores it to the \c Options.
std::error_code parseLineFilter(StringRef LineFilter,
                                clang::tidy::ClangTidyGlobalOptions &Options) {
  llvm::yaml::Input Input(LineFilter);
  Input >> Options.LineFilter;
  return Input.error();
}

llvm::ErrorOr<ClangTidyOptions>
parseConfiguration(llvm::MemoryBufferRef Config) {
  llvm::yaml::Input Input(Config);
  ClangTidyOptions Options;
  Input >> Options;
  if (Input.error())
    return Input.error();
  return Options;
}

static void diagHandlerImpl(const llvm::SMDiagnostic &Diag, void *Ctx) {
  (*reinterpret_cast<DiagCallback *>(Ctx))(Diag);
}

llvm::ErrorOr<ClangTidyOptions>
parseConfigurationWithDiags(llvm::MemoryBufferRef Config,
                            DiagCallback Handler) {
  llvm::yaml::Input Input(Config, nullptr, Handler ? diagHandlerImpl : nullptr,
                          &Handler);
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
