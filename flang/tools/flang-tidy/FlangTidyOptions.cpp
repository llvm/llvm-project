//===--- FlangTidyOptions.cpp - flang-tidy --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FlangTidyOptions.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/YAMLTraits.h"
#include <algorithm>
#include <sstream>

#define DEBUG_TYPE "flang-tidy-options"

using Fortran::tidy::FlangTidyOptions;
using OptionsSource = Fortran::tidy::FlangTidyOptionsProvider::OptionsSource;

namespace llvm::yaml {

template <>
struct MappingTraits<FlangTidyOptions::StringPair> {
  static void mapping(IO &IO, FlangTidyOptions::StringPair &KeyValue) {
    IO.mapRequired("key", KeyValue.first);
    IO.mapRequired("value", KeyValue.second);
  }
};

struct NOptionMap {
  NOptionMap(IO &) {}
  NOptionMap(IO &, const FlangTidyOptions::OptionMap &OptionMap) {
    Options.reserve(OptionMap.size());
    for (const auto &KeyValue : OptionMap)
      Options.emplace_back(std::string(KeyValue.getKey()),
                           KeyValue.getValue().Value);
  }
  FlangTidyOptions::OptionMap denormalize(IO &) {
    FlangTidyOptions::OptionMap Map;
    for (const auto &KeyValue : Options)
      Map[KeyValue.first] = FlangTidyOptions::FlangTidyValue(KeyValue.second);
    return Map;
  }
  std::vector<FlangTidyOptions::StringPair> Options;
};

template <>
void yamlize(IO &IO, FlangTidyOptions::OptionMap &Val, bool,
             EmptyContext &Ctx) {
  if (IO.outputting()) {
    // Ensure check options are sorted
    std::vector<std::pair<llvm::StringRef, llvm::StringRef>> SortedOptions;
    SortedOptions.reserve(Val.size());
    for (auto &Key : Val) {
      SortedOptions.emplace_back(Key.getKey(), Key.getValue().Value);
    }
    std::sort(SortedOptions.begin(), SortedOptions.end());

    IO.beginMapping();
    // Only output as a map
    for (auto &Option : SortedOptions) {
      bool UseDefault = false;
      void *SaveInfo = nullptr;
      IO.preflightKey(Option.first.data(), true, false, UseDefault, SaveInfo);
      IO.scalarString(Option.second, needsQuotes(Option.second));
      IO.postflightKey(SaveInfo);
    }
    IO.endMapping();
  } else {
    // Support both old list format and new map format for reading
    auto &I = reinterpret_cast<Input &>(IO);
    if (isa<SequenceNode>(I.getCurrentNode())) {
      MappingNormalization<NOptionMap, FlangTidyOptions::OptionMap> NOpts(IO,
                                                                          Val);
      EmptyContext Ctx;
      yamlize(IO, NOpts->Options, true, Ctx);
    } else if (isa<MappingNode>(I.getCurrentNode())) {
      IO.beginMapping();
      for (llvm::StringRef Key : IO.keys()) {
        IO.mapRequired(Key.data(), Val[Key].Value);
      }
      IO.endMapping();
    } else {
      IO.setError("expected a sequence or map");
    }
  }
}

template <>
struct MappingTraits<FlangTidyOptions> {
  static void mapping(IO &IO, FlangTidyOptions &Options) {
    IO.mapOptional("Checks", Options.Checks);
    IO.mapOptional("WarningsAsErrors", Options.WarningsAsErrors);
    IO.mapOptional("CheckOptions", Options.CheckOptions);
    IO.mapOptional("ExtraArgs", Options.ExtraArgs);
    IO.mapOptional("ExtraArgsBefore", Options.ExtraArgsBefore);
    IO.mapOptional("InheritParentConfig", Options.InheritParentConfig);
  }
};

} // namespace llvm::yaml

namespace Fortran::tidy {

const char FlangTidyOptionsProvider::OptionsSourceTypeDefaultBinary[] =
    "flang-tidy binary";
const char FlangTidyOptionsProvider::OptionsSourceTypeCheckCommandLineOption[] =
    "command-line option '-checks'";
const char
    FlangTidyOptionsProvider::OptionsSourceTypeConfigCommandLineOption[] =
        "command-line option '-config'";

FlangTidyOptions FlangTidyOptions::getDefaults() {
  FlangTidyOptions Options;
  Options.Checks = "*";
  Options.WarningsAsErrors = "";
  return Options;
}

template <typename T>
static void mergeVectors(std::optional<T> &Dest, const std::optional<T> &Src) {
  if (Src) {
    if (Dest)
      Dest->insert(Dest->end(), Src->begin(), Src->end());
    else
      Dest = Src;
  }
}

template <typename T>
static void overrideValue(std::optional<T> &Dest, const std::optional<T> &Src) {
  if (Src)
    Dest = Src;
}

FlangTidyOptions &FlangTidyOptions::mergeWith(const FlangTidyOptions &Other,
                                              unsigned Order) {
  // For checks: if Other has checks defined, override completely (don't merge)
  // This ensures config file checks override defaults instead of appending
  if (Other.Checks) {
    Checks = Other.Checks;
  }
  // Same for WarningsAsErrors. Needs to be consistent with Checks.
  if (Other.WarningsAsErrors) {
    WarningsAsErrors = Other.WarningsAsErrors;
  }

  mergeVectors(ExtraArgs, Other.ExtraArgs);
  mergeVectors(ExtraArgsBefore, Other.ExtraArgsBefore);
  overrideValue(InheritParentConfig, Other.InheritParentConfig);

  for (const auto &KeyValue : Other.CheckOptions) {
    CheckOptions.insert_or_assign(
        KeyValue.getKey(),
        FlangTidyValue(KeyValue.getValue().Value,
                       KeyValue.getValue().Priority + Order));
  }
  return *this;
}

FlangTidyOptions FlangTidyOptions::merge(const FlangTidyOptions &Other,
                                         unsigned Order) const {
  FlangTidyOptions Result = *this;
  Result.mergeWith(Other, Order);
  return Result;
}

void FlangTidyOptions::parseChecksString() {
  enabledChecks.clear();
  if (!Checks || Checks->empty())
    return;

  std::stringstream ss(*Checks);
  std::string check;
  while (std::getline(ss, check, ',')) {
    // Trim whitespace
    check.erase(0, check.find_first_not_of(" \t"));
    check.erase(check.find_last_not_of(" \t") + 1);
    if (!check.empty()) {
      enabledChecks.push_back(check);
    }
  }
}

void FlangTidyOptions::parseWarningsAsErrorsString() {
  enabledWarningsAsErrors.clear();
  if (!WarningsAsErrors || WarningsAsErrors->empty())
    return;

  std::stringstream ss(*WarningsAsErrors);
  std::string check;
  while (std::getline(ss, check, ',')) {
    // Trim whitespace
    check.erase(0, check.find_first_not_of(" \t"));
    check.erase(check.find_last_not_of(" \t") + 1);
    if (!check.empty()) {
      enabledWarningsAsErrors.push_back(check);
    }
  }
}

FlangTidyOptions
FlangTidyOptionsProvider::getOptions(llvm::StringRef FileName) {
  FlangTidyOptions Result;
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
    FlangTidyGlobalOptions GlobalOptions, FlangTidyOptions DefaultOptions,
    FlangTidyOptions ConfigOptions, FlangTidyOptions OverrideOptions,
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

    llvm::ErrorOr<llvm::SmallString<128>> AbsoluteFilePath =
        getNormalizedAbsolutePath(FileName);
    if (AbsoluteFilePath) {
      addRawFileOptions(AbsoluteFilePath->str(), RawOptions);
    }
  }
  RawOptions.emplace_back(ConfigOptions,
                          OptionsSourceTypeConfigCommandLineOption);
  // Give command-line options very high priority so they override everything
  RawOptions.emplace_back(OverrideOptions,
                          OptionsSourceTypeCheckCommandLineOption);
  return RawOptions;
}

FileOptionsBaseProvider::FileOptionsBaseProvider(
    FlangTidyGlobalOptions GlobalOptions, FlangTidyOptions DefaultOptions,
    FlangTidyOptions OverrideOptions,
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS)
    : DefaultOptionsProvider(std::move(GlobalOptions),
                             std::move(DefaultOptions)),
      OverrideOptions(std::move(OverrideOptions)), FS(std::move(VFS)) {
  if (!FS)
    FS = llvm::vfs::getRealFileSystem();
  ConfigHandlers.emplace_back(".flang-tidy", parseConfiguration);
}

FileOptionsBaseProvider::FileOptionsBaseProvider(
    FlangTidyGlobalOptions GlobalOptions, FlangTidyOptions DefaultOptions,
    FlangTidyOptions OverrideOptions,
    FileOptionsBaseProvider::ConfigFileHandlers ConfigHandlers)
    : DefaultOptionsProvider(std::move(GlobalOptions),
                             std::move(DefaultOptions)),
      OverrideOptions(std::move(OverrideOptions)),
      ConfigHandlers(std::move(ConfigHandlers)) {}

llvm::ErrorOr<llvm::SmallString<128>>
FileOptionsBaseProvider::getNormalizedAbsolutePath(llvm::StringRef Path) {
  assert(FS && "FS must be set.");
  llvm::SmallString<128> NormalizedAbsolutePath = {Path};
  std::error_code Err = FS->makeAbsolute(NormalizedAbsolutePath);
  if (Err)
    return Err;
  llvm::sys::path::remove_dots(NormalizedAbsolutePath, /*remove_dot_dot=*/true);
  return NormalizedAbsolutePath;
}

void FileOptionsBaseProvider::addRawFileOptions(
    llvm::StringRef AbsolutePath, std::vector<OptionsSource> &CurOptions) {
  auto CurSize = CurOptions.size();
  llvm::StringRef RootPath = llvm::sys::path::parent_path(AbsolutePath);
  auto MemorizedConfigFile =
      [this,
       &RootPath](llvm::StringRef CurrentPath) -> std::optional<OptionsSource> {
    const auto Iter = CachedOptions.Memorized.find(CurrentPath);
    if (Iter != CachedOptions.Memorized.end())
      return CachedOptions.Storage[Iter->second];
    std::optional<OptionsSource> OptionsSource = tryReadConfigFile(CurrentPath);
    if (OptionsSource) {
      const size_t Index = CachedOptions.Storage.size();
      CachedOptions.Storage.emplace_back(OptionsSource.value());
      while (RootPath != CurrentPath) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Caching configuration for path " << RootPath << ".\n");
        CachedOptions.Memorized[RootPath] = Index;
        RootPath = llvm::sys::path::parent_path(RootPath);
      }
      CachedOptions.Memorized[CurrentPath] = Index;
      RootPath = llvm::sys::path::parent_path(CurrentPath);
    }
    return OptionsSource;
  };
  for (llvm::StringRef CurrentPath = RootPath; !CurrentPath.empty();
       CurrentPath = llvm::sys::path::parent_path(CurrentPath)) {
    if (std::optional<OptionsSource> Result =
            MemorizedConfigFile(CurrentPath)) {
      CurOptions.emplace_back(Result.value());
      if (!Result->first.InheritParentConfig.value_or(false))
        break;
    }
  }
  // Reverse order of file configs because closer configs should have higher
  // priority.
  std::reverse(CurOptions.begin() + CurSize, CurOptions.end());
}

FileOptionsProvider::FileOptionsProvider(
    FlangTidyGlobalOptions GlobalOptions, FlangTidyOptions DefaultOptions,
    FlangTidyOptions OverrideOptions,
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS)
    : FileOptionsBaseProvider(std::move(GlobalOptions),
                              std::move(DefaultOptions),
                              std::move(OverrideOptions), std::move(VFS)) {}

FileOptionsProvider::FileOptionsProvider(
    FlangTidyGlobalOptions GlobalOptions, FlangTidyOptions DefaultOptions,
    FlangTidyOptions OverrideOptions,
    FileOptionsBaseProvider::ConfigFileHandlers ConfigHandlers)
    : FileOptionsBaseProvider(
          std::move(GlobalOptions), std::move(DefaultOptions),
          std::move(OverrideOptions), std::move(ConfigHandlers)) {}

std::vector<OptionsSource>
FileOptionsProvider::getRawOptions(llvm::StringRef FileName) {
  LLVM_DEBUG(llvm::dbgs() << "Getting options for file " << FileName
                          << "...\n");

  llvm::ErrorOr<llvm::SmallString<128>> AbsoluteFilePath =
      getNormalizedAbsolutePath(FileName);
  if (!AbsoluteFilePath)
    return {};

  std::vector<OptionsSource> RawOptions =
      DefaultOptionsProvider::getRawOptions(AbsoluteFilePath->str());
  addRawFileOptions(AbsoluteFilePath->str(), RawOptions);

  // Give command-line options very high priority so they override everything
  OptionsSource CommandLineOptions(OverrideOptions,
                                   OptionsSourceTypeCheckCommandLineOption);
  RawOptions.push_back(CommandLineOptions);
  return RawOptions;
}

std::optional<OptionsSource>
FileOptionsBaseProvider::tryReadConfigFile(llvm::StringRef Directory) {
  assert(!Directory.empty());

  llvm::ErrorOr<llvm::vfs::Status> DirectoryStatus = FS->status(Directory);

  if (!DirectoryStatus || !DirectoryStatus->isDirectory()) {
    llvm::errs() << "Error reading configuration from " << Directory
                 << ": directory doesn't exist.\n";
    return std::nullopt;
  }

  for (const ConfigFileHandler &ConfigHandler : ConfigHandlers) {
    llvm::SmallString<128> ConfigFile(Directory);
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

    // Skip empty files
    if ((*Text)->getBuffer().empty())
      continue;
    llvm::ErrorOr<FlangTidyOptions> ParsedOptions =
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

llvm::ErrorOr<FlangTidyOptions>
parseConfiguration(llvm::MemoryBufferRef Config) {
  llvm::yaml::Input Input(Config);
  FlangTidyOptions Options;
  Input >> Options;
  if (Input.error())
    return Input.error();

  // Parse the checks string into the enabledChecks vector
  Options.parseChecksString();
  Options.parseWarningsAsErrorsString();

  return Options;
}

std::string configurationAsText(const FlangTidyOptions &Options) {
  std::string Text;
  llvm::raw_string_ostream Stream(Text);
  llvm::yaml::Output Output(Stream);
  // We use the same mapping method for input and output, so we need a non-const
  // reference here.
  FlangTidyOptions NonConstValue = Options;
  Output << NonConstValue;
  return Stream.str();
}

// Simple function to get options for a file (for backward compatibility)
FlangTidyOptions getOptionsForFile(llvm::StringRef FileName) {
  FlangTidyGlobalOptions GlobalOptions;
  FlangTidyOptions DefaultOptions = FlangTidyOptions::getDefaults();
  FlangTidyOptions OverrideOptions;

  FileOptionsProvider Provider(std::move(GlobalOptions),
                               std::move(DefaultOptions),
                               std::move(OverrideOptions));
  return Provider.getOptions(FileName);
}

} // namespace Fortran::tidy
