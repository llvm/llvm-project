//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ClangTidyOptions.h"
#include "ClangTidyModuleRegistry.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/YAMLTraits.h"
#include <algorithm>
#include <optional>
#include <utility>

#define DEBUG_TYPE "clang-tidy-options"

using clang::tidy::ClangTidyOptions;
using clang::tidy::FileFilter;
using OptionsSource = clang::tidy::ClangTidyOptionsProvider::OptionsSource;

LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(FileFilter)
LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(FileFilter::LineRange)

namespace llvm::yaml {

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
      Options.emplace_back(std::string(KeyValue.getKey()),
                           KeyValue.getValue().Value);
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
void yamlize(IO &IO, ClangTidyOptions::OptionMap &Val, bool,
             EmptyContext &Ctx) {
  if (IO.outputting()) {
    // Ensure check options are sorted
    std::vector<std::pair<StringRef, StringRef>> SortedOptions;
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
      // Requires 'llvm::yaml::IO' to accept 'StringRef'
      // NOLINTNEXTLINE(bugprone-suspicious-stringview-data-usage)
      IO.preflightKey(Option.first.data(), true, false, UseDefault, SaveInfo);
      IO.scalarString(Option.second, needsQuotes(Option.second));
      IO.postflightKey(SaveInfo);
    }
    IO.endMapping();
  } else {
    // We need custom logic here to support the old method of specifying check
    // options using a list of maps containing key and value keys.
    auto &I = reinterpret_cast<Input &>(IO);
    if (isa<SequenceNode>(I.getCurrentNode())) {
      MappingNormalization<NOptionMap, ClangTidyOptions::OptionMap> NOpts(IO,
                                                                          Val);
      EmptyContext Ctx;
      yamlize(IO, NOpts->Options, true, Ctx);
    } else if (isa<MappingNode>(I.getCurrentNode())) {
      IO.beginMapping();
      for (StringRef Key : IO.keys()) {
        // Requires 'llvm::yaml::IO' to accept 'StringRef'
        // NOLINTNEXTLINE(bugprone-suspicious-stringview-data-usage)
        IO.mapRequired(Key.data(), Val[Key].Value);
      }
      IO.endMapping();
    } else {
      IO.setError("expected a sequence or map");
    }
  }
}

namespace {
struct MultiLineString {
  std::string &S;
};
} // namespace

template <> struct BlockScalarTraits<MultiLineString> {
  static void output(const MultiLineString &S, void *Ctxt, raw_ostream &OS) {
    OS << S.S;
  }
  static StringRef input(StringRef Str, void *Ctxt, MultiLineString &S) {
    S.S = Str;
    return "";
  }
};

template <> struct ScalarEnumerationTraits<clang::DiagnosticIDs::Level> {
  static void enumeration(IO &IO, clang::DiagnosticIDs::Level &Level) {
    IO.enumCase(Level, "Warning", clang::DiagnosticIDs::Level::Warning);
    IO.enumCase(Level, "Note", clang::DiagnosticIDs::Level::Note);
  }
};
template <> struct SequenceElementTraits<ClangTidyOptions::CustomCheckDiag> {
  // NOLINTNEXTLINE(readability-identifier-naming) Defined by YAMLTraits.h
  static const bool flow = false;
};
template <> struct MappingTraits<ClangTidyOptions::CustomCheckDiag> {
  static void mapping(IO &IO, ClangTidyOptions::CustomCheckDiag &D) {
    IO.mapRequired("BindName", D.BindName);
    MultiLineString MLS{D.Message};
    IO.mapRequired("Message", MLS);
    IO.mapOptional("Level", D.Level);
  }
};
template <> struct SequenceElementTraits<ClangTidyOptions::CustomCheckValue> {
  // NOLINTNEXTLINE(readability-identifier-naming) Defined by YAMLTraits.h
  static const bool flow = false;
};
template <> struct MappingTraits<ClangTidyOptions::CustomCheckValue> {
  static void mapping(IO &IO, ClangTidyOptions::CustomCheckValue &V) {
    IO.mapRequired("Name", V.Name);
    MultiLineString MLS{V.Query};
    IO.mapRequired("Query", MLS);
    IO.mapRequired("Diagnostic", V.Diags);
  }
};

struct ChecksVariant {
  std::optional<std::string> AsString;
  std::optional<std::vector<std::string>> AsVector;
};

template <> void yamlize(IO &IO, ChecksVariant &Val, bool, EmptyContext &Ctx) {
  if (!IO.outputting()) {
    // Special case for reading from YAML
    // Must support reading from both a string or a list
    auto &I = reinterpret_cast<Input &>(IO);
    if (isa<ScalarNode, BlockScalarNode>(I.getCurrentNode())) {
      Val.AsString = std::string();
      yamlize(IO, *Val.AsString, true, Ctx);
    } else if (isa<SequenceNode>(I.getCurrentNode())) {
      Val.AsVector = std::vector<std::string>();
      yamlize(IO, *Val.AsVector, true, Ctx);
    } else {
      IO.setError("expected string or sequence");
    }
  }
}

static void mapChecks(IO &IO, std::optional<std::string> &Checks) {
  if (IO.outputting()) {
    // Output always a string
    IO.mapOptional("Checks", Checks);
  } else {
    // Input as either a string or a list
    ChecksVariant ChecksAsVariant;
    IO.mapOptional("Checks", ChecksAsVariant);
    if (ChecksAsVariant.AsString)
      Checks = ChecksAsVariant.AsString;
    else if (ChecksAsVariant.AsVector)
      Checks = llvm::join(*ChecksAsVariant.AsVector, ",");
  }
}

template <> struct MappingTraits<ClangTidyOptions> {
  static void mapping(IO &IO, ClangTidyOptions &Options) {
    mapChecks(IO, Options.Checks);
    IO.mapOptional("WarningsAsErrors", Options.WarningsAsErrors);
    IO.mapOptional("HeaderFileExtensions", Options.HeaderFileExtensions);
    IO.mapOptional("ImplementationFileExtensions",
                   Options.ImplementationFileExtensions);
    IO.mapOptional("HeaderFilterRegex", Options.HeaderFilterRegex);
    IO.mapOptional("ExcludeHeaderFilterRegex",
                   Options.ExcludeHeaderFilterRegex);
    IO.mapOptional("FormatStyle", Options.FormatStyle);
    IO.mapOptional("User", Options.User);
    IO.mapOptional("CheckOptions", Options.CheckOptions);
    IO.mapOptional("ExtraArgs", Options.ExtraArgs);
    IO.mapOptional("ExtraArgsBefore", Options.ExtraArgsBefore);
    IO.mapOptional("InheritParentConfig", Options.InheritParentConfig);
    IO.mapOptional("UseColor", Options.UseColor);
    IO.mapOptional("SystemHeaders", Options.SystemHeaders);
    IO.mapOptional("CustomChecks", Options.CustomChecks);
  }
};

} // namespace llvm::yaml

namespace clang::tidy {

ClangTidyOptions ClangTidyOptions::getDefaults() {
  ClangTidyOptions Options;
  Options.Checks = "";
  Options.WarningsAsErrors = "";
  Options.HeaderFileExtensions = {"", "h", "hh", "hpp", "hxx"};
  Options.ImplementationFileExtensions = {"c", "cc", "cpp", "cxx"};
  Options.HeaderFilterRegex = "";
  Options.ExcludeHeaderFilterRegex = "";
  Options.SystemHeaders = false;
  Options.FormatStyle = "none";
  Options.User = std::nullopt;
  for (const ClangTidyModuleRegistry::entry &Module :
       ClangTidyModuleRegistry::entries())
    Options.mergeWith(Module.instantiate()->getModuleOptions(), 0);
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

static void mergeCommaSeparatedLists(std::optional<std::string> &Dest,
                                     const std::optional<std::string> &Src) {
  if (Src)
    Dest = (Dest && !Dest->empty() ? *Dest + "," : "") + *Src;
}

template <typename T>
static void overrideValue(std::optional<T> &Dest, const std::optional<T> &Src) {
  if (Src)
    Dest = Src;
}

ClangTidyOptions &ClangTidyOptions::mergeWith(const ClangTidyOptions &Other,
                                              unsigned Order) {
  mergeCommaSeparatedLists(Checks, Other.Checks);
  mergeCommaSeparatedLists(WarningsAsErrors, Other.WarningsAsErrors);
  overrideValue(HeaderFileExtensions, Other.HeaderFileExtensions);
  overrideValue(ImplementationFileExtensions,
                Other.ImplementationFileExtensions);
  overrideValue(HeaderFilterRegex, Other.HeaderFilterRegex);
  overrideValue(ExcludeHeaderFilterRegex, Other.ExcludeHeaderFilterRegex);
  overrideValue(SystemHeaders, Other.SystemHeaders);
  overrideValue(FormatStyle, Other.FormatStyle);
  overrideValue(User, Other.User);
  overrideValue(UseColor, Other.UseColor);
  mergeVectors(ExtraArgs, Other.ExtraArgs);
  mergeVectors(ExtraArgsBefore, Other.ExtraArgsBefore);
  // FIXME: how to handle duplicate names check?
  mergeVectors(CustomChecks, Other.CustomChecks);
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

    llvm::ErrorOr<llvm::SmallString<128>> AbsoluteFilePath =
        getNormalizedAbsolutePath(FileName);
    if (AbsoluteFilePath) {
      addRawFileOptions(AbsoluteFilePath->str(), RawOptions);
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
  // Look for a suitable configuration file in all parent directories of the
  // file. Start with the immediate parent directory and move up.
  StringRef RootPath = llvm::sys::path::parent_path(AbsolutePath);
  auto MemorizedConfigFile =
      [this, &RootPath](StringRef CurrentPath) -> std::optional<OptionsSource> {
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
  for (StringRef CurrentPath = RootPath; !CurrentPath.empty();
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

  llvm::ErrorOr<llvm::SmallString<128>> AbsoluteFilePath =
      getNormalizedAbsolutePath(FileName);
  if (!AbsoluteFilePath)
    return {};

  std::vector<OptionsSource> RawOptions =
      DefaultOptionsProvider::getRawOptions(AbsoluteFilePath->str());
  addRawFileOptions(AbsoluteFilePath->str(), RawOptions);
  OptionsSource CommandLineOptions(OverrideOptions,
                                   OptionsSourceTypeCheckCommandLineOption);

  RawOptions.push_back(CommandLineOptions);
  return RawOptions;
}

std::optional<OptionsSource>
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

} // namespace clang::tidy
