//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
///  \file This file implements a clang-tidy tool.
///
///  This tool uses the Clang Tooling infrastructure, see
///    https://clang.llvm.org/docs/HowToSetupToolingForLLVM.html
///  for details on setting it up with LLVM source tree.
///
//===----------------------------------------------------------------------===//

#include "ClangTidyMain.h"
#include "../ClangTidy.h"
#include "../ClangTidyForceLinker.h" // IWYU pragma: keep
#include "../GlobList.h"
#include "clang/Basic/Version.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/CommandLine.h"
#define DONT_GET_PLUGIN_LOADER_OPTION
#include "llvm/Support/PluginLoader.h" // IWYU pragma: keep
#include "llvm/Support/Process.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/WithColor.h"
#include "llvm/TargetParser/Host.h"
#include <optional>

using namespace clang::tooling;
using namespace llvm;

namespace {
using namespace llvm::opt;

enum ID {
  OPT_INVALID = 0,
#define OPTION(...) LLVM_MAKE_OPT_ID(__VA_ARGS__),
#include "Opts.inc"
#undef OPTION
};

#define OPTTABLE_STR_TABLE_CODE
#include "Opts.inc"
#undef OPTTABLE_STR_TABLE_CODE

#define OPTTABLE_PREFIXES_TABLE_CODE
#include "Opts.inc"
#undef OPTTABLE_PREFIXES_TABLE_CODE

static constexpr opt::OptTable::Info InfoTable[] = {
#define OPTION(...) LLVM_CONSTRUCT_OPT_INFO(__VA_ARGS__),
#include "Opts.inc"
#undef OPTION
};

class ClangTidyOptTable : public opt::GenericOptTable {
public:
  ClangTidyOptTable()
      : opt::GenericOptTable(OptionStrTable, OptionPrefixesTable, InfoTable) {}
};

static constexpr llvm::StringLiteral CommonHelp = R"(
-p <build-path> is used to read a compile command database.

  For example, it can be a CMake build directory in which a file named
  compile_commands.json exists (use -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
  CMake option to get this output). When no build path is specified,
  a search for compile_commands.json will be attempted through all
  parent paths of the first input file. See:
  https://clang.llvm.org/docs/HowToSetupToolingForLLVM.html for an
  example of setting up Clang Tooling on a source tree.

<source0> ... specify the paths of source files. These paths are
  looked up in the compile command database. If the path of a file is
  absolute, it needs to point into CMake's source tree. If the path is
  relative, the current working directory needs to be in the CMake
  source tree and the file must be in a subdirectory of the current
  working directory. "./" prefixes in the relative files will be
  automatically removed, but the rest of a relative path must be a
  suffix of a path in the compile command database.
)";
static constexpr llvm::StringLiteral ClangTidyParameterFileHelp = R"(
Parameters files:
  A large number of options or source files can be passed as parameter files
  by use '@parameter-file' in the command line.
)";
static constexpr llvm::StringLiteral ClangTidyHelp = R"(
Configuration files:
  clang-tidy attempts to read configuration for each source file from a
  .clang-tidy file located in the closest parent directory of the source
  file. The .clang-tidy file is specified in YAML format. If any configuration
  options have a corresponding command-line option, command-line option takes
  precedence.

  The following configuration options may be used in a .clang-tidy file:

  CheckOptions                 - List of key-value pairs defining check-specific
                                 options. Example:
                                   CheckOptions:
                                     some-check.SomeOption: 'some value'
  Checks                       - Same as '--checks'. Additionally, the list of
                                 globs can be specified as a list instead of a
                                 string.
  CustomChecks                 - Array of user defined checks based on
                                 Clang-Query syntax.
  ExcludeHeaderFilterRegex     - Same as '--exclude-header-filter'.
  ExtraArgs                    - Same as '--extra-arg'.
  ExtraArgsBefore              - Same as '--extra-arg-before'.
  FormatStyle                  - Same as '--format-style'.
  HeaderFileExtensions         - File extensions to consider to determine if a
                                 given diagnostic is located in a header file.
  HeaderFilterRegex            - Same as '--header-filter'.
  ImplementationFileExtensions - File extensions to consider to determine if a
                                 given diagnostic is located in an
                                 implementation file.
  InheritParentConfig          - If this option is true in a config file, the
                                 configuration file in the parent directory
                                 (if any exists) will be taken and the current
                                 config file will be applied on top of the
                                 parent one.
  RemovedArgs                  - Same as '--removed-arg'.
  SystemHeaders                - Same as '--system-headers'.
  UseColor                     - Same as '--use-color'.
  User                         - Specifies the name or e-mail of the user
                                 running clang-tidy. This option is used, for
                                 example, to place the correct user name in
                                 TODO() comments in the relevant check.
  WarningsAsErrors             - Same as '--warnings-as-errors'.

  The effective configuration can be inspected using --dump-config:

    $ clang-tidy --dump-config
    ---
    Checks:                       '-*,some-check'
    WarningsAsErrors:             ''
    HeaderFileExtensions:         ['', 'h','hh','hpp','hxx']
    ImplementationFileExtensions: ['c','cc','cpp','cxx']
    HeaderFilterRegex:            '.*'
    FormatStyle:                  none
    InheritParentConfig:          true
    User:                         user
    CheckOptions:
      some-check.SomeOption: 'some value'
    ...

)";

const char DefaultChecks[] = // Enable these checks by default:
    "clang-diagnostic-*";    //   * compiler diagnostics

static std::string Checks;
static bool ChecksSpecified;
static std::string WarningsAsErrors;
static bool WarningsAsErrorsSpecified;
static std::string HeaderFilter;
static bool HeaderFilterSpecified;
static std::string ExcludeHeaderFilter;
static bool ExcludeHeaderFilterSpecified;
static bool SystemHeaders;
static bool SystemHeadersSpecified;
static std::string LineFilter;
static bool Fix;
static bool FixErrors;
static bool FixNotes;
static std::string FormatStyle;
static bool FormatStyleSpecified;
static bool ListChecks;
static bool ExplainConfig;
static std::string Config;
static bool ConfigSpecified;
static std::string ConfigFile;
static bool ConfigFileSpecified;
static bool DumpConfig;
static bool EnableCheckProfile;
static std::string StoreCheckProfile;
static bool AllowEnablingAnalyzerAlphaCheckers;
static bool EnableModuleHeadersParsing;
static std::string ExportFixes;
static bool Quiet;
static std::string VfsOverlay;
static bool UseColor;
static bool UseColorSpecified;
static bool VerifyConfig;
static bool AllowNoChecks;
static bool ExperimentalCustomChecks;
static std::vector<std::string> RemovedArgs;
static bool RemovedArgsSpecified;

static void printHelp(bool ShowHidden = false) {
  ClangTidyOptTable Tbl;
  Tbl.printHelp(outs(),
                "clang-tidy [options] <source0> [... <sourceN>] "
                "[-- <compiler arguments>]",
                "clang-tidy", ShowHidden);
  outs() << CommonHelp << ClangTidyParameterFileHelp << ClangTidyHelp;
}

static bool parseBoolArg(const opt::Arg *A, unsigned ValueID, bool &Value) {
  if (!A->getOption().matches(ValueID)) {
    Value = true;
    return true;
  }
  std::optional<bool> Parsed = StringSwitch<std::optional<bool>>(A->getValue())
                                   .CaseLower("true", true)
                                   .Case("1", true)
                                   .CaseLower("false", false)
                                   .Case("0", false)
                                   .Default(std::nullopt);
  if (Parsed) {
    Value = *Parsed;
    return true;
  }
  errs() << "clang-tidy: invalid value '" << A->getValue() << "' for option '"
         << A->getSpelling() << "'\n";
  return false;
}

static bool parseBoolArg(const opt::InputArgList &Args, unsigned FlagID,
                         unsigned ValueID, bool &Value,
                         bool *Specified = nullptr) {
  const opt::Arg *A = Args.getLastArg(FlagID, ValueID);
  if (Specified)
    *Specified = A != nullptr;
  Value = false;
  return !A || parseBoolArg(A, ValueID, Value);
}

static bool parseCommandLine(int argc, char **argv, BumpPtrAllocator &Allocator,
                             StringSaver &Saver, ClangTidyOptTable &Tbl,
                             opt::InputArgList &Args,
                             std::unique_ptr<CompilationDatabase> &Compilations,
                             std::vector<std::string> &SourcePaths) {
  SmallVector<const char *> ExpandedArgs(argv, argv + argc);
  cl::TokenizerCallback Tokenizer =
      Triple(sys::getProcessTriple()).isOSWindows()
          ? cl::TokenizeWindowsCommandLine
          : cl::TokenizeGNUCommandLine;
  cl::ExpansionContext ECtx(Allocator, Tokenizer);
  if (Error Err = ECtx.expandResponseFiles(ExpandedArgs)) {
    WithColor::error() << toString(std::move(Err)) << "\n";
    return false;
  }

  int ToolArgc = static_cast<int>(ExpandedArgs.size());
  std::string FixedDatabaseError;
  Compilations = FixedCompilationDatabase::loadFromCommandLine(
      ToolArgc, ExpandedArgs.data(), FixedDatabaseError);

  SmallVector<char *> ToolArgv;
  ToolArgv.reserve(ToolArgc);
  for (const char *Arg : ArrayRef(ExpandedArgs).take_front(ToolArgc))
    ToolArgv.push_back(const_cast<char *>(Arg));

  bool HasError = false;
  Args = Tbl.parseArgs(ToolArgc, ToolArgv.data(), OPT_UNKNOWN, Saver,
                       [&](StringRef Message) {
                         errs() << "clang-tidy: " << Message << '\n';
                         HasError = true;
                       });
  if (HasError) {
    if (!FixedDatabaseError.empty())
      errs() << FixedDatabaseError;
    return false;
  }

  Checks = Args.getLastArgValue(OPT_checks_EQ);
  ChecksSpecified = Args.hasArg(OPT_checks_EQ);
  WarningsAsErrors = Args.getLastArgValue(OPT_warnings_as_errors_EQ);
  WarningsAsErrorsSpecified = Args.hasArg(OPT_warnings_as_errors_EQ);
  HeaderFilter = Args.getLastArgValue(OPT_header_filter_EQ, ".*");
  HeaderFilterSpecified = Args.hasArg(OPT_header_filter_EQ);
  ExcludeHeaderFilter = Args.getLastArgValue(OPT_exclude_header_filter_EQ);
  ExcludeHeaderFilterSpecified = Args.hasArg(OPT_exclude_header_filter_EQ);
  LineFilter = Args.getLastArgValue(OPT_line_filter_EQ);
  FormatStyle = Args.getLastArgValue(OPT_format_style_EQ, "none");
  FormatStyleSpecified = Args.hasArg(OPT_format_style_EQ);
  Config = Args.getLastArgValue(OPT_config_EQ);
  ConfigSpecified = Args.hasArg(OPT_config_EQ);
  ConfigFile = Args.getLastArgValue(OPT_config_file_EQ);
  ConfigFileSpecified = Args.hasArg(OPT_config_file_EQ);
  StoreCheckProfile = Args.getLastArgValue(OPT_store_check_profile_EQ);
  ExportFixes = Args.getLastArgValue(OPT_export_fixes_EQ);
  VfsOverlay = Args.getLastArgValue(OPT_vfsoverlay_EQ);
  RemovedArgs = Args.getAllArgValues(OPT_removed_arg_EQ);
  RemovedArgsSpecified = Args.hasArg(OPT_removed_arg_EQ);
  SourcePaths = Args.getAllArgValues(OPT_INPUT);

  if (!parseBoolArg(Args, OPT_system_headers, OPT_system_headers_EQ,
                    SystemHeaders, &SystemHeadersSpecified) ||
      !parseBoolArg(Args, OPT_fix, OPT_fix_EQ, Fix) ||
      !parseBoolArg(Args, OPT_fix_errors, OPT_fix_errors_EQ, FixErrors) ||
      !parseBoolArg(Args, OPT_fix_notes, OPT_fix_notes_EQ, FixNotes) ||
      !parseBoolArg(Args, OPT_list_checks, OPT_list_checks_EQ, ListChecks) ||
      !parseBoolArg(Args, OPT_explain_config, OPT_explain_config_EQ,
                    ExplainConfig) ||
      !parseBoolArg(Args, OPT_dump_config, OPT_dump_config_EQ, DumpConfig) ||
      !parseBoolArg(Args, OPT_enable_check_profile, OPT_enable_check_profile_EQ,
                    EnableCheckProfile) ||
      !parseBoolArg(Args, OPT_allow_enabling_analyzer_alpha_checkers,
                    OPT_allow_enabling_analyzer_alpha_checkers_EQ,
                    AllowEnablingAnalyzerAlphaCheckers) ||
      !parseBoolArg(Args, OPT_enable_module_headers_parsing,
                    OPT_enable_module_headers_parsing_EQ,
                    EnableModuleHeadersParsing) ||
      !parseBoolArg(Args, OPT_quiet, OPT_quiet_EQ, Quiet) ||
      !parseBoolArg(Args, OPT_use_color, OPT_use_color_EQ, UseColor,
                    &UseColorSpecified) ||
      !parseBoolArg(Args, OPT_verify_config, OPT_verify_config_EQ,
                    VerifyConfig) ||
      !parseBoolArg(Args, OPT_allow_no_checks, OPT_allow_no_checks_EQ,
                    AllowNoChecks) ||
      !parseBoolArg(Args, OPT_experimental_custom_checks,
                    OPT_experimental_custom_checks_EQ,
                    ExperimentalCustomChecks))
    return false;

  for (StringRef Plugin : Args.getAllArgValues(OPT_load_EQ)) {
    PluginLoader Loader;
    Loader = Plugin.str();
  }

  if (SourcePaths.empty()) {
    if (!Compilations)
      Compilations = std::make_unique<FixedCompilationDatabase>(
          ".", std::vector<std::string>());
    return true;
  }

  if (!Compilations) {
    std::string ErrorMessage;
    StringRef BuildPath = Args.getLastArgValue(OPT_p);
    if (!BuildPath.empty())
      Compilations =
          CompilationDatabase::autoDetectFromDirectory(BuildPath, ErrorMessage);
    else
      Compilations = CompilationDatabase::autoDetectFromSource(
          SourcePaths.front(), ErrorMessage);
    if (!Compilations) {
      errs() << "Error while trying to load a compilation database:\n"
             << ErrorMessage << "Running without flags.\n";
      Compilations = std::make_unique<FixedCompilationDatabase>(
          ".", std::vector<std::string>());
    }
  }

  auto AdjustingCompilations =
      std::make_unique<ArgumentsAdjustingCompilations>(std::move(Compilations));
  ArgumentsAdjuster Adjuster =
      getInsertArgumentAdjuster(Args.getAllArgValues(OPT_extra_arg_before_EQ),
                                ArgumentInsertPosition::BEGIN);
  Adjuster = combineAdjusters(
      std::move(Adjuster),
      getInsertArgumentAdjuster(Args.getAllArgValues(OPT_extra_arg_EQ),
                                ArgumentInsertPosition::END));
  AdjustingCompilations->appendArgumentsAdjuster(std::move(Adjuster));
  Compilations = std::move(AdjustingCompilations);
  return true;
}

} // namespace

namespace clang::tidy {

static void printStats(const ClangTidyStats &Stats) {
  if (Stats.errorsIgnored()) {
    llvm::errs() << "Suppressed " << Stats.errorsIgnored() << " warnings (";
    StringRef Separator = "";
    if (Stats.ErrorsIgnoredNonUserCode) {
      llvm::errs() << Stats.ErrorsIgnoredNonUserCode << " in non-user code";
      Separator = ", ";
    }
    if (Stats.ErrorsIgnoredLineFilter) {
      llvm::errs() << Separator << Stats.ErrorsIgnoredLineFilter
                   << " due to line filter";
      Separator = ", ";
    }
    if (Stats.ErrorsIgnoredNOLINT) {
      llvm::errs() << Separator << Stats.ErrorsIgnoredNOLINT << " NOLINT";
      Separator = ", ";
    }
    if (Stats.ErrorsIgnoredCheckFilter)
      llvm::errs() << Separator << Stats.ErrorsIgnoredCheckFilter
                   << " with check filters";
    llvm::errs() << ").\n";
    if (Stats.ErrorsIgnoredNonUserCode)
      llvm::errs() << "Use -header-filter=.* or leave it as default to display "
                      "errors from all non-system headers. Use -system-headers "
                      "to display errors from system headers as well.\n";
  }
}

static std::unique_ptr<ClangTidyOptionsProvider>
createOptionsProvider(llvm::IntrusiveRefCntPtr<vfs::FileSystem> FS) {
  ClangTidyGlobalOptions GlobalOptions;
  if (const std::error_code Err = parseLineFilter(LineFilter, GlobalOptions)) {
    llvm::errs() << "Invalid LineFilter: " << Err.message() << "\n\nUsage:\n";
    printHelp();
    return nullptr;
  }

  ClangTidyOptions DefaultOptions;
  DefaultOptions.Checks = DefaultChecks;
  DefaultOptions.WarningsAsErrors = "";
  DefaultOptions.HeaderFilterRegex = HeaderFilter;
  DefaultOptions.ExcludeHeaderFilterRegex = ExcludeHeaderFilter;
  DefaultOptions.SystemHeaders = SystemHeaders;
  DefaultOptions.FormatStyle = FormatStyle;
  DefaultOptions.User = llvm::sys::Process::GetEnv("USER");
  // USERNAME is used on Windows.
  if (!DefaultOptions.User)
    DefaultOptions.User = llvm::sys::Process::GetEnv("USERNAME");

  ClangTidyOptions OverrideOptions;
  if (ChecksSpecified)
    OverrideOptions.Checks = Checks;
  if (WarningsAsErrorsSpecified)
    OverrideOptions.WarningsAsErrors = WarningsAsErrors;
  if (HeaderFilterSpecified)
    OverrideOptions.HeaderFilterRegex = HeaderFilter;
  if (ExcludeHeaderFilterSpecified)
    OverrideOptions.ExcludeHeaderFilterRegex = ExcludeHeaderFilter;
  if (SystemHeadersSpecified)
    OverrideOptions.SystemHeaders = SystemHeaders;
  if (FormatStyleSpecified)
    OverrideOptions.FormatStyle = FormatStyle;
  if (UseColorSpecified)
    OverrideOptions.UseColor = UseColor;
  if (RemovedArgsSpecified)
    OverrideOptions.RemovedArgs = RemovedArgs;

  auto LoadConfig =
      [&](StringRef Configuration,
          StringRef Source) -> std::unique_ptr<ClangTidyOptionsProvider> {
    llvm::ErrorOr<ClangTidyOptions> ParsedConfig =
        parseConfiguration(MemoryBufferRef(Configuration, Source));
    if (ParsedConfig)
      return std::make_unique<ConfigOptionsProvider>(
          std::move(GlobalOptions),
          ClangTidyOptions::getDefaults().merge(DefaultOptions, 0),
          std::move(*ParsedConfig), std::move(OverrideOptions), std::move(FS));
    llvm::errs() << "Error: invalid configuration specified.\n"
                 << ParsedConfig.getError().message() << "\n";
    return nullptr;
  };

  if (ConfigFileSpecified) {
    if (ConfigSpecified) {
      llvm::errs() << "Error: --config-file and --config are "
                      "mutually exclusive. Specify only one.\n";
      return nullptr;
    }

    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> Text =
        llvm::MemoryBuffer::getFile(ConfigFile);
    if (const std::error_code EC = Text.getError()) {
      llvm::errs() << "Error: can't read config-file '" << ConfigFile
                   << "': " << EC.message() << "\n";
      return nullptr;
    }

    return LoadConfig((*Text)->getBuffer(), ConfigFile);
  }

  if (ConfigSpecified)
    return LoadConfig(Config, "<command-line-config>");

  return std::make_unique<FileOptionsProvider>(
      std::move(GlobalOptions), std::move(DefaultOptions),
      std::move(OverrideOptions), std::move(FS));
}

static llvm::IntrusiveRefCntPtr<vfs::FileSystem>
getVfsFromFile(const std::string &OverlayFile, vfs::FileSystem &BaseFS) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> Buffer =
      BaseFS.getBufferForFile(OverlayFile);
  if (!Buffer) {
    llvm::errs() << "Can't load virtual filesystem overlay file '"
                 << OverlayFile << "': " << Buffer.getError().message()
                 << ".\n";
    return nullptr;
  }

  IntrusiveRefCntPtr<vfs::FileSystem> FS = vfs::getVFSFromYAML(
      std::move(Buffer.get()), /*DiagHandler*/ nullptr, OverlayFile);
  if (!FS) {
    llvm::errs() << "Error: invalid virtual filesystem overlay file '"
                 << OverlayFile << "'.\n";
    return nullptr;
  }
  return FS;
}

static StringRef closest(StringRef Value, const StringSet<> &Allowed) {
  unsigned MaxEdit = 5U;
  StringRef Closest;
  for (auto Item : Allowed.keys()) {
    const unsigned Cur = Value.edit_distance_insensitive(Item, true, MaxEdit);
    if (Cur < MaxEdit) {
      Closest = Item;
      MaxEdit = Cur;
    }
  }
  return Closest;
}

static constexpr llvm::StringLiteral VerifyConfigWarningEnd =
    " [-verify-config]\n";

static bool verifyChecks(const StringSet<> &AllChecks, StringRef CheckGlob,
                         StringRef Source) {
  const GlobList Globs(CheckGlob);
  bool AnyInvalid = false;
  for (const auto &Item : Globs.getItems()) {
    if (Item.Text.starts_with("clang-diagnostic"))
      continue;
    if (llvm::none_of(AllChecks.keys(),
                      [&Item](StringRef S) { return Item.Regex.match(S); })) {
      AnyInvalid = true;
      if (Item.Text.contains('*')) {
        llvm::WithColor::warning(llvm::errs(), Source)
            << "check glob '" << Item.Text << "' doesn't match any known check"
            << VerifyConfigWarningEnd;
      } else {
        llvm::raw_ostream &Output =
            llvm::WithColor::warning(llvm::errs(), Source)
            << "unknown check '" << Item.Text << '\'';
        const StringRef Closest = closest(Item.Text, AllChecks);
        if (!Closest.empty())
          Output << "; did you mean '" << Closest << '\'';
        Output << VerifyConfigWarningEnd;
      }
    }
  }
  return AnyInvalid;
}

static bool verifyFileExtensions(
    const std::vector<std::string> &HeaderFileExtensions,
    const std::vector<std::string> &ImplementationFileExtensions,
    StringRef Source) {
  bool AnyInvalid = false;
  for (const auto &HeaderExtension : HeaderFileExtensions) {
    for (const auto &ImplementationExtension : ImplementationFileExtensions) {
      if (HeaderExtension == ImplementationExtension) {
        AnyInvalid = true;
        auto &Output = llvm::WithColor::warning(llvm::errs(), Source)
                       << "HeaderFileExtension '" << HeaderExtension << '\''
                       << " is the same as ImplementationFileExtension '"
                       << ImplementationExtension << '\'';
        Output << VerifyConfigWarningEnd;
      }
    }
  }
  return AnyInvalid;
}

static bool verifyOptions(const llvm::StringSet<> &ValidOptions,
                          const ClangTidyOptions::OptionMap &OptionMap,
                          StringRef Source) {
  bool AnyInvalid = false;
  for (auto Key : OptionMap.keys()) {
    if (ValidOptions.contains(Key))
      continue;
    AnyInvalid = true;
    auto &Output = llvm::WithColor::warning(llvm::errs(), Source)
                   << "unknown check option '" << Key << '\'';
    const StringRef Closest = closest(Key, ValidOptions);
    if (!Closest.empty())
      Output << "; did you mean '" << Closest << '\'';
    Output << VerifyConfigWarningEnd;
  }
  return AnyInvalid;
}

static SmallString<256> makeAbsolute(StringRef Input) {
  if (Input.empty())
    return {};
  SmallString<256> AbsolutePath(Input);
  if (const std::error_code EC = llvm::sys::fs::make_absolute(AbsolutePath)) {
    llvm::errs() << "Can't make absolute path from " << Input << ": "
                 << EC.message() << "\n";
  }
  return AbsolutePath;
}

static llvm::IntrusiveRefCntPtr<vfs::OverlayFileSystem> createBaseFS() {
  llvm::IntrusiveRefCntPtr<vfs::OverlayFileSystem> BaseFS(
      new vfs::OverlayFileSystem(vfs::getRealFileSystem()));

  if (!VfsOverlay.empty()) {
    IntrusiveRefCntPtr<vfs::FileSystem> VfsFromFile =
        getVfsFromFile(VfsOverlay, *BaseFS);
    if (!VfsFromFile)
      return nullptr;
    BaseFS->pushOverlay(std::move(VfsFromFile));
  }
  return BaseFS;
}

int clangTidyMain(int argc, char **argv) {
  BumpPtrAllocator Alloc;
  StringSaver Saver(Alloc);
  ClangTidyOptTable Tbl;
  opt::InputArgList Args;
  std::unique_ptr<CompilationDatabase> Compilations;
  std::vector<std::string> PathList;
  if (!parseCommandLine(argc, argv, Alloc, Saver, Tbl, Args, Compilations,
                        PathList))
    return 1;

  if (Args.hasArg(OPT_help, OPT_help_hidden)) {
    printHelp(Args.hasArg(OPT_help_hidden));
    return 0;
  }

  if (Args.hasArg(OPT_version)) {
    outs() << clang::getClangToolFullVersion("clang-tidy") << '\n';
    return 0;
  }

  const llvm::IntrusiveRefCntPtr<vfs::OverlayFileSystem> BaseFS =
      createBaseFS();
  if (!BaseFS)
    return 1;

  auto OwningOptionsProvider = createOptionsProvider(BaseFS);
  auto *OptionsProvider = OwningOptionsProvider.get();
  if (!OptionsProvider)
    return 1;

  const SmallString<256> ProfilePrefix = makeAbsolute(StoreCheckProfile);

  StringRef FileName("dummy");
  if (!PathList.empty())
    FileName = PathList.front();

  const SmallString<256> FilePath = makeAbsolute(FileName);
  ClangTidyOptions EffectiveOptions = OptionsProvider->getOptions(FilePath);

  const std::vector<std::string> EnabledChecks =
      getCheckNames(EffectiveOptions, AllowEnablingAnalyzerAlphaCheckers,
                    ExperimentalCustomChecks);

  if (ExplainConfig) {
    // FIXME: Show other ClangTidyOptions' fields, like ExtraArg.
    std::vector<ClangTidyOptionsProvider::OptionsSource> RawOptions =
        OptionsProvider->getRawOptions(FilePath);
    for (const std::string &Check : EnabledChecks) {
      for (const auto &[Opts, Source] : llvm::reverse(RawOptions)) {
        if (Opts.Checks && GlobList(*Opts.Checks).contains(Check)) {
          llvm::outs() << "'" << Check << "' is enabled in the " << Source
                       << ".\n";
          break;
        }
      }
    }
    return 0;
  }

  if (ListChecks) {
    if (EnabledChecks.empty() && !AllowNoChecks) {
      llvm::errs() << "No checks enabled.\n";
      return 1;
    }
    llvm::outs() << "Enabled checks:";
    for (const auto &CheckName : EnabledChecks)
      llvm::outs() << "\n    " << CheckName;
    llvm::outs() << "\n\n";
    return 0;
  }

  if (DumpConfig) {
    EffectiveOptions.CheckOptions =
        getCheckOptions(EffectiveOptions, AllowEnablingAnalyzerAlphaCheckers,
                        ExperimentalCustomChecks);
    ClangTidyOptions OptionsToDump =
        ClangTidyOptions::getDefaults().merge(EffectiveOptions, 0);
    filterCheckOptions(OptionsToDump, EnabledChecks);
    llvm::outs() << configurationAsText(OptionsToDump) << "\n";
    return 0;
  }

  if (VerifyConfig) {
    const std::vector<ClangTidyOptionsProvider::OptionsSource> RawOptions =
        OptionsProvider->getRawOptions(FileName);
    const ChecksAndOptions Valid = getAllChecksAndOptions(
        AllowEnablingAnalyzerAlphaCheckers, ExperimentalCustomChecks);
    bool AnyInvalid = false;
    for (const auto &[Opts, Source] : RawOptions) {
      if (Opts.Checks)
        AnyInvalid |= verifyChecks(Valid.Checks, *Opts.Checks, Source);
      if (Opts.HeaderFileExtensions && Opts.ImplementationFileExtensions)
        AnyInvalid |=
            verifyFileExtensions(*Opts.HeaderFileExtensions,
                                 *Opts.ImplementationFileExtensions, Source);
      AnyInvalid |= verifyOptions(Valid.Options, Opts.CheckOptions, Source);
    }
    if (AnyInvalid)
      return 1;
    llvm::outs() << "No config errors detected.\n";
    return 0;
  }

  if (EnabledChecks.empty() && !AllowNoChecks) {
    llvm::errs() << "Error: no checks enabled.\n";
    printHelp();
    return 1;
  }

  if (PathList.empty()) {
    llvm::errs() << "Error: no input files specified.\n";
    printHelp();
    return 1;
  }

  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();

  ClangTidyContext Context(
      std::move(OwningOptionsProvider), AllowEnablingAnalyzerAlphaCheckers,
      EnableModuleHeadersParsing, ExperimentalCustomChecks);
  std::vector<ClangTidyError> Errors =
      runClangTidy(Context, *Compilations, PathList, BaseFS, FixNotes,
                   EnableCheckProfile, ProfilePrefix, Quiet);
  const bool FoundErrors = llvm::any_of(Errors, [](const ClangTidyError &E) {
    return E.DiagLevel == ClangTidyError::Error;
  });

  // --fix-errors and --fix-notes imply --fix.
  const FixBehaviour Behaviour = FixNotes             ? FB_FixNotes
                                 : (Fix || FixErrors) ? FB_Fix
                                                      : FB_NoFix;

  const bool DisableFixes = FoundErrors && !FixErrors;

  unsigned WErrorCount = 0;

  handleErrors(Errors, Context, DisableFixes ? FB_NoFix : Behaviour,
               WErrorCount, BaseFS);

  if (!ExportFixes.empty() && !Errors.empty()) {
    std::error_code EC;
    llvm::raw_fd_ostream OS(ExportFixes, EC, llvm::sys::fs::OF_None);
    if (EC) {
      llvm::errs() << "Error opening output file: " << EC.message() << '\n';
      return 1;
    }
    exportReplacements(FilePath.str(), Errors, OS);
  }

  if (!Quiet) {
    printStats(Context.getStats());
    if (DisableFixes && Behaviour != FB_NoFix)
      llvm::errs()
          << "Found compiler errors, but -fix-errors was not specified.\n"
             "Fixes have NOT been applied.\n\n";
  }

  if (WErrorCount) {
    if (!Quiet) {
      const StringRef Plural = WErrorCount == 1 ? "" : "s";
      llvm::errs() << WErrorCount << " warning" << Plural << " treated as error"
                   << Plural << "\n";
    }
    return 1;
  }

  if (FoundErrors) {
    // TODO: Figure out when zero exit code should be used with -fix-errors:
    //   a. when a fix has been applied for an error
    //   b. when a fix has been applied for all errors
    //   c. some other condition.
    // For now always returning zero when -fix-errors is used.
    if (FixErrors)
      return 0;
    if (!Quiet)
      llvm::errs() << "Found compiler error(s).\n";
    return 1;
  }

  return 0;
}

} // namespace clang::tidy
