//===--- tools/extra/clang-tidy/ClangTidyMain.cpp - Clang tidy tool -------===//
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
///    http://clang.llvm.org/docs/HowToSetupToolingForLLVM.html
///  for details on setting it up with LLVM source tree.
///
//===----------------------------------------------------------------------===//

#include "ClangTidyMain.h"
#include "../ClangTidy.h"
#include "../ClangTidyForceLinker.h"
#include "../GlobList.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/PluginLoader.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/WithColor.h"
#include <optional>

using namespace clang::tooling;
using namespace llvm;

static cl::OptionCategory ClangTidyCategory("clang-tidy options");

static cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);
static cl::extrahelp ClangTidyHelp(R"(
Configuration files:
  clang-tidy attempts to read configuration for each source file from a
  .clang-tidy file located in the closest parent directory of the source
  file. If InheritParentConfig is true in a config file, the configuration file
  in the parent directory (if any exists) will be taken and current config file
  will be applied on top of the parent one. If any configuration options have
  a corresponding command-line option, command-line option takes precedence.
  The effective configuration can be inspected using -dump-config:

    $ clang-tidy -dump-config
    ---
    Checks:                       '-*,some-check'
    WarningsAsErrors:             ''
    HeaderFileExtensions:         ['', 'h','hh','hpp','hxx']
    ImplementationFileExtensions: ['c','cc','cpp','cxx']
    HeaderFilterRegex:            ''
    FormatStyle:                  none
    InheritParentConfig:          true
    User:                         user
    CheckOptions:
      some-check.SomeOption: 'some value'
    ...

)");

const char DefaultChecks[] = // Enable these checks by default:
    "clang-diagnostic-*,"    //   * compiler diagnostics
    "clang-analyzer-*";      //   * Static Analyzer checks

static cl::opt<std::string> Checks("checks", cl::desc(R"(
Comma-separated list of globs with optional '-'
prefix. Globs are processed in order of
appearance in the list. Globs without '-'
prefix add checks with matching names to the
set, globs with the '-' prefix remove checks
with matching names from the set of enabled
checks. This option's value is appended to the
value of the 'Checks' option in .clang-tidy
file, if any.
)"),
                                   cl::init(""), cl::cat(ClangTidyCategory));

static cl::opt<std::string> WarningsAsErrors("warnings-as-errors", cl::desc(R"(
Upgrades warnings to errors. Same format as
'-checks'.
This option's value is appended to the value of
the 'WarningsAsErrors' option in .clang-tidy
file, if any.
)"),
                                             cl::init(""),
                                             cl::cat(ClangTidyCategory));

static cl::opt<std::string> HeaderFilter("header-filter", cl::desc(R"(
Regular expression matching the names of the
headers to output diagnostics from. Diagnostics
from the main file of each translation unit are
always displayed.
Can be used together with -line-filter.
This option overrides the 'HeaderFilterRegex'
option in .clang-tidy file, if any.
)"),
                                         cl::init(""),
                                         cl::cat(ClangTidyCategory));

static cl::opt<bool>
    SystemHeaders("system-headers",
                  cl::desc("Display the errors from system headers."),
                  cl::init(false), cl::cat(ClangTidyCategory));
static cl::opt<std::string> LineFilter("line-filter", cl::desc(R"(
List of files with line ranges to filter the
warnings. Can be used together with
-header-filter. The format of the list is a
JSON array of objects:
  [
    {"name":"file1.cpp","lines":[[1,3],[5,7]]},
    {"name":"file2.h"}
  ]
)"),
                                       cl::init(""),
                                       cl::cat(ClangTidyCategory));

static cl::opt<bool> Fix("fix", cl::desc(R"(
Apply suggested fixes. Without -fix-errors
clang-tidy will bail out if any compilation
errors were found.
)"),
                         cl::init(false), cl::cat(ClangTidyCategory));

static cl::opt<bool> FixErrors("fix-errors", cl::desc(R"(
Apply suggested fixes even if compilation
errors were found. If compiler errors have
attached fix-its, clang-tidy will apply them as
well.
)"),
                               cl::init(false), cl::cat(ClangTidyCategory));

static cl::opt<bool> FixNotes("fix-notes", cl::desc(R"(
If a warning has no fix, but a single fix can
be found through an associated diagnostic note,
apply the fix.
Specifying this flag will implicitly enable the
'--fix' flag.
)"),
                              cl::init(false), cl::cat(ClangTidyCategory));

static cl::opt<std::string> FormatStyle("format-style", cl::desc(R"(
Style for formatting code around applied fixes:
  - 'none' (default) turns off formatting
  - 'file' (literally 'file', not a placeholder)
    uses .clang-format file in the closest parent
    directory
  - '{ <json> }' specifies options inline, e.g.
    -format-style='{BasedOnStyle: llvm, IndentWidth: 8}'
  - 'llvm', 'google', 'webkit', 'mozilla'
See clang-format documentation for the up-to-date
information about formatting styles and options.
This option overrides the 'FormatStyle` option in
.clang-tidy file, if any.
)"),
                                   cl::init("none"),
                                   cl::cat(ClangTidyCategory));

static cl::opt<bool> ListChecks("list-checks", cl::desc(R"(
List all enabled checks and exit. Use with
-checks=* to list all available checks.
)"),
                                cl::init(false), cl::cat(ClangTidyCategory));

static cl::opt<bool> ExplainConfig("explain-config", cl::desc(R"(
For each enabled check explains, where it is
enabled, i.e. in clang-tidy binary, command
line or a specific configuration file.
)"),
                                   cl::init(false), cl::cat(ClangTidyCategory));

static cl::opt<std::string> Config("config", cl::desc(R"(
Specifies a configuration in YAML/JSON format:
  -config="{Checks: '*',
            CheckOptions: {x: y}}"
When the value is empty, clang-tidy will
attempt to find a file named .clang-tidy for
each source file in its parent directories.
)"),
                                   cl::init(""), cl::cat(ClangTidyCategory));

static cl::opt<std::string> ConfigFile("config-file", cl::desc(R"(
Specify the path of .clang-tidy or custom config file:
 e.g. --config-file=/some/path/myTidyConfigFile
This option internally works exactly the same way as
 --config option after reading specified config file.
Use either --config-file or --config, not both.
)"),
                                       cl::init(""),
                                       cl::cat(ClangTidyCategory));

static cl::opt<bool> DumpConfig("dump-config", cl::desc(R"(
Dumps configuration in the YAML format to
stdout. This option can be used along with a
file name (and '--' if the file is outside of a
project with configured compilation database).
The configuration used for this file will be
printed.
Use along with -checks=* to include
configuration of all checks.
)"),
                                cl::init(false), cl::cat(ClangTidyCategory));

static cl::opt<bool> EnableCheckProfile("enable-check-profile", cl::desc(R"(
Enable per-check timing profiles, and print a
report to stderr.
)"),
                                        cl::init(false),
                                        cl::cat(ClangTidyCategory));

static cl::opt<std::string> StoreCheckProfile("store-check-profile",
                                              cl::desc(R"(
By default reports are printed in tabulated
format to stderr. When this option is passed,
these per-TU profiles are instead stored as JSON.
)"),
                                              cl::value_desc("prefix"),
                                              cl::cat(ClangTidyCategory));

/// This option allows enabling the experimental alpha checkers from the static
/// analyzer. This option is set to false and not visible in help, because it is
/// highly not recommended for users.
static cl::opt<bool>
    AllowEnablingAnalyzerAlphaCheckers("allow-enabling-analyzer-alpha-checkers",
                                       cl::init(false), cl::Hidden,
                                       cl::cat(ClangTidyCategory));

static cl::opt<std::string> ExportFixes("export-fixes", cl::desc(R"(
YAML file to store suggested fixes in. The
stored fixes can be applied to the input source
code with clang-apply-replacements.
)"),
                                        cl::value_desc("filename"),
                                        cl::cat(ClangTidyCategory));

static cl::opt<bool> Quiet("quiet", cl::desc(R"(
Run clang-tidy in quiet mode. This suppresses
printing statistics about ignored warnings and
warnings treated as errors if the respective
options are specified.
)"),
                           cl::init(false),
                           cl::cat(ClangTidyCategory));

static cl::opt<std::string> VfsOverlay("vfsoverlay", cl::desc(R"(
Overlay the virtual filesystem described by file
over the real file system.
)"),
                                       cl::value_desc("filename"),
                                       cl::cat(ClangTidyCategory));

static cl::opt<bool> UseColor("use-color", cl::desc(R"(
Use colors in diagnostics. If not set, colors
will be used if the terminal connected to
standard output supports colors.
This option overrides the 'UseColor' option in
.clang-tidy file, if any.
)"),
                              cl::init(false), cl::cat(ClangTidyCategory));

static cl::opt<bool> VerifyConfig("verify-config", cl::desc(R"(
Check the config files to ensure each check and
option is recognized.
)"),
                                  cl::init(false), cl::cat(ClangTidyCategory));

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
      llvm::errs() << "Use -header-filter=.* to display errors from all "
                      "non-system headers. Use -system-headers to display "
                      "errors from system headers as well.\n";
  }
}

static std::unique_ptr<ClangTidyOptionsProvider> createOptionsProvider(
   llvm::IntrusiveRefCntPtr<vfs::FileSystem> FS) {
  ClangTidyGlobalOptions GlobalOptions;
  if (std::error_code Err = parseLineFilter(LineFilter, GlobalOptions)) {
    llvm::errs() << "Invalid LineFilter: " << Err.message() << "\n\nUsage:\n";
    llvm::cl::PrintHelpMessage(/*Hidden=*/false, /*Categorized=*/true);
    return nullptr;
  }

  ClangTidyOptions DefaultOptions;
  DefaultOptions.Checks = DefaultChecks;
  DefaultOptions.WarningsAsErrors = "";
  DefaultOptions.HeaderFilterRegex = HeaderFilter;
  DefaultOptions.SystemHeaders = SystemHeaders;
  DefaultOptions.FormatStyle = FormatStyle;
  DefaultOptions.User = llvm::sys::Process::GetEnv("USER");
  // USERNAME is used on Windows.
  if (!DefaultOptions.User)
    DefaultOptions.User = llvm::sys::Process::GetEnv("USERNAME");

  ClangTidyOptions OverrideOptions;
  if (Checks.getNumOccurrences() > 0)
    OverrideOptions.Checks = Checks;
  if (WarningsAsErrors.getNumOccurrences() > 0)
    OverrideOptions.WarningsAsErrors = WarningsAsErrors;
  if (HeaderFilter.getNumOccurrences() > 0)
    OverrideOptions.HeaderFilterRegex = HeaderFilter;
  if (SystemHeaders.getNumOccurrences() > 0)
    OverrideOptions.SystemHeaders = SystemHeaders;
  if (FormatStyle.getNumOccurrences() > 0)
    OverrideOptions.FormatStyle = FormatStyle;
  if (UseColor.getNumOccurrences() > 0)
    OverrideOptions.UseColor = UseColor;

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

  if (ConfigFile.getNumOccurrences() > 0) {
    if (Config.getNumOccurrences() > 0) {
      llvm::errs() << "Error: --config-file and --config are "
                      "mutually exclusive. Specify only one.\n";
      return nullptr;
    }

    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> Text =
        llvm::MemoryBuffer::getFile(ConfigFile);
    if (std::error_code EC = Text.getError()) {
      llvm::errs() << "Error: can't read config-file '" << ConfigFile
                   << "': " << EC.message() << "\n";
      return nullptr;
    }

    return LoadConfig((*Text)->getBuffer(), ConfigFile);
  }

  if (Config.getNumOccurrences() > 0)
    return LoadConfig(Config, "<command-line-config>");

  return std::make_unique<FileOptionsProvider>(
      std::move(GlobalOptions), std::move(DefaultOptions),
      std::move(OverrideOptions), std::move(FS));
}

llvm::IntrusiveRefCntPtr<vfs::FileSystem>
getVfsFromFile(const std::string &OverlayFile,
               llvm::IntrusiveRefCntPtr<vfs::FileSystem> BaseFS) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> Buffer =
      BaseFS->getBufferForFile(OverlayFile);
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
    unsigned Cur = Value.edit_distance_insensitive(Item, true, MaxEdit);
    if (Cur < MaxEdit) {
      Closest = Item;
      MaxEdit = Cur;
    }
  }
  return Closest;
}

static constexpr StringLiteral VerifyConfigWarningEnd = " [-verify-config]\n";

static bool verifyChecks(const StringSet<> &AllChecks, StringRef CheckGlob,
                         StringRef Source) {
  llvm::StringRef Cur, Rest;
  bool AnyInvalid = false;
  for (std::tie(Cur, Rest) = CheckGlob.split(',');
       !(Cur.empty() && Rest.empty()); std::tie(Cur, Rest) = Rest.split(',')) {
    Cur = Cur.trim();
    if (Cur.empty())
      continue;
    Cur.consume_front("-");
    if (Cur.startswith("clang-diagnostic"))
      continue;
    if (Cur.contains('*')) {
      SmallString<128> RegexText("^");
      StringRef MetaChars("()^$|*+?.[]\\{}");
      for (char C : Cur) {
        if (C == '*')
          RegexText.push_back('.');
        else if (MetaChars.contains(C))
          RegexText.push_back('\\');
        RegexText.push_back(C);
      }
      RegexText.push_back('$');
      llvm::Regex Glob(RegexText);
      std::string Error;
      if (!Glob.isValid(Error)) {
        AnyInvalid = true;
        llvm::WithColor::error(llvm::errs(), Source)
            << "building check glob '" << Cur << "' " << Error << "'\n";
        continue;
      }
      if (llvm::none_of(AllChecks.keys(),
                        [&Glob](StringRef S) { return Glob.match(S); })) {
        AnyInvalid = true;
        llvm::WithColor::warning(llvm::errs(), Source)
            << "check glob '" << Cur << "' doesn't match any known check"
            << VerifyConfigWarningEnd;
      }
    } else {
      if (AllChecks.contains(Cur))
        continue;
      AnyInvalid = true;
      llvm::raw_ostream &Output = llvm::WithColor::warning(llvm::errs(), Source)
                                  << "unknown check '" << Cur << '\'';
      llvm::StringRef Closest = closest(Cur, AllChecks);
      if (!Closest.empty())
        Output << "; did you mean '" << Closest << '\'';
      Output << VerifyConfigWarningEnd;
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

int clangTidyMain(int argc, const char **argv) {
  llvm::InitLLVM X(argc, argv);

  // Enable help for -load option, if plugins are enabled.
  if (cl::Option *LoadOpt = cl::getRegisteredOptions().lookup("load"))
    LoadOpt->addCategory(ClangTidyCategory);

  llvm::Expected<CommonOptionsParser> OptionsParser =
      CommonOptionsParser::create(argc, argv, ClangTidyCategory,
                                  cl::ZeroOrMore);
  if (!OptionsParser) {
    llvm::WithColor::error() << llvm::toString(OptionsParser.takeError());
    return 1;
  }

  llvm::IntrusiveRefCntPtr<vfs::OverlayFileSystem> BaseFS(
      new vfs::OverlayFileSystem(vfs::getRealFileSystem()));

  if (!VfsOverlay.empty()) {
    IntrusiveRefCntPtr<vfs::FileSystem> VfsFromFile =
        getVfsFromFile(VfsOverlay, BaseFS);
    if (!VfsFromFile)
      return 1;
    BaseFS->pushOverlay(std::move(VfsFromFile));
  }

  auto OwningOptionsProvider = createOptionsProvider(BaseFS);
  auto *OptionsProvider = OwningOptionsProvider.get();
  if (!OptionsProvider)
    return 1;

  auto MakeAbsolute = [](const std::string &Input) -> SmallString<256> {
    if (Input.empty())
      return {};
    SmallString<256> AbsolutePath(Input);
    if (std::error_code EC = llvm::sys::fs::make_absolute(AbsolutePath)) {
      llvm::errs() << "Can't make absolute path from " << Input << ": "
                   << EC.message() << "\n";
    }
    return AbsolutePath;
  };

  SmallString<256> ProfilePrefix = MakeAbsolute(StoreCheckProfile);

  StringRef FileName("dummy");
  auto PathList = OptionsParser->getSourcePathList();
  if (!PathList.empty()) {
    FileName = PathList.front();
  }

  SmallString<256> FilePath = MakeAbsolute(std::string(FileName));

  ClangTidyOptions EffectiveOptions = OptionsProvider->getOptions(FilePath);
  std::vector<std::string> EnabledChecks =
      getCheckNames(EffectiveOptions, AllowEnablingAnalyzerAlphaCheckers);

  if (ExplainConfig) {
    // FIXME: Show other ClangTidyOptions' fields, like ExtraArg.
    std::vector<clang::tidy::ClangTidyOptionsProvider::OptionsSource>
        RawOptions = OptionsProvider->getRawOptions(FilePath);
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
    if (EnabledChecks.empty()) {
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
        getCheckOptions(EffectiveOptions, AllowEnablingAnalyzerAlphaCheckers);
    llvm::outs() << configurationAsText(ClangTidyOptions::getDefaults().merge(
                        EffectiveOptions, 0))
                 << "\n";
    return 0;
  }

  if (VerifyConfig) {
    std::vector<ClangTidyOptionsProvider::OptionsSource> RawOptions =
        OptionsProvider->getRawOptions(FileName);
    NamesAndOptions Valid =
        getAllChecksAndOptions(AllowEnablingAnalyzerAlphaCheckers);
    bool AnyInvalid = false;
    for (const auto &[Opts, Source] : RawOptions) {
      if (Opts.Checks)
        AnyInvalid |= verifyChecks(Valid.Names, *Opts.Checks, Source);

      if (Opts.HeaderFileExtensions && Opts.ImplementationFileExtensions)
        AnyInvalid |=
            verifyFileExtensions(*Opts.HeaderFileExtensions,
                                 *Opts.ImplementationFileExtensions, Source);

      for (auto Key : Opts.CheckOptions.keys()) {
        if (Valid.Options.contains(Key))
          continue;
        AnyInvalid = true;
        auto &Output = llvm::WithColor::warning(llvm::errs(), Source)
                       << "unknown check option '" << Key << '\'';
        llvm::StringRef Closest = closest(Key, Valid.Options);
        if (!Closest.empty())
          Output << "; did you mean '" << Closest << '\'';
        Output << VerifyConfigWarningEnd;
      }
    }
    if (AnyInvalid)
      return 1;
    llvm::outs() << "No config errors detected.\n";
    return 0;
  }

  if (EnabledChecks.empty()) {
    llvm::errs() << "Error: no checks enabled.\n";
    llvm::cl::PrintHelpMessage(/*Hidden=*/false, /*Categorized=*/true);
    return 1;
  }

  if (PathList.empty()) {
    llvm::errs() << "Error: no input files specified.\n";
    llvm::cl::PrintHelpMessage(/*Hidden=*/false, /*Categorized=*/true);
    return 1;
  }

  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();

  ClangTidyContext Context(std::move(OwningOptionsProvider),
                           AllowEnablingAnalyzerAlphaCheckers);
  std::vector<ClangTidyError> Errors =
      runClangTidy(Context, OptionsParser->getCompilations(), PathList, BaseFS,
                   FixNotes, EnableCheckProfile, ProfilePrefix);
  bool FoundErrors = llvm::any_of(Errors, [](const ClangTidyError &E) {
    return E.DiagLevel == ClangTidyError::Error;
  });

  // --fix-errors and --fix-notes imply --fix.
  FixBehaviour Behaviour = FixNotes             ? FB_FixNotes
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
      StringRef Plural = WErrorCount == 1 ? "" : "s";
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
