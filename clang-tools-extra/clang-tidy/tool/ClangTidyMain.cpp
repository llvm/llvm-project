//===--- tools/extra/clang-tidy/ClangTidyMain.cpp - Clang tidy tool -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

#include "../ClangTidy.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "llvm/Support/Process.h"

using namespace clang::ast_matchers;
using namespace clang::driver;
using namespace clang::tooling;
using namespace llvm;

static cl::OptionCategory ClangTidyCategory("clang-tidy options");

static cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);
static cl::extrahelp ClangTidyHelp(R"(
Configuration files:
  clang-tidy attempts to read configuration for each source file from a
  .clang-tidy file located in the closest parent directory of the source
  file. If any configuration options have a corresponding command-line
  option, command-line option takes precedence. The effective
  configuration can be inspected using -dump-config:

    $ clang-tidy -dump-config - --
    ---
    Checks:          '-*,some-check'
    WarningsAsErrors: ''
    HeaderFilterRegex: ''
    AnalyzeTemporaryDtors: false
    User:            user
    CheckOptions:
      - key:             some-check.SomeOption
        value:           'some value'
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
checks.  This option's value is appended to the
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
This option overrides the 'HeaderFilter' option
in .clang-tidy file, if any.
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

static cl::opt<std::string> FormatStyle("style", cl::desc(R"(
Fallback style for reformatting after inserting fixes
if there is no clang-format config file found.
)"),
                                        cl::init("llvm"),
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
            CheckOptions: [{key: x,
                            value: y}]}"
When the value is empty, clang-tidy will
attempt to find a file named .clang-tidy for
each source file in its parent directories.
)"),
                                   cl::init(""), cl::cat(ClangTidyCategory));

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

static cl::opt<bool> AnalyzeTemporaryDtors("analyze-temporary-dtors",
                                           cl::desc(R"(
Enable temporary destructor-aware analysis in
clang-analyzer- checks.
This option overrides the value read from a
.clang-tidy file.
)"),
                                           cl::init(false),
                                           cl::cat(ClangTidyCategory));

static cl::opt<std::string> ExportFixes("export-fixes", cl::desc(R"(
YAML file to store suggested fixes in. The
stored fixes can be applied to the input source
code with clang-apply-replacements.
)"),
                                        cl::value_desc("filename"),
                                        cl::cat(ClangTidyCategory));

static cl::opt<bool> Quiet("quiet", cl::desc(R"(
Run clang-tidy in quiet mode.  This suppresses
printing statistics about ignored warnings and
warnings treated as errors if the respective
options are specified.
)"),
                           cl::init(false),
                           cl::cat(ClangTidyCategory));

namespace clang {
namespace tidy {

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

static void printProfileData(const ProfileData &Profile,
                             llvm::raw_ostream &OS) {
  // Time is first to allow for sorting by it.
  std::vector<std::pair<llvm::TimeRecord, StringRef>> Timers;
  TimeRecord Total;

  for (const auto &P : Profile.Records) {
    Timers.emplace_back(P.getValue(), P.getKey());
    Total += P.getValue();
  }

  std::sort(Timers.begin(), Timers.end());

  std::string Line = "===" + std::string(73, '-') + "===\n";
  OS << Line;

  if (Total.getUserTime())
    OS << "   ---User Time---";
  if (Total.getSystemTime())
    OS << "   --System Time--";
  if (Total.getProcessTime())
    OS << "   --User+System--";
  OS << "   ---Wall Time---";
  if (Total.getMemUsed())
    OS << "  ---Mem---";
  OS << "  --- Name ---\n";

  // Loop through all of the timing data, printing it out.
  for (auto I = Timers.rbegin(), E = Timers.rend(); I != E; ++I) {
    I->first.print(Total, OS);
    OS << I->second << '\n';
  }

  Total.print(Total, OS);
  OS << "Total\n";
  OS << Line << "\n";
  OS.flush();
}

static std::unique_ptr<ClangTidyOptionsProvider> createOptionsProvider() {
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
  DefaultOptions.AnalyzeTemporaryDtors = AnalyzeTemporaryDtors;
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
  if (AnalyzeTemporaryDtors.getNumOccurrences() > 0)
    OverrideOptions.AnalyzeTemporaryDtors = AnalyzeTemporaryDtors;

  if (!Config.empty()) {
    if (llvm::ErrorOr<ClangTidyOptions> ParsedConfig =
            parseConfiguration(Config)) {
      return llvm::make_unique<ConfigOptionsProvider>(
          GlobalOptions,
          ClangTidyOptions::getDefaults().mergeWith(DefaultOptions),
          *ParsedConfig, OverrideOptions);
    } else {
      llvm::errs() << "Error: invalid configuration specified.\n"
                   << ParsedConfig.getError().message() << "\n";
      return nullptr;
    }
  }
  return llvm::make_unique<FileOptionsProvider>(GlobalOptions, DefaultOptions,
                                                OverrideOptions);
}

static int clangTidyMain(int argc, const char **argv) {
  CommonOptionsParser OptionsParser(argc, argv, ClangTidyCategory,
                                    cl::ZeroOrMore);

  auto OptionsProvider = createOptionsProvider();
  if (!OptionsProvider)
    return 1;

  StringRef FileName("dummy");
  auto PathList = OptionsParser.getSourcePathList();
  if (!PathList.empty()) {
    FileName = PathList.front();
  }

  SmallString<256> FilePath(FileName);
  if (std::error_code EC = llvm::sys::fs::make_absolute(FilePath)) {
    llvm::errs() << "Can't make absolute path from " << FileName << ": "
                 << EC.message() << "\n";
  }
  ClangTidyOptions EffectiveOptions = OptionsProvider->getOptions(FilePath);
  std::vector<std::string> EnabledChecks = getCheckNames(EffectiveOptions);

  if (ExplainConfig) {
    // FIXME: Show other ClangTidyOptions' fields, like ExtraArg.
    std::vector<clang::tidy::ClangTidyOptionsProvider::OptionsSource>
        RawOptions = OptionsProvider->getRawOptions(FilePath);
    for (const std::string &Check : EnabledChecks) {
      for (auto It = RawOptions.rbegin(); It != RawOptions.rend(); ++It) {
        if (It->first.Checks && GlobList(*It->first.Checks).contains(Check)) {
          llvm::outs() << "'" << Check << "' is enabled in the " << It->second
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
    EffectiveOptions.CheckOptions = getCheckOptions(EffectiveOptions);
    llvm::outs() << configurationAsText(
                        ClangTidyOptions::getDefaults().mergeWith(
                            EffectiveOptions))
                 << "\n";
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

  ProfileData Profile;

  std::vector<ClangTidyError> Errors;
  ClangTidyStats Stats =
      runClangTidy(std::move(OptionsProvider), OptionsParser.getCompilations(),
                   PathList, &Errors, EnableCheckProfile ? &Profile : nullptr);
  bool FoundErrors =
      std::find_if(Errors.begin(), Errors.end(), [](const ClangTidyError &E) {
        return E.DiagLevel == ClangTidyError::Error;
      }) != Errors.end();

  const bool DisableFixes = Fix && FoundErrors && !FixErrors;

  unsigned WErrorCount = 0;

  // -fix-errors implies -fix.
  handleErrors(Errors, (FixErrors || Fix) && !DisableFixes, FormatStyle,
               WErrorCount);

  if (!ExportFixes.empty() && !Errors.empty()) {
    std::error_code EC;
    llvm::raw_fd_ostream OS(ExportFixes, EC, llvm::sys::fs::F_None);
    if (EC) {
      llvm::errs() << "Error opening output file: " << EC.message() << '\n';
      return 1;
    }
    exportReplacements(FilePath.str(), Errors, OS);
  }

  if (!Quiet) {
    printStats(Stats);
    if (DisableFixes)
      llvm::errs()
          << "Found compiler errors, but -fix-errors was not specified.\n"
             "Fixes have NOT been applied.\n\n";
  }

  if (EnableCheckProfile)
    printProfileData(Profile, llvm::errs());

  if (WErrorCount) {
    if (!Quiet) {
      StringRef Plural = WErrorCount == 1 ? "" : "s";
      llvm::errs() << WErrorCount << " warning" << Plural << " treated as error"
                   << Plural << "\n";
    }
    return WErrorCount;
  }

  return 0;
}

// This anchor is used to force the linker to link the CERTModule.
extern volatile int CERTModuleAnchorSource;
static int LLVM_ATTRIBUTE_UNUSED CERTModuleAnchorDestination =
    CERTModuleAnchorSource;

// This anchor is used to force the linker to link the BoostModule.
extern volatile int BoostModuleAnchorSource;
static int LLVM_ATTRIBUTE_UNUSED BoostModuleAnchorDestination =
    BoostModuleAnchorSource;

// This anchor is used to force the linker to link the LLVMModule.
extern volatile int LLVMModuleAnchorSource;
static int LLVM_ATTRIBUTE_UNUSED LLVMModuleAnchorDestination =
    LLVMModuleAnchorSource;

// This anchor is used to force the linker to link the CppCoreGuidelinesModule.
extern volatile int CppCoreGuidelinesModuleAnchorSource;
static int LLVM_ATTRIBUTE_UNUSED CppCoreGuidelinesModuleAnchorDestination =
    CppCoreGuidelinesModuleAnchorSource;

// This anchor is used to force the linker to link the GoogleModule.
extern volatile int GoogleModuleAnchorSource;
static int LLVM_ATTRIBUTE_UNUSED GoogleModuleAnchorDestination =
    GoogleModuleAnchorSource;

// This anchor is used to force the linker to link the MiscModule.
extern volatile int MiscModuleAnchorSource;
static int LLVM_ATTRIBUTE_UNUSED MiscModuleAnchorDestination =
    MiscModuleAnchorSource;

// This anchor is used to force the linker to link the ModernizeModule.
extern volatile int ModernizeModuleAnchorSource;
static int LLVM_ATTRIBUTE_UNUSED ModernizeModuleAnchorDestination =
    ModernizeModuleAnchorSource;

// This anchor is used to force the linker to link the MPIModule.
extern volatile int MPIModuleAnchorSource;
static int LLVM_ATTRIBUTE_UNUSED MPIModuleAnchorDestination =
    MPIModuleAnchorSource;

// This anchor is used to force the linker to link the PerformanceModule.
extern volatile int PerformanceModuleAnchorSource;
static int LLVM_ATTRIBUTE_UNUSED PerformanceModuleAnchorDestination =
    PerformanceModuleAnchorSource;

// This anchor is used to force the linker to link the ReadabilityModule.
extern volatile int ReadabilityModuleAnchorSource;
static int LLVM_ATTRIBUTE_UNUSED ReadabilityModuleAnchorDestination =
    ReadabilityModuleAnchorSource;

// This anchor is used to force the linker to link the SafetyModule.
extern volatile int SafetyModuleAnchorSource;
static int LLVM_ATTRIBUTE_UNUSED SafetyModuleAnchorDestination =
    SafetyModuleAnchorSource;

} // namespace tidy
} // namespace clang

int main(int argc, const char **argv) {
  return clang::tidy::clangTidyMain(argc, argv);
}
