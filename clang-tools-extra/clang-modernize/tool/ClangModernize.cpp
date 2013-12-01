//===-- ClangModernize.cpp - Main file for Clang modernization tool -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file implements the C++11 feature migration tool main function
/// and transformation framework.
///
/// See user documentation for usage instructions.
///
//===----------------------------------------------------------------------===//

#include "Core/PerfSupport.h"
#include "Core/ReplacementHandling.h"
#include "Core/Transform.h"
#include "Core/Transforms.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/Version.h"
#include "clang/Format/Format.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Signals.h"

namespace cl = llvm::cl;
using namespace clang;
using namespace clang::tooling;

TransformOptions GlobalOptions;

// All options must belong to locally defined categories for them to get shown
// by -help. We explicitly hide everything else (except -help and -version).
static cl::OptionCategory GeneralCategory("Modernizer Options");
static cl::OptionCategory FormattingCategory("Formatting Options");
static cl::OptionCategory IncludeExcludeCategory("Inclusion/Exclusion Options");
static cl::OptionCategory SerializeCategory("Serialization Options");

const cl::OptionCategory *VisibleCategories[] = {
  &GeneralCategory,   &FormattingCategory, &IncludeExcludeCategory,
  &SerializeCategory, &TransformCategory,  &TransformsOptionsCategory,
};

static cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);
static cl::extrahelp MoreHelp(
    "EXAMPLES:\n\n"
    "Apply all transforms on a file that doesn't require compilation arguments:\n\n"
    "  clang-modernize file.cpp\n"
    "\n"
    "Convert for loops to ranged-based for loops for all files in the compilation\n"
    "database that belong in a project subtree and then reformat the code\n"
    "automatically using the LLVM style:\n\n"
    "    clang-modernize -p build/path -include project/path -format -loop-convert\n"
    "\n"
    "Make use of both nullptr and the override specifier, using git ls-files:\n"
    "\n"
    "  git ls-files '*.cpp' | xargs -I{} clang-modernize -p build/path \\\n"
    "      -use-nullptr -add-override -override-macros {}\n"
    "\n"
    "Apply all transforms supported by both clang >= 3.0 and gcc >= 4.7 to\n"
    "foo.cpp and any included headers in bar:\n\n"
    "  clang-modernize -for-compilers=clang-3.0,gcc-4.7 foo.cpp \\\n"
    "      -include bar -- -std=c++11 -Ibar\n\n");

////////////////////////////////////////////////////////////////////////////////
/// General Options

// This is set to hidden on purpose. The actual help text for this option is
// included in CommonOptionsParser::HelpMessage.
static cl::opt<std::string> BuildPath("p", cl::desc("Build Path"), cl::Optional,
                                      cl::Hidden, cl::cat(GeneralCategory));

static cl::list<std::string> SourcePaths(cl::Positional,
                                         cl::desc("[<sources>...]"),
                                         cl::ZeroOrMore,
                                         cl::cat(GeneralCategory));

static cl::opt<RiskLevel, /*ExternalStorage=*/true> MaxRiskLevel(
    "risk", cl::desc("Select a maximum risk level:"),
    cl::values(clEnumValN(RL_Safe, "safe", "Only safe transformations"),
               clEnumValN(RL_Reasonable, "reasonable",
                          "Enable transformations that might change "
                          "semantics (default)"),
               clEnumValN(RL_Risky, "risky",
                          "Enable transformations that are likely to "
                          "change semantics"),
               clEnumValEnd),
    cl::location(GlobalOptions.MaxRiskLevel), cl::init(RL_Reasonable),
    cl::cat(GeneralCategory));

static cl::opt<bool> FinalSyntaxCheck(
    "final-syntax-check",
    cl::desc("Check for correct syntax after applying transformations"),
    cl::init(false), cl::cat(GeneralCategory));

static cl::opt<bool> SummaryMode("summary", cl::desc("Print transform summary"),
                                 cl::init(false), cl::cat(GeneralCategory));

static cl::opt<std::string>
TimingDirectoryName("perf",
                    cl::desc("Capture performance data and output to specified "
                             "directory. Default: ./migrate_perf"),
                    cl::ValueOptional, cl::value_desc("directory name"),
                    cl::cat(GeneralCategory));

cl::opt<std::string> SupportedCompilers(
    "for-compilers", cl::value_desc("string"),
    cl::desc("Select transforms targeting the intersection of\n"
             "language features supported by the given compilers.\n"
             "Takes a comma-separated list of <compiler>-<version>.\n"
             "\t<compiler> can be any of: clang, gcc, icc, msvc\n"
             "\t<version> is <major>[.<minor>]\n"),
    cl::cat(GeneralCategory));

////////////////////////////////////////////////////////////////////////////////
/// Format Options
static cl::opt<bool> DoFormat(
    "format",
    cl::desc("Enable formatting of code changed by applying replacements.\n"
             "Use -style to choose formatting style.\n"),
    cl::cat(FormattingCategory));

static cl::opt<std::string>
FormatStyleOpt("style", cl::desc(format::StyleOptionHelpDescription),
               cl::init("LLVM"), cl::cat(FormattingCategory));

// FIXME: Consider making the default behaviour for finding a style
// configuration file to start the search anew for every file being changed to
// handle situations where the style is different for different parts of a
// project.

static cl::opt<std::string> FormatStyleConfig(
    "style-config",
    cl::desc("Path to a directory containing a .clang-format file\n"
             "describing a formatting style to use for formatting\n"
             "code when -style=file.\n"),
    cl::init(""), cl::cat(FormattingCategory));

////////////////////////////////////////////////////////////////////////////////
/// Include/Exclude Options
static cl::opt<std::string>
IncludePaths("include",
             cl::desc("Comma-separated list of paths to consider to be "
                      "transformed"),
             cl::cat(IncludeExcludeCategory));

static cl::opt<std::string>
ExcludePaths("exclude", cl::desc("Comma-separated list of paths that can not "
                                 "be transformed"),
             cl::cat(IncludeExcludeCategory));

static cl::opt<std::string>
IncludeFromFile("include-from", cl::value_desc("filename"),
                cl::desc("File containing a list of paths to consider to "
                         "be transformed"),
                cl::cat(IncludeExcludeCategory));

static cl::opt<std::string>
ExcludeFromFile("exclude-from", cl::value_desc("filename"),
                cl::desc("File containing a list of paths that can not be "
                         "transformed"),
                cl::cat(IncludeExcludeCategory));

////////////////////////////////////////////////////////////////////////////////
/// Serialization Options

static cl::opt<bool>
SerializeOnly("serialize-replacements",
              cl::desc("Serialize translation unit replacements to "
                       "disk instead of changing files."),
              cl::init(false),
              cl::cat(SerializeCategory));

static cl::opt<std::string>
SerializeLocation("serialize-dir",
                  cl::desc("Path to an existing directory in which to write\n"
                           "serialized replacements. Default behaviour is to\n"
                           "write to a temporary directory.\n"),
                  cl::cat(SerializeCategory));

////////////////////////////////////////////////////////////////////////////////

void printVersion() {
  llvm::outs() << "clang-modernizer version " CLANG_VERSION_STRING
               << "\n";
}

/// \brief Extract the minimum compiler versions as requested on the command
/// line by the switch \c -for-compilers.
///
/// \param ProgName The name of the program, \c argv[0], used to print errors.
/// \param Error If an error occur while parsing the versions this parameter is
/// set to \c true, otherwise it will be left untouched.
static CompilerVersions handleSupportedCompilers(const char *ProgName,
                                                 bool &Error) {
  if (SupportedCompilers.getNumOccurrences() == 0)
    return CompilerVersions();
  CompilerVersions RequiredVersions;
  llvm::SmallVector<llvm::StringRef, 4> Compilers;

  llvm::StringRef(SupportedCompilers).split(Compilers, ",");

  for (llvm::SmallVectorImpl<llvm::StringRef>::iterator I = Compilers.begin(),
                                                        E = Compilers.end();
       I != E; ++I) {
    llvm::StringRef Compiler, VersionStr;
    llvm::tie(Compiler, VersionStr) = I->split('-');
    Version *V = llvm::StringSwitch<Version *>(Compiler)
        .Case("clang", &RequiredVersions.Clang)
        .Case("gcc", &RequiredVersions.Gcc).Case("icc", &RequiredVersions.Icc)
        .Case("msvc", &RequiredVersions.Msvc).Default(NULL);

    if (V == NULL) {
      llvm::errs() << ProgName << ": " << Compiler
                   << ": unsupported platform\n";
      Error = true;
      continue;
    }
    if (VersionStr.empty()) {
      llvm::errs() << ProgName << ": " << *I
                   << ": missing version number in platform\n";
      Error = true;
      continue;
    }

    Version Version = Version::getFromString(VersionStr);
    if (Version.isNull()) {
      llvm::errs()
          << ProgName << ": " << *I
          << ": invalid version, please use \"<major>[.<minor>]\" instead of \""
          << VersionStr << "\"\n";
      Error = true;
      continue;
    }
    // support the lowest version given
    if (V->isNull() || Version < *V)
      *V = Version;
  }
  return RequiredVersions;
}

CompilationDatabase *autoDetectCompilations(std::string &ErrorMessage) {
  // Auto-detect a compilation database from BuildPath.
  if (BuildPath.getNumOccurrences() > 0)
    return CompilationDatabase::autoDetectFromDirectory(BuildPath,
                                                        ErrorMessage);
  // Try to auto-detect a compilation database from the first source.
  if (!SourcePaths.empty()) {
    if (CompilationDatabase *Compilations =
            CompilationDatabase::autoDetectFromSource(SourcePaths[0],
                                                      ErrorMessage)) {
      // FIXME: just pass SourcePaths[0] once getCompileCommands supports
      // non-absolute paths.
      SmallString<64> Path(SourcePaths[0]);
      llvm::sys::fs::make_absolute(Path);
      std::vector<CompileCommand> Commands =
          Compilations->getCompileCommands(Path);
      // Ignore a detected compilation database that doesn't contain source0
      // since it is probably an unrelated compilation database.
      if (!Commands.empty())
        return Compilations;
    }
    // Reset ErrorMessage since a fix compilation database will be created if
    // it fails to detect one from source.
    ErrorMessage = "";
    // If no compilation database can be detected from source then we create a
    // fixed compilation database with c++11 support.
    std::string CommandLine[] = { "-std=c++11" };
    return new FixedCompilationDatabase(".", CommandLine);
  }

  ErrorMessage = "Could not determine sources to transform";
  return 0;
}

// Predicate definition for determining whether a file is not included.
static bool isFileNotIncludedPredicate(llvm::StringRef FilePath) {
  return !GlobalOptions.ModifiableFiles.isFileIncluded(FilePath);
}

// Predicate definition for determining if a file was explicitly excluded.
static bool isFileExplicitlyExcludedPredicate(llvm::StringRef FilePath) {
  if (GlobalOptions.ModifiableFiles.isFileExplicitlyExcluded(FilePath)) {
    llvm::errs() << "Warning \"" << FilePath << "\" will not be transformed "
                 << "because it's in the excluded list.\n";
    return true;
  }
  return false;
}

int main(int argc, const char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal();
  Transforms TransformManager;
  ReplacementHandling ReplacementHandler;

  TransformManager.registerTransforms();

  // Hide all options we don't define ourselves. Move pre-defined 'help',
  // 'help-list', and 'version' to our general category.
  llvm::StringMap<cl::Option*> Options;
  cl::getRegisteredOptions(Options);
  const cl::OptionCategory **CategoryEnd =
      VisibleCategories + llvm::array_lengthof(VisibleCategories);
  for (llvm::StringMap<cl::Option *>::iterator I = Options.begin(),
                                               E = Options.end();
       I != E; ++I) {
    if (I->first() == "help" || I->first() == "version" ||
        I->first() == "help-list")
      I->second->setCategory(GeneralCategory);
    else if (std::find(VisibleCategories, CategoryEnd, I->second->Category) ==
             CategoryEnd)
      I->second->setHiddenFlag(cl::ReallyHidden);
  }
  cl::SetVersionPrinter(&printVersion);

  // Parse options and generate compilations.
  OwningPtr<CompilationDatabase> Compilations(
      FixedCompilationDatabase::loadFromCommandLine(argc, argv));
  cl::ParseCommandLineOptions(argc, argv);

  // Populate the ModifiableFiles structure.
  GlobalOptions.ModifiableFiles.readListFromString(IncludePaths, ExcludePaths);
  GlobalOptions.ModifiableFiles.readListFromFile(IncludeFromFile,
                                                 ExcludeFromFile);

  if (!Compilations) {
    std::string ErrorMessage;
    Compilations.reset(autoDetectCompilations(ErrorMessage));
    if (!Compilations) {
      llvm::errs() << llvm::sys::path::filename(argv[0]) << ": " << ErrorMessage
                   << "\n";
      return 1;
    }
  }

  // Populate source files.
  std::vector<std::string> Sources;
  if (!SourcePaths.empty()) {
    // Use only files that are not explicitly excluded.
    std::remove_copy_if(SourcePaths.begin(), SourcePaths.end(),
                        std::back_inserter(Sources),
                        isFileExplicitlyExcludedPredicate);
  } else {
    if (GlobalOptions.ModifiableFiles.isIncludeListEmpty()) {
      llvm::errs() << llvm::sys::path::filename(argv[0])
                   << ": Use -include to indicate which files of "
                   << "the compilatiion database to transform.\n";
      return 1;
    }
    // Use source paths from the compilation database.
    // We only transform files that are explicitly included.
    Sources = Compilations->getAllFiles();
    std::vector<std::string>::iterator E = std::remove_if(
        Sources.begin(), Sources.end(), isFileNotIncludedPredicate);
    Sources.erase(E, Sources.end());
  }

  if (Sources.empty()) {
    llvm::errs() << llvm::sys::path::filename(argv[0])
                 << ": Could not determine sources to transform.\n";
    return 1;
  }

  // Enable timming.
  GlobalOptions.EnableTiming = TimingDirectoryName.getNumOccurrences() > 0;

  bool CmdSwitchError = false;
  CompilerVersions RequiredVersions =
      handleSupportedCompilers(argv[0], CmdSwitchError);
  if (CmdSwitchError)
    return 1;

  TransformManager.createSelectedTransforms(GlobalOptions, RequiredVersions);

  if (TransformManager.begin() == TransformManager.end()) {
    if (SupportedCompilers.empty())
      llvm::errs() << llvm::sys::path::filename(argv[0])
                   << ": no selected transforms\n";
    else
      llvm::errs() << llvm::sys::path::filename(argv[0])
                   << ": no transforms available for specified compilers\n";
    return 1;
  }

  llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> DiagOpts(
      new DiagnosticOptions());
  DiagnosticsEngine Diagnostics(
      llvm::IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs()),
      DiagOpts.getPtr());

  // FIXME: Make this DiagnosticsEngine available to all Transforms probably via
  // GlobalOptions.

  // If SerializeReplacements is requested, then code reformatting must be
  // turned off and only one transform should be requested.
  if (SerializeOnly &&
      (std::distance(TransformManager.begin(), TransformManager.end()) > 1 ||
       DoFormat)) {
    llvm::errs() << "Serialization of replacements requested for multiple "
                    "transforms.\nChanges from only one transform can be "
                    "serialized.\n";
    return 1;
  }

  // If we're asked to apply changes to files on disk, need to locate
  // clang-apply-replacements.
  if (!SerializeOnly) {
    if (!ReplacementHandler.findClangApplyReplacements(argv[0])) {
      llvm::errs() << "Could not find clang-apply-replacements\n";
      return 1;
    }

    if (DoFormat)
      ReplacementHandler.enableFormatting(FormatStyleOpt, FormatStyleConfig);
  }

  StringRef TempDestinationDir;
  if (SerializeLocation.getNumOccurrences() > 0)
    ReplacementHandler.setDestinationDir(SerializeLocation);
  else
    TempDestinationDir = ReplacementHandler.useTempDestinationDir();

  SourcePerfData PerfData;

  for (Transforms::const_iterator I = TransformManager.begin(),
                                  E = TransformManager.end();
       I != E; ++I) {
    Transform *T = *I;

    if (T->apply(*Compilations, Sources) != 0) {
      // FIXME: Improve ClangTool to not abort if just one file fails.
      return 1;
    }

    if (GlobalOptions.EnableTiming)
      collectSourcePerfData(*T, PerfData);

    if (SummaryMode) {
      llvm::outs() << "Transform: " << T->getName()
                   << " - Accepted: " << T->getAcceptedChanges();
      if (T->getChangesNotMade()) {
        llvm::outs() << " - Rejected: " << T->getRejectedChanges()
                     << " - Deferred: " << T->getDeferredChanges();
      }
      llvm::outs() << "\n";
    }

    if (!ReplacementHandler.serializeReplacements(T->getAllReplacements()))
      return 1;

    if (!SerializeOnly)
      if (!ReplacementHandler.applyReplacements())
        return 1;
  }

  // Let the user know which temporary directory the replacements got written
  // to.
  if (SerializeOnly && !TempDestinationDir.empty())
    llvm::errs() << "Replacements serialized to: " << TempDestinationDir << "\n";

  if (FinalSyntaxCheck) {
    ClangTool SyntaxTool(*Compilations, SourcePaths);
    if (SyntaxTool.run(newFrontendActionFactory<SyntaxOnlyAction>()) != 0)
      return 1;
  }

  // Report execution times.
  if (GlobalOptions.EnableTiming && !PerfData.empty()) {
    std::string DirectoryName = TimingDirectoryName;
    // Use default directory name.
    if (DirectoryName.empty())
      DirectoryName = "./migrate_perf";
    writePerfDataJSON(DirectoryName, PerfData);
  }

  return 0;
}

// These anchors are used to force the linker to link the transforms
extern volatile int AddOverrideTransformAnchorSource;
extern volatile int LoopConvertTransformAnchorSource;
extern volatile int PassByValueTransformAnchorSource;
extern volatile int ReplaceAutoPtrTransformAnchorSource;
extern volatile int UseAutoTransformAnchorSource;
extern volatile int UseNullptrTransformAnchorSource;

static int TransformsAnchorsDestination[] = {
  AddOverrideTransformAnchorSource,
  LoopConvertTransformAnchorSource,
  PassByValueTransformAnchorSource,
  ReplaceAutoPtrTransformAnchorSource,
  UseAutoTransformAnchorSource,
  UseNullptrTransformAnchorSource
};
