//===- CheckerRegistry.cpp - Maintains all available checkers -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Frontend/CheckerRegistry.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LLVM.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/AnalyzerOptions.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

using namespace clang;
using namespace ento;
using llvm::sys::DynamicLibrary;

using RegisterCheckersFn = void (*)(CheckerRegistry &);

static bool isCompatibleAPIVersion(const char *VersionString) {
  // If the version string is null, its not an analyzer plugin.
  if (!VersionString)
    return false;

  // For now, none of the static analyzer API is considered stable.
  // Versions must match exactly.
  return strcmp(VersionString, CLANG_ANALYZER_API_VERSION_STRING) == 0;
}

namespace {
template <class T> struct FullNameLT {
  bool operator()(const T &Lhs, const T &Rhs) {
    return Lhs.FullName < Rhs.FullName;
  }
};

using PackageNameLT = FullNameLT<CheckerRegistry::PackageInfo>;
using CheckerNameLT = FullNameLT<CheckerRegistry::CheckerInfo>;
} // end of anonymous namespace

template <class CheckerOrPackageInfoList>
static
    typename std::conditional<std::is_const<CheckerOrPackageInfoList>::value,
                              typename CheckerOrPackageInfoList::const_iterator,
                              typename CheckerOrPackageInfoList::iterator>::type
    binaryFind(CheckerOrPackageInfoList &Collection, StringRef FullName) {

  using CheckerOrPackage = typename CheckerOrPackageInfoList::value_type;
  using CheckerOrPackageFullNameLT = FullNameLT<CheckerOrPackage>;

  assert(std::is_sorted(Collection.begin(), Collection.end(),
                        CheckerOrPackageFullNameLT{}) &&
         "In order to efficiently gather checkers/packages, this function "
         "expects them to be already sorted!");

  return llvm::lower_bound(Collection, CheckerOrPackage(FullName),
                           CheckerOrPackageFullNameLT{});
}

static constexpr char PackageSeparator = '.';

static bool isInPackage(const CheckerRegistry::CheckerInfo &Checker,
                        StringRef PackageName) {
  // Does the checker's full name have the package as a prefix?
  if (!Checker.FullName.startswith(PackageName))
    return false;

  // Is the package actually just the name of a specific checker?
  if (Checker.FullName.size() == PackageName.size())
    return true;

  // Is the checker in the package (or a subpackage)?
  if (Checker.FullName[PackageName.size()] == PackageSeparator)
    return true;

  return false;
}

CheckerRegistry::CheckerInfoListRange
CheckerRegistry::getMutableCheckersForCmdLineArg(StringRef CmdLineArg) {
  auto It = binaryFind(Checkers, CmdLineArg);

  if (!isInPackage(*It, CmdLineArg))
    return {Checkers.end(), Checkers.end()};

  // See how large the package is.
  // If the package doesn't exist, assume the option refers to a single
  // checker.
  size_t Size = 1;
  llvm::StringMap<size_t>::const_iterator PackageSize =
      PackageSizes.find(CmdLineArg);

  if (PackageSize != PackageSizes.end())
    Size = PackageSize->getValue();

  return {It, It + Size};
}

CheckerRegistry::CheckerRegistry(
    ArrayRef<std::string> Plugins, DiagnosticsEngine &Diags,
    AnalyzerOptions &AnOpts, const LangOptions &LangOpts,
    ArrayRef<std::function<void(CheckerRegistry &)>> CheckerRegistrationFns)
    : Diags(Diags), AnOpts(AnOpts), LangOpts(LangOpts) {

  // Register builtin checkers.
#define GET_CHECKERS
#define CHECKER(FULLNAME, CLASS, HELPTEXT, DOC_URI, IS_HIDDEN)                 \
  addChecker(register##CLASS, shouldRegister##CLASS, FULLNAME, HELPTEXT,       \
             DOC_URI, IS_HIDDEN);

#define GET_PACKAGES
#define PACKAGE(FULLNAME) addPackage(FULLNAME);

#include "clang/StaticAnalyzer/Checkers/Checkers.inc"
#undef CHECKER
#undef GET_CHECKERS
#undef PACKAGE
#undef GET_PACKAGES

  // Register checkers from plugins.
  for (const std::string &Plugin : Plugins) {
    // Get access to the plugin.
    std::string ErrorMsg;
    DynamicLibrary Lib =
        DynamicLibrary::getPermanentLibrary(Plugin.c_str(), &ErrorMsg);
    if (!Lib.isValid()) {
      Diags.Report(diag::err_fe_unable_to_load_plugin) << Plugin << ErrorMsg;
      continue;
    }

    // See if its compatible with this build of clang.
    const char *PluginAPIVersion = static_cast<const char *>(
        Lib.getAddressOfSymbol("clang_analyzerAPIVersionString"));

    if (!isCompatibleAPIVersion(PluginAPIVersion)) {
      Diags.Report(diag::warn_incompatible_analyzer_plugin_api)
          << llvm::sys::path::filename(Plugin);
      Diags.Report(diag::note_incompatible_analyzer_plugin_api)
          << CLANG_ANALYZER_API_VERSION_STRING << PluginAPIVersion;
      continue;
    }

    // Register its checkers.
    RegisterCheckersFn RegisterPluginCheckers =
        reinterpret_cast<RegisterCheckersFn>(
            Lib.getAddressOfSymbol("clang_registerCheckers"));
    if (RegisterPluginCheckers)
      RegisterPluginCheckers(*this);
  }

  // Register statically linked checkers, that aren't generated from the tblgen
  // file, but rather passed their registry function as a parameter in
  // checkerRegistrationFns.

  for (const auto &Fn : CheckerRegistrationFns)
    Fn(*this);

  // Sort checkers for efficient collection.
  // FIXME: Alphabetical sort puts 'experimental' in the middle.
  // Would it be better to name it '~experimental' or something else
  // that's ASCIIbetically last?
  llvm::sort(Packages, PackageNameLT{});
  llvm::sort(Checkers, CheckerNameLT{});

#define GET_CHECKER_DEPENDENCIES

#define CHECKER_DEPENDENCY(FULLNAME, DEPENDENCY)                               \
  addDependency(FULLNAME, DEPENDENCY);

#define GET_CHECKER_OPTIONS
#define CHECKER_OPTION(TYPE, FULLNAME, CMDFLAG, DESC, DEFAULT_VAL, DEVELOPMENT_STATUS, IS_HIDDEN)  \
  addCheckerOption(TYPE, FULLNAME, CMDFLAG, DEFAULT_VAL, DESC, DEVELOPMENT_STATUS, IS_HIDDEN);

#define GET_PACKAGE_OPTIONS
#define PACKAGE_OPTION(TYPE, FULLNAME, CMDFLAG, DESC, DEFAULT_VAL, DEVELOPMENT_STATUS, IS_HIDDEN)  \
  addPackageOption(TYPE, FULLNAME, CMDFLAG, DEFAULT_VAL, DESC, DEVELOPMENT_STATUS, IS_HIDDEN);

#include "clang/StaticAnalyzer/Checkers/Checkers.inc"
#undef CHECKER_DEPENDENCY
#undef GET_CHECKER_DEPENDENCIES
#undef CHECKER_OPTION
#undef GET_CHECKER_OPTIONS
#undef PACKAGE_OPTION
#undef GET_PACKAGE_OPTIONS

  resolveDependencies();
  resolveCheckerAndPackageOptions();

  // Parse '-analyzer-checker' and '-analyzer-disable-checker' options from the
  // command line.
  for (const std::pair<std::string, bool> &Opt : AnOpts.CheckersAndPackages) {
    CheckerInfoListRange CheckerForCmdLineArg =
        getMutableCheckersForCmdLineArg(Opt.first);

    if (CheckerForCmdLineArg.begin() == CheckerForCmdLineArg.end()) {
      Diags.Report(diag::err_unknown_analyzer_checker_or_package) << Opt.first;
      Diags.Report(diag::note_suggest_disabling_all_checkers);
    }

    for (CheckerInfo &checker : CheckerForCmdLineArg) {
      checker.State = Opt.second ? StateFromCmdLine::State_Enabled
                                 : StateFromCmdLine::State_Disabled;
    }
  }
}

/// Collects dependencies in \p ret, returns false on failure.
static bool
collectDependenciesImpl(const CheckerRegistry::ConstCheckerInfoList &Deps,
                        const LangOptions &LO,
                        CheckerRegistry::CheckerInfoSet &Ret);

/// Collects dependenies in \p enabledCheckers. Return None on failure.
LLVM_NODISCARD
static llvm::Optional<CheckerRegistry::CheckerInfoSet>
collectDependencies(const CheckerRegistry::CheckerInfo &checker,
                    const LangOptions &LO) {

  CheckerRegistry::CheckerInfoSet Ret;
  // Add dependencies to the enabled checkers only if all of them can be
  // enabled.
  if (!collectDependenciesImpl(checker.Dependencies, LO, Ret))
    return None;

  return Ret;
}

static bool
collectDependenciesImpl(const CheckerRegistry::ConstCheckerInfoList &Deps,
                        const LangOptions &LO,
                        CheckerRegistry::CheckerInfoSet &Ret) {

  for (const CheckerRegistry::CheckerInfo *Dependency : Deps) {

    if (Dependency->isDisabled(LO))
      return false;

    // Collect dependencies recursively.
    if (!collectDependenciesImpl(Dependency->Dependencies, LO, Ret))
      return false;

    Ret.insert(Dependency);
  }

  return true;
}

CheckerRegistry::CheckerInfoSet CheckerRegistry::getEnabledCheckers() const {

  CheckerInfoSet EnabledCheckers;

  for (const CheckerInfo &Checker : Checkers) {
    if (!Checker.isEnabled(LangOpts))
      continue;

    // Recursively enable its dependencies.
    llvm::Optional<CheckerInfoSet> Deps =
        collectDependencies(Checker, LangOpts);

    if (!Deps) {
      // If we failed to enable any of the dependencies, don't enable this
      // checker.
      continue;
    }

    // Note that set_union also preserves the order of insertion.
    EnabledCheckers.set_union(*Deps);

    // Enable the checker.
    EnabledCheckers.insert(&Checker);
  }

  return EnabledCheckers;
}

void CheckerRegistry::resolveDependencies() {
  for (const std::pair<StringRef, StringRef> &Entry : Dependencies) {
    auto CheckerIt = binaryFind(Checkers, Entry.first);
    assert(CheckerIt != Checkers.end() && CheckerIt->FullName == Entry.first &&
           "Failed to find the checker while attempting to set up its "
           "dependencies!");

    auto DependencyIt = binaryFind(Checkers, Entry.second);
    assert(DependencyIt != Checkers.end() &&
           DependencyIt->FullName == Entry.second &&
           "Failed to find the dependency of a checker!");

    CheckerIt->Dependencies.emplace_back(&*DependencyIt);
  }

  Dependencies.clear();
}

void CheckerRegistry::addDependency(StringRef FullName, StringRef Dependency) {
  Dependencies.emplace_back(FullName, Dependency);
}

/// Insert the checker/package option to AnalyzerOptions' config table, and
/// validate it, if the user supplied it on the command line.
static void insertAndValidate(StringRef FullName,
                              const CheckerRegistry::CmdLineOption &Option,
                              AnalyzerOptions &AnOpts,
                              DiagnosticsEngine &Diags) {

  std::string FullOption = (FullName + ":" + Option.OptionName).str();

  auto It = AnOpts.Config.insert({FullOption, Option.DefaultValStr});

  // Insertation was successful -- CmdLineOption's constructor will validate
  // whether values received from plugins or TableGen files are correct.
  if (It.second)
    return;

  // Insertion failed, the user supplied this package/checker option on the
  // command line. If the supplied value is invalid, we'll restore the option
  // to it's default value, and if we're in non-compatibility mode, we'll also
  // emit an error.

  StringRef SuppliedValue = It.first->getValue();

  if (Option.OptionType == "bool") {
    if (SuppliedValue != "true" && SuppliedValue != "false") {
      if (AnOpts.ShouldEmitErrorsOnInvalidConfigValue) {
        Diags.Report(diag::err_analyzer_checker_option_invalid_input)
            << FullOption << "a boolean value";
      }

      It.first->setValue(Option.DefaultValStr);
    }
    return;
  }

  if (Option.OptionType == "int") {
    int Tmp;
    bool HasFailed = SuppliedValue.getAsInteger(0, Tmp);
    if (HasFailed) {
      if (AnOpts.ShouldEmitErrorsOnInvalidConfigValue) {
        Diags.Report(diag::err_analyzer_checker_option_invalid_input)
            << FullOption << "an integer value";
      }

      It.first->setValue(Option.DefaultValStr);
    }
    return;
  }
}

template <class T>
static void
insertOptionToCollection(StringRef FullName, T &Collection,
                         const CheckerRegistry::CmdLineOption &Option,
                         AnalyzerOptions &AnOpts, DiagnosticsEngine &Diags) {
  auto It = binaryFind(Collection, FullName);
  assert(It != Collection.end() &&
         "Failed to find the checker while attempting to add a command line "
         "option to it!");

  insertAndValidate(FullName, Option, AnOpts, Diags);

  It->CmdLineOptions.emplace_back(Option);
}

void CheckerRegistry::resolveCheckerAndPackageOptions() {
  for (const std::pair<StringRef, CmdLineOption> &CheckerOptEntry :
       CheckerOptions) {
    insertOptionToCollection(CheckerOptEntry.first, Checkers,
                             CheckerOptEntry.second, AnOpts, Diags);
  }
  CheckerOptions.clear();

  for (const std::pair<StringRef, CmdLineOption> &PackageOptEntry :
       PackageOptions) {
    insertOptionToCollection(PackageOptEntry.first, Packages,
                             PackageOptEntry.second, AnOpts, Diags);
  }
  PackageOptions.clear();
}

void CheckerRegistry::addPackage(StringRef FullName) {
  Packages.emplace_back(PackageInfo(FullName));
}

void CheckerRegistry::addPackageOption(StringRef OptionType,
                                       StringRef PackageFullName,
                                       StringRef OptionName,
                                       StringRef DefaultValStr,
                                       StringRef Description,
                                       StringRef DevelopmentStatus,
                                       bool IsHidden) {
  PackageOptions.emplace_back(
      PackageFullName, CmdLineOption{OptionType, OptionName, DefaultValStr,
                                     Description, DevelopmentStatus, IsHidden});
}

void CheckerRegistry::addChecker(InitializationFunction Rfn,
                                 ShouldRegisterFunction Sfn, StringRef Name,
                                 StringRef Desc, StringRef DocsUri,
                                 bool IsHidden) {
  Checkers.emplace_back(Rfn, Sfn, Name, Desc, DocsUri, IsHidden);

  // Record the presence of the checker in its packages.
  StringRef PackageName, LeafName;
  std::tie(PackageName, LeafName) = Name.rsplit(PackageSeparator);
  while (!LeafName.empty()) {
    PackageSizes[PackageName] += 1;
    std::tie(PackageName, LeafName) = PackageName.rsplit(PackageSeparator);
  }
}

void CheckerRegistry::addCheckerOption(StringRef OptionType,
                                       StringRef CheckerFullName,
                                       StringRef OptionName,
                                       StringRef DefaultValStr,
                                       StringRef Description,
                                       StringRef DevelopmentStatus,
                                       bool IsHidden) {
  CheckerOptions.emplace_back(
      CheckerFullName, CmdLineOption{OptionType, OptionName, DefaultValStr,
                                     Description, DevelopmentStatus, IsHidden});
}

void CheckerRegistry::initializeManager(CheckerManager &CheckerMgr) const {
  // Collect checkers enabled by the options.
  CheckerInfoSet enabledCheckers = getEnabledCheckers();

  // Initialize the CheckerManager with all enabled checkers.
  for (const auto *Checker : enabledCheckers) {
    CheckerMgr.setCurrentCheckerName(CheckerNameRef(Checker->FullName));
    Checker->Initialize(CheckerMgr);
  }
}

static void
isOptionContainedIn(const CheckerRegistry::CmdLineOptionList &OptionList,
                    StringRef SuppliedChecker, StringRef SuppliedOption,
                    const AnalyzerOptions &AnOpts, DiagnosticsEngine &Diags) {

  if (!AnOpts.ShouldEmitErrorsOnInvalidConfigValue)
    return;

  using CmdLineOption = CheckerRegistry::CmdLineOption;

  auto SameOptName = [SuppliedOption](const CmdLineOption &Opt) {
    return Opt.OptionName == SuppliedOption;
  };

  auto OptionIt = llvm::find_if(OptionList, SameOptName);

  if (OptionIt == OptionList.end()) {
    Diags.Report(diag::err_analyzer_checker_option_unknown)
        << SuppliedChecker << SuppliedOption;
    return;
  }
}

void CheckerRegistry::validateCheckerOptions() const {
  for (const auto &Config : AnOpts.Config) {

    StringRef SuppliedCheckerOrPackage;
    StringRef SuppliedOption;
    std::tie(SuppliedCheckerOrPackage, SuppliedOption) =
        Config.getKey().split(':');

    if (SuppliedOption.empty())
      continue;

    // AnalyzerOptions' config table contains the user input, so an entry could
    // look like this:
    //
    //   cor:NoFalsePositives=true
    //
    // Since lower_bound would look for the first element *not less* than "cor",
    // it would return with an iterator to the first checker in the core, so we
    // we really have to use find here, which uses operator==.
    auto CheckerIt =
        llvm::find(Checkers, CheckerInfo(SuppliedCheckerOrPackage));
    if (CheckerIt != Checkers.end()) {
      isOptionContainedIn(CheckerIt->CmdLineOptions, SuppliedCheckerOrPackage,
                          SuppliedOption, AnOpts, Diags);
      continue;
    }

    auto PackageIt =
        llvm::find(Packages, PackageInfo(SuppliedCheckerOrPackage));
    if (PackageIt != Packages.end()) {
      isOptionContainedIn(PackageIt->CmdLineOptions, SuppliedCheckerOrPackage,
                          SuppliedOption, AnOpts, Diags);
      continue;
    }

    Diags.Report(diag::err_unknown_analyzer_checker_or_package)
        << SuppliedCheckerOrPackage;
  }
}

void CheckerRegistry::printCheckerWithDescList(raw_ostream &Out,
                                               size_t MaxNameChars) const {
  // FIXME: Print available packages.

  Out << "CHECKERS:\n";

  // Find the maximum option length.
  size_t OptionFieldWidth = 0;
  for (const auto &Checker : Checkers) {
    // Limit the amount of padding we are willing to give up for alignment.
    //   Package.Name     Description  [Hidden]
    size_t NameLength = Checker.FullName.size();
    if (NameLength <= MaxNameChars)
      OptionFieldWidth = std::max(OptionFieldWidth, NameLength);
  }

  const size_t InitialPad = 2;

  auto Print = [=](llvm::raw_ostream &Out, const CheckerInfo &Checker,
                   StringRef Description) {
    AnalyzerOptions::printFormattedEntry(Out, {Checker.FullName, Description},
                                         InitialPad, OptionFieldWidth);
    Out << '\n';
  };

  for (const auto &Checker : Checkers) {
    // The order of this if branches is significant, we wouldn't like to display
    // developer checkers even in the alpha output. For example,
    // alpha.cplusplus.IteratorModeling is a modeling checker, hence it's hidden
    // by default, and users (even when the user is a developer of an alpha
    // checker) shouldn't normally tinker with whether they should be enabled.

    if (Checker.IsHidden) {
      if (AnOpts.ShowCheckerHelpDeveloper)
        Print(Out, Checker, Checker.Desc);
      continue;
    }

    if (Checker.FullName.startswith("alpha")) {
      if (AnOpts.ShowCheckerHelpAlpha)
        Print(Out, Checker,
              ("(Enable only for development!) " + Checker.Desc).str());
      continue;
    }

    if (AnOpts.ShowCheckerHelp)
        Print(Out, Checker, Checker.Desc);
  }
}

void CheckerRegistry::printEnabledCheckerList(raw_ostream &Out) const {
  // Collect checkers enabled by the options.
  CheckerInfoSet EnabledCheckers = getEnabledCheckers();

  for (const auto *i : EnabledCheckers)
    Out << i->FullName << '\n';
}

void CheckerRegistry::printCheckerOptionList(raw_ostream &Out) const {
  Out << "OVERVIEW: Clang Static Analyzer Checker and Package Option List\n\n";
  Out << "USAGE: -analyzer-config <OPTION1=VALUE,OPTION2=VALUE,...>\n\n";
  Out << "       -analyzer-config OPTION1=VALUE, -analyzer-config "
         "OPTION2=VALUE, ...\n\n";
  Out << "OPTIONS:\n\n";

  std::multimap<StringRef, const CmdLineOption &> OptionMap;

  for (const CheckerInfo &Checker : Checkers) {
    for (const CmdLineOption &Option : Checker.CmdLineOptions) {
      OptionMap.insert({Checker.FullName, Option});
    }
  }

  for (const PackageInfo &Package : Packages) {
    for (const CmdLineOption &Option : Package.CmdLineOptions) {
      OptionMap.insert({Package.FullName, Option});
    }
  }

  auto Print = [] (llvm::raw_ostream &Out, StringRef FullOption, StringRef Desc) {
    AnalyzerOptions::printFormattedEntry(Out, {FullOption, Desc},
                                         /*InitialPad*/ 2,
                                         /*EntryWidth*/ 50,
                                         /*MinLineWidth*/ 90);
    Out << "\n\n";
  };
  for (const std::pair<const StringRef, const CmdLineOption &> &Entry :
       OptionMap) {
    const CmdLineOption &Option = Entry.second;
    std::string FullOption = (Entry.first + ":" + Option.OptionName).str();

    std::string Desc =
        ("(" + Option.OptionType + ") " + Option.Description + " (default: " +
         (Option.DefaultValStr.empty() ? "\"\"" : Option.DefaultValStr) + ")")
            .str();

    // The list of these if branches is significant, we wouldn't like to
    // display hidden alpha checker options for
    // -analyzer-checker-option-help-alpha.

    if (Option.IsHidden) {
      if (AnOpts.ShowCheckerOptionDeveloperList)
        Print(Out, FullOption, Desc);
      continue;
    }

    if (Option.DevelopmentStatus == "alpha" ||
        Entry.first.startswith("alpha")) {
      if (AnOpts.ShowCheckerOptionAlphaList)
        Print(Out, FullOption,
              llvm::Twine("(Enable only for development!) " + Desc).str());
      continue;
    }

    if (AnOpts.ShowCheckerOptionList)
      Print(Out, FullOption, Desc);
  }
}
