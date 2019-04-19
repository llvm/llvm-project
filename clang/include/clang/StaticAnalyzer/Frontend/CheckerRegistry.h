//===- CheckerRegistry.h - Maintains all available checkers -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_STATICANALYZER_CORE_CHECKERREGISTRY_H
#define LLVM_CLANG_STATICANALYZER_CORE_CHECKERREGISTRY_H

#include "clang/Basic/LLVM.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <cstddef>
#include <vector>

// FIXME: move this information to an HTML file in docs/.
// At the very least, a checker plugin is a dynamic library that exports
// clang_analyzerAPIVersionString. This should be defined as follows:
//
//   extern "C"
//   const char clang_analyzerAPIVersionString[] =
//     CLANG_ANALYZER_API_VERSION_STRING;
//
// This is used to check whether the current version of the analyzer is known to
// be incompatible with a plugin. Plugins with incompatible version strings,
// or without a version string at all, will not be loaded.
//
// To add a custom checker to the analyzer, the plugin must also define the
// function clang_registerCheckers. For example:
//
//    extern "C"
//    void clang_registerCheckers (CheckerRegistry &registry) {
//      registry.addChecker<MainCallChecker>("example.MainCallChecker",
//        "Disallows calls to functions called main");
//    }
//
// The first method argument is the full name of the checker, including its
// enclosing package. By convention, the registered name of a checker is the
// name of the associated class (the template argument).
// The second method argument is a short human-readable description of the
// checker.
//
// The clang_registerCheckers function may add any number of checkers to the
// registry. If any checkers require additional initialization, use the three-
// argument form of CheckerRegistry::addChecker.
//
// To load a checker plugin, specify the full path to the dynamic library as
// the argument to the -load option in the cc1 frontend. You can then enable
// your custom checker using the -analyzer-checker:
//
//   clang -cc1 -load </path/to/plugin.dylib> -analyze
//     -analyzer-checker=<example.MainCallChecker>
//
// For a complete working example, see examples/analyzer-plugin.

#ifndef CLANG_ANALYZER_API_VERSION_STRING
// FIXME: The Clang version string is not particularly granular;
// the analyzer infrastructure can change a lot between releases.
// Unfortunately, this string has to be statically embedded in each plugin,
// so we can't just use the functions defined in Version.h.
#include "clang/Basic/Version.h"
#define CLANG_ANALYZER_API_VERSION_STRING CLANG_VERSION_STRING
#endif

namespace clang {

class AnalyzerOptions;
class DiagnosticsEngine;
class LangOptions;

namespace ento {

/// Manages a set of available checkers for running a static analysis.
/// The checkers are organized into packages by full name, where including
/// a package will recursively include all subpackages and checkers within it.
/// For example, the checker "core.builtin.NoReturnFunctionChecker" will be
/// included if initializeManager() is called with an option of "core",
/// "core.builtin", or the full name "core.builtin.NoReturnFunctionChecker".
class CheckerRegistry {
public:
  CheckerRegistry(ArrayRef<std::string> plugins, DiagnosticsEngine &diags,
                  AnalyzerOptions &AnOpts, const LangOptions &LangOpts,
                  ArrayRef<std::function<void(CheckerRegistry &)>>
                      checkerRegistrationFns = {});

  /// Initialization functions perform any necessary setup for a checker.
  /// They should include a call to CheckerManager::registerChecker.
  using InitializationFunction = void (*)(CheckerManager &);
  using ShouldRegisterFunction = bool (*)(const LangOptions &);

  /// Specifies a command line option. It may either belong to a checker or a
  /// package.
  struct CmdLineOption {
    StringRef OptionType;
    StringRef OptionName;
    StringRef DefaultValStr;
    StringRef Description;

    CmdLineOption(StringRef OptionType, StringRef OptionName,
                  StringRef DefaultValStr, StringRef Description)
        : OptionType(OptionType), OptionName(OptionName),
          DefaultValStr(DefaultValStr), Description(Description) {

      assert((OptionType == "bool" || OptionType == "string" ||
              OptionType == "int") &&
             "Unknown command line option type!");
    }
  };

  using CmdLineOptionList = llvm::SmallVector<CmdLineOption, 0>;

  struct CheckerInfo;

  using CheckerInfoList = std::vector<CheckerInfo>;
  using CheckerInfoListRange = llvm::iterator_range<CheckerInfoList::iterator>;
  using ConstCheckerInfoList = llvm::SmallVector<const CheckerInfo *, 0>;
  using CheckerInfoSet = llvm::SetVector<const CheckerInfo *>;

  /// Specifies a checker. Note that this isn't what we call a checker object,
  /// it merely contains everything required to create one.
  struct CheckerInfo {
    enum class StateFromCmdLine {
      // This checker wasn't explicitly enabled or disabled.
      State_Unspecified,
      // This checker was explicitly disabled.
      State_Disabled,
      // This checker was explicitly enabled.
      State_Enabled
    };

    InitializationFunction Initialize = nullptr;
    ShouldRegisterFunction ShouldRegister = nullptr;
    StringRef FullName;
    StringRef Desc;
    StringRef DocumentationUri;
    CmdLineOptionList CmdLineOptions;
    StateFromCmdLine State = StateFromCmdLine::State_Unspecified;

    ConstCheckerInfoList Dependencies;

    bool isEnabled(const LangOptions &LO) const {
      return State == StateFromCmdLine::State_Enabled && ShouldRegister(LO);
    }

    bool isDisabled(const LangOptions &LO) const {
      return State == StateFromCmdLine::State_Disabled && ShouldRegister(LO);
    }

    CheckerInfo(InitializationFunction Fn, ShouldRegisterFunction sfn,
                StringRef Name, StringRef Desc, StringRef DocsUri)
        : Initialize(Fn), ShouldRegister(sfn), FullName(Name), Desc(Desc),
          DocumentationUri(DocsUri) {}

    // Used for lower_bound.
    explicit CheckerInfo(StringRef FullName) : FullName(FullName) {}
  };

  using StateFromCmdLine = CheckerInfo::StateFromCmdLine;

  /// Specifies a package. Each package option is implicitly an option for all
  /// checkers within the package.
  struct PackageInfo {
    StringRef FullName;
    CmdLineOptionList CmdLineOptions;

    // Since each package must have a different full name, we can identify
    // CheckerInfo objects by them.
    bool operator==(const PackageInfo &Rhs) const {
      return FullName == Rhs.FullName;
    }

    explicit PackageInfo(StringRef FullName) : FullName(FullName) {}
  };

  using PackageInfoList = llvm::SmallVector<PackageInfo, 0>;

private:
  template <typename T> static void initializeManager(CheckerManager &mgr) {
    mgr.registerChecker<T>();
  }

  template <typename T> static bool returnTrue(const LangOptions &LO) {
    return true;
  }

public:
  /// Adds a checker to the registry. Use this non-templated overload when your
  /// checker requires custom initialization.
  void addChecker(InitializationFunction Fn, ShouldRegisterFunction sfn,
                  StringRef FullName, StringRef Desc, StringRef DocsUri);

  /// Adds a checker to the registry. Use this templated overload when your
  /// checker does not require any custom initialization.
  template <class T>
  void addChecker(StringRef FullName, StringRef Desc, StringRef DocsUri) {
    // Avoid MSVC's Compiler Error C2276:
    // http://msdn.microsoft.com/en-us/library/850cstw1(v=VS.80).aspx
    addChecker(&CheckerRegistry::initializeManager<T>,
               &CheckerRegistry::returnTrue<T>, FullName, Desc, DocsUri);
  }

  /// Makes the checker with the full name \p fullName depends on the checker
  /// called \p dependency.
  void addDependency(StringRef FullName, StringRef Dependency);

  /// Registers an option to a given checker. A checker option will always have
  /// the following format:
  ///   CheckerFullName:OptionName=Value
  /// And can be specified from the command line like this:
  ///   -analyzer-config CheckerFullName:OptionName=Value
  ///
  /// Options for unknown checkers, or unknown options for a given checker, or
  /// invalid value types for that given option are reported as an error in
  /// non-compatibility mode.
  void addCheckerOption(StringRef OptionType, StringRef CheckerFullName,
                        StringRef OptionName, StringRef DefaultValStr,
                        StringRef Description);

  /// Adds a package to the registry.
  void addPackage(StringRef FullName);

  /// Registers an option to a given package. A package option will always have
  /// the following format:
  ///   PackageFullName:OptionName=Value
  /// And can be specified from the command line like this:
  ///   -analyzer-config PackageFullName:OptionName=Value
  ///
  /// Options for unknown packages, or unknown options for a given package, or
  /// invalid value types for that given option are reported as an error in
  /// non-compatibility mode.
  void addPackageOption(StringRef OptionType, StringRef PackageFullName,
                        StringRef OptionName, StringRef DefaultValStr,
                        StringRef Description);

  // FIXME: This *really* should be added to the frontend flag descriptions.
  /// Initializes a CheckerManager by calling the initialization functions for
  /// all checkers specified by the given CheckerOptInfo list. The order of this
  /// list is significant; later options can be used to reverse earlier ones.
  /// This can be used to exclude certain checkers in an included package.
  void initializeManager(CheckerManager &CheckerMgr) const;

  /// Check if every option corresponds to a specific checker or package.
  void validateCheckerOptions() const;

  /// Prints the name and description of all checkers in this registry.
  /// This output is not intended to be machine-parseable.
  void printCheckerWithDescList(raw_ostream &Out,
                                size_t MaxNameChars = 30) const;
  void printEnabledCheckerList(raw_ostream &Out) const;

private:
  /// Collect all enabled checkers. The returned container preserves the order
  /// of insertion, as dependencies have to be enabled before the checkers that
  /// depend on them.
  CheckerInfoSet getEnabledCheckers() const;

  /// Return an iterator range of mutable CheckerInfos \p CmdLineArg applies to.
  /// For example, it'll return the checkers for the core package, if
  /// \p CmdLineArg is "core".
  CheckerInfoListRange getMutableCheckersForCmdLineArg(StringRef CmdLineArg);

  CheckerInfoList Checkers;
  PackageInfoList Packages;
  /// Used for couting how many checkers belong to a certain package in the
  /// \c Checkers field. For convenience purposes.
  llvm::StringMap<size_t> PackageSizes;

  /// Contains all (Dependendent checker, Dependency) pairs. We need this, as
  /// we'll resolve dependencies after all checkers were added first.
  llvm::SmallVector<std::pair<StringRef, StringRef>, 0> Dependencies;
  void resolveDependencies();

  /// Contains all (FullName, CmdLineOption) pairs. Similarly to dependencies,
  /// we only modify the actual CheckerInfo and PackageInfo objects once all
  /// of them have been added.
  llvm::SmallVector<std::pair<StringRef, CmdLineOption>, 0> PackageOptions;
  llvm::SmallVector<std::pair<StringRef, CmdLineOption>, 0> CheckerOptions;

  void resolveCheckerAndPackageOptions();

  DiagnosticsEngine &Diags;
  AnalyzerOptions &AnOpts;
  const LangOptions &LangOpts;
};

} // namespace ento
} // namespace clang

#endif // LLVM_CLANG_STATICANALYZER_CORE_CHECKERREGISTRY_H
