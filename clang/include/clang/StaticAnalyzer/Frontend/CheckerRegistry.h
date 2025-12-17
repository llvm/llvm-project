//===- CheckerRegistry.h - Maintains all available checkers -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the logic for parsing the TableGen file Checkers.td, and parsing the
// specific invocation of the analyzer (which checker/package is enabled, values
// of their options, etc). This is in the frontend library because checker
// registry functions are called from here but are defined in the dependent
// library libStaticAnalyzerCheckers, but the actual data structure that holds
// the parsed information is in the Core library.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_STATICANALYZER_FRONTEND_CHECKERREGISTRY_H
#define LLVM_CLANG_STATICANALYZER_FRONTEND_CHECKERREGISTRY_H

#include "clang/Basic/LLVM.h"
#include "clang/StaticAnalyzer/Core/CheckerRegistryData.h"
#include "llvm/ADT/StringRef.h"

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
//    void clang_registerCheckers(CheckerRegistry &Registry) {
//      Registry.addChecker<MainCallChecker>(
//                    "example.MainCallChecker",
//                    "Disallows calls to functions called main");
//    }
//
// The first argument of this templated method is the full name of the checker
// (including its package), while the second argument is a short description
// that is printed by `-analyzer-checker-help`.
//
// A plugin may register several separate checkers by calling `addChecker()`
// multiple times. If a checker requires custom registration functions (e.g.
// checker option handling) use the non-templated overload of `addChecker` that
// takes two callback functions as the first two parameters.
//
// To load a checker plugin, specify the full path to the dynamic library as
// the argument to the -load option in the cc1 frontend. You can then enable
// your custom checker using the -analyzer-checker:
//
//   clang -cc1 -load /path/to/plugin.dylib -analyze
//     -analyzer-checker=example.MainCallChecker
//
// For complete examples, see clang/lib/Analysis/plugins/SampleAnalyzer

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

namespace ento {

class CheckerManager;

/// Manages a set of available checkers for running a static analysis.
/// The checkers are organized into packages by full name, where including
/// a package will recursively include all subpackages and checkers within it.
/// For example, the checker "core.builtin.NoReturnFunctionChecker" will be
/// included if initializeManager() is called with an option of "core",
/// "core.builtin", or the full name "core.builtin.NoReturnFunctionChecker".
class CheckerRegistry {
public:
  CheckerRegistry(CheckerRegistryData &Data, ArrayRef<std::string> Plugins,
                  DiagnosticsEngine &Diags, AnalyzerOptions &AnOpts,
                  ArrayRef<std::function<void(CheckerRegistry &)>>
                      CheckerRegistrationFns = {});

  /// Collects all enabled checkers in the field EnabledCheckers. It preserves
  /// the order of insertion, as dependencies have to be enabled before the
  /// checkers that depend on them.
  void initializeRegistry(const CheckerManager &Mgr);


private:
  /// Default initialization function for checkers -- since CheckerManager
  /// includes this header, we need to make it a template parameter, and since
  /// the checker must be a template parameter as well, we can't put this in the
  /// cpp file.
  template <typename MGR, typename T> static void initializeManager(MGR &mgr) {
    mgr.template registerChecker<T>();
  }

  static bool returnTrue(const CheckerManager &) { return true; }

public:
  /// Adds a checker to the registry.
  /// Use this for a checker defined in a plugin if it requires custom
  /// registration functions (e.g. for handling checker options).
  /// NOTE: As of now `DocsUri` is never queried from the checker registry.
  void addChecker(RegisterCheckerFn Fn, ShouldRegisterFunction Sfn,
                  StringRef FullName, StringRef Desc,
                  StringRef DocsUri = "NoDocsUri", bool IsHidden = false);

  /// Adds a checker to the registry.
  /// Use this for a checker defined in a plugin if it doesn't require custom
  /// registration functions.
  template <class T>
  void addChecker(StringRef FullName, StringRef Desc,
                  StringRef DocsUri = "NoDocsUri", bool IsHidden = false) {
    addChecker(&CheckerRegistry::initializeManager<CheckerManager, T>,
               &CheckerRegistry::returnTrue, FullName, Desc, DocsUri, IsHidden);
  }

  /// Makes the checker with the full name \p fullName depend on the checker
  /// called \p dependency.
  void addDependency(StringRef FullName, StringRef Dependency);

  /// Makes the checker with the full name \p fullName weak depend on the
  /// checker called \p dependency.
  void addWeakDependency(StringRef FullName, StringRef Dependency);

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
                        StringRef Description, StringRef DevelopmentStatus,
                        bool IsHidden = false);

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
                        StringRef Description, StringRef DevelopmentStatus,
                        bool IsHidden = false);

  // FIXME: This *really* should be added to the frontend flag descriptions.
  /// Initializes a CheckerManager by calling the initialization functions for
  /// all checkers specified by the given CheckerOptInfo list. The order of this
  /// list is significant; later options can be used to reverse earlier ones.
  /// This can be used to exclude certain checkers in an included package.
  void initializeManager(CheckerManager &CheckerMgr) const;

  /// Check if every option corresponds to a specific checker or package.
  void validateCheckerOptions() const;

private:
  template <bool IsWeak> void resolveDependencies();
  void resolveCheckerAndPackageOptions();

  CheckerRegistryData &Data;

  DiagnosticsEngine &Diags;
  AnalyzerOptions &AnOpts;
};

} // namespace ento
} // namespace clang

#endif // LLVM_CLANG_STATICANALYZER_FRONTEND_CHECKERREGISTRY_H
