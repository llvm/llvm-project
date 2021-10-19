//===--- ConfigProvider.h - Loading of user configuration --------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Various clangd features have configurable behaviour (or can be disabled).
// The configuration system allows users to control this:
//  - in a user config file, a project config file, via LSP, or via flags
//  - specifying different settings for different files
// This file defines the structures used for this, that produce a Config.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_CONFIGPROVIDER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_CONFIGPROVIDER_H

#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include <chrono>
#include <string>
#include <vector>

namespace clang {
namespace clangd {
struct Config;
class ThreadsafeFS;
namespace config {

/// Describes the context used to evaluate configuration fragments.
struct Params {
  /// Absolute path to a source file we're applying the config to. Unix slashes.
  /// Empty if not configuring a particular file.
  llvm::StringRef Path;
  /// Hint that stale data is OK to improve performance (e.g. avoid IO).
  /// FreshTime sets a bound for how old the data can be.
  /// By default, providers should validate caches against the data source.
  std::chrono::steady_clock::time_point FreshTime =
      std::chrono::steady_clock::time_point::max();
};

/// Used to report problems in parsing or interpreting a config.
/// Errors reflect structurally invalid config that should be user-visible.
/// Warnings reflect e.g. unknown properties that are recoverable.
/// Notes are used to report files and fragments.
/// (This can be used to track when previous warnings/errors have been "fixed").
using DiagnosticCallback = llvm::function_ref<void(const llvm::SMDiagnostic &)>;

/// A chunk of configuration that has been fully analyzed and is ready to apply.
/// Typically this is obtained from a Fragment by calling Fragment::compile().
///
/// Calling it updates the configuration to reflect settings from the fragment.
/// Returns true if the condition was met and the settings were used.
using CompiledFragment = std::function<bool(const Params &, Config &)>;

/// A source of configuration fragments.
/// Generally these providers reflect a fixed policy for obtaining config,
/// but return different concrete configuration over time.
/// e.g. a provider that reads config from files is responsive to file changes.
class Provider {
public:
  virtual ~Provider() = default;

  /// Reads fragments from a single YAML file with a fixed path. If non-empty,
  /// Directory will be used to resolve relative paths in the fragments.
  static std::unique_ptr<Provider> fromYAMLFile(llvm::StringRef AbsPath,
                                                llvm::StringRef Directory,
                                                const ThreadsafeFS &,
                                                bool Trusted = false);
  // Reads fragments from YAML files found relative to ancestors of Params.Path.
  //
  // All fragments that exist are returned, starting from distant ancestors.
  // For instance, given RelPath of ".clangd", then for source file /foo/bar.cc,
  // the searched fragments are [/.clangd, /foo/.clangd].
  //
  // If Params does not specify a path, no fragments are returned.
  static std::unique_ptr<Provider>
  fromAncestorRelativeYAMLFiles(llvm::StringRef RelPath, const ThreadsafeFS &,
                                bool Trusted = false);

  /// A provider that includes fragments from all the supplied providers.
  /// Order is preserved; later providers take precedence over earlier ones.
  static std::unique_ptr<Provider> combine(std::vector<const Provider *>);

  /// Build a config based on this provider.
  Config getConfig(const Params &, DiagnosticCallback) const;

private:
  /// Provide fragments that may be relevant to the file.
  /// The configuration provider is not responsible for testing conditions.
  ///
  /// Providers are expected to cache compiled fragments, and only
  /// reparse/recompile when the source data has changed.
  /// Despite the need for caching, this function must be threadsafe.
  ///
  /// When parsing/compiling, the DiagnosticCallback is used to report errors.
  virtual std::vector<CompiledFragment>
  getFragments(const Params &, DiagnosticCallback) const = 0;
};

} // namespace config
} // namespace clangd
} // namespace clang

#endif
