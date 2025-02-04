//===--- Config.h - User configuration of clangd behavior --------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Various clangd features have configurable behaviour (or can be disabled).
// This file defines "resolved" configuration seen by features within clangd.
// For example, settings may vary per-file, the resolved Config only contains
// settings that apply to the current file.
//
// This is distinct from how the config is specified by the user (Fragment)
// interpreted (CompiledFragment), and combined (Provider).
// ConfigFragment.h describes the steps to add a new configuration option.
//
// Because this structure is shared throughout clangd, it's a potential source
// of layering problems. Config should be expressed in terms of simple
// vocabulary types where possible.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_CONFIG_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_CONFIG_H

#include "support/Context.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace clang {
namespace clangd {

/// Settings that express user/project preferences and control clangd behavior.
///
/// Generally, features should consume Config::current() and the caller is
/// responsible for setting it appropriately. In practice these callers are
/// ClangdServer, TUScheduler, and BackgroundQueue.
struct Config {
  /// Returns the Config of the current Context, or an empty configuration.
  static const Config &current();
  /// Context key which can be used to set the current Config.
  static clangd::Key<Config> Key;

  Config() = default;
  Config(const Config &) = delete;
  Config &operator=(const Config &) = delete;
  Config(Config &&) = default;
  Config &operator=(Config &&) = default;

  struct CDBSearchSpec {
    enum { Ancestors, FixedDir, NoCDBSearch } Policy = Ancestors;
    // Absolute, native slashes, no trailing slash.
    std::optional<std::string> FixedCDBPath;
  };

  /// Controls how the compile command for the current file is determined.
  struct {
    /// Edits to apply to the compile command, in sequence.
    std::vector<llvm::unique_function<void(std::vector<std::string> &) const>>
        Edits;
    /// Where to search for compilation databases for this file's flags.
    CDBSearchSpec CDBSearch = {CDBSearchSpec::Ancestors, std::nullopt};
  } CompileFlags;

  enum class BackgroundPolicy { Build, Skip };
  /// Describes an external index configuration.
  struct ExternalIndexSpec {
    enum { None, File, Server } Kind = None;
    /// This is one of:
    /// - Address of a clangd-index-server, in the form of "ip:port".
    /// - Absolute path to an index produced by clangd-indexer.
    std::string Location;
    /// Absolute path to source root this index is associated with, uses
    /// forward-slashes.
    std::string MountPoint;
  };
  /// Controls index behavior.
  struct {
    /// Whether this TU should be background-indexed.
    BackgroundPolicy Background = BackgroundPolicy::Build;
    ExternalIndexSpec External;
    bool StandardLibrary = true;
  } Index;

  enum class IncludesPolicy {
    /// Diagnose missing and unused includes.
    Strict,
    None,
  };
  enum class FastCheckPolicy { Strict, Loose, None };
  /// Controls warnings and errors when parsing code.
  struct {
    bool SuppressAll = false;
    llvm::StringSet<> Suppress;

    /// Configures what clang-tidy checks to run and options to use with them.
    struct {
      // A comma-separated list of globs specify which clang-tidy checks to run.
      std::string Checks;
      llvm::StringMap<std::string> CheckOptions;
      FastCheckPolicy FastCheckFilter = FastCheckPolicy::Strict;
    } ClangTidy;

    IncludesPolicy UnusedIncludes = IncludesPolicy::Strict;
    IncludesPolicy MissingIncludes = IncludesPolicy::None;

    struct {
      /// IncludeCleaner will not diagnose usages of these headers matched by
      /// these regexes.
      std::vector<std::function<bool(llvm::StringRef)>> IgnoreHeader;
      bool AnalyzeAngledIncludes = false;
    } Includes;
  } Diagnostics;

  /// Style of the codebase.
  struct {
    // Namespaces that should always be fully qualified, meaning no "using"
    // declarations, always spell out the whole name (with or without leading
    // ::). All nested namespaces are affected as well.
    std::vector<std::string> FullyQualifiedNamespaces;

    // List of matcher functions for inserting certain headers with <> or "".
    std::vector<std::function<bool(llvm::StringRef)>> QuotedHeaders;
    std::vector<std::function<bool(llvm::StringRef)>> AngledHeaders;
  } Style;

  /// controls the completion options for argument lists.
  enum class ArgumentListsPolicy {
    /// nothing, no argument list and also NO Delimiters "()" or "<>".
    None,
    /// open, only opening delimiter "(" or "<".
    OpenDelimiter,
    /// empty pair of delimiters "()" or "<>".
    Delimiters,
    /// full name of both type and variable.
    FullPlaceholders,
  };

  /// Configures code completion feature.
  struct {
    /// Whether code completion includes results that are not visible in current
    /// scopes.
    bool AllScopes = true;
    /// controls the completion options for argument lists.
    ArgumentListsPolicy ArgumentLists = ArgumentListsPolicy::FullPlaceholders;
  } Completion;

  /// Configures hover feature.
  struct {
    /// Whether hover show a.k.a type.
    bool ShowAKA = true;
  } Hover;

  struct {
    /// If false, inlay hints are completely disabled.
    bool Enabled = true;

    // Whether specific categories of hints are enabled.
    bool Parameters = true;
    bool DeducedTypes = true;
    bool Designators = true;
    bool BlockEnd = false;
    bool DefaultArguments = false;
    // Limit the length of type names in inlay hints. (0 means no limit)
    uint32_t TypeNameLimit = 32;
  } InlayHints;

  struct {
    /// Controls highlighting kinds that are disabled.
    std::vector<std::string> DisabledKinds;
    /// Controls highlighting modifiers that are disabled.
    std::vector<std::string> DisabledModifiers;
  } SemanticTokens;
};

} // namespace clangd
} // namespace clang

namespace llvm {
template <> struct DenseMapInfo<clang::clangd::Config::ExternalIndexSpec> {
  using ExternalIndexSpec = clang::clangd::Config::ExternalIndexSpec;
  static inline ExternalIndexSpec getEmptyKey() {
    return {ExternalIndexSpec::File, "", ""};
  }
  static inline ExternalIndexSpec getTombstoneKey() {
    return {ExternalIndexSpec::File, "TOMB", "STONE"};
  }
  static unsigned getHashValue(const ExternalIndexSpec &Val) {
    return llvm::hash_combine(Val.Kind, Val.Location, Val.MountPoint);
  }
  static bool isEqual(const ExternalIndexSpec &LHS,
                      const ExternalIndexSpec &RHS) {
    return std::tie(LHS.Kind, LHS.Location, LHS.MountPoint) ==
           std::tie(RHS.Kind, RHS.Location, RHS.MountPoint);
  }
};
} // namespace llvm

#endif
