//===--- ConfigFragment.h - Unit of user-specified configuration -*- C++-*-===//
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
//
// This file defines the config::Fragment structure which models one piece of
// configuration as obtained from a source like a file.
//
// This is distinct from how the config is interpreted (CompiledFragment),
// combined (Provider) and exposed to the rest of clangd (Config).
//
//===----------------------------------------------------------------------===//
//
// To add a new configuration option, you must:
//  - add its syntactic form to Fragment
//  - update ConfigYAML.cpp to parse it
//  - add its semantic form to Config (in Config.h)
//  - update ConfigCompile.cpp to map Fragment -> Config
//  - make use of the option inside clangd
//  - document the new option (config.md in the llvm/clangd-www repository)
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_CONFIGFRAGMENT_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_CONFIGFRAGMENT_H

#include "ConfigProvider.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include <optional>
#include <string>
#include <vector>

namespace clang {
namespace clangd {
namespace config {

/// An entity written in config along, with its optional location in the file.
template <typename T> struct Located {
  Located(T Value, llvm::SMRange Range = {})
      : Range(Range), Value(std::move(Value)) {}

  llvm::SMRange Range;
  T *operator->() { return &Value; }
  const T *operator->() const { return &Value; }
  T &operator*() { return Value; }
  const T &operator*() const { return Value; }

private:
  T Value;
};

/// A chunk of configuration obtained from a config file, LSP, or elsewhere.
struct Fragment {
  /// Parses fragments from a YAML file (one from each --- delimited document).
  /// Documents that contained fatal errors are omitted from the results.
  /// BufferName is used for the SourceMgr and diagnostics.
  static std::vector<Fragment> parseYAML(llvm::StringRef YAML,
                                         llvm::StringRef BufferName,
                                         DiagnosticCallback);

  /// Analyzes and consumes this fragment, possibly yielding more diagnostics.
  /// This always produces a usable result (errors are recovered).
  ///
  /// Typically, providers will compile a Fragment once when it's first loaded,
  /// caching the result for reuse.
  /// Like a compiled program, this is good for performance and also encourages
  /// errors to be reported early and only once.
  ///
  /// The returned function is a cheap-copyable wrapper of refcounted internals.
  CompiledFragment compile(DiagnosticCallback) &&;

  /// These fields are not part of the user-specified configuration, but
  /// instead are populated by the parser to describe the configuration source.
  struct SourceInfo {
    /// Retains a buffer of the original source this fragment was parsed from.
    /// Locations within Located<T> objects point into this SourceMgr.
    /// Shared because multiple fragments are often parsed from one (YAML) file.
    /// May be null, then all locations should be ignored.
    std::shared_ptr<llvm::SourceMgr> Manager;
    /// The start of the original source for this fragment.
    /// Only valid if SourceManager is set.
    llvm::SMLoc Location;
    /// Absolute path to directory the fragment is associated with. Relative
    /// paths mentioned in the fragment are resolved against this.
    std::string Directory;
    /// Whether this fragment is allowed to make critical security/privacy
    /// decisions.
    bool Trusted = false;
  };
  SourceInfo Source;

  /// Conditions in the If block restrict when a Fragment applies.
  ///
  /// Each separate condition must match (combined with AND).
  /// When one condition has multiple values, any may match (combined with OR).
  /// e.g. `PathMatch: [foo/.*, bar/.*]` matches files in either directory.
  ///
  /// Conditions based on a file's path use the following form:
  /// - if the fragment came from a project directory, the path is relative
  /// - if the fragment is global (e.g. user config), the path is absolute
  /// - paths always use forward-slashes (UNIX-style)
  /// If no file is being processed, these conditions will not match.
  struct IfBlock {
    /// The file being processed must fully match a regular expression.
    std::vector<Located<std::string>> PathMatch;
    /// The file being processed must *not* fully match a regular expression.
    std::vector<Located<std::string>> PathExclude;

    /// An unrecognized key was found while parsing the condition.
    /// The condition will evaluate to false.
    bool HasUnrecognizedCondition = false;
  };
  IfBlock If;

  /// Conditions in the CompileFlags block affect how a file is parsed.
  ///
  /// clangd emulates how clang would interpret a file.
  /// By default, it behaves roughly like `clang $FILENAME`, but real projects
  /// usually require setting the include path (with the `-I` flag), defining
  /// preprocessor symbols, configuring warnings etc.
  /// Often, a compilation database specifies these compile commands. clangd
  /// searches for compile_commands.json in parents of the source file.
  ///
  /// This section modifies how the compile command is constructed.
  struct CompileFlagsBlock {
    /// Override the compiler executable name to simulate.
    ///
    /// The name can affect how flags are parsed (clang++ vs clang).
    /// If the executable name is in the --query-driver allowlist, then it will
    /// be invoked to extract include paths.
    ///
    /// (That this simply replaces argv[0], and may mangle commands that use
    /// more complicated drivers like ccache).
    std::optional<Located<std::string>> Compiler;

    /// List of flags to append to the compile command.
    std::vector<Located<std::string>> Add;
    /// List of flags to remove from the compile command.
    ///
    /// - If the value is a recognized clang flag (like "-I") then it will be
    ///   removed along with any arguments. Synonyms like --include-dir= will
    ///   also be removed.
    /// - Otherwise, if the value ends in * (like "-DFOO=*") then any argument
    ///   with the prefix will be removed.
    /// - Otherwise any argument exactly matching the value is removed.
    ///
    /// In all cases, -Xclang is also removed where needed.
    ///
    /// Example:
    ///   Command: clang++ --include-directory=/usr/include -DFOO=42 foo.cc
    ///   Remove: [-I, -DFOO=*]
    ///   Result: clang++ foo.cc
    ///
    /// Flags added by the same CompileFlags entry will not be removed.
    std::vector<Located<std::string>> Remove;

    /// Directory to search for compilation database (compile_commands.json
    /// etc). Valid values are:
    /// - A single path to a directory (absolute, or relative to the fragment)
    /// - Ancestors: search all parent directories (the default)
    /// - std::nullopt: do not use a compilation database, just default flags.
    std::optional<Located<std::string>> CompilationDatabase;
  };
  CompileFlagsBlock CompileFlags;

  /// Controls how clangd understands code outside the current file.
  /// clangd's indexes provide information about symbols that isn't available
  /// to clang's parser, such as incoming references.
  struct IndexBlock {
    /// Whether files are built in the background to produce a project index.
    /// This is checked for translation units only, not headers they include.
    /// Legal values are "Build" or "Skip".
    std::optional<Located<std::string>> Background;
    /// An external index uses data source outside of clangd itself. This is
    /// usually prepared using clangd-indexer.
    /// Exactly one source (File/Server) should be configured.
    struct ExternalBlock {
      /// Whether the block is explicitly set to `None`. Can be used to clear
      /// any external index specified before.
      Located<bool> IsNone = false;
      /// Path to an index file generated by clangd-indexer. Relative paths may
      /// be used, if config fragment is associated with a directory.
      std::optional<Located<std::string>> File;
      /// Address and port number for a clangd-index-server. e.g.
      /// `123.1.1.1:13337`.
      std::optional<Located<std::string>> Server;
      /// Source root governed by this index. Default is the directory
      /// associated with the config fragment. Absolute in case of user config
      /// and relative otherwise. Should always use forward-slashes.
      std::optional<Located<std::string>> MountPoint;
    };
    std::optional<Located<ExternalBlock>> External;
    // Whether the standard library visible from this file should be indexed.
    // This makes all standard library symbols available, included or not.
    std::optional<Located<bool>> StandardLibrary;
  };
  IndexBlock Index;

  /// Controls behavior of diagnostics (errors and warnings).
  struct DiagnosticsBlock {
    /// Diagnostic codes that should be suppressed.
    ///
    /// Valid values are:
    /// - *, to disable all diagnostics
    /// - diagnostic codes exposed by clangd (e.g unknown_type, -Wunused-result)
    /// - clang internal diagnostic codes (e.g. err_unknown_type)
    /// - warning categories (e.g. unused-result)
    /// - clang-tidy check names (e.g. bugprone-narrowing-conversions)
    ///
    /// This is a simple filter. Diagnostics can be controlled in other ways
    /// (e.g. by disabling a clang-tidy check, or the -Wunused compile flag).
    /// This often has other advantages, such as skipping some analysis.
    std::vector<Located<std::string>> Suppress;

    /// Controls how clangd will correct "unnecessary" #include directives.
    /// clangd can warn if a header is `#include`d but not used, and suggest
    /// removing it.
    //
    /// Strict means a header is unused if it does not *directly* provide any
    /// symbol used in the file. Removing it may still break compilation if it
    /// transitively includes headers that are used. This should be fixed by
    /// including those headers directly.
    ///
    /// Valid values are:
    /// - Strict
    /// - std::nullopt
    std::optional<Located<std::string>> UnusedIncludes;

    /// Controls if clangd should analyze missing #include directives.
    /// clangd will warn if no header providing a symbol is `#include`d
    /// (missing) directly, and suggest adding it.
    ///
    /// Strict means a header providing a symbol is missing if it is not
    /// *directly #include'd. The file might still compile if the header is
    /// included transitively.
    ///
    /// Valid values are:
    /// - Strict
    /// - std::nullopt
    std::optional<Located<std::string>> MissingIncludes;

    /// Controls IncludeCleaner diagnostics.
    struct IncludesBlock {
      /// Regexes that will be used to avoid diagnosing certain includes as
      /// unused or missing. These can match any suffix of the header file in
      /// question.
      std::vector<Located<std::string>> IgnoreHeader;
    };
    IncludesBlock Includes;

    /// Controls how clang-tidy will run over the code base.
    ///
    /// The settings are merged with any settings found in .clang-tidy
    /// configuration files with these ones taking precedence.
    struct ClangTidyBlock {
      std::vector<Located<std::string>> Add;
      /// List of checks to disable.
      /// Takes precedence over Add. To enable all llvm checks except include
      /// order:
      ///   Add: llvm-*
      ///   Remove: llvm-include-order
      std::vector<Located<std::string>> Remove;

      /// A Key-Value pair list of options to pass to clang-tidy checks
      /// These take precedence over options specified in clang-tidy
      /// configuration files. Example:
      ///   CheckOptions:
      ///     readability-braces-around-statements.ShortStatementLines: 2
      std::vector<std::pair<Located<std::string>, Located<std::string>>>
          CheckOptions;
    };
    ClangTidyBlock ClangTidy;
  };
  DiagnosticsBlock Diagnostics;

  // Describes the style of the codebase, beyond formatting.
  struct StyleBlock {
    // Namespaces that should always be fully qualified, meaning no "using"
    // declarations, always spell out the whole name (with or without leading
    // ::). All nested namespaces are affected as well.
    // Affects availability of the AddUsing tweak.
    std::vector<Located<std::string>> FullyQualifiedNamespaces;
  };
  StyleBlock Style;

  /// Describes code completion preferences.
  struct CompletionBlock {
    /// Whether code completion should include suggestions from scopes that are
    /// not visible. The required scope prefix will be inserted.
    std::optional<Located<bool>> AllScopes;
  };
  CompletionBlock Completion;

  /// Describes hover preferences.
  struct HoverBlock {
    /// Whether hover show a.k.a type.
    std::optional<Located<bool>> ShowAKA;
  };
  HoverBlock Hover;

  /// Configures labels shown inline with the code.
  struct InlayHintsBlock {
    /// Enables/disables the inlay-hints feature.
    std::optional<Located<bool>> Enabled;

    /// Show parameter names before function arguments.
    std::optional<Located<bool>> ParameterNames;
    /// Show deduced types for `auto`.
    std::optional<Located<bool>> DeducedTypes;
    /// Show designators in aggregate initialization.
    std::optional<Located<bool>> Designators;
    /// Show defined symbol names at the end of a definition block.
    std::optional<Located<bool>> BlockEnd;
    /// Limit the length of type name hints. (0 means no limit)
    std::optional<Located<uint32_t>> TypeNameLimit;
  };
  InlayHintsBlock InlayHints;

  /// Configures semantic tokens that are produced by clangd.
  struct SemanticTokensBlock {
    /// Disables clangd to produce semantic tokens for the given kinds.
    std::vector<Located<std::string>> DisabledKinds;
    /// Disables clangd to assign semantic tokens with the given modifiers.
    std::vector<Located<std::string>> DisabledModifiers;
  };
  SemanticTokensBlock SemanticTokens;
};

} // namespace config
} // namespace clangd
} // namespace clang

#endif
