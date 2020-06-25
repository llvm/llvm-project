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

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
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
  T &operator->() { return Value; }
  const T &operator->() const { return Value; }
  T &operator*() { return Value; }
  const T &operator*() const { return Value; }

private:
  T Value;
};

/// Used to report problems in parsing or interpreting a config.
/// Errors reflect structurally invalid config that should be user-visible.
/// Warnings reflect e.g. unknown properties that are recoverable.
using DiagnosticCallback = llvm::function_ref<void(const llvm::SMDiagnostic &)>;

/// A chunk of configuration obtained from a config file, LSP, or elsewhere.
struct Fragment {
  /// Parses fragments from a YAML file (one from each --- delimited document).
  /// Documents that contained fatal errors are omitted from the results.
  /// BufferName is used for the SourceMgr and diagnostics.
  static std::vector<Fragment> parseYAML(llvm::StringRef YAML,
                                         llvm::StringRef BufferName,
                                         DiagnosticCallback);

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
  };
  SourceInfo Source;

  /// Conditions restrict when a Fragment applies.
  ///
  /// Each separate condition must match (combined with AND).
  /// When one condition has multiple values, any may match (combined with OR).
  ///
  /// Conditions based on a file's path use the following form:
  /// - if the fragment came from a project directory, the path is relative
  /// - if the fragment is global (e.g. user config), the path is absolute
  /// - paths always use forward-slashes (UNIX-style)
  /// If no file is being processed, these conditions will not match.
  struct ConditionBlock {
    /// The file being processed must fully match a regular expression.
    std::vector<Located<std::string>> PathMatch;
    /// An unrecognized key was found while parsing the condition.
    /// The condition will evaluate to false.
    bool HasUnrecognizedCondition = false;
  };
  ConditionBlock Condition;

  struct CompileFlagsBlock {
    std::vector<Located<std::string>> Add;
  } CompileFlags;
};

} // namespace config
} // namespace clangd
} // namespace clang

#endif
