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
// vocubulary types where possible.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_CONFIG_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_CONFIG_H

#include "support/Context.h"
#include "llvm/ADT/FunctionExtras.h"
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

  /// Controls how the compile command for the current file is determined.
  struct {
    // Edits to apply to the compile command, in sequence.
    std::vector<llvm::unique_function<void(std::vector<std::string> &) const>>
        Edits;
  } CompileFlags;

  enum class BackgroundPolicy { Build, Skip };
  /// Controls background-index behavior.
  struct {
    /// Whether this TU should be indexed.
    BackgroundPolicy Background = BackgroundPolicy::Build;
  } Index;
};

} // namespace clangd
} // namespace clang

#endif
