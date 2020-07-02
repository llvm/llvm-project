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
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include <string>
#include <vector>

namespace clang {
namespace clangd {
struct Config;
namespace config {

/// Describes the context used to evaluate configuration fragments.
struct Params {
  /// Absolute path to a source file we're applying the config to. Unix slashes.
  /// Empty if not configuring a particular file.
  llvm::StringRef Path;
};

/// Used to report problems in parsing or interpreting a config.
/// Errors reflect structurally invalid config that should be user-visible.
/// Warnings reflect e.g. unknown properties that are recoverable.
using DiagnosticCallback = llvm::function_ref<void(const llvm::SMDiagnostic &)>;

/// A chunk of configuration that has been fully analyzed and is ready to apply.
/// Typically this is obtained from a Fragment by calling Fragment::compile().
///
/// Calling it updates the configuration to reflect settings from the fragment.
/// Returns true if the condition was met and the settings were used.
using CompiledFragment = std::function<bool(const Params &, Config &)>;

} // namespace config
} // namespace clangd
} // namespace clang

#endif
