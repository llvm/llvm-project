//===--- TidyProvider.h - create options for running clang-tidy------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_TIDYPROVIDER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_TIDYPROVIDER_H

#include "../clang-tidy/ClangTidyOptions.h"
#include "support/ThreadsafeFS.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/StringRef.h"

namespace clang {
namespace clangd {

/// A factory to modify a \ref tidy::ClangTidyOptions.
using TidyProvider =
    llvm::unique_function<void(tidy::ClangTidyOptions &,
                               /*Filename=*/llvm::StringRef) const>;

/// A factory to modify a \ref tidy::ClangTidyOptions that doesn't hold any
/// state.
using TidyProviderRef = llvm::function_ref<void(tidy::ClangTidyOptions &,
                                                /*Filename=*/llvm::StringRef)>;

TidyProvider combine(std::vector<TidyProvider> Providers);

/// Provider that just sets the defaults.
TidyProviderRef provideEnvironment();

/// Provider that will enable a nice set of default checks if none are
/// specified.
TidyProviderRef provideDefaultChecks();

/// Provider the enables a specific set of checks and warnings as errors.
TidyProvider addTidyChecks(llvm::StringRef Checks,
                           llvm::StringRef WarningsAsErrors = {});

/// Provider that will disable checks known to not work with clangd. \p
/// ExtraBadChecks specifies any other checks that should be always
/// disabled.
TidyProvider
disableUnusableChecks(llvm::ArrayRef<std::string> ExtraBadChecks = {});

/// Provider that searches for .clang-tidy configuration files in the directory
/// tree.
TidyProvider provideClangTidyFiles(ThreadsafeFS &);

// Provider that uses clangd configuration files.
TidyProviderRef provideClangdConfig();

tidy::ClangTidyOptions getTidyOptionsForFile(TidyProviderRef Provider,
                                             llvm::StringRef Filename);

/// Returns if \p Check is a registered clang-tidy check
/// \pre \p must not be empty, must not contain '*' or ',' or start with '-'.
bool isRegisteredTidyCheck(llvm::StringRef Check);

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_TIDYPROVIDER_H
