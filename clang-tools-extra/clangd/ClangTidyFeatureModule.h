//===--- ClangTidyFeatureModule.h - clang-tidy integration -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_CLANGTIDYFEATUREMODULE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_CLANGTIDYFEATUREMODULE_H

#include "FeatureModule.h"
#include "TidyProvider.h"
#include <memory>
#include <mutex>
#include <utility>

namespace clang {
namespace clangd {

/// Runs clang-tidy checks as part of clangd's AST build.
class ClangTidyFeatureModule final : public FeatureModule {
public:
  /// Construct explicitly with the application's clang-tidy options provider.
  /// Takes ownership of the provider; an empty provider disables checks.
  explicit ClangTidyFeatureModule(TidyProvider OptionsProvider) {
    setProvider(std::move(OptionsProvider));
  }

  /// Swap provider snapshots, e.g. to temporarily override the configuration
  /// for benchmarking and restore it afterwards. Existing listeners retain
  /// their snapshot.
  void swapProvider(std::shared_ptr<const TidyProvider> &OptionsProvider);

  std::unique_ptr<ASTListener> astListeners() override;

private:
  void setProvider(TidyProvider OptionsProvider);

  std::mutex ProviderMu;
  std::shared_ptr<const TidyProvider> Provider;
};

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_CLANGTIDYFEATUREMODULE_H
