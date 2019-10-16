//===--- Transformer.h - Transformer class ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_TRANSFORMER_TRANSFORMER_H_
#define LLVM_CLANG_TOOLING_TRANSFORMER_TRANSFORMER_H_

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Refactoring/AtomicChange.h"
#include "clang/Tooling/Transformer/RewriteRule.h"
#include "llvm/Support/Error.h"
#include <functional>
#include <utility>

namespace clang {
namespace tooling {
/// Handles the matcher and callback registration for a single `RewriteRule`, as
/// defined by the arguments of the constructor.
class Transformer : public ast_matchers::MatchFinder::MatchCallback {
public:
  using ChangeConsumer =
      std::function<void(Expected<clang::tooling::AtomicChange> Change)>;

  /// \param Consumer Receives each rewrite or error.  Will not necessarily be
  /// called for each match; for example, if the rewrite is not applicable
  /// because of macros, but doesn't fail.  Note that clients are responsible
  /// for handling the case that independent \c AtomicChanges conflict with each
  /// other.
  Transformer(transformer::RewriteRule Rule, ChangeConsumer Consumer)
      : Rule(std::move(Rule)), Consumer(std::move(Consumer)) {}

  /// N.B. Passes `this` pointer to `MatchFinder`.  So, this object should not
  /// be moved after this call.
  void registerMatchers(ast_matchers::MatchFinder *MatchFinder);

  /// Not called directly by users -- called by the framework, via base class
  /// pointer.
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  transformer::RewriteRule Rule;
  /// Receives each successful rewrites as an \c AtomicChange.
  ChangeConsumer Consumer;
};
} // namespace tooling
} // namespace clang

#endif // LLVM_CLANG_TOOLING_TRANSFORMER_TRANSFORMER_H_
