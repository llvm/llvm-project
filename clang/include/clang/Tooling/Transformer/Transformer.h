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
  /// Provides the set of changes to the consumer.  The callback is free to move
  /// or destructively consume the changes as needed.
  ///
  /// We use \c MutableArrayRef as an abstraction to provide decoupling, and we
  /// expect the majority of consumers to copy or move the individual values
  /// into a separate data structure.
  using ChangeSetConsumer = std::function<void(
      Expected<llvm::MutableArrayRef<AtomicChange>> Changes)>;

  /// \param Consumer Receives all rewrites for a single match, or an error.
  /// Will not necessarily be called for each match; for example, if the rule
  /// generates no edits but does not fail.  Note that clients are responsible
  /// for handling the case that independent \c AtomicChanges conflict with each
  /// other.
  explicit Transformer(transformer::RewriteRule Rule,
                       ChangeSetConsumer Consumer)
      : Rule(std::move(Rule)), Consumer(std::move(Consumer)) {
    assert(this->Consumer && "Consumer is empty");
  }

  /// N.B. Passes `this` pointer to `MatchFinder`.  So, this object should not
  /// be moved after this call.
  void registerMatchers(ast_matchers::MatchFinder *MatchFinder);

  /// Not called directly by users -- called by the framework, via base class
  /// pointer.
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  transformer::RewriteRule Rule;
  /// Receives sets of successful rewrites as an
  /// \c llvm::ArrayRef<AtomicChange>.
  ChangeSetConsumer Consumer;
};
} // namespace tooling
} // namespace clang

#endif // LLVM_CLANG_TOOLING_TRANSFORMER_TRANSFORMER_H_
