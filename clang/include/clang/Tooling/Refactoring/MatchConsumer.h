//===--- MatchConsumer.h - MatchConsumer abstraction ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file defines the *MatchConsumer* abstraction: a computation over
/// match results, specifically the `ast_matchers::MatchFinder::MatchResult`
/// class.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_REFACTOR_MATCH_CONSUMER_H_
#define LLVM_CLANG_TOOLING_REFACTOR_MATCH_CONSUMER_H_

#include "clang/AST/ASTTypeTraits.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"

namespace clang {
namespace tooling {

/// A failable computation over nodes bound by AST matchers.
///
/// The computation should report any errors though its return value (rather
/// than terminating the program) to enable usage in interactive scenarios like
/// clang-query.
///
/// This is a central abstraction of the Transformer framework.
template <typename T>
using MatchConsumer =
    std::function<Expected<T>(const ast_matchers::MatchFinder::MatchResult &)>;

/// Creates an error that signals that a `MatchConsumer` expected a certain node
/// to be bound by AST matchers, but it was not actually bound.
inline llvm::Error notBoundError(llvm::StringRef Id) {
  return llvm::make_error<llvm::StringError>(llvm::errc::invalid_argument,
                                             "Id not bound: " + Id);
}

/// Chooses between the two consumers, based on whether \p ID is bound in the
/// match.
template <typename T>
MatchConsumer<T> ifBound(std::string ID, MatchConsumer<T> TrueC,
                         MatchConsumer<T> FalseC) {
  return [=](const ast_matchers::MatchFinder::MatchResult &Result) {
    auto &Map = Result.Nodes.getMap();
    return (Map.find(ID) != Map.end() ? TrueC : FalseC)(Result);
  };
}

} // namespace tooling
} // namespace clang
#endif // LLVM_CLANG_TOOLING_REFACTOR_MATCH_CONSUMER_H_
