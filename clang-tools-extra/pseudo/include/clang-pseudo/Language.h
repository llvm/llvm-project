//===--- Language.h -------------------------------------------- -*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_PSEUDO_LANGUAGE_H
#define CLANG_PSEUDO_LANGUAGE_H

#include "clang-pseudo/Token.h"
#include "clang-pseudo/grammar/Grammar.h"
#include "clang-pseudo/grammar/LRTable.h"

namespace clang {
namespace pseudo {
class ForestNode;
class TokenStream;
class LRTable;

// A guard restricts when a grammar rule can be used.
//
// The GLR parser will use the guard to determine whether a rule reduction will
// be conducted. For example, e.g. a guard may allow the rule
// `virt-specifier := IDENTIFIER` only if the identifier's text is 'override`.
//
// Return true if the guard is satisfied.
using RuleGuard = llvm::function_ref<bool(
    llvm::ArrayRef<const ForestNode *> RHS, const TokenStream &)>;

// A recovery strategy determines a region of code to skip when parsing fails.
//
// For example, given `class-def := CLASS IDENT { body [recover=Brackets] }`,
// if parsing fails while attempting to parse `body`, we may skip up to the
// matching `}` and assume everything between was a `body`.
//
// The provided index is the token where the skipped region begins.
// Returns the (excluded) end of the range, or Token::Invalid for no recovery.
using RecoveryStrategy =
    llvm::function_ref<Token::Index(Token::Index Start, const TokenStream &)>;

// Specify a language that can be parsed by the pseduoparser.
struct Language {
  Grammar G;
  LRTable Table;

  // Binding extension ids to corresponding implementations.
  llvm::DenseMap<RuleID, RuleGuard> Guards;
  llvm::DenseMap<ExtensionID, RecoveryStrategy> RecoveryStrategies;

  // FIXME: add clang::LangOptions.
  // FIXME: add default start symbols.
};

} // namespace pseudo
} // namespace clang

#endif // CLANG_PSEUDO_LANGUAGE_H
