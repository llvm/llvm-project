//===--- Language.h -------------------------------------------- -*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_PSEUDO_GRAMMAR_LANGUAGE_H
#define CLANG_PSEUDO_GRAMMAR_LANGUAGE_H

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

// Specify a language that can be parsed by the pseduoparser.
struct Language {
  Grammar G;
  LRTable Table;

  // Binding "guard" extension id to a piece of C++ code.
  llvm::DenseMap<ExtensionID, RuleGuard> Guards;

  // FIXME: add clang::LangOptions.
  // FIXME: add default start symbols.
};

} // namespace pseudo
} // namespace clang

#endif // CLANG_PSEUDO_GRAMMAR_LANGUAGE_H
