//===--- CXX.h - Public interfaces for the C++ grammar -----------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines public interfaces for the C++ grammar
//  (pseudo/lib/cxx/cxx.bnf). It provides a fast way to access core building
//  pieces of the LR parser, e.g. Grammar, LRTable, rather than parsing the
//  grammar file at the runtime.
//
//  We do a compilation of the C++ BNF grammar at build time, and generate
//  critical data sources. The implementation of the interfaces are based on the
//  generated data sources.
//
//  FIXME: not everything is fully compiled yet. The implementation of the
//  interfaces are still parsing the grammar file at the runtime.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_PSEUDO_CXX_CXX_H
#define CLANG_PSEUDO_CXX_CXX_H

#include "clang-pseudo/Language.h"
#include "clang-pseudo/grammar/Grammar.h"

namespace clang {
namespace pseudo {
namespace cxx {

// We want enums to be scoped but implicitly convertible to RuleID etc.
// So create regular (unscoped) enums inside subnamespaces of `detail`.
// Then add aliases for them outside `detail`.
namespace detail {
namespace symbols {
enum Symbol : SymbolID {
#define NONTERMINAL(X, Y) X = Y,
#include "CXXSymbols.inc"
#undef NONTERMINAL
};
} // namespace symbols

namespace extensions {
enum Extension : ExtensionID {
#define EXTENSION(X, Y) X = Y,
#include "CXXSymbols.inc"
#undef EXTENSION
};
} // namespace extensions

namespace rules {
// For each symbol we close the last symbol's enum+namespace and open new ones.
// We need a dummy namespace+enum so that this works for the first rule.
namespace dummy {
enum Dummy {
//clang-format off
#define NONTERMINAL(NAME, ID) \
};                            \
}                             \
namespace NAME {              \
enum Rule : RuleID {
//clang-format on
#define RULE(LHS, RHS, ID) RHS = ID,
#include "CXXSymbols.inc"
};
}
} // namespace rules
} // namespace detail

// Symbol represents nonterminal symbols in the C++ grammar.
// It provides a simple uniform way to access a particular nonterminal.
using Symbol = detail::symbols::Symbol;

using Extension = detail::extensions::Extension;

namespace rule {
#define NONTERMINAL(NAME, ID) using NAME = detail::rules::NAME::Rule;
#include "CXXSymbols.inc"
} // namespace rule

// Returns the Language for the cxx.bnf grammar.
const Language &getLanguage();

} // namespace cxx

} // namespace pseudo
} // namespace clang

#endif // CLANG_PSEUDO_CXX_CXX_H
