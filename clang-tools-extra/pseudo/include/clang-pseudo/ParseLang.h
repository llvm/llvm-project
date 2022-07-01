//===--- ParseLang.h ------------------------------------------- -*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_PSEUDO_PARSELANG_H
#define CLANG_PSEUDO_PARSELANG_H

#include "clang-pseudo/grammar/Grammar.h"
#include "clang-pseudo/grammar/LRTable.h"

namespace clang {
namespace pseudo {

// Specify a language that can be parsed by the pseduoparser.
// Manifest generated from a bnf grammar file.
struct ParseLang {
  Grammar G;
  LRTable Table;

  // FIXME: add clang::LangOptions.
  // FIXME: add default start symbols.
};

} // namespace pseudo
} // namespace clang

#endif // CLANG_PSEUDO_PARSELANG_H
