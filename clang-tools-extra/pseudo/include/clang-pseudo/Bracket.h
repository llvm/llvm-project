//===--- Bracket.h - Analyze bracket structure --------------------*-C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Bracket structure (particularly braces) is key to isolating broken regions
// of code and preventing parsing from going "off the rails".
//
// For correct C++ code, brackets are well-nested and identifying pairs and
// therefore blocks is simple. In broken code, brackets are not properly nested.
// We cannot match them all and must choose which pairs to form.
//
// Rather than have the grammar-based parser make these choices, we pair
// brackets up-front based on textual features like indentation.
// This mirrors the way humans read code, and so is likely to produce the
// "correct" interpretation of broken code.
//
// This interpretation then guides the parse: a rule containing a bracket pair
// must match against paired bracket tokens.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_PSEUDO_BRACKET_H
#define CLANG_PSEUDO_BRACKET_H

#include "clang-pseudo/Token.h"

namespace clang {
namespace pseudo {

/// Identifies bracket token in the stream which should be paired.
/// Sets Token::Pair accordingly.
void pairBrackets(TokenStream &);

} // namespace pseudo
} // namespace clang

#endif
