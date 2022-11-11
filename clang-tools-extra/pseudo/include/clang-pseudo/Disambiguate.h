//===--- Disambiguate.h - Find the best tree in the forest -------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A GLR parse forest represents every possible parse tree for the source code.
//
// Before we can do useful analysis/editing of the code, we need to pick a
// single tree which we think is accurate. We use three main types of clues:
//
// A) Semantic language rules may restrict which parses are allowed.
//    For example, `string string string X` is *grammatical* C++, but only a
//    single type-name is allowed in a decl-specifier-sequence.
//    Where possible, these interpretations are forbidden by guards.
//    Sometimes this isn't possible, or we want our parser to be lenient.
//
// B) Some constructs are rarer, while others are common.
//    For example `a<b>::c` is often a template specialization, and rarely a
//    double comparison between a, b, and c.
//
// C) Identifier text hints whether they name types/values/templates etc.
//    "std" is usually a namespace, a project index may also guide us.
//    Hints may be within the document: if one occurrence of 'foo' is a variable
//    then the others probably are too.
//    (Text need not match: similar CaseStyle can be a weak hint, too).
//
//----------------------------------------------------------------------------//
//
// Mechanically, we replace each ambiguous node with its best alternative.
//
// "Best" is determined by assigning bonuses/penalties to nodes, to express
// the clues of type A and B above. A forest node representing an unlikely
// parse would apply a penalty to every subtree is is present in.
// Disambiguation proceeds bottom-up, so that the score of each alternative
// is known when a decision is made.
//
// Identifier-based hints within the document mean some nodes should be
// *correlated*. Rather than resolve these simultaneously, we make the most
// certain decisions first and use these results to update bonuses elsewhere.
//
//===----------------------------------------------------------------------===//

#include "clang-pseudo/Forest.h"

namespace clang::pseudo {

struct DisambiguateParams {};

// Maps ambiguous nodes onto the index of their preferred alternative.
using Disambiguation = llvm::DenseMap<const ForestNode *, unsigned>;

// Resolve each ambiguous node in the forest.
// Maps each ambiguous node to the index of the chosen alternative.
// FIXME: current implementation is a placeholder and chooses arbitrarily.
Disambiguation disambiguate(const ForestNode *Root,
                            const DisambiguateParams &Params);

// Remove all ambiguities from the forest, resolving them according to Disambig.
void removeAmbiguities(ForestNode *&Root, const Disambiguation &Disambig);

} // namespace clang::pseudo
