//===- BuildTree.h - build syntax trees -----------------------*- C++ -*-=====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Functions to construct a syntax tree from an AST.
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_TOOLING_SYNTAX_TREE_H
#define LLVM_CLANG_TOOLING_SYNTAX_TREE_H

#include "clang/AST/Decl.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Tooling/Syntax/Nodes.h"
#include "clang/Tooling/Syntax/Tree.h"

namespace clang {
namespace syntax {

/// Build a syntax tree for the main file.
syntax::TranslationUnit *buildSyntaxTree(Arena &A,
                                         const clang::TranslationUnitDecl &TU);

// Create syntax trees from subtrees not backed by the source code.

// Synthesis of Leafs
/// Create `Leaf` from token with `Spelling` and assert it has the desired
/// `TokenKind`.
syntax::Leaf *createLeaf(syntax::Arena &A, tok::TokenKind K,
                         StringRef Spelling);

/// Infer the token spelling from its `TokenKind`, then create `Leaf` from
/// this token
syntax::Leaf *createLeaf(syntax::Arena &A, tok::TokenKind K);

// Synthesis of Trees
/// Creates the concrete syntax node according to the specified `NodeKind` `K`.
/// Returns it as a pointer to the base class `Tree`.
syntax::Tree *
createTree(syntax::Arena &A,
           ArrayRef<std::pair<syntax::Node *, syntax::NodeRole>> Children,
           syntax::NodeKind K);

// Synthesis of Syntax Nodes
syntax::EmptyStatement *createEmptyStatement(syntax::Arena &A);

/// Creates a completely independent copy of `N` with its macros expanded.
///
/// The copy is:
/// * Detached, i.e. `Parent == NextSibling == nullptr` and
/// `Role == Detached`.
/// * Synthesized, i.e. `Original == false`.
syntax::Node *deepCopyExpandingMacros(syntax::Arena &A, const syntax::Node *N);
} // namespace syntax
} // namespace clang
#endif
