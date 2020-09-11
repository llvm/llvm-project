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
syntax::Tree *
createTree(Arena &A,
           std::vector<std::pair<syntax::Node *, syntax::NodeRole>> Children,
           syntax::NodeKind K);

// Synthesis of Syntax Nodes
clang::syntax::EmptyStatement *createEmptyStatement(clang::syntax::Arena &A);

} // namespace syntax
} // namespace clang
#endif
