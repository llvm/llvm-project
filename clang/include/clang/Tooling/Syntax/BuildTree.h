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
#include "clang/Tooling/Syntax/Nodes.h"

namespace clang {
namespace syntax {

/// Build a syntax tree for the main file.
syntax::TranslationUnit *buildSyntaxTree(Arena &A,
                                         const clang::TranslationUnitDecl &TU);
} // namespace syntax
} // namespace clang
#endif
