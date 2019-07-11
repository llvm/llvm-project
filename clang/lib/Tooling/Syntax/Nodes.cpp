//===- Nodes.cpp ----------------------------------------------*- C++ -*-=====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "clang/Tooling/Syntax/Nodes.h"
#include "clang/Basic/TokenKinds.h"

using namespace clang;

llvm::raw_ostream &syntax::operator<<(llvm::raw_ostream &OS, NodeKind K) {
  switch (K) {
  case NodeKind::Leaf:
    return OS << "Leaf";
  case NodeKind::TranslationUnit:
    return OS << "TranslationUnit";
  case NodeKind::TopLevelDeclaration:
    return OS << "TopLevelDeclaration";
  case NodeKind::CompoundStatement:
    return OS << "CompoundStatement";
  }
  llvm_unreachable("unknown node kind");
}

syntax::Leaf *syntax::CompoundStatement::lbrace() {
  return llvm::cast_or_null<syntax::Leaf>(
      findChild(NodeRole::CompoundStatement_lbrace));
}

syntax::Leaf *syntax::CompoundStatement::rbrace() {
  return llvm::cast_or_null<syntax::Leaf>(
      findChild(NodeRole::CompoundStatement_rbrace));
}
