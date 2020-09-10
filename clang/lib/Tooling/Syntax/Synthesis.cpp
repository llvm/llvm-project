//===- Synthesis.cpp ------------------------------------------*- C++ -*-=====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "clang/Basic/TokenKinds.h"
#include "clang/Tooling/Syntax/BuildTree.h"

using namespace clang;

/// Exposes private syntax tree APIs required to implement node synthesis.
/// Should not be used for anything else.
class clang::syntax::FactoryImpl {
public:
  static void setCanModify(syntax::Node *N) { N->CanModify = true; }

  static void prependChildLowLevel(syntax::Tree *T, syntax::Node *Child,
                                   syntax::NodeRole R) {
    T->prependChildLowLevel(Child, R);
  }
};

syntax::Leaf *clang::syntax::createLeaf(syntax::Arena &A, tok::TokenKind K,
                                        StringRef Spelling) {
  auto Tokens = A.lexBuffer(llvm::MemoryBuffer::getMemBuffer(Spelling)).second;
  assert(Tokens.size() == 1);
  assert(Tokens.front().kind() == K &&
         "spelling is not lexed into the expected kind of token");

  auto *Leaf = new (A.getAllocator()) syntax::Leaf(Tokens.begin());
  syntax::FactoryImpl::setCanModify(Leaf);
  Leaf->assertInvariants();
  return Leaf;
}

syntax::Leaf *clang::syntax::createLeaf(syntax::Arena &A, tok::TokenKind K) {
  const auto *Spelling = tok::getPunctuatorSpelling(K);
  if (!Spelling)
    Spelling = tok::getKeywordSpelling(K);
  assert(Spelling &&
         "Cannot infer the spelling of the token from its token kind.");
  return createLeaf(A, K, Spelling);
}

syntax::EmptyStatement *clang::syntax::createEmptyStatement(syntax::Arena &A) {
  auto *S = new (A.getAllocator()) syntax::EmptyStatement;
  FactoryImpl::setCanModify(S);
  FactoryImpl::prependChildLowLevel(S, createLeaf(A, tok::semi),
                                    NodeRole::Unknown);
  S->assertInvariants();
  return S;
}
