//===- Mutations.cpp ------------------------------------------*- C++ -*-=====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "clang/Tooling/Syntax/Mutations.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Lex/Token.h"
#include "clang/Tooling/Core/Replacement.h"
#include "clang/Tooling/Syntax/BuildTree.h"
#include "clang/Tooling/Syntax/Nodes.h"
#include "clang/Tooling/Syntax/Tokens.h"
#include "clang/Tooling/Syntax/Tree.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include <cassert>
#include <string>

using namespace clang;

// This class has access to the internals of tree nodes. Its sole purpose is to
// define helpers that allow implementing the high-level mutation operations.
class syntax::MutationsImpl {
public:
  /// Add a new node with a specified role.
  static void addAfter(syntax::Node *Anchor, syntax::Node *New, NodeRole Role) {
    assert(New->Parent == nullptr);
    assert(New->NextSibling == nullptr);
    assert(!New->isDetached());
    assert(Role != NodeRole::Detached);

    New->Role = static_cast<unsigned>(Role);
    Anchor->parent()->replaceChildRangeLowLevel(Anchor, Anchor, New);
  }

  /// Replace the node, keeping the role.
  static void replace(syntax::Node *Old, syntax::Node *New) {
    assert(New->Parent == nullptr);
    assert(New->NextSibling == nullptr);
    assert(New->isDetached());

    New->Role = Old->Role;
    Old->parent()->replaceChildRangeLowLevel(findPrevious(Old),
                                             Old->nextSibling(), New);
  }

  /// Completely remove the node from its parent.
  static void remove(syntax::Node *N) {
    N->parent()->replaceChildRangeLowLevel(findPrevious(N), N->nextSibling(),
                                           /*New=*/nullptr);
  }

private:
  static syntax::Node *findPrevious(syntax::Node *N) {
    if (N->parent()->firstChild() == N)
      return nullptr;
    for (syntax::Node *C = N->parent()->firstChild(); C != nullptr;
         C = C->nextSibling()) {
      if (C->nextSibling() == N)
        return C;
    }
    llvm_unreachable("could not find a child node");
  }
};

void syntax::removeStatement(syntax::Arena &A, syntax::Statement *S) {
  assert(S->canModify());

  if (isa<CompoundStatement>(S->parent())) {
    // A child of CompoundStatement can just be safely removed.
    MutationsImpl::remove(S);
    return;
  }
  // For the rest, we have to replace with an empty statement.
  if (isa<EmptyStatement>(S))
    return; // already an empty statement, nothing to do.

  MutationsImpl::replace(S, createEmptyStatement(A));
}
