//===- Scope.cpp - Lexical scope information --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Scope class, which is used for recording
// information about a lexical scope.
//
//===----------------------------------------------------------------------===//

#include "clang/Sema/Scope.h"

using namespace clang;

void Scope::Init(Scope *parent, unsigned flags) {
  AnyParent = parent;
  Flags = flags;

  if (parent && !(flags & FnScope)) {
    BreakParent    = parent->BreakParent;
    ContinueParent = parent->ContinueParent;
  } else {
    // Control scopes do not contain the contents of nested function scopes for
    // control flow purposes.
    BreakParent = ContinueParent = 0;
  }

  if (parent) {
    Depth = parent->Depth + 1;
    PrototypeDepth = parent->PrototypeDepth;
    PrototypeIndex = 0;
    FnParent       = parent->FnParent;
    BlockParent    = parent->BlockParent;
    TemplateParamParent = parent->TemplateParamParent;
  } else {
    Depth = 0;
    PrototypeDepth = 0;
    PrototypeIndex = 0;
    FnParent = BlockParent = 0;
    TemplateParamParent = 0;
  }

  // If this scope is a function or contains breaks/continues, remember it.
  if (flags & FnScope)            FnParent = this;
  if (flags & BreakScope)         BreakParent = this;
  if (flags & ContinueScope)      ContinueParent = this;
  if (flags & BlockScope)         BlockParent = this;
  if (flags & TemplateParamScope) TemplateParamParent = this;

  // If this is a prototype scope, record that.
  if (flags & FunctionPrototypeScope) PrototypeDepth++;

  DeclsInScope.clear();
  UsingDirectives.clear();
  Entity = 0;
  ErrorTrap.reset();
}

bool Scope::containedInPrototypeScope() const {
  const Scope *S = this;
  while (S) {
    if (S->isFunctionPrototypeScope())
      return true;
    S = S->getParent();
  }
  return false;
}
