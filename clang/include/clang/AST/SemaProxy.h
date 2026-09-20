//===--- SemaProxy.h - Interface to language semantics ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the SemaProxy interface, used during language-mandated
//  constant evaluation to act on, and query, the representation of the program
//  according to language-defined semantics.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_SEMA_PROXY_H
#define LLVM_CLANG_AST_SEMA_PROXY_H

#include "clang/Basic/SourceLocation.h"

namespace clang {

class FunctionDecl;

/// Classes implementing SemaProxy present a restricted view of the (possibly
/// mutating) actions and queries defined by language semantics against the
/// representation of the program (i.e., the AST). Such a view is required in
/// order to evaluate certain expressions (e.g., C++'s manifestly
/// constant-evaluated expressions) according to language rules.
class SemaProxy {
public:
  virtual ~SemaProxy() = default;

  virtual void
  instantiateFunctionDefinition(SourceLocation PointOfInstantiation,
                                FunctionDecl *Function) = 0;
};

} // end namespace clang

#endif // LLVM_CLANG_AST_SEMA_PROXY_H
