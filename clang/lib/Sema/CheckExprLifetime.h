//===- CheckExprLifetime.h -----------------------------------  -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
//  This files implements a statement-local lifetime analysis.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_CHECK_EXPR_LIFETIME_H
#define LLVM_CLANG_SEMA_CHECK_EXPR_LIFETIME_H

#include "clang/AST/Expr.h"
#include "clang/Sema/Initialization.h"
#include "clang/Sema/Sema.h"

namespace clang::sema {

struct CapturingEntity {
  // The expression of the entity which captures another entity.
  // For example:
  //  1. In an assignment, this would be the left-hand side expression.
  //    std::string_view sv;
  //    sv = std::string(); // Here 'sv' is the 'Entity'.
  //
  //  2. In an function call involving a lifetime capture, this would be the
  //  argument capturing the lifetime of another argument.
  //    void addToSet(std::string_view s [[clang::lifetime_capture_by(sv)]],
  //                  set<std::string_view>& setsv);
  //    set<std::string_view> ssv;
  //    addToSet(std::string(), ssv); // Here 'ssv' is the 'Entity'.
  Expr *Expression = nullptr;
  CXXMethodDecl *AssignmentOperator = nullptr;
};

/// Check that the lifetime of the given expr (and its subobjects) is
/// sufficient for initializing the entity, and perform lifetime extension
/// (when permitted) if not.
void checkInitLifetime(Sema &SemaRef, const InitializedEntity &Entity,
                       Expr *Init);

/// Check that the lifetime of the given expr (and its subobjects) is
/// sufficient for assigning to the entity.
void checkAssignmentLifetime(Sema &SemaRef, const CapturingEntity &Entity,
                             Expr *RHS);

void checkCaptureLifetime(Sema &SemaRef, const CapturingEntity &Entity,
                          Expr *Captured);

/// Check that the lifetime of the given expr (and its subobjects) is
/// sufficient, assuming that it is passed as an argument to a musttail
/// function.
void checkExprLifetimeMustTailArg(Sema &SemaRef,
                                  const InitializedEntity &Entity, Expr *Init);
} // namespace clang::sema

#endif // LLVM_CLANG_SEMA_CHECK_EXPR_LIFETIME_H
