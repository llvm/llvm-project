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

// Tells whether the type is annotated with [[gsl::Pointer]].
bool isGLSPointerType(QualType QT);

/// Describes an entity that is being assigned.
struct AssignedEntity {
  // The left-hand side expression of the assignment.
  Expr *LHS = nullptr;
  CXXMethodDecl *AssignmentOperator = nullptr;
};

struct CapturingEntity {
  // In an function call involving a lifetime capture, this would be the
  // argument capturing the lifetime of another argument.
  //    void addToSet(std::string_view sv [[clang::lifetime_capture_by(setsv)]],
  //                  set<std::string_view>& setsv);
  //    set<std::string_view> setsv;
  //    addToSet(std::string(), setsv); // Here 'setsv' is the 'Entity'.
  //
  // This is 'nullptr' when the capturing entity is 'global' or 'unknown'.
  Expr *Entity = nullptr;
};

/// Check that the lifetime of the given expr (and its subobjects) is
/// sufficient for initializing the entity, and perform lifetime extension
/// (when permitted) if not.
void checkInitLifetime(Sema &SemaRef, const InitializedEntity &Entity,
                       Expr *Init);

/// Check that the lifetime of the given expr (and its subobjects) is
/// sufficient for assigning to the entity.
void checkAssignmentLifetime(Sema &SemaRef, const AssignedEntity &Entity,
                             Expr *Init);

void checkCaptureByLifetime(Sema &SemaRef, const CapturingEntity &Entity,
                            Expr *Init);

/// Check that the lifetime of the given expr (and its subobjects) is
/// sufficient, assuming that it is passed as an argument to a musttail
/// function.
void checkExprLifetimeMustTailArg(Sema &SemaRef,
                                  const InitializedEntity &Entity, Expr *Init);

bool implicitObjectParamIsLifetimeBound(const FunctionDecl *FD);

} // namespace clang::sema

#endif // LLVM_CLANG_SEMA_CHECK_EXPR_LIFETIME_H
