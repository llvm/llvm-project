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
#include <variant>

namespace clang::sema {

/// Describes an entity that is being assigned.
struct AssignedEntity {
  // The left-hand side expression of the assignment.
  Expr *LHS = nullptr;
};

using CheckingEntity =
    std::variant<const InitializedEntity *, const AssignedEntity *>;

/// Check that the lifetime of the given expr (and its subobjects) is
/// sufficient for initializing or assigning to the entity.
///
/// If the entity is being initialized and its lifetime is insufficient, perform
/// lifetime extension (when permitted).
void checkExprLifetime(Sema &SemaRef, const CheckingEntity &CEntity,
                       Expr *Init);

} // namespace clang::sema

#endif // LLVM_CLANG_SEMA_CHECK_EXPR_LIFETIME_H
