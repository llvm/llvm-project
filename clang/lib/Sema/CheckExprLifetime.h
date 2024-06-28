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

/// Check that the lifetime of the given expr (and its subobjects) is
/// sufficient for initializing the entity, and perform lifetime extension
/// (when permitted) if not.
void checkExprLifetime(Sema &SemaRef, const InitializedEntity &Entity,
                       Expr *Init);

} // namespace clang::sema

#endif // LLVM_CLANG_SEMA_CHECK_EXPR_LIFETIME_H
