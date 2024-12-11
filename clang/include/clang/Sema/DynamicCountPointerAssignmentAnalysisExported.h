//===--- DynamicCountPointerAssignmentAnalysisExported.h --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file exports DynamicCountPointer Analysis utilities to other libraries.
//
// This is a workaround necessitated in the fact that Sema/TreeTransform.h is a
// private header file (not in the include path that other libraries can use).
// The "right" solution here would be to move the header so it is public but
// that would create a conflict with upstream which is really not desirable.
//
// So instead this file exports a really simple interface that hides the
// `TreeTransform` type (and its dependencies)
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_SEMA_DYNAMIC_COUNT_POINTER_ASSIGNMENT_ANALYSIS_EXPORTED_H
#define LLVM_CLANG_SEMA_DYNAMIC_COUNT_POINTER_ASSIGNMENT_ANALYSIS_EXPORTED_H
#include "clang/Sema/Ownership.h"
#include "clang/Sema/Sema.h"

namespace clang {

ExprResult ReplaceCountExprParamsWithArgsFromCall(const Expr *CountExpr,
                                                  const CallExpr *CE, Sema &S);

}

#endif
