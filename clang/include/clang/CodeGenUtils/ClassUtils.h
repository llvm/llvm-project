//===--- ClassUtils.h - Shared C++ class emission queries -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file holds the AST queries about C++ class construction and destruction
// that both classic CodeGen and CIR CodeGen need, chiefly to decide when a
// vtable pointer has to be established before running member initializers or
// a destructor body.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CODEGENUTILS_CLASSUTILS_H
#define LLVM_CLANG_CODEGENUTILS_CLASSUTILS_H

#include "clang/AST/ASTContext.h"

namespace clang::CodeGenUtils {

/// Check whether \p Init uses 'this' in a way which requires the vtable to be
/// properly set.
bool baseInitializerUsesThis(ASTContext &Ctx, const Expr *Init);

/// Check whether we need to initialize any vtable pointers before calling this
/// destructor.
bool canSkipVTablePointerInitialization(ASTContext &Ctx,
                                        const CXXDestructorDecl *Dtor);

/// Check whether destructing \p Field has no observable behaviors, and thus can
/// be skipped when creating a destructor body. So non-record types, anonymous
/// structs/unions, or record types where the destructor doesnt DO anything are
/// considered as this version of 'trivial'.
/// Note: This is a more liberal definition of trivial destruction than the C++
/// Standard's version, and thus cannot be used as a substitute for C++ Standard
/// requirements.
bool fieldHasTrivialDestructorBody(ASTContext &Context, const FieldDecl *Field);

bool isInitializerOfDynamicClass(const CXXCtorInitializer *BaseInit);

} // namespace clang::CodeGenUtils

#endif // LLVM_CLANG_CODEGENUTILS_CLASSUTILS_H
