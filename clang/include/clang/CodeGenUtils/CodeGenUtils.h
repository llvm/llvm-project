//===--- CodeGenUtils.h - Shared Classic CodeGen/CIR CodeGen Utils--C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CODEGENUTILS_CODEGENUTILS_H
#define LLVM_CLANG_CODEGENUTILS_CODEGENUTILS_H

#include "clang/AST/ASTContext.h"

namespace clang::CodeGenUtils {
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

/// Determines whether the language options require us to model
/// unwind exceptions.  We treat -fexceptions as mandating this
/// except under the fragile ObjC ABI with only ObjC exceptions
/// enabled.  This means, for example, that C with -fexceptions
/// enables this.
bool hasUnwindExceptions(const LangOptions &LangOpts);

/// Helper method to check if the underlying ABI is AAPCS
bool isAAPCS(const TargetInfo &TargetInfo);

bool isInitializerOfDynamicClass(const CXXCtorInitializer *BaseInit);
} // namespace clang::CodeGenUtils

#endif // LLVM_CLANG_CODEGENUTILS_CODEGENUTILS_H
