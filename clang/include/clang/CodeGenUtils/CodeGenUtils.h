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
bool CanSkipVTablePointerInitialization(ASTContext &Ctx,
                                        const CXXDestructorDecl *Dtor);

bool FieldHasTrivialDestructorBody(ASTContext &Context, const FieldDecl *Field);

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
