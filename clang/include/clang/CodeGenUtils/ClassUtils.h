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

} // namespace clang::CodeGenUtils

#endif // LLVM_CLANG_CODEGENUTILS_CLASSUTILS_H
