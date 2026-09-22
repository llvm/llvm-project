//===--- CallUtils.h - Shared call emission queries -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file holds the AST and language option queries that both classic
// CodeGen and CIR CodeGen need while building a call and its argument and
// return value attributes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CODEGENUTILS_CALLUTILS_H
#define LLVM_CLANG_CODEGENUTILS_CALLUTILS_H

#include "clang/AST/ASTContext.h"
#include "llvm/ADT/FloatingPointMode.h"

namespace clang::CodeGenUtils {

/// Returns the canonical formal type of the given C++ method.
CanQual<FunctionProtoType> getFormalType(const CXXMethodDecl *MD);

/// Returns the set of floating-point value kinds that the language options
/// promise never reach a function's arguments or return value.
llvm::FPClassTest getNoFPClassTestMask(const LangOptions &LangOpts);

} // namespace clang::CodeGenUtils

#endif // LLVM_CLANG_CODEGENUTILS_CALLUTILS_H
