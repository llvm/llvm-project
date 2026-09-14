//===--- FunctionUtils.h - Shared function emission queries -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file holds the queries and checks that both classic CodeGen and CIR
// CodeGen need while emitting a function body.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CODEGENUTILS_FUNCTIONUTILS_H
#define LLVM_CLANG_CODEGENUTILS_FUNCTIONUTILS_H

#include "clang/Basic/CodeGenOptions.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"

namespace clang {
class ASTContext;
class CallExpr;
class DiagnosticsEngine;
class FunctionDecl;
} // namespace clang

namespace clang::CodeGenUtils {

/// Decide whether we need to emit the lifetime markers.
bool shouldEmitLifetimeMarkers(const CodeGenOptions &CGOpts,
                               const LangOptions &LangOpts);

/// Check that a call to a target-specific builtin has the required target
/// features enabled in the caller, emitting an error diagnostic if not.
/// \p caller is the FunctionDecl of the enclosing function (may be null).
void checkTargetFeatures(ASTContext &Ctx, DiagnosticsEngine &Diags,
                         const LangOptions &LangOpts, const CallExpr *E,
                         const FunctionDecl *Caller,
                         const FunctionDecl *TargetDecl);

/// Overload taking a raw source location instead of a CallExpr.
void checkTargetFeatures(ASTContext &Ctx, DiagnosticsEngine &Diags,
                         const LangOptions &LangOpts, SourceLocation Loc,
                         const FunctionDecl *Caller,
                         const FunctionDecl *TargetDecl);

} // namespace clang::CodeGenUtils

#endif // LLVM_CLANG_CODEGENUTILS_FUNCTIONUTILS_H
