//===----- SemaOpenACC.h 000- Semantic Analysis for SYCL constructs -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file declares semantic analysis for SYCL constructs.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_SEMASYCL_H
#define LLVM_CLANG_SEMA_SEMASYCL_H

#include "clang/AST/Decl.h"
#include "clang/AST/Type.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/Ownership.h"
#include "clang/Sema/SemaBase.h"
#include "llvm/ADT/DenseSet.h"

namespace clang {

class SemaSYCL : public SemaBase {
public:
  SemaSYCL(Sema &S);

  /// Creates a SemaDiagnosticBuilder that emits the diagnostic if the current
  /// context is "used as device code".
  ///
  /// - If CurLexicalContext is a kernel function or it is known that the
  ///   function will be emitted for the device, emits the diagnostics
  ///   immediately.
  /// - If CurLexicalContext is a function and we are compiling
  ///   for the device, but we don't know that this function will be codegen'ed
  ///   for devive yet, creates a diagnostic which is emitted if and when we
  ///   realize that the function will be codegen'ed.
  ///
  /// Example usage:
  ///
  /// Diagnose __float128 type usage only from SYCL device code if the current
  /// target doesn't support it
  /// if (!S.Context.getTargetInfo().hasFloat128Type() &&
  ///     S.getLangOpts().SYCLIsDevice)
  ///   SYCLDiagIfDeviceCode(Loc, diag::err_type_unsupported) << "__float128";
  SemaDiagnosticBuilder SYCLDiagIfDeviceCode(SourceLocation Loc,
                                             unsigned DiagID);

  void deepTypeCheckForSYCLDevice(SourceLocation UsedAt,
                                  llvm::DenseSet<QualType> Visited,
                                  ValueDecl *DeclToCheck);

  ExprResult BuildSYCLUniqueStableNameExpr(SourceLocation OpLoc,
                                           SourceLocation LParen,
                                           SourceLocation RParen,
                                           TypeSourceInfo *TSI);
  ExprResult ActOnSYCLUniqueStableNameExpr(SourceLocation OpLoc,
                                           SourceLocation LParen,
                                           SourceLocation RParen,
                                           ParsedType ParsedTy);
};

} // namespace clang

#endif // LLVM_CLANG_SEMA_SEMASYCL_H
