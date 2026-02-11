//===----- SemaSYCL.h ------- Semantic Analysis for SYCL constructs -------===//
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

#include "clang/AST/ASTFwd.h"
#include "clang/AST/Type.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/Ownership.h"
#include "clang/Sema/SemaBase.h"
#include "llvm/ADT/DenseSet.h"

namespace clang {
class Decl;
class ParsedAttr;

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
  ///   for the device, but we don't know yet that this function will be
  ///   codegen'ed for the devive, creates a diagnostic which is emitted if and
  ///   when we realize that the function will be codegen'ed.
  ///
  /// Example usage:
  ///
  /// Diagnose __float128 type usage only from SYCL device code if the current
  /// target doesn't support it
  /// if (!S.Context.getTargetInfo().hasFloat128Type() &&
  ///     S.getLangOpts().SYCLIsDevice)
  ///   DiagIfDeviceCode(Loc, diag::err_type_unsupported) << "__float128";
  SemaDiagnosticBuilder DiagIfDeviceCode(SourceLocation Loc, unsigned DiagID);

  void deepTypeCheckForDevice(SourceLocation UsedAt,
                              llvm::DenseSet<QualType> Visited,
                              ValueDecl *DeclToCheck);

  ExprResult BuildUniqueStableNameExpr(SourceLocation OpLoc,
                                       SourceLocation LParen,
                                       SourceLocation RParen,
                                       TypeSourceInfo *TSI);
  ExprResult ActOnUniqueStableNameExpr(SourceLocation OpLoc,
                                       SourceLocation LParen,
                                       SourceLocation RParen,
                                       ParsedType ParsedTy);

  void handleKernelAttr(Decl *D, const ParsedAttr &AL);
  void handleKernelEntryPointAttr(Decl *D, const ParsedAttr &AL);

  /// Issues a deferred diagnostic if use of the declaration designated
  /// by 'D' is invalid in a device context.
  void CheckDeviceUseOfDecl(NamedDecl *D, SourceLocation Loc);

  void CheckSYCLExternalFunctionDecl(FunctionDecl *FD);
  void CheckSYCLEntryPointFunctionDecl(FunctionDecl *FD);

  /// Builds an expression for the lookup of a 'sycl_kernel_launch' template
  /// with 'KernelName' as an explicit template argument. Lookup is performed
  /// as if from the first statement of the body of 'FD' and thus requires
  /// searching the scopes that exist at parse time. This function therefore
  /// requires the current semantic context to be the definition of 'FD'. In a
  /// dependent context, the returned expression will be an UnresolvedLookupExpr
  /// or an UnresolvedMemberExpr. In a non-dependent context, the returned
  /// expression will be a DeclRefExpr or MemberExpr. If lookup fails, a null
  /// error result is returned. The resulting expression is intended to be
  /// passed as the 'LaunchIdExpr' argument in a call to either
  /// BuildSYCLKernelCallStmt() or BuildUnresolvedSYCLKernelCallStmt() after
  /// the function body has been parsed.
  ExprResult BuildSYCLKernelLaunchIdExpr(FunctionDecl *FD, QualType KernelName);

  /// Builds a SYCLKernelCallStmt to wrap 'Body' and to be used as the body of
  /// 'FD'. 'LaunchIdExpr' specifies the lookup result returned by a previous
  /// call to BuildSYCLKernelLaunchIdExpr().
  StmtResult BuildSYCLKernelCallStmt(FunctionDecl *FD, CompoundStmt *Body,
                                     Expr *LaunchIdExpr);

  /// Builds an UnresolvedSYCLKernelCallStmt to wrap 'Body'. 'LaunchIdExpr'
  /// specifies the lookup result returned by a previous call to
  /// BuildSYCLKernelLaunchIdExpr().
  StmtResult BuildUnresolvedSYCLKernelCallStmt(CompoundStmt *Body,
                                               Expr *LaunchIdExpr);
};

} // namespace clang

#endif // LLVM_CLANG_SEMA_SEMASYCL_H
