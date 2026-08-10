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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

namespace clang {
class Decl;
class ParsedAttr;

class SemaSYCL : public SemaBase {
  /// SYCLKernelInfoClassTemplate caches the class template declaration built
  /// by GetSYCLKernelInfoClassTemplate().
  ClassTemplateDecl *SYCLKernelInfoClassTemplate = nullptr;

  /// SYCLKernelInfoClassTemplateSpecializations maps SYCL kernel name types
  /// to their corresponding synthesized explicit specialization declarations
  /// of the SYCL kernel info class template.
  llvm::DenseMap<CanQualType, ClassTemplateSpecializationDecl *>
      SYCLKernelInfoClassTemplateSpecializations;

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
  /// by 'ND' is invalid in a device context.
  void CheckDeviceUseOfDecl(NamedDecl *ND, SourceLocation Loc);

  void CheckSYCLExternalFunctionDecl(FunctionDecl *FD);
  void CheckSYCLEntryPointFunctionDecl(FunctionDecl *FD);

  /// GetSYCLKernelInfoClassTemplate builds a class template used to pass SYCL
  /// kernel information in synthesized calls to SYCL runtime library functions.
  /// The class template has an unspecified name, is an incomplete class
  /// (explicit specializations are synthesized as needed), built on first call,
  /// and cached for return in subsequent calls.
  ClassTemplateDecl *GetSYCLKernelInfoClassTemplate();

  /// GetSYCLKernelInfoClassSpecializationType returns a type for the class
  /// template returned by GetSYCLKernelInfoClassTemplate() specialized for
  /// 'KNT'. 'KNT' may be a dependent type. If 'KNT' is not a dependent type,
  /// an explicit specialization declaration for the specialization is
  /// synthesized and a canonical type returned. A definition for the explicit
  /// specialization is synthesized when SYCLKernelCallStmt is constructed for
  /// 'KNT'.
  QualType GetSYCLKernelInfoClassSpecializationType(QualType KNT);

  /// SYCLKernelCallStmtASTFragments holds portions of the AST used for lookup
  /// of SYCL runtime library functions. These are constructed at the beginning
  /// of a function definition and later used in synthesized calls to the
  /// SYCL runtime library. In a dependent context, these are used to construct
  /// an UnresolvedSYCLKernelCallStmt. During template instantiation, or in a
  /// non-dependent context, these are used to construct a SYCLKernelCallStmt.
  ///
  /// KernelInfoType is the type to be passed as the explicit template argument
  /// to SYCL runtime libraries.
  ///
  /// KernelLaunchIdExpr is an UnresolvedLookupExpr or UnresolvedMemberExpr
  /// for the SYCL kernel launch function, with KernelInfoType used as the
  /// explicit template argument.
  struct SYCLKernelCallStmtASTFragments {
    QualType KernelInfoType;
    Expr *KernelLaunchIdExpr;
  };

  /// BuildSYCLKernelCallStmtASTFragments builds portions of the AST needed
  /// to construct SYCLKernelCallStmt or UnresolvedSYCLKernelCallStmt that
  /// must be constructed early in the definition of a function before the
  /// body of the function has been parsed.
  ///
  /// 'FD' must be a function declared with a valid sycl_kernel_entry_point
  /// attribute and the current Sema context must match 'FD'.
  ///
  /// If construction of the AST fragments fails, diagnostics are issued and
  /// a disengaged optional value is returned.
  std::optional<SYCLKernelCallStmtASTFragments>
  BuildSYCLKernelCallStmtASTFragments(FunctionDecl *FD);

  /// BuildSYCLKernelCallStmt constructs a SYCLKernelCallStmt or an
  /// UnresolvedSYCLKernelCallStmt depending on whether 'FD' is a templated
  /// function.
  ///
  /// 'FD' must be a function declared with a valid sycl_kernel_entry_point
  /// attribute, must not have had its body assigned yet, and the current
  /// Sema context must match 'FD'.
  ///
  /// 'Body' is the parsed compound statement body of 'FD' to be wrapped by
  /// the new SYCLKernelCallStmt or UnresolvedSYCLKernelCallStmt.
  ///
  /// 'ASTFragments' contains the AST fragments previously returned by a call
  /// to BuildSYCLKernelCallStmtASTFragments() for 'FD'.
  StmtResult
  BuildSYCLKernelCallStmt(FunctionDecl *FD, CompoundStmt *Body,
                          const SYCLKernelCallStmtASTFragments &ASTFragments);
};

} // namespace clang

#endif // LLVM_CLANG_SEMA_SEMASYCL_H
