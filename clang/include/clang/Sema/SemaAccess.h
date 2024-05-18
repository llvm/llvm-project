//===----- SemaAccess.h --------- C++ Access Control ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file declares routines for C++ access control semantics.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_SEMAACCESS_H
#define LLVM_CLANG_SEMA_SEMAACCESS_H

#include "clang/AST/CXXInheritance.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclAccessPair.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DependentDiagnostic.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Type.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/PartialDiagnostic.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Sema/DelayedDiagnostic.h"
#include "clang/Sema/Initialization.h"
#include "clang/Sema/SemaBase.h"

namespace clang {
class LookupResult;
class MultiLevelTemplateArgumentList;

class SemaAccess : public SemaBase {
public:
  SemaAccess(Sema &S);

  enum AccessResult {
    AR_accessible,
    AR_inaccessible,
    AR_dependent,
    AR_delayed
  };

  bool SetMemberAccessSpecifier(NamedDecl *MemberDecl,
                                NamedDecl *PrevMemberDecl,
                                AccessSpecifier LexicalAS);

  AccessResult CheckUnresolvedMemberAccess(UnresolvedMemberExpr *E,
                                           DeclAccessPair FoundDecl);
  AccessResult CheckUnresolvedLookupAccess(UnresolvedLookupExpr *E,
                                           DeclAccessPair FoundDecl);
  AccessResult CheckAllocationAccess(SourceLocation OperatorLoc,
                                     SourceRange PlacementRange,
                                     CXXRecordDecl *NamingClass,
                                     DeclAccessPair FoundDecl,
                                     bool Diagnose = true);
  AccessResult CheckConstructorAccess(SourceLocation Loc, CXXConstructorDecl *D,
                                      DeclAccessPair FoundDecl,
                                      const InitializedEntity &Entity,
                                      bool IsCopyBindingRefToTemp = false);
  AccessResult CheckConstructorAccess(SourceLocation Loc, CXXConstructorDecl *D,
                                      DeclAccessPair FoundDecl,
                                      const InitializedEntity &Entity,
                                      const PartialDiagnostic &PDiag);
  AccessResult CheckDestructorAccess(SourceLocation Loc,
                                     CXXDestructorDecl *Dtor,
                                     const PartialDiagnostic &PDiag,
                                     QualType objectType = QualType());
  AccessResult CheckFriendAccess(NamedDecl *D);
  AccessResult CheckMemberAccess(SourceLocation UseLoc,
                                 CXXRecordDecl *NamingClass,
                                 DeclAccessPair Found);
  AccessResult
  CheckStructuredBindingMemberAccess(SourceLocation UseLoc,
                                     CXXRecordDecl *DecomposedClass,
                                     DeclAccessPair Field);
  AccessResult CheckMemberOperatorAccess(SourceLocation Loc, Expr *ObjectExpr,
                                         const SourceRange &,
                                         DeclAccessPair FoundDecl);
  AccessResult CheckMemberOperatorAccess(SourceLocation Loc, Expr *ObjectExpr,
                                         Expr *ArgExpr,
                                         DeclAccessPair FoundDecl);
  AccessResult CheckMemberOperatorAccess(SourceLocation Loc, Expr *ObjectExpr,
                                         ArrayRef<Expr *> ArgExprs,
                                         DeclAccessPair FoundDecl);
  AccessResult CheckAddressOfMemberAccess(Expr *OvlExpr,
                                          DeclAccessPair FoundDecl);
  AccessResult CheckBaseClassAccess(SourceLocation AccessLoc, QualType Base,
                                    QualType Derived, const CXXBasePath &Path,
                                    unsigned DiagID, bool ForceCheck = false,
                                    bool ForceUnprivileged = false);
  void CheckLookupAccess(const LookupResult &R);
  bool IsSimplyAccessible(NamedDecl *Decl, CXXRecordDecl *NamingClass,
                          QualType BaseType);
  bool isMemberAccessibleForDeletion(CXXRecordDecl *NamingClass,
                                     DeclAccessPair Found, QualType ObjectType,
                                     SourceLocation Loc,
                                     const PartialDiagnostic &Diag);
  bool isMemberAccessibleForDeletion(CXXRecordDecl *NamingClass,
                                     DeclAccessPair Found, QualType ObjectType);

  void HandleDependentAccessCheck(
      const DependentDiagnostic &DD,
      const MultiLevelTemplateArgumentList &TemplateArgs);
  void HandleDelayedAccessCheck(sema::DelayedDiagnostic &DD, Decl *Ctx);
};
} // namespace clang

#endif // LLVM_CLANG_SEMA_SEMAACCESS_H