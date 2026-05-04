//===--- SemaEJIT.cpp - EmbeddedJIT Attribute Handling --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements EmbeddedJIT attribute processing.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/DiagnosticSema.h"
#include "clang/Sema/ParsedAttr.h"
#include "clang/Sema/Sema.h"
#include "llvm/ADT/SmallVector.h"

using namespace clang;

namespace {

/// RecursiveASTVisitor that checks if a function calls itself.
class RecursiveCallVisitor
    : public RecursiveASTVisitor<RecursiveCallVisitor> {
  const FunctionDecl *FD;
public:
  bool FoundRecursiveCall = false;

  explicit RecursiveCallVisitor(const FunctionDecl *FD) : FD(FD) {}

  bool VisitCallExpr(CallExpr *CE) {
    if (auto *Callee = dyn_cast<DeclRefExpr>(CE->getCallee()->IgnoreParens())) {
      if (Callee->getDecl() == FD) {
        FoundRecursiveCall = true;
        return false; // Stop traversal
      }
    }
    return true;
  }
};

} // anonymous namespace

/// handleEjitMayConstAttr - Process the ejit_may_const attribute.
/// Checks:
///   1. Applies only to FieldDecl
///   2. Field type must be integer, boolean, floating-point, struct, or array
void handleEjitMayConstAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  auto *FD = dyn_cast<FieldDecl>(D);
  if (!FD) {
    S.Diag(AL.getLoc(), diag::warn_attribute_wrong_decl_type_str)
        << AL << AL.isRegularKeywordAttribute() << "field declarations";
    return;
  }

  QualType FT = FD->getType();

  // Check type: integer, boolean, floating-point, struct/class, or array
  if (!FT->isIntegerType() && !FT->isBooleanType() &&
      !FT->isFloatingType() && !FT->isStructureOrClassType() &&
      !FT->isArrayType()) {
    S.Diag(AL.getLoc(), diag::warn_attribute_wrong_decl_type_str)
        << AL << AL.isRegularKeywordAttribute()
        << "fields of integer, boolean, floating-point, or struct type";
    return;
  }

  // volatile fields are silently skipped — their loads won't get
  // !ejit.may_const metadata, so the JIT naturally ignores them.

  D->addAttr(::new (S.Context) EjitMayConstAttr(S.Context, AL));
}

/// handleEjitPeriodAttr - Process the ejit_period(name) attribute.
/// Checks:
///   1. Applies only to VarDecl
///   2. Must have global storage
///   3. Cannot be an array (use ejit_period_arr)
///   4. No duplicate period/period_arr attributes
void handleEjitPeriodAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  auto *VD = dyn_cast<VarDecl>(D);
  if (!VD) {
    S.Diag(AL.getLoc(), diag::warn_attribute_wrong_decl_type_str)
        << AL << AL.isRegularKeywordAttribute() << "variable declarations";
    return;
  }

  // Must be a global variable
  if (!VD->hasGlobalStorage()) {
    S.Diag(AL.getLoc(), diag::warn_attribute_wrong_decl_type_str)
        << AL << AL.isRegularKeywordAttribute() << "global variables";
    return;
  }

  // Cannot be an array — use ejit_period_arr for arrays
  if (VD->getType()->isArrayType()) {
    S.Diag(AL.getLoc(), diag::err_ejit_period_not_array) << VD;
    return;
  }

  // Check for duplicate period/period_arr attributes
  if (VD->hasAttr<EjitPeriodAttr>() || VD->hasAttr<EjitPeriodArrAttr>()) {
    S.Diag(AL.getLoc(), diag::err_ejit_period_conflict) << VD;
    return;
  }

  // Extract period name
  StringRef PeriodName;
  if (!S.checkStringLiteralArgumentAttr(AL, 0, PeriodName))
    return;

  VD->addAttr(::new (S.Context) EjitPeriodAttr(S.Context, AL, PeriodName));
}

/// handleEjitPeriodArrAttr - Process the ejit_period_arr(name) attribute.
/// Checks:
///   1. Applies only to VarDecl
///   2. Must have global storage
///   3. Must be an array type with constant size < 100
///   4. No duplicate period/period_arr attributes
void handleEjitPeriodArrAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  auto *VD = dyn_cast<VarDecl>(D);
  if (!VD) {
    S.Diag(AL.getLoc(), diag::warn_attribute_wrong_decl_type_str)
        << AL << AL.isRegularKeywordAttribute() << "variable declarations";
    return;
  }

  // Must be a global variable
  if (!VD->hasGlobalStorage()) {
    S.Diag(AL.getLoc(), diag::warn_attribute_wrong_decl_type_str)
        << AL << AL.isRegularKeywordAttribute() << "global variables";
    return;
  }

  // Must be an array type
  const ArrayType *AT = S.Context.getAsArrayType(VD->getType());
  if (!AT) {
    S.Diag(AL.getLoc(), diag::err_ejit_period_arr_not_scalar) << VD;
    return;
  }

  // Check array size < 100
  if (const auto *CAT = dyn_cast<ConstantArrayType>(AT)) {
    uint64_t Size = CAT->getSize().getZExtValue();
    if (Size > 100) {
      S.Diag(AL.getLoc(), diag::err_ejit_period_arr_too_large)
          << VD << static_cast<unsigned>(Size);
      return;
    }
  } else {
    // Non-constant array size (VLA, etc.) — error
    S.Diag(AL.getLoc(), diag::err_ejit_period_arr_not_scalar) << VD;
    return;
  }

  // Check for duplicate period/period_arr attributes
  if (VD->hasAttr<EjitPeriodAttr>() || VD->hasAttr<EjitPeriodArrAttr>()) {
    S.Diag(AL.getLoc(), diag::err_ejit_period_conflict) << VD;
    return;
  }

  // Extract period name
  StringRef PeriodName;
  if (!S.checkStringLiteralArgumentAttr(AL, 0, PeriodName))
    return;

  VD->addAttr(::new (S.Context) EjitPeriodArrAttr(S.Context, AL, PeriodName));
}

/// handleEjitPeriodArrIndAttr - Process the ejit_period_arr_ind(name) attribute.
/// Checks:
///   1. Applies only to ParmVarDecl
///   2. Parameter type must be integer
///   3. At most 4 such parameters per function
void handleEjitPeriodArrIndAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  auto *PVD = dyn_cast<ParmVarDecl>(D);
  if (!PVD) {
    S.Diag(AL.getLoc(), diag::warn_attribute_wrong_decl_type_str)
        << AL << AL.isRegularKeywordAttribute() << "function parameters";
    return;
  }

  // Parameter type must be integer
  QualType PT = PVD->getType();
  if (!PT->isIntegerType()) {
    S.Diag(AL.getLoc(), diag::err_ejit_period_arr_ind_invalid_type) << PVD;
    return;
  }

  // Max count check (4 per function) is deferred to
  // checkEjitPeriodArrIndLimit() in ActOnFunctionDeclarator because the
  // FunctionDecl is not yet set as ParmVarDecl DeclContext during parsing.

  // Extract period name
  StringRef PeriodName;
  if (!S.checkStringLiteralArgumentAttr(AL, 0, PeriodName))
    return;

  PVD->addAttr(::new (S.Context)
      EjitPeriodArrIndAttr(S.Context, AL, PeriodName));
}

/// handleEjitEntryAttr - Process the ejit_entry attribute.
/// Checks:
///   1. Applies only to FunctionDecl
///   2. Function must not be recursive
void handleEjitEntryAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  auto *FD = dyn_cast<FunctionDecl>(D);
  if (!FD) {
    S.Diag(AL.getLoc(), diag::warn_attribute_wrong_decl_type_str)
        << AL << AL.isRegularKeywordAttribute() << "functions";
    return;
  }

  // Check for recursive function (only when body is available)
  if (FD->hasBody()) {
    RecursiveCallVisitor Visitor(FD);
    Visitor.TraverseStmt(FD->getBody());
    if (Visitor.FoundRecursiveCall) {
      S.Diag(AL.getLoc(), diag::err_ejit_entry_recursive) << FD;
      return;
    }
  }

  D->addAttr(::new (S.Context) EjitEntryAttr(S.Context, AL));
}

/// handleEjitPeriodLcAttr - Process the ejit_period_lc(name) attribute.
/// Checks:
///   1. Applies only to FunctionDecl
///   2. Must have a corresponding ejit_period_arr_ind(name) parameter
void handleEjitPeriodLcAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  auto *FD = dyn_cast<FunctionDecl>(D);
  if (!FD) {
    S.Diag(AL.getLoc(), diag::warn_attribute_wrong_decl_type_str)
        << AL << AL.isRegularKeywordAttribute() << "functions";
    return;
  }

  // Extract period name
  StringRef PeriodName;
  if (!S.checkStringLiteralArgumentAttr(AL, 0, PeriodName))
    return;

  // Check for matching ejit_period_arr_ind parameter
  bool HasMatchingIdx = false;
  for (const ParmVarDecl *P : FD->parameters()) {
    if (auto *IdxAttr = P->getAttr<EjitPeriodArrIndAttr>()) {
      if (IdxAttr->getPeriodName() == PeriodName) {
        HasMatchingIdx = true;
        break;
      }
    }
  }

  if (!HasMatchingIdx) {
    S.Diag(AL.getLoc(), diag::err_ejit_period_lc_no_index) << PeriodName;
    return;
  }

  FD->addAttr(::new (S.Context) EjitPeriodLcAttr(S.Context, AL, PeriodName));
}

/// checkEjitPeriodArrIndLimit - Enforce the limit of at most 4
/// ejit_period_arr_ind parameters per function. Called from
/// ActOnFunctionDeclarator after all parameter attributes have been processed.
void checkEjitPeriodArrIndLimit(Sema &S, const FunctionDecl *FD) {
  if (!FD)
    return;

  unsigned Count = 0;
  const ParmVarDecl *OverflowPVD = nullptr;
  for (const ParmVarDecl *P : FD->parameters()) {
    if (P->hasAttr<EjitPeriodArrIndAttr>()) {
      Count++;
      if (Count > 4)
        OverflowPVD = P;
    }
  }

  if (Count > 4 && OverflowPVD) {
    // Get the attribute location from the overflow parameter
    if (auto *A = OverflowPVD->getAttr<EjitPeriodArrIndAttr>()) {
      S.Diag(A->getLocation(), diag::err_ejit_period_arr_ind_too_many)
          << FD << Count;
    }
  }
}
