//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseSizeTypeCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

void UseSizeTypeCheck::registerMatchers(MatchFinder *Finder) {
  // Match local variables with signed integer type initialized from an
  // unsigned expression (via implicit cast).
  Finder->addMatcher(
      varDecl(hasLocalStorage(),
              hasType(qualType(hasCanonicalType(isSignedInteger()))),
              hasInitializer(ignoringImplicit(expr(
                  hasType(qualType(hasCanonicalType(isUnsignedInteger())))))),
              unless(isConstexpr()),
              hasParent(declStmt(hasParent(compoundStmt().bind("scope")))))
          .bind("var"),
      this);
}

/// Return true if every use of \p VD within \p Scope is in a context that
/// expects or is compatible with an unsigned / size_t type.
static bool allUsesAreUnsignedCompatible(const VarDecl *VD,
                                         const CompoundStmt *Scope,
                                         ASTContext &Ctx) {
  auto Refs =
      match(findAll(declRefExpr(to(varDecl(equalsNode(VD)))).bind("ref")),
            *Scope, Ctx);

  if (Refs.empty())
    return false;

  for (const auto &Ref : Refs) {
    const auto *DRE = Ref.getNodeAs<DeclRefExpr>("ref");
    if (!DRE)
      return false;

    const auto Parents = Ctx.getParents(*DRE);
    if (Parents.empty())
      return false;

    // Walk up through implicit casts to find the "real" parent context.
    // The DeclRefExpr is typically wrapped in LValueToRValue and
    // IntegralCast implicit casts.
    DynTypedNodeList CurrentParents = Parents;
    while (CurrentParents.size() == 1) {
      if (const auto *ICE = CurrentParents[0].get<ImplicitCastExpr>()) {
        CurrentParents = Ctx.getParents(*ICE);
        continue;
      }
      break;
    }

    bool UsageOk = false;
    for (const auto &Parent : CurrentParents) {
      // Used in binary comparison with unsigned operand.
      if (const auto *BO = Parent.get<BinaryOperator>()) {
        if (BO->isComparisonOp()) {
          UsageOk = true;
          break;
        }
        // Also accept arithmetic where the result feeds into an
        // unsigned context (but not standalone).
      }

      // Used as function argument (CallExpr or CXXMemberCallExpr).
      if (const auto *CE = Parent.get<CallExpr>()) {
        for (unsigned I = 0; I < CE->getNumArgs(); ++I) {
          if (CE->getArg(I)->IgnoreParenImpCasts() == DRE) {
            if (const auto *FD = CE->getDirectCallee()) {
              if (I < FD->getNumParams()) {
                const QualType PT =
                    FD->getParamDecl(I)->getType().getCanonicalType();
                if (PT->isUnsignedIntegerType())
                  UsageOk = true;
              }
            }
            break;
          }
        }
        if (UsageOk)
          break;
      }

      // Used as array subscript index.
      if (const auto *ASE = Parent.get<ArraySubscriptExpr>()) {
        if (ASE->getIdx()->IgnoreParenImpCasts() == DRE) {
          UsageOk = true;
          break;
        }
      }

      // Used in operator[] (CXXOperatorCallExpr).
      if (const auto *OCE = Parent.get<CXXOperatorCallExpr>()) {
        if (OCE->getOperator() == OO_Subscript && OCE->getNumArgs() > 1 &&
            OCE->getArg(1)->IgnoreParenImpCasts() == DRE) {
          UsageOk = true;
          break;
        }
        if (UsageOk)
          break;
      }
    }

    if (!UsageOk)
      return false;
  }

  return true;
}

void UseSizeTypeCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *VD = Result.Nodes.getNodeAs<VarDecl>("var");
  const auto *Scope = Result.Nodes.getNodeAs<CompoundStmt>("scope");
  if (!VD || !Scope)
    return;

  // Skip dependent types.
  if (VD->getType()->isDependentType())
    return;

  // Skip macros.
  if (VD->getLocation().isMacroID())
    return;

  // Check all uses are unsigned-compatible.
  if (!allUsesAreUnsignedCompatible(VD, Scope, *Result.Context))
    return;

  // Get the type specifier source range to replace.
  const SourceLocation TypeStart = VD->getTypeSpecStartLoc();
  const SourceLocation TypeEnd = VD->getTypeSpecEndLoc();
  if (TypeStart.isInvalid() || TypeEnd.isInvalid())
    return;
  if (TypeStart.isMacroID() || TypeEnd.isMacroID())
    return;

  diag(VD->getLocation(),
       "variable %0 is of signed type %1 but is initialized from and "
       "used as an unsigned value; consider using 'size_t'")
      << VD << VD->getType()
      << FixItHint::CreateReplacement(
             CharSourceRange::getTokenRange(TypeStart, TypeEnd), "size_t");
}

} // namespace clang::tidy::modernize
