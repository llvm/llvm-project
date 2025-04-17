//===--- InvalidEnumDefaultInitializationCheck.cpp - clang-tidy -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InvalidEnumDefaultInitializationCheck.h"
// #include "../utils/Matchers.h"
// #include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include <algorithm>

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

namespace {

AST_MATCHER(EnumDecl, isCompleteAndHasNoZeroValue) {
  const EnumDecl *Definition = Node.getDefinition();
  return Definition && Node.isComplete() &&
         std::none_of(Definition->enumerator_begin(),
                      Definition->enumerator_end(),
                      [](const EnumConstantDecl *Value) {
                        return Value->getInitVal().isZero();
                      });
}

AST_MATCHER(Expr, isEmptyInit) {
  if (isa<CXXScalarValueInitExpr>(&Node))
    return true;
  if (isa<ImplicitValueInitExpr>(&Node))
    return true;
  if (const auto *Init = dyn_cast<InitListExpr>(&Node))
    return Init->getNumInits() == 0;
  return false;
}

} // namespace

InvalidEnumDefaultInitializationCheck::InvalidEnumDefaultInitializationCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context) {}

bool InvalidEnumDefaultInitializationCheck::isLanguageVersionSupported(
    const LangOptions &LangOpts) const {
  return LangOpts.CPlusPlus;
}

void InvalidEnumDefaultInitializationCheck::registerMatchers(
    MatchFinder *Finder) {
  Finder->addMatcher(
      expr(isEmptyInit(),
           hasType(hasUnqualifiedDesugaredType(enumType(hasDeclaration(
               enumDecl(isCompleteAndHasNoZeroValue()).bind("enum"))))))
          .bind("expr"),
      this);
}

void InvalidEnumDefaultInitializationCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *InitExpr = Result.Nodes.getNodeAs<Expr>("expr");
  const auto *Enum = Result.Nodes.getNodeAs<EnumDecl>("enum");
  if (!InitExpr || !Enum)
    return;

  ASTContext &ACtx = Enum->getASTContext();
  SourceLocation Loc = InitExpr->getExprLoc();
  if (Loc.isInvalid()) {
    if (isa<ImplicitValueInitExpr>(InitExpr) || isa<InitListExpr>(InitExpr)) {
      DynTypedNodeList Parents = ACtx.getParents(*InitExpr);
      if (Parents.empty())
        return;

      if (const auto *Ctor = Parents[0].get<CXXConstructorDecl>()) {
        // Try to find member initializer with the found expression and get the
        // source location from it.
        CXXCtorInitializer *const *CtorInit = std::find_if(
            Ctor->init_begin(), Ctor->init_end(),
            [InitExpr](const CXXCtorInitializer *Init) {
              return Init->isMemberInitializer() && Init->getInit() == InitExpr;
            });
        if (!CtorInit)
          return;
        Loc = (*CtorInit)->getLParenLoc();
      } else if (const auto *InitList = Parents[0].get<InitListExpr>()) {
        // The expression may be implicitly generated for an initialization.
        // Search for a parent initialization list with valid source location.
        while (InitList->getExprLoc().isInvalid()) {
          DynTypedNodeList Parents = ACtx.getParents(*InitList);
          if (Parents.empty())
            return;
          InitList = Parents[0].get<InitListExpr>();
          if (!InitList)
            return;
        }
        Loc = InitList->getExprLoc();
      }
    }
    // If still not found a source location, omit the warning.
    // FIXME: All such cases should be fixed to make the checker more precise.
    if (Loc.isInvalid())
      return;
  }
  diag(Loc, "enum value of type %0 initialized with invalid value of 0, "
            "enum doesn't have a zero-value enumerator")
      << Enum;
  diag(Enum->getLocation(), "enum is defined here", DiagnosticIDs::Note);
}

} // namespace clang::tidy::bugprone
