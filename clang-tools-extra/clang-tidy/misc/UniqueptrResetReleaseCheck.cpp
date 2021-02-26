//===--- UniqueptrResetReleaseCheck.cpp - clang-tidy ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UniqueptrResetReleaseCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace misc {

void UniqueptrResetReleaseCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      cxxMemberCallExpr(
          on(expr().bind("left")), callee(memberExpr().bind("reset_member")),
          callee(
              cxxMethodDecl(hasName("reset"),
                            ofClass(cxxRecordDecl(hasName("::std::unique_ptr"),
                                                  decl().bind("left_class"))))),
          has(ignoringParenImpCasts(cxxMemberCallExpr(
              on(expr().bind("right")),
              callee(memberExpr().bind("release_member")),
              callee(cxxMethodDecl(
                  hasName("release"),
                  ofClass(cxxRecordDecl(hasName("::std::unique_ptr"),
                                        decl().bind("right_class")))))))))
          .bind("reset_call"),
      this);
}

namespace {
const Type *getDeleterForUniquePtr(const MatchFinder::MatchResult &Result,
                                   StringRef ID) {
  const auto *Class =
      Result.Nodes.getNodeAs<ClassTemplateSpecializationDecl>(ID);
  if (!Class)
    return nullptr;
  auto DeleterArgument = Class->getTemplateArgs()[1];
  if (DeleterArgument.getKind() != TemplateArgument::Type)
    return nullptr;
  return DeleterArgument.getAsType().getTypePtr();
}

bool areDeletersCompatible(const MatchFinder::MatchResult &Result) {
  const Type *LeftDeleterType = getDeleterForUniquePtr(Result, "left_class");
  const Type *RightDeleterType = getDeleterForUniquePtr(Result, "right_class");

  if (LeftDeleterType->getUnqualifiedDesugaredType() ==
      RightDeleterType->getUnqualifiedDesugaredType()) {
    // Same type. We assume they are compatible.
    // This check handles the case where the deleters are function pointers.
    return true;
  }

  const CXXRecordDecl *LeftDeleter = LeftDeleterType->getAsCXXRecordDecl();
  const CXXRecordDecl *RightDeleter = RightDeleterType->getAsCXXRecordDecl();
  if (!LeftDeleter || !RightDeleter)
    return false;

  if (LeftDeleter->getCanonicalDecl() == RightDeleter->getCanonicalDecl()) {
    // Same class. We assume they are compatible.
    return true;
  }

  const auto *LeftAsTemplate =
      dyn_cast<ClassTemplateSpecializationDecl>(LeftDeleter);
  const auto *RightAsTemplate =
      dyn_cast<ClassTemplateSpecializationDecl>(RightDeleter);
  if (LeftAsTemplate && RightAsTemplate &&
      LeftAsTemplate->getSpecializedTemplate() ==
          RightAsTemplate->getSpecializedTemplate()) {
    // They are different instantiations of the same template. We assume they
    // are compatible.
    // This handles things like std::default_delete<Base> vs.
    // std::default_delete<Derived>.
    return true;
  }
  return false;
}

} // namespace

void UniqueptrResetReleaseCheck::check(const MatchFinder::MatchResult &Result) {
  if (!areDeletersCompatible(Result))
    return;

  const auto *ResetMember = Result.Nodes.getNodeAs<MemberExpr>("reset_member");
  const auto *ReleaseMember =
      Result.Nodes.getNodeAs<MemberExpr>("release_member");
  const auto *Right = Result.Nodes.getNodeAs<Expr>("right");
  const auto *Left = Result.Nodes.getNodeAs<Expr>("left");
  const auto *ResetCall =
      Result.Nodes.getNodeAs<CXXMemberCallExpr>("reset_call");

  std::string LeftText = std::string(clang::Lexer::getSourceText(
      CharSourceRange::getTokenRange(Left->getSourceRange()),
      *Result.SourceManager, getLangOpts()));
  std::string RightText = std::string(clang::Lexer::getSourceText(
      CharSourceRange::getTokenRange(Right->getSourceRange()),
      *Result.SourceManager, getLangOpts()));

  if (ResetMember->isArrow())
    LeftText = "*" + LeftText;
  if (ReleaseMember->isArrow())
    RightText = "*" + RightText;
  bool IsMove = false;
  // Even if x was rvalue, *x is not rvalue anymore.
  if (!Right->isRValue() || ReleaseMember->isArrow()) {
    RightText = "std::move(" + RightText + ")";
    IsMove = true;
  }

  std::string NewText = LeftText + " = " + RightText;

  diag(ResetMember->getExprLoc(),
       "prefer ptr = %select{std::move(ptr2)|ReturnUnique()}0 over "
       "ptr.reset(%select{ptr2|ReturnUnique()}0.release())")
      << !IsMove
      << FixItHint::CreateReplacement(
             CharSourceRange::getTokenRange(ResetCall->getSourceRange()),
             NewText);
}

} // namespace misc
} // namespace tidy
} // namespace clang
