//===----------------------------------------------------------------------===//
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

namespace clang::tidy::misc {

UniqueptrResetReleaseCheck::UniqueptrResetReleaseCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      Inserter(Options.getLocalOrGlobal("IncludeStyle",
                                        utils::IncludeSorter::IS_LLVM),
               areDiagsSelfContained()) {}

void UniqueptrResetReleaseCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IncludeStyle", Inserter.getStyle());
}

void UniqueptrResetReleaseCheck::registerPPCallbacks(
    const SourceManager &SM, Preprocessor *PP, Preprocessor *ModuleExpanderPP) {
  Inserter.registerPreprocessor(PP);
}

void UniqueptrResetReleaseCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      cxxMemberCallExpr(
          callee(memberExpr(
                     member(cxxMethodDecl(
                         hasName("reset"),
                         ofClass(cxxRecordDecl(hasName("::std::unique_ptr"),
                                               decl().bind("left_class"))))))
                     .bind("reset_member")),
          hasArgument(
              0, ignoringParenImpCasts(cxxMemberCallExpr(
                     on(expr().bind("right")),
                     callee(memberExpr(member(cxxMethodDecl(
                                           hasName("release"),
                                           ofClass(cxxRecordDecl(
                                               hasName("::std::unique_ptr"),
                                               decl().bind("right_class"))))))
                                .bind("release_member"))))))
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
  const auto *ResetCall =
      Result.Nodes.getNodeAs<CXXMemberCallExpr>("reset_call");

  StringRef AssignmentText = " = ";
  StringRef TrailingText = "";
  bool NeedsUtilityInclude = false;
  if (ReleaseMember->isArrow()) {
    AssignmentText = " = std::move(*";
    TrailingText = ")";
    NeedsUtilityInclude = true;
  } else if (!Right->isPRValue()) {
    AssignmentText = " = std::move(";
    TrailingText = ")";
    NeedsUtilityInclude = true;
  }

  auto D = diag(ResetMember->getExprLoc(),
                "prefer 'unique_ptr<>' assignment over 'release' and 'reset'");
  if (ResetMember->isArrow())
    D << FixItHint::CreateInsertion(ResetMember->getBeginLoc(), "*");
  D << FixItHint::CreateReplacement(
           CharSourceRange::getCharRange(ResetMember->getOperatorLoc(),
                                         Right->getBeginLoc()),
           AssignmentText)
    << FixItHint::CreateReplacement(
           CharSourceRange::getTokenRange(ReleaseMember->getOperatorLoc(),
                                          ResetCall->getEndLoc()),
           TrailingText);
  if (NeedsUtilityInclude)
    D << Inserter.createIncludeInsertion(
        Result.SourceManager->getFileID(ResetMember->getBeginLoc()),
        "<utility>");
}
} // namespace clang::tidy::misc
