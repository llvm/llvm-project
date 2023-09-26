//===--- MoveSharedPointerContentsCheck.cpp - clang-tidy ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MoveSharedPointerContentsCheck.h"
#include "../ClangTidyCheck.h"
#include "../utils/Matchers.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

MoveSharedPointerContentsCheck::MoveSharedPointerContentsCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      SharedPointerClasses(utils::options::parseStringList(
          Options.get("SharedPointerClasses", "::std::shared_ptr"))) {}

void MoveSharedPointerContentsCheck::registerMatchers(MatchFinder *Finder) {
  auto isStdMove = callee(functionDecl(hasName("::std::move")));

  // Resolved type, direct move.
  Finder->addMatcher(
      callExpr(isStdMove, hasArgument(0, cxxOperatorCallExpr(
                                             hasOverloadedOperatorName("*"),
                                             callee(cxxMethodDecl(ofClass(
                                                 matchers::matchesAnyListedName(
                                                     SharedPointerClasses)))))))
          .bind("call"),
      this);

  // Resolved type, move out of get().
  Finder->addMatcher(
      callExpr(
          isStdMove,
          hasArgument(
              0, unaryOperator(
                     hasOperatorName("*"),
                     hasUnaryOperand(cxxMemberCallExpr(callee(cxxMethodDecl(
                         hasName("get"), ofClass(matchers::matchesAnyListedName(
                                             SharedPointerClasses)))))))))
          .bind("get_call"),
      this);

  auto isStdMoveUnresolved = callee(unresolvedLookupExpr(
      hasAnyDeclaration(namedDecl(hasUnderlyingDecl(hasName("::std::move"))))));

  // Unresolved type, direct move.
  Finder->addMatcher(
      callExpr(
          isStdMoveUnresolved,
          hasArgument(0, unaryOperator(hasOperatorName("*"),
                                       hasUnaryOperand(declRefExpr(hasType(
                                           qualType().bind("unresolved_p")))))))
          .bind("unresolved_call"),
      this);
  // Annoyingly, the declRefExpr in the unresolved-move-of-get() case
  // is of <dependent type> rather than shared_ptr<T>, so we have to
  // just fetch the variable. This does leave a gap where a temporary
  // shared_ptr wouldn't be caught, but moving out of a temporary
  // shared pointer is a truly wild thing to do so it should be okay.

  // Unresolved type, move out of get().
  Finder->addMatcher(
      callExpr(isStdMoveUnresolved,
               hasArgument(
                   0, unaryOperator(hasOperatorName("*"),
                                    hasDescendant(cxxDependentScopeMemberExpr(
                                        hasMemberName("get"))),
                                    hasDescendant(declRefExpr(to(
                                        varDecl().bind("unresolved_get_p")))))))
          .bind("unresolved_get_call"),
      this);
}

bool MoveSharedPointerContentsCheck::isSharedPointerClass(
    const VarDecl *VD) const {
  if (VD == nullptr)
    return false;

  const QualType QT = VD->getType();
  return isSharedPointerClass(&QT);
}

bool MoveSharedPointerContentsCheck::isSharedPointerClass(
    const QualType *QT) const {
  if (QT == nullptr)
    return false;

  // We want the qualified name without template parameters,
  // const/volatile, or reference/pointer qualifiers so we can look
  // it up in SharedPointerClasses. This is a bit messy, but gets us
  // to the underlying type without template parameters (eg
  // std::shared_ptr) or const/volatile qualifiers even in the face of
  // typedefs.

  bool found = false;
  const auto *Template = llvm::dyn_cast<TemplateSpecializationType>(
      QT->getSplitDesugaredType().Ty);
  if (Template != nullptr) {
    const std::string TypeName = Template->getTemplateName()
                                     .getAsTemplateDecl()
                                     ->getQualifiedNameAsString();
    for (const llvm::StringRef SharedPointer : SharedPointerClasses) {
      // SharedPointer entries may or may not have leading ::, but TypeName
      // definitely won't.
      if (SharedPointer == TypeName || SharedPointer.substr(2) == TypeName) {
        found = true;
        break;
      }
    }
  }

  return found;
}

void MoveSharedPointerContentsCheck::check(
    const MatchFinder::MatchResult &Result) {
  const bool Unresolved =
      isSharedPointerClass(Result.Nodes.getNodeAs<QualType>("unresolved_p"));
  const bool UnresolvedGet =
      isSharedPointerClass(Result.Nodes.getNodeAs<VarDecl>("unresolved_get_p"));

  clang::SourceLocation Loc;
  if (const auto *UnresolvedCall =
          Result.Nodes.getNodeAs<CallExpr>("unresolved_call");
      UnresolvedCall != nullptr && Unresolved) {
    Loc = UnresolvedCall->getBeginLoc();
  } else if (const auto *UnresolvedGetCall =
                 Result.Nodes.getNodeAs<CallExpr>("unresolved_get_call");
             UnresolvedGetCall != nullptr && UnresolvedGet) {
    Loc = UnresolvedGetCall->getBeginLoc();
  } else if (const auto *GetCall = Result.Nodes.getNodeAs<CallExpr>("get_call");
             GetCall != nullptr) {
    Loc = GetCall->getBeginLoc();
  } else if (const auto *Call = Result.Nodes.getNodeAs<CallExpr>("call");
             Call != nullptr) {
    Loc = Call->getBeginLoc();
  } else {
    return;
  }

  if (Loc.isValid()) {
    diag(Loc,
         "don't move the contents out of a shared pointer, as other accessors "
         "expect them to remain in a determinate state");
  }
}

} // namespace clang::tidy::bugprone
