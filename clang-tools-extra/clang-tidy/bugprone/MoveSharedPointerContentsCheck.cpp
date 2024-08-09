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
namespace {

// Reports whether the QualType matches the inner matcher, which is expected to be
// matchesAnyListedName. The QualType is expected to either point to a RecordDecl
// (for concrete types) or an ElaboratedType (for dependent ones).
AST_MATCHER_P(QualType, isSharedPointer,
              clang::ast_matchers::internal::Matcher<NamedDecl>, InnerMatcher) {
  if (const auto *RD = Node.getTypePtr()->getAsCXXRecordDecl(); RD != nullptr) {
    return InnerMatcher.matches(*RD, Finder, Builder);
  } else if (const auto *ED = Node.getTypePtr()->getAs<ElaboratedType>();
             ED != nullptr) {
    if (const auto *TS = ED->getNamedType()
                             .getTypePtr()
                             ->getAs<TemplateSpecializationType>();
        TS != nullptr) {
      return InnerMatcher.matches(*TS->getTemplateName().getAsTemplateDecl(),
                                  Finder, Builder);
    }
  }

  return false;
}

} // namespace

MoveSharedPointerContentsCheck::MoveSharedPointerContentsCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      SharedPointerClasses(utils::options::parseStringList(
          Options.get("SharedPointerClasses", "::std::shared_ptr;::boost::shared_pointer"))) {}

void MoveSharedPointerContentsCheck::registerMatchers(MatchFinder *Finder) {
  auto isStdMove = callee(functionDecl(hasAnyName("::std::move", "::std::forward")));

  auto resolvedType = callExpr(anyOf(
      // Resolved type, direct move.
      callExpr(
          isStdMove,
          hasArgument(
              0,
              cxxOperatorCallExpr(
                  hasOverloadedOperatorName("*"),
                  hasDescendant(declRefExpr(hasType(qualType(isSharedPointer(
                      matchers::matchesAnyListedName(SharedPointerClasses)))))),
                  callee(cxxMethodDecl()))))
          .bind("call"),
      // Resolved type, move out of get().
      callExpr(
          isStdMove,
          hasArgument(
              0, unaryOperator(
                     hasOperatorName("*"),
                     hasUnaryOperand(allOf(
                         hasDescendant(declRefExpr(hasType(qualType(
                             isSharedPointer(matchers::matchesAnyListedName(
                                 SharedPointerClasses)))))),
                         cxxMemberCallExpr(
                             callee(cxxMethodDecl(hasName("get")))))))))
          .bind("get_call")));

  Finder->addMatcher(resolvedType, this);

  auto isStdMoveUnresolved = callee(unresolvedLookupExpr(
      hasAnyDeclaration(namedDecl(hasUnderlyingDecl(hasName("::std::move"))))));

  auto unresolvedType = callExpr(anyOf(
      // Unresolved type, direct move.
      callExpr(
          isStdMoveUnresolved,
          hasArgument(0, unaryOperator(
                             hasOperatorName("*"),
                             hasUnaryOperand(declRefExpr(hasType(qualType(
                                 isSharedPointer(matchers::matchesAnyListedName(
                                     SharedPointerClasses)))))))))
          .bind("unresolved_call"),

      // Annoyingly, the declRefExpr in the unresolved-move-of-get() case
      // is of <dependent type> rather than shared_ptr<T>, so we have to
      // just fetch the variable. This does leave a gap where a temporary
      // shared_ptr wouldn't be caught, but moving out of a temporary
      // shared pointer is a truly wild thing to do so it should be okay.
      callExpr(isStdMoveUnresolved,
               hasArgument(
                   0, unaryOperator(
                          hasOperatorName("*"),
                          hasDescendant(cxxDependentScopeMemberExpr(
                              hasMemberName("get"))),
                          hasDescendant(declRefExpr(to(varDecl(hasType(qualType(
                              isSharedPointer(matchers::matchesAnyListedName(
                                  SharedPointerClasses)))))))))))
          .bind("unresolved_get_call")));

  Finder->addMatcher(unresolvedType, this);
}

void MoveSharedPointerContentsCheck::check(
    const MatchFinder::MatchResult &Result) {
  clang::SourceLocation Loc;
  if (const auto *UnresolvedCall =
          Result.Nodes.getNodeAs<CallExpr>("unresolved_call");
      UnresolvedCall != nullptr) {
    Loc = UnresolvedCall->getBeginLoc();
  } else if (const auto *UnresolvedGetCall =
                 Result.Nodes.getNodeAs<CallExpr>("unresolved_get_call");
             UnresolvedGetCall != nullptr) {
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
