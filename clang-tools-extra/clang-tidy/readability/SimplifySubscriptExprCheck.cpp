//===--- SimplifySubscriptExprCheck.cpp - clang-tidy-----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SimplifySubscriptExprCheck.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

static const char KDefaultTypes[] =
    "::std::basic_string;::std::basic_string_view;::std::vector;::std::array;::"
    "std::span";

SimplifySubscriptExprCheck::SimplifySubscriptExprCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context), Types(utils::options::parseStringList(
                                         Options.get("Types", KDefaultTypes))) {
}

void SimplifySubscriptExprCheck::registerMatchers(MatchFinder *Finder) {
  const auto TypesMatcher = hasUnqualifiedDesugaredType(
      recordType(hasDeclaration(cxxRecordDecl(hasAnyName(Types)))));

  Finder->addMatcher(
      arraySubscriptExpr(hasBase(
          cxxMemberCallExpr(
              has(memberExpr().bind("member")),
              on(hasType(qualType(
                  unless(anyOf(substTemplateTypeParmType(),
                               hasDescendant(substTemplateTypeParmType()))),
                  anyOf(TypesMatcher, pointerType(pointee(TypesMatcher)))))),
              callee(namedDecl(hasName("data"))))
              .bind("call"))),
      this);
}

void SimplifySubscriptExprCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Call = Result.Nodes.getNodeAs<CXXMemberCallExpr>("call");
  if (Result.Context->getSourceManager().isMacroBodyExpansion(
          Call->getExprLoc()))
    return;

  const auto *Member = Result.Nodes.getNodeAs<MemberExpr>("member");
  auto DiagBuilder =
      diag(Member->getMemberLoc(),
           "accessing an element of the container does not require a call to "
           "'data()'; did you mean to use 'operator[]'?");
  if (Member->isArrow())
    DiagBuilder << FixItHint::CreateInsertion(Member->getBeginLoc(), "(*")
                << FixItHint::CreateInsertion(Member->getOperatorLoc(), ")");
  DiagBuilder << FixItHint::CreateRemoval(
      {Member->getOperatorLoc(), Call->getEndLoc()});
}

void SimplifySubscriptExprCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "Types", utils::options::serializeStringList(Types));
}

} // namespace clang::tidy::readability
