//===--- SimplifySubscriptExprCheck.cpp - clang-tidy-----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SimplifySubscriptExprCheck.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace readability {

static const char kDefaultTypes[] =
    "::std::basic_string;::std::basic_string_view;::std::vector;::std::array";

SimplifySubscriptExprCheck::SimplifySubscriptExprCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context), Types(utils::options::parseStringList(
                                         Options.get("Types", kDefaultTypes))) {
}

void SimplifySubscriptExprCheck::registerMatchers(MatchFinder *Finder) {
  if (!getLangOpts().CPlusPlus)
    return;

  const auto TypesMatcher = hasUnqualifiedDesugaredType(
      recordType(hasDeclaration(cxxRecordDecl(hasAnyName(
          llvm::SmallVector<StringRef, 8>(Types.begin(), Types.end()))))));

  Finder->addMatcher(
      arraySubscriptExpr(hasBase(ignoringParenImpCasts(
          cxxMemberCallExpr(
              has(memberExpr().bind("member")),
              on(hasType(qualType(
                  unless(anyOf(substTemplateTypeParmType(),
                               hasDescendant(substTemplateTypeParmType()))),
                  anyOf(TypesMatcher, pointerType(pointee(TypesMatcher)))))),
              callee(namedDecl(hasName("data"))))
              .bind("call")))),
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

} // namespace readability
} // namespace tidy
} // namespace clang
