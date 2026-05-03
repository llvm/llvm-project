//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "StringviewNullptrCheck.h"
#include "../utils/LexerUtils.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

namespace {

AST_MATCHER(CXXConstructExpr, constructedFromNullptr) {
  if (Node.getNumArgs() != 1)
    return false;

  const Expr *Arg = Node.getArg(0);
  bool ArgValue; // NOLINT(cppcoreguidelines-init-variables)
  return !Arg->isValueDependent() &&
         Arg->EvaluateAsBooleanCondition(ArgValue, Finder->getASTContext()) &&
         !ArgValue;
}

} // namespace

void StringviewNullptrCheck::registerMatchers(MatchFinder *Finder) {
  const auto HasBasicStringViewType =
      hasType(hasUnqualifiedDesugaredType(recordType(
          hasDeclaration(cxxRecordDecl(hasName("::std::basic_string_view"))))));

  Finder->addMatcher(
      cxxConstructExpr(HasBasicStringViewType, constructedFromNullptr())
          .bind("construct_expr"),
      this);
}

void StringviewNullptrCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *ConstructExpr =
      Result.Nodes.getNodeAs<CXXConstructExpr>("construct_expr");
  const Expr *NullArg = ConstructExpr->getArg(0);
  const SourceLocation NullArgExpansionLoc =
      Result.SourceManager->getExpansionLoc(NullArg->getBeginLoc());

  auto Diag = diag(NullArgExpansionLoc,
                   "constructing basic_string_view from null is undefined");

  const std::optional<Token> TokenBeforeNullArg =
      utils::lexer::getPreviousToken(NullArgExpansionLoc, *Result.SourceManager,
                                     getLangOpts());

  if (!TokenBeforeNullArg)
    return;

  const auto NullArgReplacement = [&]() -> StringRef {
    // 'sv = nullptr;' -> 'sv = {};'
    // '(std::string_view)nullptr' -> '(std::string_view){}'
    // 'return nullptr;' -> 'return {};'
    if (TokenBeforeNullArg->isOneOf(tok::equal, tok::r_paren) ||
        (TokenBeforeNullArg->is(tok::raw_identifier) &&
         TokenBeforeNullArg->getRawIdentifier() == "return"))
      return "{}";

    // Implicit constructor call.
    if (ConstructExpr->getSourceRange() == NullArg->getSourceRange())
      return "\"\"";

    // 'std::string_view {nullptr}' -> 'std::string_view {}'
    // 'std::string_view sv {nullptr};' -> 'std::string_view sv {};'
    if (TokenBeforeNullArg->is(tok::l_brace))
      return {};

    if (TokenBeforeNullArg->is(tok::l_paren)) {
      const DynTypedNodeList Parents =
          Result.Context->getParentMapContext().getParents(*ConstructExpr);

      if (Parents.empty())
        return {};

      // 'static_cast<std::string_view>(nullptr)'
      //     -> 'static_cast<std::string_view>("")'
      if (Parents[0].get<CXXStaticCastExpr>())
        return "\"\"";

      // Avoid causing a most vexing parse.
      if (const auto *Var = Parents[0].get<VarDecl>())
        if (Var->getInitStyle() == VarDecl::InitializationStyle::CallInit)
          Diag << FixItHint::CreateRemoval(
              ConstructExpr->getParenOrBraceRange());

      return {};
    }

    return "\"\"";
  }();

  Diag << FixItHint::CreateReplacement(NullArg->getSourceRange(),
                                       NullArgReplacement);
}

} // namespace clang::tidy::bugprone
