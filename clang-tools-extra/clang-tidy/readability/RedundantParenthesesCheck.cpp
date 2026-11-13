//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RedundantParenthesesCheck.h"
#include "../utils/Matchers.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/AST/TypeLoc.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchersMacros.h"
#include "clang/Lex/Lexer.h"
#include <cassert>

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

namespace {

AST_MATCHER_P(ParenExpr, subExpr, ast_matchers::internal::Matcher<Expr>,
              InnerMatcher) {
  return InnerMatcher.matches(*Node.getSubExpr(), Finder, Builder);
}

AST_MATCHER(ParenExpr, isInMacro) {
  const Expr *E = Node.getSubExpr();
  return Node.getLParen().isMacroID() || Node.getRParen().isMacroID() ||
         E->getBeginLoc().isMacroID() || E->getEndLoc().isMacroID();
}

AST_MATCHER(ParenExpr, isWritten) {
  return Node.getLParen() != Node.getRParen();
}

AST_MATCHER(TypeLoc, isTypeOfExprTypeLoc) {
  return !Node.getUnqualifiedLoc().getAs<TypeOfExprTypeLoc>().isNull();
}

AST_MATCHER(AutoType, isDecltypeAuto) { return Node.isDecltypeAuto(); }

AST_MATCHER(MemberExpr, isQualified) { return Node.hasQualifier(); }

AST_MATCHER(MemberExpr, isImplicitAccess) { return Node.isImplicitAccess(); }

// Skips the implicit casts, copies, temporaries, and conversion calls that
// wrap an operand; they add no tokens, so they share the operand's range.
AST_MATCHER_P(Expr, hasParentIgnoringImplicit,
              ast_matchers::internal::Matcher<Expr>, InnerMatcher) {
  const SourceRange Range = Node.getSourceRange();
  const Expr *E = &Node;
  while (true) {
    const DynTypedNodeList Parents = Finder->getASTContext().getParents(*E);
    if (Parents.size() != 1)
      return false;
    E = Parents[0].get<Expr>();
    if (!E)
      return false;
    if (E->getSourceRange() != Range)
      return InnerMatcher.matches(*E, Finder, Builder);
  }
}

} // namespace

static FixItHint createSpacedRemoval(SourceLocation Loc,
                                     const SourceManager &SM,
                                     const LangOptions &LangOpts) {
  if (Loc.isValid() && !Loc.isMacroID()) {
    const auto LocInfo = SM.getDecomposedLoc(Loc);
    bool Invalid = false;
    StringRef Buffer = SM.getBufferData(LocInfo.first, &Invalid);
    if (!Invalid && LocInfo.second > 0 && LocInfo.second + 1 < Buffer.size() &&
        Lexer::isAsciiIdentifierContinueChar(Buffer[LocInfo.second - 1],
                                             LangOpts) &&
        Lexer::isAsciiIdentifierContinueChar(Buffer[LocInfo.second + 1],
                                             LangOpts))
      return FixItHint::CreateReplacement(SourceRange(Loc, Loc), " ");
  }
  return FixItHint::CreateRemoval(Loc);
}

RedundantParenthesesCheck::RedundantParenthesesCheck(StringRef Name,
                                                     ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      AllowedDecls(utils::options::parseStringList(
          Options.get("AllowedDecls", "std::max;std::min"))) {}

void RedundantParenthesesCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "AllowedDecls",
                utils::options::serializeStringList(AllowedDecls));
}

void RedundantParenthesesCheck::registerMatchers(MatchFinder *Finder) {
  const auto ConstantExpr =
      expr(anyOf(integerLiteral(), floatLiteral(), characterLiteral(),
                 cxxBoolLiteral(), stringLiteral(), cxxNullPtrLiteralExpr()));
  // Include dependent and substituted operands so templates and their
  // instantiations protect the same outer pair of parentheses.
  const auto DecltypeOperand =
      expr(anyOf(declRefExpr(), memberExpr(), dependentScopeDeclRefExpr(),
                 cxxDependentScopeMemberExpr(), unresolvedLookupExpr(),
                 substNonTypeTemplateParmExpr()));
  const auto DecltypeAutoType = ignoringParens(autoType(isDecltypeAuto()));
  const auto DecltypeAutoVar = varDecl(hasType(DecltypeAutoType));
  // Operands that create temporaries are wrapped in ExprWithCleanups.
  const auto InDecltypeAutoContext = anyOf(
      hasParent(DecltypeAutoVar),
      hasParent(
          expr(anyOf(initListExpr(), parenListExpr()),
               anyOf(hasParent(DecltypeAutoVar),
                     hasParent(exprWithCleanups(hasParent(DecltypeAutoVar)))))),
      hasParent(
          returnStmt(forCallable(functionDecl(returns(DecltypeAutoType))))));
  const auto IsDecltypeOperand = allOf(
      subExpr(ignoringParens(DecltypeOperand)),
      anyOf(hasParent(typeLoc(loc(decltypeType()))), InDecltypeAutoContext,
            hasParent(exprWithCleanups(InDecltypeAutoContext))));
  // &S::value is a pointer-to-member; T::value may instantiate to one.
  const auto MemberAddressOperand = expr(anyOf(
      memberExpr(isQualified(), isImplicitAccess()),
      dependentScopeDeclRefExpr(), declRefExpr(isInTemplateInstantiation())));
  const auto AddressOf = expr(
      anyOf(unaryOperator(hasOperatorName("&")),
            cxxOperatorCallExpr(hasOperatorName("&"), argumentCountIs(1))));
  const auto IsMemberAddressOperand =
      allOf(subExpr(ignoringParens(MemberAddressOperand)),
            hasParentIgnoringImplicit(AddressOf));
  Finder->addMatcher(
      parenExpr(subExpr(anyOf(
                    parenExpr(), ConstantExpr,
                    declRefExpr(to(namedDecl(unless(
                        matchers::matchesAnyListedRegexName(AllowedDecls))))),
                    memberExpr(),
                    callExpr(unless(cxxOperatorCallExpr(
                        unless(hasAnyOperatorName("()", "[]"))))),
                    arraySubscriptExpr())),
                isWritten(),
                unless(anyOf(isInMacro(),
                             // sizeof(...) is common used.
                             hasParent(unaryExprOrTypeTraitExpr()),
                             // typeof(...) parentheses are required syntax.
                             hasParent(typeLoc(isTypeOfExprTypeLoc())),
                             // decltype((x)) differs from decltype(x).
                             IsDecltypeOperand,
                             // &(S::value) differs from &S::value.
                             IsMemberAddressOperand)))
          .bind("dup"),
      this);
}

void RedundantParenthesesCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *PE = Result.Nodes.getNodeAs<ParenExpr>("dup");
  diag(PE->getBeginLoc(), "redundant parentheses around expression")
      << createSpacedRemoval(PE->getLParen(), *Result.SourceManager,
                             getLangOpts())
      << FixItHint::CreateRemoval(PE->getRParen());
}

} // namespace clang::tidy::readability
