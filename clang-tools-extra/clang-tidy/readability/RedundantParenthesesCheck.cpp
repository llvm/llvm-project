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
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprConcepts.h"
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

} // namespace

// Returns true if `E` is an operand of a requires-clause or of a concept
// definition.
static bool isConstraintOperand(const Expr *E, ASTContext &Ctx) {
  for (const DynTypedNode &Parent : Ctx.getParents(*E)) {
    const auto *ParentExpr = Parent.get<Expr>();
    const auto *BO = dyn_cast_or_null<BinaryOperator>(ParentExpr);
    if (ParentExpr &&
        (ParentExpr->IgnoreImplicit() == E || (BO && BO->isLogicalOp()))) {
      if (isConstraintOperand(ParentExpr, Ctx))
        return true;
      continue;
    }
    // The 'requires (E);' of a nested requirement. Other expressions inside a
    // requires-expression are unrestricted.
    if (const auto *RE = Parent.get<RequiresExpr>()) {
      if (llvm::any_of(
              RE->getRequirements(), [E](const concepts::Requirement *R) {
                const auto *NR = dyn_cast<concepts::NestedRequirement>(R);
                return NR && !NR->hasInvalidConstraint() &&
                       NR->getConstraintExpr() == E;
              }))
        return true;
      continue;
    }
    const auto *D = Parent.get<Decl>();
    if (!D)
      continue;
    // The requires-clause of a template parameter list.
    const auto *TD = dyn_cast<TemplateDecl>(D);
    const TemplateParameterList *TPL =
        TD ? TD->getTemplateParameters() : D->getDescribedTemplateParams();
    if (TPL && TPL->getRequiresClause() == E)
      return true;
    // The right hand side of a concept definition.
    if (const auto *CD = dyn_cast<ConceptDecl>(D);
        CD && CD->getConstraintExpr() == E)
      return true;
    // A trailing requires-clause.
    if (const auto *FD = dyn_cast<FunctionDecl>(D);
        FD && FD->getTrailingRequiresClause().ConstraintExpr == E)
      return true;
  }
  return false;
}

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
  const auto ConstraintSensitiveExpr =
      expr(anyOf(parenExpr(), memberExpr(),
                 callExpr(unless(cxxOperatorCallExpr(
                     unless(hasAnyOperatorName("()", "[]"))))),
                 arraySubscriptExpr()));
  Finder->addMatcher(
      parenExpr(subExpr(anyOf(
                    ConstraintSensitiveExpr.bind("constraint_sensitive"),
                    ConstantExpr,
                    declRefExpr(to(namedDecl(unless(
                        matchers::matchesAnyListedRegexName(AllowedDecls))))))),
                unless(anyOf(isInMacro(),
                             // sizeof(...) is common used.
                             hasParent(unaryExprOrTypeTraitExpr()))))
          .bind("dup"),
      this);
}

void RedundantParenthesesCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *PE = Result.Nodes.getNodeAs<ParenExpr>("dup");
  if (Result.Nodes.getNodeAs<Expr>("constraint_sensitive") &&
      isConstraintOperand(PE, *Result.Context))
    return;
  diag(PE->getBeginLoc(), "redundant parentheses around expression")
      << createSpacedRemoval(PE->getLParen(), *Result.SourceManager,
                             getLangOpts())
      << FixItHint::CreateRemoval(PE->getRParen());
}

} // namespace clang::tidy::readability
