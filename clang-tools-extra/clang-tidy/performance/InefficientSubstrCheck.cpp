//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InefficientSubstrCheck.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ExprCXX.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include <optional>
#include <string>

using namespace clang::ast_matchers;

namespace clang::tidy::performance {

InefficientSubstrCheck::InefficientSubstrCheck(StringRef Name,
                                               ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      StringLikeClasses(utils::options::parseStringList(
          Options.get("StringLikeClasses", "::std::basic_string"))) {}

void InefficientSubstrCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "StringLikeClasses",
                utils::options::serializeStringList(StringLikeClasses));
}

void InefficientSubstrCheck::registerMatchers(MatchFinder *Finder) {
  const auto LhsRef =
      ignoringParens(declRefExpr(to(varDecl().bind("lhs-var"))).bind("lhs"));

  const auto SubstrCall =
      cxxMemberCallExpr(
          callee(cxxMethodDecl(hasName("substr"),
                               ofClass(hasAnyName(StringLikeClasses)))),
          on(ignoringParens(
              declRefExpr(to(varDecl().bind("src-var"))).bind("src"))))
          .bind("substr");

  // Match: lhs = src.substr(...) and lhs += src.substr(...)
  Finder->addMatcher(cxxOperatorCallExpr(hasAnyOperatorName("=", "+="),
                                         hasArgument(0, LhsRef),
                                         hasArgument(1, SubstrCall))
                         .bind("op"),
                     this);
}

void InefficientSubstrCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Op = Result.Nodes.getNodeAs<CXXOperatorCallExpr>("op");
  const auto *LHS = Result.Nodes.getNodeAs<DeclRefExpr>("lhs");
  const auto *Src = Result.Nodes.getNodeAs<DeclRefExpr>("src");
  const auto *LHSVar = Result.Nodes.getNodeAs<VarDecl>("lhs-var");
  const auto *SrcVar = Result.Nodes.getNodeAs<VarDecl>("src-var");
  const auto *SubstrExpr = Result.Nodes.getNodeAs<CXXMemberCallExpr>("substr");
  const SourceManager &SM = *Result.SourceManager;
  const LangOptions &LangOpts = Result.Context->getLangOpts();

  const bool IsAppend = Op->getOperator() == OO_PlusEqual;
  const bool SameVar = declaresSameEntity(LHSVar, SrcVar);

  // s = s.substr(...) is excluded: the 'assign' rewrite would self-alias,
  // and the strictly better rewrite is an in-place 'erase' of the prefix.
  if (!IsAppend && SameVar)
    return;

  // The (str, pos, count) overload must belong to the same basic_string
  // specialization as the destination so that it applies without
  // conversions.
  if (!ASTContext::hasSameUnqualifiedType(LHS->getType(), Src->getType()))
    return;

  // Count only explicitly-written arguments (exclude CXXDefaultArgExpr).
  // s += t.substr() is just s += t in disguise; leave it alone.
  SmallVector<const Expr *, 2> ExplicitArgs;
  for (const Expr *Arg : SubstrExpr->arguments())
    if (!isa<CXXDefaultArgExpr>(Arg))
      ExplicitArgs.push_back(Arg);
  if (ExplicitArgs.empty())
    return;

  // substr(pos, count) and assign/append(str, pos, count) throw and clamp
  // identically, so the arguments pass through verbatim. Self-appends
  // (s += s.substr(...)) are not rewritten because the replacement would
  // introduce a self-aliasing call, and macro expansions are not rewritten
  // because editing them is unsafe; both still get the warning.
  std::optional<std::string> Replacement;
  if (!SameVar && !Op->getBeginLoc().isMacroID() &&
      !Op->getEndLoc().isMacroID()) {
    const auto GetText = [&](SourceRange R) {
      return Lexer::getSourceText(CharSourceRange::getTokenRange(R), SM,
                                  LangOpts);
    };
    StringRef LHSText = GetText(LHS->getSourceRange());
    StringRef SrcText = GetText(Src->getSourceRange());
    bool Valid = !LHSText.empty() && !SrcText.empty();
    std::string Args;
    for (const Expr *Arg : ExplicitArgs) {
      StringRef ArgText = GetText(Arg->getSourceRange());
      Valid = Valid && !ArgText.empty();
      Args += ", ";
      Args += ArgText;
    }
    if (Valid)
      Replacement = (LHSText + (IsAppend ? ".append(" : ".assign(") + SrcText +
                     Args + ")")
                        .str();
  }

  const auto Diag = diag(Op->getOperatorLoc(),
                         "inefficient %select{assignment|concatenation}0 via "
                         "'substr' temporary; use '%select{assign|append}0' to "
                         "avoid the temporary string")
                    << (IsAppend ? 1 : 0);
  if (Replacement)
    Diag << FixItHint::CreateReplacement(Op->getSourceRange(), *Replacement);
}

} // namespace clang::tidy::performance
