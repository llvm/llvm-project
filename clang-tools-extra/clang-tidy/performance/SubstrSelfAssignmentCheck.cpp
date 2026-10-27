//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SubstrSelfAssignmentCheck.h"
#include "../utils/Matchers.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ExprCXX.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/FixIt.h"
#include "llvm/ADT/STLExtras.h"
#include <optional>
#include <string>

using namespace clang::ast_matchers;

namespace clang::tidy::performance {

SubstrSelfAssignmentCheck::SubstrSelfAssignmentCheck(StringRef Name,
                                                     ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      StringLikeClasses(utils::options::parseStringList(
          Options.get("StringLikeClasses", "::std::basic_string"))) {}

void SubstrSelfAssignmentCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "StringLikeClasses",
                utils::options::serializeStringList(StringLikeClasses));
}

void SubstrSelfAssignmentCheck::registerMatchers(MatchFinder *Finder) {
  const auto VarRef = declRefExpr(to(varDecl().bind("var"))).bind("lhs");

  const auto StringClass =
      cxxRecordDecl(hasAnyName(StringLikeClasses)).bind("string-class");

  // The static member 'npos' of the matched string class. A bare name
  // comparison is not enough: a local variable or parameter that happens to
  // be called 'npos' carries a real count, which has no single-'erase'
  // rewrite.
  const auto NposDecl =
      varDecl(hasName("npos"),
              hasDeclContext(cxxRecordDecl(equalsBoundNode("string-class"))));

  const auto Npos = expr(ignoringParenImpCasts(
      mapAnyOf(declRefExpr, memberExpr).with(hasDeclaration(NposDecl))));

  const auto SubstrCall =
      cxxMemberCallExpr(
          callee(cxxMethodDecl(hasName("substr"), ofClass(StringClass))),
          on(declRefExpr(to(varDecl(equalsBoundNode("var"))))),
          optionally(hasArgument(1, Npos.bind("npos-count"))))
          .bind("substr");

  // Match: s = s.substr(...), except in unevaluated contexts such as
  // decltype or sizeof, where no temporary is ever materialized.
  Finder->addMatcher(
      cxxOperatorCallExpr(
          hasOperatorName("="), hasArgument(0, VarRef),
          hasArgument(1, SubstrCall),
          unless(anyOf(hasAncestor(typeLoc()),
                       hasAncestor(expr(matchers::hasUnevaluatedContext())))))
          .bind("assign"),
      this);
}

void SubstrSelfAssignmentCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *AssignExpr =
      Result.Nodes.getNodeAs<CXXOperatorCallExpr>("assign");
  const auto *LHS = Result.Nodes.getNodeAs<DeclRefExpr>("lhs");
  const auto *SubstrExpr = Result.Nodes.getNodeAs<CXXMemberCallExpr>("substr");

  // Count only explicitly-written arguments (exclude CXXDefaultArgExpr).
  const unsigned NumExplicitArgs =
      llvm::count_if(SubstrExpr->arguments(), [](const Expr *Arg) {
        return !isa<CXXDefaultArgExpr>(Arg);
      });

  // s = s.substr(pos) and s = s.substr(pos, npos) strip a prefix and can be
  // rewritten as s.erase(0, pos). The truncation form s = s.substr(0, count)
  // is diagnosed without a fix-it: s.erase(count) throws std::out_of_range
  // for count > s.size() where substr merely clamps. The general
  // two-argument form has no single-'erase' equivalent and is ignored.
  // TODO: Also emit a fix-it for the truncation form when
  // `count <= size()` can be proven.
  // TODO(C++23): P2438R2 makes 's = std::move(s).substr(pos, count)' an
  // exact, allocation-free rewrite for all forms; emit it as the fix-it in
  // C++23 mode.
  const bool IsPrefixStrip =
      NumExplicitArgs == 1 ||
      (NumExplicitArgs == 2 &&
       Result.Nodes.getNodeAs<Expr>("npos-count") != nullptr);
  bool IsTruncation = false;
  if (!IsPrefixStrip && NumExplicitArgs == 2) {
    const auto *PosLiteral =
        dyn_cast<IntegerLiteral>(SubstrExpr->getArg(0)->IgnoreParenImpCasts());
    IsTruncation = PosLiteral && PosLiteral->getValue() == 0;
  }
  if (!IsPrefixStrip && !IsTruncation)
    return;

  // Rewriting a macro expansion is unsafe; emit the warning without a fix-it.
  // TODO: Only emit this fix-it when `pos <= size()` can be proven.
  std::optional<std::string> Replacement;
  if (IsPrefixStrip && !AssignExpr->getBeginLoc().isMacroID() &&
      !AssignExpr->getEndLoc().isMacroID()) {
    StringRef VarName = tooling::fixit::getText(*LHS, *Result.Context);
    StringRef PosText =
        tooling::fixit::getText(*SubstrExpr->getArg(0), *Result.Context);
    if (!VarName.empty() && !PosText.empty())
      Replacement = (VarName + ".erase(0, " + PosText + ")").str();
  }

  const auto Diag =
      diag(AssignExpr->getOperatorLoc(),
           "inefficient self-assignment via 'substr'; use 'erase' to "
           "modify the string in-place");
  if (Replacement)
    Diag << FixItHint::CreateReplacement(AssignExpr->getSourceRange(),
                                         *Replacement);
}

} // namespace clang::tidy::performance
