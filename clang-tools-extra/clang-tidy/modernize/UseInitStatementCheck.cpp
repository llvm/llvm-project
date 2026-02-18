//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseInitStatementCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

void UseInitStatementCheck::registerMatchers(MatchFinder *Finder) {
  // Match if/switch statements that:
  //   - don't already have an init-statement
  //   - are inside a compound statement
  //   - are not in template instantiations
  Finder->addMatcher(
      compoundStmt(forEach(ifStmt(unless(hasInitStatement(anything())),
                                  unless(isInTemplateInstantiation()))
                               .bind("if")))
          .bind("compound"),
      this);

  Finder->addMatcher(
      compoundStmt(forEach(switchStmt(unless(hasInitStatement(anything())),
                                      unless(isInTemplateInstantiation()))
                               .bind("switch")))
          .bind("compound"),
      this);
}

/// Check if the variable declared in \p DS is referenced anywhere in the
/// subtree rooted at \p S.
static bool isVarReferencedIn(const VarDecl *VD, const Stmt *S,
                              ASTContext &Ctx) {
  return !match(findAll(declRefExpr(to(varDecl(equalsNode(VD))))), *S, Ctx)
              .empty();
}

void UseInitStatementCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Compound = Result.Nodes.getNodeAs<CompoundStmt>("compound");
  const Stmt *ConditionalStmt = Result.Nodes.getNodeAs<IfStmt>("if");
  if (!ConditionalStmt)
    ConditionalStmt = Result.Nodes.getNodeAs<SwitchStmt>("switch");
  if (!ConditionalStmt || !Compound)
    return;

  // Find the position of the if/switch in the compound statement.
  const auto *PrevDeclStmt = static_cast<const DeclStmt *>(nullptr);
  for (auto It = Compound->body_begin(), End = Compound->body_end(); It != End;
       ++It) {
    if (*It == ConditionalStmt) {
      // Check if the previous statement is a DeclStmt.
      if (It == Compound->body_begin())
        return;
      PrevDeclStmt = dyn_cast<DeclStmt>(*(It - 1));
      break;
    }
  }

  if (!PrevDeclStmt)
    return;

  // Only handle single declarations.
  if (!PrevDeclStmt->isSingleDecl())
    return;

  const auto *VD = dyn_cast<VarDecl>(PrevDeclStmt->getSingleDecl());
  if (!VD)
    return;

  // Variable must have an initializer.
  if (!VD->hasInit())
    return;

  // Skip if the variable is static or extern.
  if (VD->hasGlobalStorage())
    return;

  // The variable must be referenced in the condition of the if/switch.
  const Expr *Condition = nullptr;
  if (const auto *If = dyn_cast<IfStmt>(ConditionalStmt))
    Condition = If->getCond();
  else if (const auto *Switch = dyn_cast<SwitchStmt>(ConditionalStmt))
    Condition = Switch->getCond();

  if (!Condition || !isVarReferencedIn(VD, Condition, *Result.Context))
    return;

  // The variable must NOT be referenced after the if/switch statement.
  bool FoundConditional = false;
  for (const Stmt *Child : Compound->body()) {
    if (Child == ConditionalStmt) {
      FoundConditional = true;
      continue;
    }
    if (FoundConditional && isVarReferencedIn(VD, Child, *Result.Context))
      return;
  }

  // Skip if the DeclStmt or ConditionalStmt are in macros.
  if (PrevDeclStmt->getBeginLoc().isMacroID() ||
      ConditionalStmt->getBeginLoc().isMacroID())
    return;

  const bool IsIf = isa<IfStmt>(ConditionalStmt);
  const StringRef StmtKind = IsIf ? "if" : "switch";

  const SourceLocation DeclStart = PrevDeclStmt->getBeginLoc();

  // Get source text of the declaration (up to the VarDecl's end, not
  // the DeclStmt's semicolon).
  const SourceLocation VarDeclEnd = VD->getEndLoc();
  const StringRef DeclText = Lexer::getSourceText(
      CharSourceRange::getTokenRange(DeclStart, VarDeclEnd),
      *Result.SourceManager, Result.Context->getLangOpts());

  // Get the location right after the opening paren/keyword of if/switch.
  SourceLocation CondStart;
  if (IsIf) {
    const auto *If = cast<IfStmt>(ConditionalStmt);
    // The LParenLoc gives us the '(' position.
    CondStart = If->getLParenLoc().getLocWithOffset(1);
  } else {
    const auto *Switch = cast<SwitchStmt>(ConditionalStmt);
    CondStart = Switch->getLParenLoc().getLocWithOffset(1);
  }

  if (!CondStart.isValid())
    return;

  auto Diag = diag(PrevDeclStmt->getBeginLoc(),
                   "variable %0 can be declared in the '%1' init-statement")
              << VD << StmtKind;

  // Fix 1: Remove the declaration statement.
  // Remove everything from DeclStart to just before the if/switch keyword,
  // i.e., "int result = compute();\n  " -> leaves the if/switch in place.
  Diag << FixItHint::CreateRemoval(
      CharSourceRange::getCharRange(DeclStart, ConditionalStmt->getBeginLoc()));

  // Fix 2: Insert "decl; " at the start of the condition.
  const std::string InitStmt = (DeclText + "; ").str();
  Diag << FixItHint::CreateInsertion(CondStart, InitStmt);
}

} // namespace clang::tidy::modernize
