//===--- AssertSideEffectCheck.cpp - clang-tidy ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AssertSideEffectCheck.h"
#include "../utils/Matchers.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <string>

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

namespace {

AST_MATCHER_P2(Expr, hasSideEffect, bool, CheckFunctionCalls,
               clang::ast_matchers::internal::Matcher<NamedDecl>,
               IgnoredFunctionsMatcher) {
  const Expr *E = &Node;

  if (const auto *Op = dyn_cast<UnaryOperator>(E)) {
    UnaryOperator::Opcode OC = Op->getOpcode();
    return OC == UO_PostInc || OC == UO_PostDec || OC == UO_PreInc ||
           OC == UO_PreDec;
  }

  if (const auto *Op = dyn_cast<BinaryOperator>(E)) {
    return Op->isAssignmentOp();
  }

  if (const auto *OpCallExpr = dyn_cast<CXXOperatorCallExpr>(E)) {
    OverloadedOperatorKind OpKind = OpCallExpr->getOperator();
    return OpKind == OO_Equal || OpKind == OO_PlusEqual ||
           OpKind == OO_MinusEqual || OpKind == OO_StarEqual ||
           OpKind == OO_SlashEqual || OpKind == OO_AmpEqual ||
           OpKind == OO_PipeEqual || OpKind == OO_CaretEqual ||
           OpKind == OO_LessLessEqual || OpKind == OO_GreaterGreaterEqual ||
           OpKind == OO_PlusPlus || OpKind == OO_MinusMinus ||
           OpKind == OO_PercentEqual || OpKind == OO_New ||
           OpKind == OO_Delete || OpKind == OO_Array_New ||
           OpKind == OO_Array_Delete;
  }

  if (const auto *CExpr = dyn_cast<CallExpr>(E)) {
    bool Result = CheckFunctionCalls;
    if (const auto *FuncDecl = CExpr->getDirectCallee()) {
      if (FuncDecl->getDeclName().isIdentifier() &&
          IgnoredFunctionsMatcher.matches(*FuncDecl, Finder,
                                          Builder)) // exceptions come here
        Result = false;
      else if (const auto *MethodDecl = dyn_cast<CXXMethodDecl>(FuncDecl))
        Result &= !MethodDecl->isConst();
    }
    return Result;
  }

  return isa<CXXNewExpr>(E) || isa<CXXDeleteExpr>(E) || isa<CXXThrowExpr>(E);
}

} // namespace

AssertSideEffectCheck::AssertSideEffectCheck(StringRef Name,
                                             ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      CheckFunctionCalls(Options.get("CheckFunctionCalls", false)),
      RawAssertList(Options.get("AssertMacros", "assert,NSAssert,NSCAssert")),
      IgnoredFunctions(utils::options::parseListPair(
          "__builtin_expect;", Options.get("IgnoredFunctions", ""))) {
  StringRef(RawAssertList).split(AssertMacros, ",", -1, false);
}

// The options are explained in AssertSideEffectCheck.h.
void AssertSideEffectCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "CheckFunctionCalls", CheckFunctionCalls);
  Options.store(Opts, "AssertMacros", RawAssertList);
  Options.store(Opts, "IgnoredFunctions",
                utils::options::serializeStringList(IgnoredFunctions));
}

void AssertSideEffectCheck::registerMatchers(MatchFinder *Finder) {
  auto IgnoredFunctionsMatcher =
      matchers::matchesAnyListedName(IgnoredFunctions);

  auto DescendantWithSideEffect =
      traverse(TK_AsIs, hasDescendant(expr(hasSideEffect(
                            CheckFunctionCalls, IgnoredFunctionsMatcher))));
  auto ConditionWithSideEffect = hasCondition(DescendantWithSideEffect);
  Finder->addMatcher(
      stmt(
          anyOf(conditionalOperator(ConditionWithSideEffect),
                ifStmt(ConditionWithSideEffect),
                unaryOperator(hasOperatorName("!"),
                              hasUnaryOperand(unaryOperator(
                                  hasOperatorName("!"),
                                  hasUnaryOperand(DescendantWithSideEffect))))))
          .bind("condStmt"),
      this);
}

void AssertSideEffectCheck::check(const MatchFinder::MatchResult &Result) {
  const SourceManager &SM = *Result.SourceManager;
  const LangOptions LangOpts = getLangOpts();
  SourceLocation Loc = Result.Nodes.getNodeAs<Stmt>("condStmt")->getBeginLoc();

  StringRef AssertMacroName;
  while (Loc.isValid() && Loc.isMacroID()) {
    StringRef MacroName = Lexer::getImmediateMacroName(Loc, SM, LangOpts);

    // Check if this macro is an assert.
    if (llvm::is_contained(AssertMacros, MacroName)) {
      AssertMacroName = MacroName;
      break;
    }
    Loc = SM.getImmediateMacroCallerLoc(Loc);
  }
  if (AssertMacroName.empty())
    return;

  diag(Loc, "side effect in %0() condition discarded in release builds")
      << AssertMacroName;
}

} // namespace clang::tidy::bugprone
