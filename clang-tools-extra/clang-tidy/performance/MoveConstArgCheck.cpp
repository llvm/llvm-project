//===--- MoveConstArgCheck.cpp - clang-tidy -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MoveConstArgCheck.h"

#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace performance {

static void ReplaceCallWithArg(const CallExpr *Call, DiagnosticBuilder &Diag,
                               const SourceManager &SM,
                               const LangOptions &LangOpts) {
  const Expr *Arg = Call->getArg(0);

  CharSourceRange BeforeArgumentsRange = Lexer::makeFileCharRange(
      CharSourceRange::getCharRange(Call->getBeginLoc(), Arg->getBeginLoc()),
      SM, LangOpts);
  CharSourceRange AfterArgumentsRange = Lexer::makeFileCharRange(
      CharSourceRange::getCharRange(Call->getEndLoc(),
                                    Call->getEndLoc().getLocWithOffset(1)),
      SM, LangOpts);

  if (BeforeArgumentsRange.isValid() && AfterArgumentsRange.isValid()) {
    Diag << FixItHint::CreateRemoval(BeforeArgumentsRange)
         << FixItHint::CreateRemoval(AfterArgumentsRange);
  }
}

void MoveConstArgCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "CheckTriviallyCopyableMove", CheckTriviallyCopyableMove);
}

void MoveConstArgCheck::registerMatchers(MatchFinder *Finder) {
  auto MoveCallMatcher =
      callExpr(callee(functionDecl(hasName("::std::move"))), argumentCountIs(1),
               unless(isInTemplateInstantiation()))
          .bind("call-move");

  Finder->addMatcher(MoveCallMatcher, this);

  auto ConstParamMatcher = forEachArgumentWithParam(
      MoveCallMatcher, parmVarDecl(hasType(references(isConstQualified()))));

  Finder->addMatcher(callExpr(ConstParamMatcher).bind("receiving-expr"), this);
  Finder->addMatcher(
      traverse(ast_type_traits::TK_AsIs,
               cxxConstructExpr(ConstParamMatcher).bind("receiving-expr")),
      this);
}

void MoveConstArgCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *CallMove = Result.Nodes.getNodeAs<CallExpr>("call-move");
  const auto *ReceivingExpr = Result.Nodes.getNodeAs<Expr>("receiving-expr");
  const Expr *Arg = CallMove->getArg(0);
  SourceManager &SM = Result.Context->getSourceManager();

  CharSourceRange MoveRange =
      CharSourceRange::getCharRange(CallMove->getSourceRange());
  CharSourceRange FileMoveRange =
      Lexer::makeFileCharRange(MoveRange, SM, getLangOpts());
  if (!FileMoveRange.isValid())
    return;

  bool IsConstArg = Arg->getType().isConstQualified();
  bool IsTriviallyCopyable =
      Arg->getType().isTriviallyCopyableType(*Result.Context);

  if (IsConstArg || IsTriviallyCopyable) {
    if (const CXXRecordDecl *R = Arg->getType()->getAsCXXRecordDecl()) {
      // According to [expr.prim.lambda]p3, "whether the closure type is
      // trivially copyable" property can be changed by the implementation of
      // the language, so we shouldn't rely on it when issuing diagnostics.
      if (R->isLambda())
        return;
      // Don't warn when the type is not copyable.
      for (const auto *Ctor : R->ctors()) {
        if (Ctor->isCopyConstructor() && Ctor->isDeleted())
          return;
      }
    }

    if (!IsConstArg && IsTriviallyCopyable && !CheckTriviallyCopyableMove)
      return;

    bool IsVariable = isa<DeclRefExpr>(Arg);
    const auto *Var =
        IsVariable ? dyn_cast<DeclRefExpr>(Arg)->getDecl() : nullptr;
    auto Diag = diag(FileMoveRange.getBegin(),
                     "std::move of the %select{|const }0"
                     "%select{expression|variable %4}1 "
                     "%select{|of the trivially-copyable type %5 }2"
                     "has no effect; remove std::move()"
                     "%select{| or make the variable non-const}3")
                << IsConstArg << IsVariable << IsTriviallyCopyable
                << (IsConstArg && IsVariable && !IsTriviallyCopyable) << Var
                << Arg->getType();

    ReplaceCallWithArg(CallMove, Diag, SM, getLangOpts());
  } else if (ReceivingExpr) {
    auto Diag = diag(FileMoveRange.getBegin(),
                     "passing result of std::move() as a const reference "
                     "argument; no move will actually happen");

    ReplaceCallWithArg(CallMove, Diag, SM, getLangOpts());
  }
}

} // namespace performance
} // namespace tidy
} // namespace clang
