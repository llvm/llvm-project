//===--- MoveConstantArgumentCheck.cpp - clang-tidy -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MoveConstantArgumentCheck.h"

#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace misc {

static void ReplaceCallWithArg(const CallExpr *Call, DiagnosticBuilder &Diag,
                               const SourceManager &SM,
                               const LangOptions &LangOpts) {
  const Expr *Arg = Call->getArg(0);

  CharSourceRange BeforeArgumentsRange = Lexer::makeFileCharRange(
      CharSourceRange::getCharRange(Call->getLocStart(), Arg->getLocStart()),
      SM, LangOpts);
  CharSourceRange AfterArgumentsRange = Lexer::makeFileCharRange(
      CharSourceRange::getCharRange(Call->getLocEnd(),
                                    Call->getLocEnd().getLocWithOffset(1)),
      SM, LangOpts);

  if (BeforeArgumentsRange.isValid() && AfterArgumentsRange.isValid()) {
    Diag << FixItHint::CreateRemoval(BeforeArgumentsRange)
         << FixItHint::CreateRemoval(AfterArgumentsRange);
  }
}

void MoveConstantArgumentCheck::registerMatchers(MatchFinder *Finder) {
  if (!getLangOpts().CPlusPlus)
    return;

  auto MoveCallMatcher =
      callExpr(callee(functionDecl(hasName("::std::move"))), argumentCountIs(1),
               unless(isInTemplateInstantiation()))
          .bind("call-move");

  Finder->addMatcher(MoveCallMatcher, this);

  auto ConstParamMatcher = forEachArgumentWithParam(
      MoveCallMatcher, parmVarDecl(hasType(references(isConstQualified()))));

  Finder->addMatcher(callExpr(ConstParamMatcher).bind("receiving-expr"), this);
  Finder->addMatcher(cxxConstructExpr(ConstParamMatcher).bind("receiving-expr"),
                     this);
}

void MoveConstantArgumentCheck::check(const MatchFinder::MatchResult &Result) {
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
                << (IsConstArg && IsVariable && !IsTriviallyCopyable)
                << Var << Arg->getType();

    ReplaceCallWithArg(CallMove, Diag, SM, getLangOpts());
  } else if (ReceivingExpr) {
    auto Diag = diag(FileMoveRange.getBegin(),
                     "passing result of std::move() as a const reference "
                     "argument; no move will actually happen");

    ReplaceCallWithArg(CallMove, Diag, SM, getLangOpts());
  }
}

} // namespace misc
} // namespace tidy
} // namespace clang
