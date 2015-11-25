//===--- MoveConstandArgumentCheck.cpp - clang-tidy -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MoveConstantArgumentCheck.h"

#include <clang/Lex/Lexer.h>

namespace clang {
namespace tidy {
namespace misc {

using namespace ast_matchers;

void MoveConstantArgumentCheck::registerMatchers(MatchFinder *Finder) {
  if (!getLangOpts().CPlusPlus)
    return;
  Finder->addMatcher(callExpr(unless(isInTemplateInstantiation()),
                              callee(functionDecl(hasName("::std::move"))))
                         .bind("call-move"),
                     this);
}

void MoveConstantArgumentCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *CallMove = Result.Nodes.getNodeAs<CallExpr>("call-move");
  if (CallMove->getNumArgs() != 1)
    return;
  const Expr *Arg = CallMove->getArg(0);
  SourceManager &SM = Result.Context->getSourceManager();

  bool IsConstArg = Arg->getType().isConstQualified();
  bool IsTriviallyCopyable =
      Arg->getType().isTriviallyCopyableType(*Result.Context);

  if (IsConstArg || IsTriviallyCopyable) {
    auto MoveRange = CharSourceRange::getCharRange(CallMove->getSourceRange());
    auto FileMoveRange = Lexer::makeFileCharRange(MoveRange, SM, getLangOpts());
    if (!FileMoveRange.isValid())
      return;
    bool IsVariable = isa<DeclRefExpr>(Arg);
    auto Diag =
        diag(FileMoveRange.getBegin(), "std::move of the %select{|const }0"
                                       "%select{expression|variable}1 "
                                       "%select{|of trivially-copyable type }2"
                                       "has no effect; remove std::move()")
        << IsConstArg << IsVariable << IsTriviallyCopyable;

    auto BeforeArgumentsRange = Lexer::makeFileCharRange(
        CharSourceRange::getCharRange(CallMove->getLocStart(),
                                      Arg->getLocStart()),
        SM, getLangOpts());
    auto AfterArgumentsRange = Lexer::makeFileCharRange(
        CharSourceRange::getCharRange(
            CallMove->getLocEnd(), CallMove->getLocEnd().getLocWithOffset(1)),
        SM, getLangOpts());

    if (BeforeArgumentsRange.isValid() && AfterArgumentsRange.isValid()) {
      Diag << FixItHint::CreateRemoval(BeforeArgumentsRange)
           << FixItHint::CreateRemoval(AfterArgumentsRange);
    }
  }
}

} // namespace misc
} // namespace tidy
} // namespace clang
