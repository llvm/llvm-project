//===--- AvoidConstParamsInDecls.cpp - clang-tidy--------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AvoidConstParamsInDecls.h"
#include "../utils/LexerUtils.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace readability {
namespace {

SourceRange getTypeRange(const ParmVarDecl &Param) {
  return SourceRange(Param.getBeginLoc(),
                     Param.getLocation().getLocWithOffset(-1));
}

} // namespace

void AvoidConstParamsInDecls::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IgnoreMacros", IgnoreMacros);
}

void AvoidConstParamsInDecls::registerMatchers(MatchFinder *Finder) {
  const auto ConstParamDecl =
      parmVarDecl(hasType(qualType(isConstQualified()))).bind("param");
  Finder->addMatcher(
      functionDecl(unless(isDefinition()),
                   has(typeLoc(forEach(ConstParamDecl))))
          .bind("func"),
      this);
}

void AvoidConstParamsInDecls::check(const MatchFinder::MatchResult &Result) {
  const auto *Func = Result.Nodes.getNodeAs<FunctionDecl>("func");
  const auto *Param = Result.Nodes.getNodeAs<ParmVarDecl>("param");

  if (!Param->getType().isLocalConstQualified())
    return;

  if (IgnoreMacros &&
      (Param->getBeginLoc().isMacroID() || Param->getEndLoc().isMacroID())) {
    // Suppress the check if macros are involved.
    return;
  }

  auto Diag = diag(Param->getBeginLoc(),
                   "parameter %0 is const-qualified in the function "
                   "declaration; const-qualification of parameters only has an "
                   "effect in function definitions");
  if (Param->getName().empty()) {
    for (unsigned int I = 0; I < Func->getNumParams(); ++I) {
      if (Param == Func->getParamDecl(I)) {
        Diag << (I + 1);
        break;
      }
    }
  } else {
    Diag << Param;
  }

  if (Param->getBeginLoc().isMacroID() != Param->getEndLoc().isMacroID()) {
    // Do not offer a suggestion if the part of the variable declaration comes
    // from a macro.
    return;
  }

  CharSourceRange FileRange = Lexer::makeFileCharRange(
      CharSourceRange::getTokenRange(getTypeRange(*Param)),
      *Result.SourceManager, getLangOpts());

  if (!FileRange.isValid())
    return;

  auto Tok = tidy::utils::lexer::getQualifyingToken(
      tok::kw_const, FileRange, *Result.Context, *Result.SourceManager);
  if (!Tok)
    return;
  Diag << FixItHint::CreateRemoval(
      CharSourceRange::getTokenRange(Tok->getLocation(), Tok->getLocation()));
}

} // namespace readability
} // namespace tidy
} // namespace clang
