//===--- AvoidConstParamsInDecls.cpp - clang-tidy--------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "AvoidConstParamsInDecls.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace readability {
namespace {

SourceRange getTypeRange(const ParmVarDecl &Param) {
  if (Param.getIdentifier() != nullptr)
    return SourceRange(Param.getLocStart(),
                       Param.getLocEnd().getLocWithOffset(-1));
  return Param.getSourceRange();
}

} // namespace


void AvoidConstParamsInDecls::registerMatchers(MatchFinder *Finder) {
  const auto ConstParamDecl =
      parmVarDecl(hasType(qualType(isConstQualified()))).bind("param");
  Finder->addMatcher(functionDecl(unless(isDefinition()),
                                  has(typeLoc(forEach(ConstParamDecl))))
                         .bind("func"),
                     this);
}

void AvoidConstParamsInDecls::check(const MatchFinder::MatchResult &Result) {
  const auto *Func = Result.Nodes.getNodeAs<FunctionDecl>("func");
  const auto *Param = Result.Nodes.getNodeAs<ParmVarDecl>("param");

  QualType Type = Param->getType();
  if (!Type.isLocalConstQualified())
    return;

  Type.removeLocalConst();

  auto Diag = diag(Param->getLocStart(),
                   "parameter %0 is const-qualified in the function "
                   "declaration; const-qualification of parameters only has an "
                   "effect in function definitions");
  if (Param->getName().empty()) {
    for (unsigned int i = 0; i < Func->getNumParams(); ++i) {
      if (Param == Func->getParamDecl(i)) {
        Diag << (i + 1);
        break;
      }
    }
  } else {
    Diag << Param;
  }
  Diag << FixItHint::CreateReplacement(getTypeRange(*Param),
                                       Type.getAsString());
}

} // namespace readability
} // namespace tidy
} // namespace clang
