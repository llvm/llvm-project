//===--- ReturnBracedInitListCheck.cpp - clang-tidy------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ReturnBracedInitListCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/FixIt.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace modernize {

void ReturnBracedInitListCheck::registerMatchers(MatchFinder *Finder) {
  // Only register the matchers for C++.
  if (!getLangOpts().CPlusPlus11)
    return;

  // Skip list initialization and constructors with an initializer list.
  auto ConstructExpr =
      cxxConstructExpr(
          unless(anyOf(hasDeclaration(cxxConstructorDecl(isExplicit())),
                       isListInitialization(), hasDescendant(initListExpr()),
                       isInTemplateInstantiation())))
          .bind("ctor");

  auto CtorAsArgument = materializeTemporaryExpr(anyOf(
      has(ConstructExpr), has(cxxFunctionalCastExpr(has(ConstructExpr)))));

  Finder->addMatcher(
      functionDecl(isDefinition(), // Declarations don't have return statements.
                   returns(unless(anyOf(builtinType(), autoType()))),
                   hasDescendant(returnStmt(hasReturnValue(
                       has(cxxConstructExpr(has(CtorAsArgument)))))))
          .bind("fn"),
      this);
}

void ReturnBracedInitListCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedFunctionDecl = Result.Nodes.getNodeAs<FunctionDecl>("fn");
  const auto *MatchedConstructExpr =
      Result.Nodes.getNodeAs<CXXConstructExpr>("ctor");

  // Don't make replacements in macro.
  SourceLocation Loc = MatchedConstructExpr->getExprLoc();
  if (Loc.isMacroID())
    return;

  // Make sure that the return type matches the constructed type.
  const QualType ReturnType =
      MatchedFunctionDecl->getReturnType().getCanonicalType();
  const QualType ConstructType =
      MatchedConstructExpr->getType().getCanonicalType();
  if (ReturnType != ConstructType)
    return;

  auto Diag = diag(Loc, "avoid repeating the return type from the "
                        "declaration; use a braced initializer list instead");

  const SourceRange CallParensRange =
      MatchedConstructExpr->getParenOrBraceRange();

  // Make sure there is an explicit constructor call.
  if (CallParensRange.isInvalid())
    return;

  // Make sure that the ctor arguments match the declaration.
  for (unsigned I = 0, NumParams = MatchedConstructExpr->getNumArgs();
       I < NumParams; ++I) {
    if (const auto *VD = dyn_cast<VarDecl>(
            MatchedConstructExpr->getConstructor()->getParamDecl(I))) {
      if (MatchedConstructExpr->getArg(I)->getType().getCanonicalType() !=
          VD->getType().getCanonicalType())
        return;
    }
  }

  // Range for constructor name and opening brace.
  CharSourceRange CtorCallSourceRange = CharSourceRange::getTokenRange(
      Loc, CallParensRange.getBegin().getLocWithOffset(-1));

  Diag << FixItHint::CreateRemoval(CtorCallSourceRange)
       << FixItHint::CreateReplacement(CallParensRange.getBegin(), "{")
       << FixItHint::CreateReplacement(CallParensRange.getEnd(), "}");
}

} // namespace modernize
} // namespace tidy
} // namespace clang
