//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseAsConstCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

UseAsConstCheck::UseAsConstCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      Inserter(Options.getLocalOrGlobal("IncludeStyle",
                                        utils::IncludeSorter::IS_LLVM),
               areDiagsSelfContained()) {}

void UseAsConstCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IncludeStyle", Inserter.getStyle());
}

void UseAsConstCheck::registerMatchers(MatchFinder *Finder) {
  // Match a static_cast to a const-qualified type; check() narrows it down.
  Finder->addMatcher(
      cxxStaticCastExpr(hasType(qualType(isConstQualified()))).bind("cast"),
      this);
}

void UseAsConstCheck::registerPPCallbacks(const SourceManager &SM,
                                          Preprocessor *PP,
                                          Preprocessor *ModuleExpanderPP) {
  Inserter.registerPreprocessor(PP);
}

void UseAsConstCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Cast = Result.Nodes.getNodeAs<CXXStaticCastExpr>("cast");
  // A cast written in a macro cannot be safely rewritten.
  if (Cast->getBeginLoc().isMacroID() || Cast->getEndLoc().isMacroID())
    return;

  // Only handle a cast written as 'const T &'.
  const auto *RefType = Cast->getTypeAsWritten()->getAs<LValueReferenceType>();
  if (!RefType)
    return;
  QualType Pointee = RefType->getPointeeType();
  if (!Pointee.isConstQualified())
    return;

  const Expr *Sub = Cast->getSubExprAsWritten();
  // In a template 'const T &' may collapse to a reference that adds no const,
  // where std::as_const is not equivalent.
  if (Sub->isTypeDependent() || Pointee->isDependentType())
    return;
  if (!Sub->isLValue())
    return;

  // Rewrite only when const is the sole difference; adding const to the
  // sub-expression type keeps any volatile.
  const ASTContext &Ctx = *Result.Context;
  QualType SubType = Sub->getType();
  if (SubType.isConstQualified() ||
      !Ctx.hasSameType(Pointee, SubType.withConst()))
    return;

  const SourceManager &SM = *Result.SourceManager;
  StringRef SubText = Lexer::getSourceText(
      CharSourceRange::getTokenRange(Sub->getSourceRange()), SM, getLangOpts());
  if (SubText.empty())
    return;

  diag(Cast->getBeginLoc(),
       "use 'std::as_const' instead of 'static_cast' to add 'const'")
      << FixItHint::CreateReplacement(Cast->getSourceRange(),
                                      ("std::as_const(" + SubText + ")").str())
      << Inserter.createIncludeInsertion(SM.getFileID(Cast->getBeginLoc()),
                                         "<utility>");
}

} // namespace clang::tidy::modernize
