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

namespace {

AST_MATCHER(Expr, isLValueExpr) { return Node.isLValue(); }

AST_MATCHER_P(ExplicitCastExpr, hasSourceExpressionAsWritten,
              ast_matchers::internal::Matcher<Expr>, InnerMatcher) {
  const Expr *Sub = Node.getSubExprAsWritten();
  return Sub != nullptr && InnerMatcher.matches(*Sub, Finder, Builder);
}

AST_MATCHER(ExplicitCastExpr, addsOnlyConst) {
  const auto *Ref = Node.getTypeAsWritten()->getAs<LValueReferenceType>();
  if (Ref == nullptr)
    return false;
  const Expr *Sub = Node.getSubExprAsWritten();
  return Sub != nullptr &&
         Finder->getASTContext().hasSameType(Ref->getPointeeType(),
                                             Sub->getType().withConst());
}

} // namespace

UseAsConstCheck::UseAsConstCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      Inserter(Options.getLocalOrGlobal("IncludeStyle",
                                        utils::IncludeSorter::IS_LLVM),
               areDiagsSelfContained()),
      IgnoreMacros(Options.getLocalOrGlobal("IgnoreMacros", true)) {}

void UseAsConstCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IncludeStyle", Inserter.getStyle());
  Options.store(Opts, "IgnoreMacros", IgnoreMacros);
}

void UseAsConstCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      cxxStaticCastExpr(
          unless(isTypeDependent()),
          hasDestinationType(qualType(hasCanonicalType(
              lValueReferenceType(pointee(qualType(isConstQualified())))))),
          hasSourceExpressionAsWritten(
              expr(unless(isTypeDependent()), isLValueExpr(),
                   unless(hasType(qualType(isConstQualified()))))
                  .bind("sub")),
          addsOnlyConst())
          .bind("cast"),
      this);
}

void UseAsConstCheck::registerPPCallbacks(const SourceManager &SM,
                                          Preprocessor *PP,
                                          Preprocessor *ModuleExpanderPP) {
  Inserter.registerPreprocessor(PP);
}

void UseAsConstCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Cast = Result.Nodes.getNodeAs<CXXStaticCastExpr>("cast");
  const auto *Sub = Result.Nodes.getNodeAs<Expr>("sub");

  const bool InMacro =
      Cast->getBeginLoc().isMacroID() || Cast->getEndLoc().isMacroID();
  if (InMacro && IgnoreMacros)
    return;

  auto Diag =
      diag(Cast->getBeginLoc(),
           "use 'std::as_const' instead of 'static_cast' to add 'const'");

  if (InMacro)
    return;

  const SourceManager &SM = *Result.SourceManager;
  StringRef SubText = Lexer::getSourceText(
      CharSourceRange::getTokenRange(Sub->getSourceRange()), SM, getLangOpts());
  if (SubText.empty())
    return;

  Diag << FixItHint::CreateReplacement(Cast->getSourceRange(),
                                       ("std::as_const(" + SubText + ")").str())
       << Inserter.createIncludeInsertion(SM.getFileID(Cast->getBeginLoc()),
                                          "<utility>");
}

} // namespace clang::tidy::modernize
