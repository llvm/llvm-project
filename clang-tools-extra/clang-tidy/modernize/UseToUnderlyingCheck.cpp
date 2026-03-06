//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseToUnderlyingCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

UseToUnderlyingCheck::UseToUnderlyingCheck(StringRef Name,
                                           ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      Inserter(Options.getLocalOrGlobal("IncludeStyle",
                                        utils::IncludeSorter::IS_LLVM),
               areDiagsSelfContained()) {}
//
void UseToUnderlyingCheck::registerPPCallbacks(const SourceManager &SM,
                                               Preprocessor *PP,
                                               Preprocessor *ModuleExpanderPP) {
  Inserter.registerPreprocessor(PP);
}

void UseToUnderlyingCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IncludeStyle", Inserter.getStyle());
}

bool UseToUnderlyingCheck::isLanguageVersionSupported(
    const LangOptions &LangOpts) const {
  return LangOpts.CPlusPlus23;
}

void UseToUnderlyingCheck::registerMatchers(MatchFinder *Finder) {
  // FIXME: Add matchers.
  Finder->addMatcher(
      cxxStaticCastExpr( // C++ cast
          hasDestinationType(
              isInteger()),    // casting to any type of integer (int,long,etc)
          hasSourceExpression( // is an enum class
              expr(hasType(enumType(hasDeclaration(enumDecl(isScoped())))))
                  .bind("enumExpr"))) // giving the name enumExpr
          .bind("castExpr"),          // giving the name castExpr
      this);
}

void UseToUnderlyingCheck::check(const MatchFinder::MatchResult &Result) {
  // Acquiring the enumExpr and castExpr using getNodeAS
  const auto *Enum = Result.Nodes.getNodeAs<Expr>("enumExpr");
  const auto *Cast = Result.Nodes.getNodeAs<CXXStaticCastExpr>("castExpr");

  // getting contents of that node using getsourcetext
  StringRef EnumExprText = Lexer::getSourceText(
      CharSourceRange::getTokenRange(Enum->getSourceRange()),
      *Result.SourceManager, getLangOpts());
  // Suggestion to the user regarding the cast expr
  std::string Replacement = ("std::to_underlying(" + EnumExprText + ")").str();
  // gives and warning message if static cast is used instead to_underlying
  auto Diag = diag(
      Cast->getBeginLoc(),
      "use 'std::to_underlying' instead of 'static_cast' for 'enum class'");
  // suggest and hint for fixing it.
  Diag << FixItHint::CreateReplacement(Cast->getSourceRange(), Replacement);

  Diag << Inserter.createIncludeInsertion(
      Result.Context->getSourceManager().getFileID(Cast->getBeginLoc()),
      "<utility>");
}

} // namespace clang::tidy::modernize
