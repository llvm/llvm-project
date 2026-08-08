//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NumericlimitmaxcheckCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

NumericlimitmaxcheckCheck::NumericlimitmaxcheckCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context), 
      Inserter(Options.getLocalOrGlobal("IncludeStyle", utils::IncludeSorter::IS_LLVM),
               areDiagsSelfContained()) {}

void NumericlimitmaxcheckCheck::registerPPCallbacks(const SourceManager &SM, Preprocessor *PP, Preprocessor *ModuleExpanderPP) {
  Inserter.registerPreprocessor(PP);
}

void NumericlimitmaxcheckCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IncludeStyle", Inserter.getStyle());
}

bool NumericlimitmaxcheckCheck::isLanguageVersionSupported(const LangOptions &LangOpts) const {
  return LangOpts.CPlusPlus;
}

void NumericlimitmaxcheckCheck::registerMatchers(MatchFinder *Finder) {
  if (!getLangOpts().CPlusPlus)
    return;

  auto NegOneLiteral = integerLiteral(equals(-1));
  auto ZeroLiteral = integerLiteral(equals(0));
  
  auto NegOneExpr = anyOf(
      NegOneLiteral,
      unaryOperator(hasOperatorName("-"),
                    hasUnaryOperand(integerLiteral(equals(1)))));

  auto BitNotZero = unaryOperator(hasOperatorName("~"),
                                  hasUnaryOperand(ZeroLiteral));

  // Match implicit cast of -1 to unsigned
  auto ImplicitNegOneToUnsigned =
      implicitCastExpr(
          hasSourceExpression(ignoringParenImpCasts(anyOf(NegOneExpr, BitNotZero))),
          hasType(isUnsignedInteger()));

  // Match explicit cast to unsigned of either -1 or ~0
  auto ExplicitCastOfNegOrBitnot =
      explicitCastExpr(
          hasDestinationType(isUnsignedInteger()),
          hasSourceExpression(ignoringParenImpCasts(anyOf(NegOneExpr, BitNotZero))));

  // Match ~0 with unsigned type
  auto UnsignedBitNotZero =
      unaryOperator(
          hasOperatorName("~"),
          hasUnaryOperand(ZeroLiteral),
          hasType(isUnsignedInteger()));

  auto UnsignedLiteralNegOne =
      integerLiteral(equals(-1), hasType(isUnsignedInteger()));

  // To handle ternary branch case
  auto TernaryBranch =
      expr(anyOf(NegOneExpr, BitNotZero),
           hasAncestor(conditionalOperator(
            hasAncestor(implicitCastExpr(hasType(isUnsignedInteger()
          ))
                                   .bind("outerCast")))))
          .bind("unsignedMaxExpr");

  auto Combined =
      expr(anyOf(
          ExplicitCastOfNegOrBitnot,
          ImplicitNegOneToUnsigned,
          UnsignedBitNotZero,
          UnsignedLiteralNegOne
      )).bind("unsignedMaxExpr");

  Finder->addMatcher(Combined, this);
  Finder->addMatcher(TernaryBranch, this);
}


void NumericlimitmaxcheckCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *E = Result.Nodes.getNodeAs<Expr>("unsignedMaxExpr");
  const auto *OuterCast = Result.Nodes.getNodeAs<CastExpr>("outerCast"); 

  if (!E)
    return;

  ASTContext &Ctx = *Result.Context; // Get context before first use

  QualType QT;
  if (OuterCast) {
    // This is ternary matcher. Get type from the cast.
    QT = OuterCast->getType();
  } else {
    // Get type from the bound expression.
    QT = E->getType();
  }

  if (E->getBeginLoc().isInvalid() || E->getBeginLoc().isMacroID())
    return;

  const SourceManager &SM = Ctx.getSourceManager();

  if (!OuterCast) {
    auto Parents = Ctx.getParents(*E);
    if (!Parents.empty()) {
      for (const auto &Parent : Parents) {
        // Check if parent is an explicit cast to unsigned
        if (const auto *ParentCast = Parent.get<ExplicitCastExpr>()) {
          if (ParentCast->getType()->isUnsignedIntegerType()) {
            // Skip this match, avoids double reporting
            return;
          }
        }
        // Also check if parent is an implicit cast that's part of an explicit cast chain
        if (const auto *ImplicitCast = Parent.get<ImplicitCastExpr>()) {
          auto GrandParents = Ctx.getParents(*ImplicitCast);
          for (const auto &GP : GrandParents) {
            if (const auto *GPCast = GP.get<ExplicitCastExpr>()) {
              if (GPCast->getType()->isUnsignedIntegerType()) {
                return;
              }
            }
          }
        }
      }
    }
  }

  if (QT.isNull() || !QT->isUnsignedIntegerType())
    return;

  std::string TypeStr = QT.getUnqualifiedType().getAsString();
  if (const auto *Typedef = QT->getAs<TypedefType>()) {
    TypeStr = Typedef->getDecl()->getName().str();
  }

  // Get original source text for diagnostic message
  StringRef OriginalText =
      Lexer::getSourceText(CharSourceRange::getTokenRange(E->getSourceRange()), 
                          SM, getLangOpts());

  // Suggestion to the user regarding the usage of unsigned ~0 or -1
  std::string Replacement = "std::numeric_limits<" + TypeStr + ">::max()";

  //gives warning message if unsigned ~0 or -1 are used instead of numeric_limit::max() 
  auto Diag = diag(E->getBeginLoc(),
                   "use 'std::numeric_limits<%0>::max()' instead of '%1'")
              << TypeStr << OriginalText;

  //suggest hint for fixing it
  Diag << FixItHint::CreateReplacement(E->getSourceRange(), Replacement);

  //includes the <limits> which contains the numeric_limits::max() 
  FileID FID = SM.getFileID(E->getBeginLoc());
  if (auto IncludeHint = Inserter.createIncludeInsertion(FID, "<limits>")) {
    Diag << *IncludeHint;
  }
}

} // namespace clang::tidy::readability
