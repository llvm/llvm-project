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

 // ... inside registerMatchers() ...

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

  // *** ADD THIS NEW MATCHER ***
  // Matches -1 or ~0 when they are a branch of a ternary operator
  // that is itself being implicitly cast to unsigned.
  auto TernaryBranch =
      expr(anyOf(NegOneExpr, BitNotZero),
           hasAncestor( // <-- Use hasAncestor to look up the tree
               conditionalOperator(
                   // Check that the conditional operator itself has an ancestor
                   // which is the implicit cast to unsigned
                   hasAncestor(implicitCastExpr(hasType(isUnsignedInteger()))
                                   .bind("outerCast")))))
          .bind("unsignedMaxExpr");

  // *** MODIFY THIS PART ***
  auto OldCombined =
      expr(anyOf(
          ExplicitCastOfNegOrBitnot,
          ImplicitNegOneToUnsigned,
          UnsignedBitNotZero,
          UnsignedLiteralNegOne
      )).bind("unsignedMaxExpr");

  Finder->addMatcher(OldCombined, this);
  Finder->addMatcher(TernaryBranch, this); // Add the new matcher
}


void NumericlimitmaxcheckCheck::check(const MatchFinder::MatchResult &Result) {
const auto *E = Result.Nodes.getNodeAs<Expr>("unsignedMaxExpr");
  const auto *OuterCast = Result.Nodes.getNodeAs<CastExpr>("outerCast"); // Get the cast

  if (!E)
    return;

  ASTContext &Ctx = *Result.Context; // Get context *before* first use

  QualType QT;
  if (OuterCast) {
    // This is our new ternary matcher. Get type from the *cast*.
    QT = OuterCast->getType();
  } else {
    // This is the old logic. Get type from the bound expression.
    QT = E->getType();
  }

  if (E->getBeginLoc().isInvalid() || E->getBeginLoc().isMacroID())
    return;

  const SourceManager &SM = Ctx.getSourceManager();

  // *** ADD THIS LOGIC BLOCK ***
  // This logic prevents double-reporting for the *old* matchers.
  // We skip it for the new ternary matcher (when OuterCast is not null)
  // because the ternary matcher binds the *inner* expression, and we
  // *do* want to report it.
  if (!OuterCast) {
    auto Parents = Ctx.getParents(*E); // This fixes the [-Wunused] warning
    if (!Parents.empty()) {
      for (const auto &Parent : Parents) {
        // Check if parent is an explicit cast to unsigned
        if (const auto *ParentCast = Parent.get<ExplicitCastExpr>()) {
          if (ParentCast->getType()->isUnsignedIntegerType()) {
            // Skip this match - the cast itself will be (or was) reported
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
  // ... (rest of parent checking logic) ...
  // ...
  // Determine the unsigned destination type
  // QualType QT = E->getType(); // This line is moved up and modified
  if (QT.isNull() || !QT->isUnsignedIntegerType())
    return;

  // Get a printable type string
  std::string TypeStr = QT.getUnqualifiedType().getAsString();
  if (const auto *Typedef = QT->getAs<TypedefType>()) {
    TypeStr = Typedef->getDecl()->getName().str();
  }

  // Get original source text for diagnostic message
  StringRef OriginalText =
      Lexer::getSourceText(CharSourceRange::getTokenRange(E->getSourceRange()), 
                          SM, getLangOpts());

  // Build replacement text
  std::string Replacement = "std::numeric_limits<" + TypeStr + ">::max()";

  // Create diagnostic
  auto Diag = diag(E->getBeginLoc(),
                   "use 'std::numeric_limits<%0>::max()' instead of '%1'")
              << TypeStr << OriginalText;

  // Add fix-it hints
  Diag << FixItHint::CreateReplacement(E->getSourceRange(), Replacement);

  // Add include for <limits>
  FileID FID = SM.getFileID(E->getBeginLoc());
  if (auto IncludeHint = Inserter.createIncludeInsertion(FID, "<limits>")) {
    Diag << *IncludeHint;
  }
}

} // namespace clang::tidy::readability