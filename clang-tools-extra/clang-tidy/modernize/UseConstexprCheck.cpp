//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseConstexprCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

void UseConstexprCheck::registerMatchers(MatchFinder *Finder) {
  // Match local const variables with initializers that are not already
  // constexpr, not static, and not dependent.
  Finder->addMatcher(
      varDecl(hasLocalStorage(), hasType(qualType(isConstQualified())),
              hasInitializer(expr().bind("init")), unless(isConstexpr()))
          .bind("var"),
      this);
}

void UseConstexprCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *VD = Result.Nodes.getNodeAs<VarDecl>("var");
  if (!VD)
    return;

  // Skip if the variable is static or thread_local.
  if (VD->isStaticLocal() || VD->getTSCSpec() != TSCS_unspecified)
    return;

  // Skip volatile variables.
  if (VD->getType().isVolatileQualified())
    return;

  // Skip dependent types (in uninstantiated templates).
  if (VD->getType()->isDependentType())
    return;

  // The type must be a literal type for constexpr.
  if (!VD->getType()->isLiteralType(*Result.Context))
    return;

  // Skip reference types -- constexpr references need static storage
  // duration objects, which is too restrictive for local variables.
  if (VD->getType()->isReferenceType())
    return;

  // Skip pointer types -- the 'const' qualifier position makes the
  // fix-it non-trivial (e.g. 'int *const p' vs 'const int *p').
  if (VD->getType()->isPointerType())
    return;

  const Expr *Init = VD->getInit();
  if (!Init)
    return;

  // Skip if the initializer is value-dependent (template context).
  if (Init->isValueDependent())
    return;

  // Check if the initializer is a C++11 constant expression.
  if (!Init->isCXX11ConstantExpr(*Result.Context))
    return;

  // Find the 'const' token to replace with 'constexpr'.
  const SourceLocation VarLoc = VD->getLocation();
  const SourceLocation DeclBegin = VD->getBeginLoc();

  if (DeclBegin.isInvalid() || DeclBegin.isMacroID() || VarLoc.isMacroID())
    return;

  const SourceManager &SM = *Result.SourceManager;
  const LangOptions &LO = Result.Context->getLangOpts();

  // Get source text from the start of the declaration to the variable
  // name. This covers both 'const int x' and 'int const x'.
  const CharSourceRange DeclRange =
      CharSourceRange::getCharRange(DeclBegin, VarLoc);
  const StringRef DeclText = Lexer::getSourceText(DeclRange, SM, LO);

  // Find 'const' as a whole word in the declaration text.
  SourceLocation ConstLoc;
  size_t Pos = 0;
  while ((Pos = DeclText.find("const", Pos)) != StringRef::npos) {
    // Make sure it's not part of 'constexpr' or 'consteval' etc.
    const size_t End = Pos + 5;
    const bool WordStart = (Pos == 0 || !isAlphanumeric(DeclText[Pos - 1]));
    const bool WordEnd =
        (End >= DeclText.size() || !isAlphanumeric(DeclText[End]));
    if (WordStart && WordEnd) {
      ConstLoc = DeclBegin.getLocWithOffset(Pos);
      break;
    }
    Pos = End;
  }

  if (ConstLoc.isValid()) {
    diag(VD->getLocation(), "variable %0 can be declared 'constexpr'")
        << VD
        << FixItHint::CreateReplacement(
               CharSourceRange::getTokenRange(ConstLoc, ConstLoc), "constexpr");
  } else {
    // Couldn't find 'const' token; emit diagnostic without fix-it.
    diag(VD->getLocation(), "variable %0 can be declared 'constexpr'") << VD;
  }
}

} // namespace clang::tidy::modernize
