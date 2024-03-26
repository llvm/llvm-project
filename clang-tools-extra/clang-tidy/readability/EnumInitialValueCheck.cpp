//===--- EnumInitialValueCheck.cpp - clang-tidy ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "EnumInitialValueCheck.h"
#include "../utils/LexerUtils.h"
#include "clang/AST/Decl.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

EnumInitialValueCheck::EnumInitialValueCheck(StringRef Name,
                                             ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      AllowExplicitZeroFirstInitialValue(
          Options.get("AllowExplicitZeroFirstInitialValue", true)),
      AllowExplicitLinearInitialValues(
          Options.get("AllowExplicitLinearInitialValues", true)) {}

void EnumInitialValueCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "AllowExplicitZeroFirstInitialValue",
                AllowExplicitZeroFirstInitialValue);
  Options.store(Opts, "AllowExplicitLinearInitialValues",
                AllowExplicitLinearInitialValues);
}

namespace {

bool isNoneEnumeratorsInitialized(const EnumDecl &Node) {
  return llvm::all_of(Node.enumerators(), [](const EnumConstantDecl *ECD) {
    return ECD->getInitExpr() == nullptr;
  });
}

bool isOnlyFirstEnumeratorsInitialized(const EnumDecl &Node) {
  bool IsFirst = true;
  for (const EnumConstantDecl *ECD : Node.enumerators())
    if (IsFirst) {
      IsFirst = false;
      if (ECD->getInitExpr() == nullptr)
        return false;
    } else {
      if (ECD->getInitExpr() != nullptr)
        return false;
    }
  return !IsFirst;
}

bool isAllEnumeratorsInitialized(const EnumDecl &Node) {
  return llvm::all_of(Node.enumerators(), [](const EnumConstantDecl *ECD) {
    return ECD->getInitExpr() != nullptr;
  });
}

/// Check if \p Enumerator is initialized with a (potentially negated) \c
/// IntegerLiteral.
bool isInitializedByLiteral(const EnumConstantDecl *Enumerator) {
  const Expr *const Init = Enumerator->getInitExpr();
  if (!Init)
    return false;
  return Init->isIntegerConstantExpr(Enumerator->getASTContext());
}

AST_MATCHER(EnumDecl, hasConsistentInitialValues) {
  return isNoneEnumeratorsInitialized(Node) ||
         isOnlyFirstEnumeratorsInitialized(Node) ||
         isAllEnumeratorsInitialized(Node);
}

AST_MATCHER(EnumDecl, hasZeroFirstInitialValue) {
  if (Node.enumerators().empty())
    return false;
  const EnumConstantDecl *ECD = *Node.enumerators().begin();
  return isOnlyFirstEnumeratorsInitialized(Node) &&
         isInitializedByLiteral(ECD) && ECD->getInitVal().isZero();
}

/// Excludes bitfields because enumerators initialized with the result of a
/// bitwise operator on enumeration values or any other expr that is not a
/// potentially negative integer literal.
/// Enumerations where it is not directly clear if they are used with
/// bitmask, evident when enumerators are only initialized with (potentially
/// negative) integer literals, are ignored. This is also the case when all
/// enumerators are powers of two (e.g., 0, 1, 2).
AST_MATCHER(EnumDecl, hasLinearInitialValues) {
  if (Node.enumerators().empty())
    return false;
  const EnumConstantDecl *const FirstEnumerator = *Node.enumerator_begin();
  llvm::APSInt PrevValue = FirstEnumerator->getInitVal();
  if (!isInitializedByLiteral(FirstEnumerator))
    return false;
  bool AllEnumeratorsArePowersOfTwo = true;
  for (const EnumConstantDecl *Enumerator :
       llvm::drop_begin(Node.enumerators())) {
    const llvm::APSInt NewValue = Enumerator->getInitVal();
    if (NewValue != ++PrevValue)
      return false;
    if (!isInitializedByLiteral(Enumerator))
      return false;
    PrevValue = NewValue;
    AllEnumeratorsArePowersOfTwo &= NewValue.isPowerOf2();
  }
  return !AllEnumeratorsArePowersOfTwo;
}

} // namespace

void EnumInitialValueCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      enumDecl(unless(hasConsistentInitialValues())).bind("inconsistent"),
      this);
  if (!AllowExplicitZeroFirstInitialValue)
    Finder->addMatcher(enumDecl(hasZeroFirstInitialValue()).bind("zero_first"),
                       this);
  if (!AllowExplicitLinearInitialValues)
    Finder->addMatcher(enumDecl(hasLinearInitialValues()).bind("linear"), this);
}

static void cleanInitialValue(DiagnosticBuilder &Diag,
                              const EnumConstantDecl *ECD,
                              const SourceManager &SM,
                              const LangOptions &LangOpts) {
  std::optional<Token> EqualToken = utils::lexer::findNextTokenSkippingComments(
      ECD->getLocation(), SM, LangOpts);
  if (!EqualToken.has_value())
    return;
  SourceLocation EqualLoc{EqualToken->getLocation()};
  if (EqualLoc.isInvalid() || EqualLoc.isMacroID())
    return;
  SourceRange InitExprRange = ECD->getInitExpr()->getSourceRange();
  if (InitExprRange.isInvalid() || InitExprRange.getBegin().isMacroID() ||
      InitExprRange.getEnd().isMacroID())
    return;
  Diag << FixItHint::CreateRemoval(EqualLoc)
       << FixItHint::CreateRemoval(InitExprRange);
  return;
}

void EnumInitialValueCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *Enum = Result.Nodes.getNodeAs<EnumDecl>("inconsistent")) {
    SourceLocation Loc = Enum->getBeginLoc();
    if (Loc.isInvalid() || Loc.isMacroID())
      return;
    DiagnosticBuilder Diag =
        diag(Loc, "inital values in enum %0 are not consistent, consider "
                  "explicit initialization first, all or none of enumerators")
        << Enum;
    for (const EnumConstantDecl *ECD : Enum->enumerators())
      if (ECD->getInitExpr() == nullptr) {
        std::optional<Token> Next = utils::lexer::findNextTokenSkippingComments(
            ECD->getLocation(), *Result.SourceManager, getLangOpts());
        if (!Next.has_value() || Next->getLocation().isMacroID())
          continue;
        llvm::SmallString<8> Str{" = "};
        ECD->getInitVal().toString(Str);
        Diag << FixItHint::CreateInsertion(Next->getLocation(), Str);
      }
    return;
  }
  if (const auto *Enum = Result.Nodes.getNodeAs<EnumDecl>("zero_first")) {
    const EnumConstantDecl *ECD = *Enum->enumerators().begin();
    SourceLocation Loc = ECD->getLocation();
    if (Loc.isInvalid() || Loc.isMacroID())
      return;
    DiagnosticBuilder Diag =
        diag(Loc, "zero fist initial value in %0 can be ignored") << Enum;
    cleanInitialValue(Diag, ECD, *Result.SourceManager, getLangOpts());
    return;
  }
  if (const auto *Enum = Result.Nodes.getNodeAs<EnumDecl>("linear")) {
    SourceLocation Loc = Enum->getBeginLoc();
    if (Loc.isInvalid() || Loc.isMacroID())
      return;
    DiagnosticBuilder Diag =
        diag(Loc, "linear initial value in %0 can be ignored") << Enum;
    for (const EnumConstantDecl *ECD : llvm::drop_begin(Enum->enumerators()))
      cleanInitialValue(Diag, ECD, *Result.SourceManager, getLangOpts());
    return;
  }
}

} // namespace clang::tidy::readability
