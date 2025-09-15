//===----------------------------------------------------------------------===//
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

static bool isNoneEnumeratorsInitialized(const EnumDecl &Node) {
  return llvm::all_of(Node.enumerators(), [](const EnumConstantDecl *ECD) {
    return ECD->getInitExpr() == nullptr;
  });
}

static bool isOnlyFirstEnumeratorInitialized(const EnumDecl &Node) {
  bool IsFirst = true;
  for (const EnumConstantDecl *ECD : Node.enumerators()) {
    if ((IsFirst && ECD->getInitExpr() == nullptr) ||
        (!IsFirst && ECD->getInitExpr() != nullptr))
      return false;
    IsFirst = false;
  }
  return !IsFirst;
}

static bool areAllEnumeratorsInitialized(const EnumDecl &Node) {
  return llvm::all_of(Node.enumerators(), [](const EnumConstantDecl *ECD) {
    return ECD->getInitExpr() != nullptr;
  });
}

/// Check if \p Enumerator is initialized with a (potentially negated) \c
/// IntegerLiteral.
static bool isInitializedByLiteral(const EnumConstantDecl *Enumerator) {
  const Expr *const Init = Enumerator->getInitExpr();
  if (!Init)
    return false;
  return Init->isIntegerConstantExpr(Enumerator->getASTContext());
}

static void cleanInitialValue(DiagnosticBuilder &Diag,
                              const EnumConstantDecl *ECD,
                              const SourceManager &SM,
                              const LangOptions &LangOpts) {
  const SourceRange InitExprRange = ECD->getInitExpr()->getSourceRange();
  if (InitExprRange.isInvalid() || InitExprRange.getBegin().isMacroID() ||
      InitExprRange.getEnd().isMacroID())
    return;
  std::optional<Token> EqualToken = utils::lexer::findNextTokenSkippingComments(
      ECD->getLocation(), SM, LangOpts);
  if (!EqualToken.has_value() ||
      EqualToken.value().getKind() != tok::TokenKind::equal)
    return;
  const SourceLocation EqualLoc{EqualToken->getLocation()};
  if (EqualLoc.isInvalid() || EqualLoc.isMacroID())
    return;
  Diag << FixItHint::CreateRemoval(EqualLoc)
       << FixItHint::CreateRemoval(InitExprRange);
}

namespace {

AST_MATCHER(EnumDecl, isMacro) {
  SourceLocation Loc = Node.getBeginLoc();
  return Loc.isMacroID();
}

AST_MATCHER(EnumDecl, hasConsistentInitialValues) {
  return isNoneEnumeratorsInitialized(Node) ||
         isOnlyFirstEnumeratorInitialized(Node) ||
         areAllEnumeratorsInitialized(Node);
}

AST_MATCHER(EnumDecl, hasZeroInitialValueForFirstEnumerator) {
  const EnumDecl::enumerator_range Enumerators = Node.enumerators();
  if (Enumerators.empty())
    return false;
  const EnumConstantDecl *ECD = *Enumerators.begin();
  return isOnlyFirstEnumeratorInitialized(Node) &&
         isInitializedByLiteral(ECD) && ECD->getInitVal().isZero();
}

/// Excludes bitfields because enumerators initialized with the result of a
/// bitwise operator on enumeration values or any other expr that is not a
/// potentially negative integer literal.
/// Enumerations where it is not directly clear if they are used with
/// bitmask, evident when enumerators are only initialized with (potentially
/// negative) integer literals, are ignored. This is also the case when all
/// enumerators are powers of two (e.g., 0, 1, 2).
AST_MATCHER(EnumDecl, hasSequentialInitialValues) {
  const EnumDecl::enumerator_range Enumerators = Node.enumerators();
  if (Enumerators.empty())
    return false;
  const EnumConstantDecl *const FirstEnumerator = *Node.enumerator_begin();
  llvm::APSInt PrevValue = FirstEnumerator->getInitVal();
  if (!isInitializedByLiteral(FirstEnumerator))
    return false;
  bool AllEnumeratorsArePowersOfTwo = true;
  for (const EnumConstantDecl *Enumerator : llvm::drop_begin(Enumerators)) {
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

static std::string getName(const EnumDecl *Decl) {
  if (!Decl->getDeclName())
    return "<unnamed>";

  return Decl->getQualifiedNameAsString();
}

EnumInitialValueCheck::EnumInitialValueCheck(StringRef Name,
                                             ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      AllowExplicitZeroFirstInitialValue(
          Options.get("AllowExplicitZeroFirstInitialValue", true)),
      AllowExplicitSequentialInitialValues(
          Options.get("AllowExplicitSequentialInitialValues", true)) {}

void EnumInitialValueCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "AllowExplicitZeroFirstInitialValue",
                AllowExplicitZeroFirstInitialValue);
  Options.store(Opts, "AllowExplicitSequentialInitialValues",
                AllowExplicitSequentialInitialValues);
}

void EnumInitialValueCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(enumDecl(isDefinition(), unless(isMacro()),
                              unless(hasConsistentInitialValues()))
                         .bind("inconsistent"),
                     this);
  if (!AllowExplicitZeroFirstInitialValue)
    Finder->addMatcher(
        enumDecl(isDefinition(), hasZeroInitialValueForFirstEnumerator())
            .bind("zero_first"),
        this);
  if (!AllowExplicitSequentialInitialValues)
    Finder->addMatcher(enumDecl(isDefinition(), unless(isMacro()),
                                hasSequentialInitialValues())
                           .bind("sequential"),
                       this);
}

void EnumInitialValueCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *Enum = Result.Nodes.getNodeAs<EnumDecl>("inconsistent")) {
    DiagnosticBuilder Diag =
        diag(
            Enum->getBeginLoc(),
            "initial values in enum '%0' are not consistent, consider explicit "
            "initialization of all, none or only the first enumerator")
        << getName(Enum);
    for (const EnumConstantDecl *ECD : Enum->enumerators())
      if (ECD->getInitExpr() == nullptr) {
        const SourceLocation EndLoc = Lexer::getLocForEndOfToken(
            ECD->getLocation(), 0, *Result.SourceManager, getLangOpts());
        if (EndLoc.isMacroID())
          continue;
        llvm::SmallString<8> Str{" = "};
        ECD->getInitVal().toString(Str);
        Diag << FixItHint::CreateInsertion(EndLoc, Str);
      }
    return;
  }

  if (const auto *Enum = Result.Nodes.getNodeAs<EnumDecl>("zero_first")) {
    const EnumConstantDecl *ECD = *Enum->enumerator_begin();
    const SourceLocation Loc = ECD->getLocation();
    if (Loc.isInvalid() || Loc.isMacroID())
      return;
    DiagnosticBuilder Diag = diag(Loc, "zero initial value for the first "
                                       "enumerator in '%0' can be disregarded")
                             << getName(Enum);
    cleanInitialValue(Diag, ECD, *Result.SourceManager, getLangOpts());
    return;
  }
  if (const auto *Enum = Result.Nodes.getNodeAs<EnumDecl>("sequential")) {
    DiagnosticBuilder Diag =
        diag(Enum->getBeginLoc(),
             "sequential initial value in '%0' can be ignored")
        << getName(Enum);
    for (const EnumConstantDecl *ECD : llvm::drop_begin(Enum->enumerators()))
      cleanInitialValue(Diag, ECD, *Result.SourceManager, getLangOpts());
    return;
  }
}

} // namespace clang::tidy::readability
