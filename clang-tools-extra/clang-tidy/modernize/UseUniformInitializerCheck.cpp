//===--- UseUniformInitializerCheck.cpp - clang-tidy ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseUniformInitializerCheck.h"
#include "../utils/LexerUtils.h"
#include "clang/Tooling/FixIt.h"

AST_MATCHER(clang::VarDecl, isVarOldStyleInitializer) {
  // If it doesn't have any initializer the initializer style is not defined.
  if (!Node.hasInit())
    return false;

  const clang::VarDecl::InitializationStyle InitStyle = Node.getInitStyle();

  return InitStyle == clang::VarDecl::InitializationStyle::CInit ||
         InitStyle == clang::VarDecl::InitializationStyle::CallInit;
}

AST_MATCHER(clang::FieldDecl, isFieldOldStyleInitializer) {
  // If it doesn't have any initializer the initializer style is not defined.
  if (!Node.hasInClassInitializer() || Node.getInClassInitializer() == nullptr)
    return false;

  const clang::InClassInitStyle InitStyle = Node.getInClassInitStyle();

  return InitStyle == clang::InClassInitStyle::ICIS_CopyInit;
}

AST_MATCHER(clang::CXXCtorInitializer, isCStyleInitializer) {
  const clang::Expr *Init = Node.getInit();
  if (Init == nullptr)
    return false;

  return !llvm::isa<clang::InitListExpr>(Init);
}

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

namespace {

constexpr const StringRef VarDeclID = "VarDecl";
constexpr const StringRef FieldDeclId = "FieldDecl";
constexpr const StringRef CtorInitID = "CtorInit";

constexpr const StringRef CStyleWarningMessage =
    "Use uniform initializer instead of C-style initializer";

constexpr StringRef
getInitializerStyleName(VarDecl::InitializationStyle InitStyle) {
  switch (InitStyle) {
  case VarDecl::InitializationStyle::CInit:
    return "C";

  case VarDecl::InitializationStyle::CallInit:
    return "call";

  default:
    llvm_unreachable("Invalid initializer style!");
  }
}

SourceRange getParenLocationsForCallInit(const Expr *InitExpr,
                                         const SourceManager &SM,
                                         const LangOptions &LangOpts) {
  // We need to handle 'CXXConstructExpr' differently
  if (isa<CXXConstructExpr>(InitExpr))
    return cast<CXXConstructExpr>(InitExpr)->getParenOrBraceRange();

  // If the init expression itself is a 'ParenExpr' then
  // 'InitExpr->getBeginLoc()' will already point to a '(' which is not the
  // opening paren of the 'CallInit' expression. So it that case we need to
  // start one character before that.
  const bool NeedOffsetForOpenParen = [&]() {
    if (!isa<ParenExpr>(InitExpr))
      return false;

    const clang::StringRef CharBeforeParenExpr =
        Lexer::getSourceText(CharSourceRange::getCharRange(
                                 InitExpr->getBeginLoc().getLocWithOffset(-1),
                                 InitExpr->getBeginLoc()),
                             SM, LangOpts);

    return llvm::isSpace(CharBeforeParenExpr[0]);
  }();

  const SourceLocation OpenParenLoc = utils::lexer::findPreviousTokenKind(
      NeedOffsetForOpenParen ? InitExpr->getBeginLoc().getLocWithOffset(-1)
                             : InitExpr->getBeginLoc(),
      SM, LangOpts, tok::l_paren);
  const SourceLocation CloseParenLoc = utils::lexer::findNextTokenKind(
      InitExpr->getEndLoc(), SM, LangOpts, tok::r_paren);

  return {OpenParenLoc, CloseParenLoc};
}

const BuiltinType *getBuiltinType(const Expr *Expr) {
  assert(Expr);
  return Expr->getType().getCanonicalType().getTypePtr()->getAs<BuiltinType>();
}

bool castRequiresStaticCast(const ImplicitCastExpr *CastExpr) {
  const auto *FromExpr = CastExpr->getSubExpr();

  if (CastExpr->isInstantiationDependent() ||
      FromExpr->isInstantiationDependent())
    return false;
  if (getBuiltinType(CastExpr) == getBuiltinType(FromExpr))
    return false;

  switch (CastExpr->getCastKind()) {
  case CastKind::CK_BaseToDerived:
  case CastKind::CK_DerivedToBaseMemberPointer:
  case CastKind::CK_IntegralCast:
  case CastKind::CK_FloatingToIntegral:
    return true;

  default:
    return false;
  }
}

std::string buildReplacementString(const Expr *InitExpr,
                                   const ASTContext &Context) {
  // TODO: This function does not correctly handle the case where you have in
  // 'ImplicitCastExpr' as an argument for a 'CXXConstructExpr'.
  // In that case the generated code will not compile due to missing explicit
  // cast of the sub expression.

  const SourceManager &SM = Context.getSourceManager();
  const LangOptions &LangOpts = Context.getLangOpts();

  const StringRef InitExprStr = [&]() {
    if (isa<CXXConstructExpr>(InitExpr)) {
      const auto *ConstructExpr = llvm::cast<CXXConstructExpr>(InitExpr);

      const SourceRange ParenRange = ConstructExpr->getParenOrBraceRange();
      if (ParenRange.isValid())
        return Lexer::getSourceText(
                   CharSourceRange::getCharRange(
                       ParenRange.getBegin().getLocWithOffset(1),
                       ParenRange.getEnd()),
                   SM, LangOpts)
            .trim();

      // In case the ParenRange is invalid we use Begin/EndLoc
      const SourceLocation BeginLocation = ConstructExpr->getBeginLoc();
      const SourceLocation EndLocation = ConstructExpr->getEndLoc();

      return Lexer::getSourceText(
                 CharSourceRange::getCharRange(BeginLocation,
                                               EndLocation.getLocWithOffset(1)),
                 SM, LangOpts)
          .trim();
    }

    return tooling::fixit::getText(*InitExpr, Context);
  }();

  // For some 'ImplicitCastExpr' we need to add an extra 'static_cast<T>' around
  // the expression since implicit conversions are not allowed with uniform
  // initializer and otherwise will lead to compile errors
  if (isa<ImplicitCastExpr>(InitExpr)) {
    const auto *CastExpr = llvm::cast<ImplicitCastExpr>(InitExpr);

    if (castRequiresStaticCast(CastExpr)) {
      const QualType Type = CastExpr->getType().getUnqualifiedType();

      return ("{static_cast<" + Type.getAsString() + ">(" + InitExprStr + ")}")
          .str();
    }
  }

  // Otherwise just add the braces around the expression
  return ("{" + InitExprStr + "}").str();
}

} // namespace

void UseUniformInitializerCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(varDecl(isVarOldStyleInitializer(), unless(parmVarDecl()))
                         .bind(VarDeclID),
                     this);
  Finder->addMatcher(fieldDecl(isFieldOldStyleInitializer()).bind(FieldDeclId),
                     this);
  Finder->addMatcher(cxxCtorInitializer(isCStyleInitializer()).bind(CtorInitID),
                     this);
}

void UseUniformInitializerCheck::check(const MatchFinder::MatchResult &Result) {
  const SourceManager &SM = *Result.SourceManager;
  const ASTContext &Context = *Result.Context;
  const LangOptions &LangOpts = Context.getLangOpts();

  const auto *VDecl = Result.Nodes.getNodeAs<VarDecl>(VarDeclID);
  const auto *FDecl = Result.Nodes.getNodeAs<FieldDecl>(FieldDeclId);
  const auto *CtorInit = Result.Nodes.getNodeAs<CXXCtorInitializer>(CtorInitID);

  // Handle variable declarations
  if (VDecl != nullptr) {
    const auto *InitExpr = VDecl->getInit();
    assert(InitExpr);

    // If the expression is an 'ExprWithCleanups' just get the subexpression
    if (isa<ExprWithCleanups>(InitExpr))
      InitExpr = llvm::cast<ExprWithCleanups>(InitExpr)->getSubExpr();

    const VarDecl::InitializationStyle InitStyle = VDecl->getInitStyle();

    // Check for the special case that a default constructor with no arguments
    // is used
    // Example: struct A{}; A a;
    const auto *ConstructorExpr = dyn_cast_or_null<CXXConstructExpr>(InitExpr);
    if (ConstructorExpr && ConstructorExpr->getNumArgs() == 0)
      return;

    // Ignore declarations inside a 'CXXForRangeStmt'
    if (VDecl->isCXXForRangeDecl())
      return;

    if (InitStyle == VarDecl::InitializationStyle::CInit) {
      const SourceLocation EqualLoc = utils::lexer::findPreviousTokenKind(
          InitExpr->getBeginLoc(), SM, LangOpts, tok::equal);
      if (!EqualLoc.isValid())
        return;

      auto Diag =
          diag(EqualLoc,
               "Use uniform initializer instead of %0-style initializer")
          << getInitializerStyleName(InitStyle);

      // If we have a C-style initializer with an 'InitListExpr', we only need
      // to remove the '='
      // For example: 'int a = {0};'
      if (isa<InitListExpr>(InitExpr))
        Diag << FixItHint::CreateRemoval(EqualLoc);
      else {
        // Otherwise with a normal c-style initializer we also need to add '{'
        // and '}'
        // For example: 'int a = 0;'
        const std::string ReplacementText =
            buildReplacementString(InitExpr, Context);

        Diag << FixItHint::CreateReplacement(
            SourceRange{EqualLoc, InitExpr->getEndLoc()}, ReplacementText);
      }
    } else if (InitStyle == VarDecl::InitializationStyle::CallInit) {
      const SourceRange ParenRange =
          getParenLocationsForCallInit(InitExpr, SM, LangOpts);
      if (ParenRange.isInvalid())
        return;

      const std::string ReplacementString =
          buildReplacementString(InitExpr, Context);

      diag(ParenRange.getBegin(),
           "Use uniform initializer instead of %0-style initializer")
          << getInitializerStyleName(InitStyle)
          << FixItHint::CreateReplacement(ParenRange, ReplacementString);
    } else {
      llvm_unreachable("Invalid initializer style!");
    }
  }
  // Handle field declarations
  else if (FDecl != nullptr) {
    const auto *InitExpr = FDecl->getInClassInitializer();
    assert(InitExpr);

    const SourceLocation EqualLoc = utils::lexer::findPreviousTokenKind(
        InitExpr->getBeginLoc(), SM, LangOpts, tok::equal);
    if (!EqualLoc.isValid())
      return;

    if (isa<InitListExpr>(InitExpr))
      // If we have a C-style initializer with an 'InitListExpr', we only need
      // to remove the '='
      diag(EqualLoc, CStyleWarningMessage)
          << FixItHint::CreateRemoval(EqualLoc);
    else {
      // Otherwise with a normal c-style initializer we also need to add '{' and
      // '}' and rewrite the initializer
      const StringRef InitExprStr = tooling::fixit::getText(*InitExpr, Context);

      const std::string ReplacementText = ("{" + InitExprStr + "}").str();

      diag(EqualLoc, CStyleWarningMessage) << FixItHint::CreateReplacement(
          SourceRange{EqualLoc, InitExpr->getEndLoc()}, ReplacementText);
    }
  }
  // Otherwise must be a CXXCtorInitializer
  else {
    assert(CtorInit != nullptr);

    const auto *InitExpr = CtorInit->getInit();
    assert(InitExpr);

    const SourceLocation LParenLoc = CtorInit->getLParenLoc();
    const SourceLocation RParenLoc = CtorInit->getRParenLoc();

    if (!LParenLoc.isValid() || !RParenLoc.isValid())
      return;

    if (LParenLoc == RParenLoc)
      return;

    const std::string ReplacementString =
        buildReplacementString(InitExpr, Context);

    diag(LParenLoc, CStyleWarningMessage) << FixItHint::CreateReplacement(
        SourceRange{LParenLoc, RParenLoc}, ReplacementString);
  }
}

} // namespace clang::tidy::modernize
