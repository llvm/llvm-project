//===--- UseDefaultMemberInitCheck.cpp - clang-tidy------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseDefaultMemberInitCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace modernize {

namespace {
AST_MATCHER_P(InitListExpr, initCountIs, unsigned, N) {
  return Node.getNumInits() == N;
}
} // namespace

static StringRef getValueOfValueInit(const QualType InitType) {
  switch (InitType->getScalarTypeKind()) {
  case Type::STK_CPointer:
  case Type::STK_BlockPointer:
  case Type::STK_ObjCObjectPointer:
  case Type::STK_MemberPointer:
    return "nullptr";

  case Type::STK_Bool:
    return "false";

  case Type::STK_Integral:
    switch (InitType->castAs<BuiltinType>()->getKind()) {
    case BuiltinType::Char_U:
    case BuiltinType::UChar:
    case BuiltinType::Char_S:
    case BuiltinType::SChar:
      return "'\\0'";
    case BuiltinType::WChar_U:
    case BuiltinType::WChar_S:
      return "L'\\0'";
    case BuiltinType::Char16:
      return "u'\\0'";
    case BuiltinType::Char32:
      return "U'\\0'";
    default:
      return "0";
    }

  case Type::STK_Floating:
    switch (InitType->castAs<BuiltinType>()->getKind()) {
    case BuiltinType::Half:
    case BuiltinType::Float:
      return "0.0f";
    default:
      return "0.0";
    }

  case Type::STK_FloatingComplex:
  case Type::STK_IntegralComplex:
    return getValueOfValueInit(
        InitType->castAs<ComplexType>()->getElementType());

  case Type::STK_FixedPoint:
    switch (InitType->castAs<BuiltinType>()->getKind()) {
    case BuiltinType::ShortAccum:
    case BuiltinType::SatShortAccum:
      return "0.0hk";
    case BuiltinType::Accum:
    case BuiltinType::SatAccum:
      return "0.0k";
    case BuiltinType::LongAccum:
    case BuiltinType::SatLongAccum:
      return "0.0lk";
    case BuiltinType::UShortAccum:
    case BuiltinType::SatUShortAccum:
      return "0.0uhk";
    case BuiltinType::UAccum:
    case BuiltinType::SatUAccum:
      return "0.0uk";
    case BuiltinType::ULongAccum:
    case BuiltinType::SatULongAccum:
      return "0.0ulk";
    case BuiltinType::ShortFract:
    case BuiltinType::SatShortFract:
      return "0.0hr";
    case BuiltinType::Fract:
    case BuiltinType::SatFract:
      return "0.0r";
    case BuiltinType::LongFract:
    case BuiltinType::SatLongFract:
      return "0.0lr";
    case BuiltinType::UShortFract:
    case BuiltinType::SatUShortFract:
      return "0.0uhr";
    case BuiltinType::UFract:
    case BuiltinType::SatUFract:
      return "0.0ur";
    case BuiltinType::ULongFract:
    case BuiltinType::SatULongFract:
      return "0.0ulr";
    default:
      llvm_unreachable("Unhandled fixed point BuiltinType");
    }
  }
  llvm_unreachable("Invalid scalar type kind");
}

static bool isZero(const Expr *E) {
  switch (E->getStmtClass()) {
  case Stmt::CXXNullPtrLiteralExprClass:
  case Stmt::ImplicitValueInitExprClass:
    return true;
  case Stmt::InitListExprClass:
    return cast<InitListExpr>(E)->getNumInits() == 0;
  case Stmt::CharacterLiteralClass:
    return !cast<CharacterLiteral>(E)->getValue();
  case Stmt::CXXBoolLiteralExprClass:
    return !cast<CXXBoolLiteralExpr>(E)->getValue();
  case Stmt::IntegerLiteralClass:
    return !cast<IntegerLiteral>(E)->getValue();
  case Stmt::FloatingLiteralClass: {
    llvm::APFloat Value = cast<FloatingLiteral>(E)->getValue();
    return Value.isZero() && !Value.isNegative();
  }
  default:
    return false;
  }
}

static const Expr *ignoreUnaryPlus(const Expr *E) {
  auto *UnaryOp = dyn_cast<UnaryOperator>(E);
  if (UnaryOp && UnaryOp->getOpcode() == UO_Plus)
    return UnaryOp->getSubExpr();
  return E;
}

static const Expr *getInitializer(const Expr *E) {
  auto *InitList = dyn_cast<InitListExpr>(E);
  if (InitList && InitList->getNumInits() == 1)
    return InitList->getInit(0)->IgnoreParenImpCasts();
  return E;
}

static bool sameValue(const Expr *E1, const Expr *E2) {
  E1 = ignoreUnaryPlus(getInitializer(E1->IgnoreParenImpCasts()));
  E2 = ignoreUnaryPlus(getInitializer(E2->IgnoreParenImpCasts()));

  if (isZero(E1) && isZero(E2))
    return true;

  if (E1->getStmtClass() != E2->getStmtClass())
    return false;

  switch (E1->getStmtClass()) {
  case Stmt::UnaryOperatorClass:
    return sameValue(cast<UnaryOperator>(E1)->getSubExpr(),
                     cast<UnaryOperator>(E2)->getSubExpr());
  case Stmt::CharacterLiteralClass:
    return cast<CharacterLiteral>(E1)->getValue() ==
           cast<CharacterLiteral>(E2)->getValue();
  case Stmt::CXXBoolLiteralExprClass:
    return cast<CXXBoolLiteralExpr>(E1)->getValue() ==
           cast<CXXBoolLiteralExpr>(E2)->getValue();
  case Stmt::IntegerLiteralClass:
    return cast<IntegerLiteral>(E1)->getValue() ==
           cast<IntegerLiteral>(E2)->getValue();
  case Stmt::FloatingLiteralClass:
    return cast<FloatingLiteral>(E1)->getValue().bitwiseIsEqual(
        cast<FloatingLiteral>(E2)->getValue());
  case Stmt::StringLiteralClass:
    return cast<StringLiteral>(E1)->getString() ==
           cast<StringLiteral>(E2)->getString();
  case Stmt::DeclRefExprClass:
    return cast<DeclRefExpr>(E1)->getDecl() == cast<DeclRefExpr>(E2)->getDecl();
  default:
    return false;
  }
}

UseDefaultMemberInitCheck::UseDefaultMemberInitCheck(StringRef Name,
                                                     ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      UseAssignment(Options.get("UseAssignment", false)),
      IgnoreMacros(Options.getLocalOrGlobal("IgnoreMacros", true)) {}

void UseDefaultMemberInitCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "UseAssignment", UseAssignment);
  Options.store(Opts, "IgnoreMacros", IgnoreMacros);
}

void UseDefaultMemberInitCheck::registerMatchers(MatchFinder *Finder) {
  auto InitBase =
      anyOf(stringLiteral(), characterLiteral(), integerLiteral(),
            unaryOperator(hasAnyOperatorName("+", "-"),
                          hasUnaryOperand(integerLiteral())),
            floatLiteral(),
            unaryOperator(hasAnyOperatorName("+", "-"),
                          hasUnaryOperand(floatLiteral())),
            cxxBoolLiteral(), cxxNullPtrLiteralExpr(), implicitValueInitExpr(),
            declRefExpr(to(enumConstantDecl())));

  auto Init =
      anyOf(initListExpr(anyOf(
                allOf(initCountIs(1), hasInit(0, ignoringImplicit(InitBase))),
                initCountIs(0))),
            InitBase);

  Finder->addMatcher(
      cxxConstructorDecl(
          isDefaultConstructor(), unless(isInstantiated()),
          forEachConstructorInitializer(
              cxxCtorInitializer(
                  forField(unless(anyOf(getLangOpts().CPlusPlus20
                                            ? unless(anything())
                                            : isBitField(),
                                        hasInClassInitializer(anything()),
                                        hasParent(recordDecl(isUnion()))))),
                  isWritten(), withInitializer(ignoringImplicit(Init)))
                  .bind("default"))),
      this);

  Finder->addMatcher(
      cxxConstructorDecl(
          unless(ast_matchers::isTemplateInstantiation()),
          forEachConstructorInitializer(
              cxxCtorInitializer(forField(hasInClassInitializer(anything())),
                                 isWritten(),
                                 withInitializer(ignoringImplicit(Init)))
                  .bind("existing"))),
      this);
}

void UseDefaultMemberInitCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *Default =
          Result.Nodes.getNodeAs<CXXCtorInitializer>("default"))
    checkDefaultInit(Result, Default);
  else if (const auto *Existing =
               Result.Nodes.getNodeAs<CXXCtorInitializer>("existing"))
    checkExistingInit(Result, Existing);
  else
    llvm_unreachable("Bad Callback. No node provided.");
}

void UseDefaultMemberInitCheck::checkDefaultInit(
    const MatchFinder::MatchResult &Result, const CXXCtorInitializer *Init) {
  const FieldDecl *Field = Init->getAnyMember();

  SourceLocation StartLoc = Field->getBeginLoc();
  if (StartLoc.isMacroID() && IgnoreMacros)
    return;

  SourceLocation FieldEnd =
      Lexer::getLocForEndOfToken(Field->getSourceRange().getEnd(), 0,
                                 *Result.SourceManager, getLangOpts());
  SourceLocation LParenEnd = Lexer::getLocForEndOfToken(
      Init->getLParenLoc(), 0, *Result.SourceManager, getLangOpts());
  CharSourceRange InitRange =
      CharSourceRange::getCharRange(LParenEnd, Init->getRParenLoc());

  bool ValueInit = isa<ImplicitValueInitExpr>(Init->getInit());
  bool CanAssign = UseAssignment && (!ValueInit || !Init->getInit()->getType()->isEnumeralType());

  auto Diag =
      diag(Field->getLocation(), "use default member initializer for %0")
      << Field
      << FixItHint::CreateInsertion(FieldEnd, CanAssign ? " = " : "{")
      << FixItHint::CreateInsertionFromRange(FieldEnd, InitRange);

  if (CanAssign && ValueInit)
    Diag << FixItHint::CreateInsertion(
        FieldEnd, getValueOfValueInit(Init->getInit()->getType()));

  if (!CanAssign)
    Diag << FixItHint::CreateInsertion(FieldEnd, "}");

  Diag << FixItHint::CreateRemoval(Init->getSourceRange());
}

void UseDefaultMemberInitCheck::checkExistingInit(
    const MatchFinder::MatchResult &Result, const CXXCtorInitializer *Init) {
  const FieldDecl *Field = Init->getAnyMember();

  if (!sameValue(Field->getInClassInitializer(), Init->getInit()))
    return;

  diag(Init->getSourceLocation(), "member initializer for %0 is redundant")
      << Field
      << FixItHint::CreateRemoval(Init->getSourceRange());
}

} // namespace modernize
} // namespace tidy
} // namespace clang
