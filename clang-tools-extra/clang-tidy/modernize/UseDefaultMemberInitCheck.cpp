//===--- UseDefaultMemberInitCheck.cpp - clang-tidy------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
    switch (InitType->getAs<BuiltinType>()->getKind()) {
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
    switch (InitType->getAs<BuiltinType>()->getKind()) {
    case BuiltinType::Half:
    case BuiltinType::Float:
      return "0.0f";
    default:
      return "0.0";
    }

  case Type::STK_FloatingComplex:
  case Type::STK_IntegralComplex:
    return getValueOfValueInit(
        InitType->getAs<ComplexType>()->getElementType());
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
    return InitList->getInit(0);
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
      UseAssignment(Options.get("UseAssignment", 0) != 0),
      IgnoreMacros(Options.getLocalOrGlobal("IgnoreMacros", true) != 0) {}

void UseDefaultMemberInitCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "UseAssignment", UseAssignment);
  Options.store(Opts, "IgnoreMacros", IgnoreMacros);
}

void UseDefaultMemberInitCheck::registerMatchers(MatchFinder *Finder) {
  if (!getLangOpts().CPlusPlus11)
    return;

  auto Init =
      anyOf(stringLiteral(), characterLiteral(), integerLiteral(),
            unaryOperator(anyOf(hasOperatorName("+"), hasOperatorName("-")),
                          hasUnaryOperand(integerLiteral())),
            floatLiteral(),
            unaryOperator(anyOf(hasOperatorName("+"), hasOperatorName("-")),
                          hasUnaryOperand(floatLiteral())),
            cxxBoolLiteral(), cxxNullPtrLiteralExpr(), implicitValueInitExpr(),
            declRefExpr(to(enumConstantDecl())));

  Finder->addMatcher(
      cxxConstructorDecl(
          isDefaultConstructor(), unless(isInstantiated()),
          forEachConstructorInitializer(
              cxxCtorInitializer(
                  forField(unless(anyOf(getLangOpts().CPlusPlus2a
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

  auto Diag =
      diag(Field->getLocation(), "use default member initializer for %0")
      << Field
      << FixItHint::CreateInsertion(FieldEnd, UseAssignment ? " = " : "{")
      << FixItHint::CreateInsertionFromRange(FieldEnd, InitRange);

  if (UseAssignment && isa<ImplicitValueInitExpr>(Init->getInit()))
    Diag << FixItHint::CreateInsertion(
        FieldEnd, getValueOfValueInit(Init->getInit()->getType()));

  if (!UseAssignment)
    Diag << FixItHint::CreateInsertion(FieldEnd, "}");

  Diag << FixItHint::CreateRemoval(Init->getSourceRange());
}

void UseDefaultMemberInitCheck::checkExistingInit(
    const MatchFinder::MatchResult &Result, const CXXCtorInitializer *Init) {
  const FieldDecl *Field = Init->getMember();

  if (!sameValue(Field->getInClassInitializer(), Init->getInit()))
    return;

  diag(Init->getSourceLocation(), "member initializer for %0 is redundant")
      << Field
      << FixItHint::CreateRemoval(Init->getSourceRange());
}

} // namespace modernize
} // namespace tidy
} // namespace clang
