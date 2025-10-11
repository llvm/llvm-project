//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FixItHintUtils.h"
#include "LexerUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Type.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Tooling/FixIt.h"
#include <optional>

namespace clang::tidy::utils::fixit {

FixItHint changeVarDeclToReference(const VarDecl &Var, ASTContext &Context) {
  SourceLocation AmpLocation = Var.getLocation();
  auto Token = utils::lexer::getPreviousToken(
      AmpLocation, Context.getSourceManager(), Context.getLangOpts());
  if (!Token.is(tok::unknown))
    AmpLocation = Lexer::getLocForEndOfToken(Token.getLocation(), 0,
                                             Context.getSourceManager(),
                                             Context.getLangOpts());
  return FixItHint::CreateInsertion(AmpLocation, "&");
}

static bool isValueType(const Type *T) {
  return !(isa<PointerType>(T) || isa<ReferenceType>(T) || isa<ArrayType>(T) ||
           isa<MemberPointerType>(T) || isa<ObjCObjectPointerType>(T));
}
static bool isValueType(QualType QT) { return isValueType(QT.getTypePtr()); }
static bool isMemberOrFunctionPointer(QualType QT) {
  return (QT->isPointerType() && QT->isFunctionPointerType()) ||
         isa<MemberPointerType>(QT.getTypePtr());
}

static bool locDangerous(SourceLocation S) {
  return S.isInvalid() || S.isMacroID();
}

static std::optional<SourceLocation>
skipLParensBackwards(SourceLocation Start, const ASTContext &Context) {
  if (locDangerous(Start))
    return std::nullopt;

  auto PreviousTokenLParen = [&Start, &Context]() {
    Token T;
    T = lexer::getPreviousToken(Start, Context.getSourceManager(),
                                Context.getLangOpts());
    return T.is(tok::l_paren);
  };

  while (Start.isValid() && PreviousTokenLParen())
    Start = lexer::findPreviousTokenStart(Start, Context.getSourceManager(),
                                          Context.getLangOpts());

  if (locDangerous(Start))
    return std::nullopt;
  return Start;
}

static std::optional<FixItHint> fixIfNotDangerous(SourceLocation Loc,
                                                  StringRef Text) {
  if (locDangerous(Loc))
    return std::nullopt;
  return FixItHint::CreateInsertion(Loc, Text);
}

// Build a string that can be emitted as FixIt with either a space in before
// or after the qualifier, either ' const' or 'const '.
static std::string buildQualifier(Qualifiers::TQ Qualifier,
                                  bool WhitespaceBefore = false) {
  if (WhitespaceBefore)
    return (llvm::Twine(' ') + Qualifiers::fromCVRMask(Qualifier).getAsString())
        .str();
  return (llvm::Twine(Qualifiers::fromCVRMask(Qualifier).getAsString()) + " ")
      .str();
}

static std::optional<FixItHint> changeValue(const VarDecl &Var,
                                            Qualifiers::TQ Qualifier,
                                            QualifierTarget QualTarget,
                                            QualifierPolicy QualPolicy,
                                            const ASTContext &Context) {
  switch (QualPolicy) {
  case QualifierPolicy::Left:
    return fixIfNotDangerous(Var.getTypeSpecStartLoc(),
                             buildQualifier(Qualifier));
  case QualifierPolicy::Right:
    std::optional<SourceLocation> IgnoredParens =
        skipLParensBackwards(Var.getLocation(), Context);

    if (IgnoredParens)
      return fixIfNotDangerous(*IgnoredParens, buildQualifier(Qualifier));
    return std::nullopt;
  }
  llvm_unreachable("Unknown QualifierPolicy enum");
}

static std::optional<FixItHint> changePointerItself(const VarDecl &Var,
                                                    Qualifiers::TQ Qualifier,
                                                    const ASTContext &Context) {
  if (locDangerous(Var.getLocation()))
    return std::nullopt;

  std::optional<SourceLocation> IgnoredParens =
      skipLParensBackwards(Var.getLocation(), Context);
  if (IgnoredParens)
    return fixIfNotDangerous(*IgnoredParens, buildQualifier(Qualifier));
  return std::nullopt;
}

static std::optional<FixItHint>
changePointer(const VarDecl &Var, Qualifiers::TQ Qualifier, const Type *Pointee,
              QualifierTarget QualTarget, QualifierPolicy QualPolicy,
              const ASTContext &Context) {
  // The pointer itself shall be marked as `const`. This is always to the right
  // of the '*' or in front of the identifier.
  if (QualTarget == QualifierTarget::Value)
    return changePointerItself(Var, Qualifier, Context);

  // Mark the pointee `const` that is a normal value (`int* p = nullptr;`).
  if (QualTarget == QualifierTarget::Pointee && isValueType(Pointee)) {
    // Adding the `const` on the left side is just the beginning of the type
    // specification. (`const int* p = nullptr;`)
    if (QualPolicy == QualifierPolicy::Left)
      return fixIfNotDangerous(Var.getTypeSpecStartLoc(),
                               buildQualifier(Qualifier));

    // Adding the `const` on the right side of the value type requires finding
    // the `*` token and placing the `const` left of it.
    // (`int const* p = nullptr;`)
    if (QualPolicy == QualifierPolicy::Right) {
      SourceLocation BeforeStar = lexer::findPreviousTokenKind(
          Var.getLocation(), Context.getSourceManager(), Context.getLangOpts(),
          tok::star);
      if (locDangerous(BeforeStar))
        return std::nullopt;

      std::optional<SourceLocation> IgnoredParens =
          skipLParensBackwards(BeforeStar, Context);

      if (IgnoredParens)
        return fixIfNotDangerous(*IgnoredParens,
                                 buildQualifier(Qualifier, true));
      return std::nullopt;
    }
  }

  if (QualTarget == QualifierTarget::Pointee && Pointee->isPointerType()) {
    // Adding the `const` to the pointee if the pointee is a pointer
    // is the same as 'QualPolicy == Right && isValueType(Pointee)'.
    // The `const` must be left of the last `*` token.
    // (`int * const* p = nullptr;`)
    SourceLocation BeforeStar = lexer::findPreviousTokenKind(
        Var.getLocation(), Context.getSourceManager(), Context.getLangOpts(),
        tok::star);
    return fixIfNotDangerous(BeforeStar, buildQualifier(Qualifier, true));
  }

  return std::nullopt;
}

static std::optional<FixItHint>
changeReferencee(const VarDecl &Var, Qualifiers::TQ Qualifier, QualType Pointee,
                 QualifierTarget QualTarget, QualifierPolicy QualPolicy,
                 const ASTContext &Context) {
  if (QualPolicy == QualifierPolicy::Left && isValueType(Pointee))
    return fixIfNotDangerous(Var.getTypeSpecStartLoc(),
                             buildQualifier(Qualifier));

  SourceLocation BeforeRef = lexer::findPreviousAnyTokenKind(
      Var.getLocation(), Context.getSourceManager(), Context.getLangOpts(),
      tok::amp, tok::ampamp);
  std::optional<SourceLocation> IgnoredParens =
      skipLParensBackwards(BeforeRef, Context);
  if (IgnoredParens)
    return fixIfNotDangerous(*IgnoredParens, buildQualifier(Qualifier, true));

  return std::nullopt;
}

std::optional<FixItHint> addQualifierToVarDecl(const VarDecl &Var,
                                               const ASTContext &Context,
                                               Qualifiers::TQ Qualifier,
                                               QualifierTarget QualTarget,
                                               QualifierPolicy QualPolicy) {
  assert((QualPolicy == QualifierPolicy::Left ||
          QualPolicy == QualifierPolicy::Right) &&
         "Unexpected Insertion Policy");
  assert((QualTarget == QualifierTarget::Pointee ||
          QualTarget == QualifierTarget::Value) &&
         "Unexpected Target");

  QualType ParenStrippedType = Var.getType().IgnoreParens();
  if (isValueType(ParenStrippedType))
    return changeValue(Var, Qualifier, QualTarget, QualPolicy, Context);

  if (ParenStrippedType->isReferenceType())
    return changeReferencee(Var, Qualifier, Var.getType()->getPointeeType(),
                            QualTarget, QualPolicy, Context);

  if (isMemberOrFunctionPointer(ParenStrippedType))
    return changePointerItself(Var, Qualifier, Context);

  if (ParenStrippedType->isPointerType())
    return changePointer(Var, Qualifier,
                         ParenStrippedType->getPointeeType().getTypePtr(),
                         QualTarget, QualPolicy, Context);

  if (ParenStrippedType->isArrayType()) {
    const Type *AT = ParenStrippedType->getBaseElementTypeUnsafe();
    assert(AT && "Did not retrieve array element type for an array.");

    if (isValueType(AT))
      return changeValue(Var, Qualifier, QualTarget, QualPolicy, Context);

    if (AT->isPointerType())
      return changePointer(Var, Qualifier, AT->getPointeeType().getTypePtr(),
                           QualTarget, QualPolicy, Context);
  }

  return std::nullopt;
}

bool areParensNeededForStatement(const Stmt &Node) {
  if (isa<ParenExpr>(&Node))
    return false;

  if (isa<clang::BinaryOperator>(&Node) || isa<UnaryOperator>(&Node))
    return true;

  if (isa<clang::ConditionalOperator>(&Node) ||
      isa<BinaryConditionalOperator>(&Node))
    return true;

  if (const auto *Op = dyn_cast<CXXOperatorCallExpr>(&Node)) {
    switch (Op->getOperator()) {
    case OO_PlusPlus:
      [[fallthrough]];
    case OO_MinusMinus:
      return Op->getNumArgs() != 2;
    case OO_Call:
      [[fallthrough]];
    case OO_Subscript:
      [[fallthrough]];
    case OO_Arrow:
      return false;
    default:
      return true;
    };
  }

  if (isa<CStyleCastExpr>(&Node))
    return true;

  return false;
}

// Return true if expr needs to be put in parens when it is an argument of a
// prefix unary operator, e.g. when it is a binary or ternary operator
// syntactically.
static bool needParensAfterUnaryOperator(const Expr &ExprNode) {
  if (isa<clang::BinaryOperator>(&ExprNode) ||
      isa<clang::ConditionalOperator>(&ExprNode)) {
    return true;
  }
  if (const auto *Op = dyn_cast<CXXOperatorCallExpr>(&ExprNode)) {
    return Op->getNumArgs() == 2 && Op->getOperator() != OO_PlusPlus &&
           Op->getOperator() != OO_MinusMinus && Op->getOperator() != OO_Call &&
           Op->getOperator() != OO_Subscript;
  }
  return false;
}

// Format a pointer to an expression: prefix with '*' but simplify
// when it already begins with '&'.  Return empty string on failure.
std::string formatDereference(const Expr &ExprNode, const ASTContext &Context) {
  if (const auto *Op = dyn_cast<clang::UnaryOperator>(&ExprNode)) {
    if (Op->getOpcode() == UO_AddrOf) {
      // Strip leading '&'.
      return std::string(
          tooling::fixit::getText(*Op->getSubExpr()->IgnoreParens(), Context));
    }
  }
  StringRef Text = tooling::fixit::getText(ExprNode, Context);

  if (Text.empty())
    return {};

  // Remove remaining '->' from overloaded operator call
  Text.consume_back("->");

  // Add leading '*'.
  if (needParensAfterUnaryOperator(ExprNode)) {
    return (llvm::Twine("*(") + Text + ")").str();
  }
  return (llvm::Twine("*") + Text).str();
}

} // namespace clang::tidy::utils::fixit
