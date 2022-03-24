//===--- RedundantExpressionCheck.cpp - clang-tidy-------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RedundantExpressionCheck.h"
#include "../utils/Matchers.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

using namespace clang::ast_matchers;
using namespace clang::tidy::matchers;

namespace clang {
namespace tidy {
namespace misc {
namespace {
using llvm::APSInt;

static constexpr llvm::StringLiteral KnownBannedMacroNames[] = {
    "EAGAIN",
    "EWOULDBLOCK",
    "SIGCLD",
    "SIGCHLD",
};

static bool incrementWithoutOverflow(const APSInt &Value, APSInt &Result) {
  Result = Value;
  ++Result;
  return Value < Result;
}

static bool areEquivalentNameSpecifier(const NestedNameSpecifier *Left,
                                       const NestedNameSpecifier *Right) {
  llvm::FoldingSetNodeID LeftID, RightID;
  Left->Profile(LeftID);
  Right->Profile(RightID);
  return LeftID == RightID;
}

static bool areEquivalentExpr(const Expr *Left, const Expr *Right) {
  if (!Left || !Right)
    return !Left && !Right;

  Left = Left->IgnoreParens();
  Right = Right->IgnoreParens();

  // Compare classes.
  if (Left->getStmtClass() != Right->getStmtClass())
    return false;

  // Compare children.
  Expr::const_child_iterator LeftIter = Left->child_begin();
  Expr::const_child_iterator RightIter = Right->child_begin();
  while (LeftIter != Left->child_end() && RightIter != Right->child_end()) {
    if (!areEquivalentExpr(dyn_cast_or_null<Expr>(*LeftIter),
                           dyn_cast_or_null<Expr>(*RightIter)))
      return false;
    ++LeftIter;
    ++RightIter;
  }
  if (LeftIter != Left->child_end() || RightIter != Right->child_end())
    return false;

  // Perform extra checks.
  switch (Left->getStmtClass()) {
  default:
    return false;

  case Stmt::CharacterLiteralClass:
    return cast<CharacterLiteral>(Left)->getValue() ==
           cast<CharacterLiteral>(Right)->getValue();
  case Stmt::IntegerLiteralClass: {
    llvm::APInt LeftLit = cast<IntegerLiteral>(Left)->getValue();
    llvm::APInt RightLit = cast<IntegerLiteral>(Right)->getValue();
    return LeftLit.getBitWidth() == RightLit.getBitWidth() &&
           LeftLit == RightLit;
  }
  case Stmt::FloatingLiteralClass:
    return cast<FloatingLiteral>(Left)->getValue().bitwiseIsEqual(
        cast<FloatingLiteral>(Right)->getValue());
  case Stmt::StringLiteralClass:
    return cast<StringLiteral>(Left)->getBytes() ==
           cast<StringLiteral>(Right)->getBytes();
  case Stmt::CXXOperatorCallExprClass:
    return cast<CXXOperatorCallExpr>(Left)->getOperator() ==
           cast<CXXOperatorCallExpr>(Right)->getOperator();
  case Stmt::DependentScopeDeclRefExprClass:
    if (cast<DependentScopeDeclRefExpr>(Left)->getDeclName() !=
        cast<DependentScopeDeclRefExpr>(Right)->getDeclName())
      return false;
    return areEquivalentNameSpecifier(
        cast<DependentScopeDeclRefExpr>(Left)->getQualifier(),
        cast<DependentScopeDeclRefExpr>(Right)->getQualifier());
  case Stmt::DeclRefExprClass:
    return cast<DeclRefExpr>(Left)->getDecl() ==
           cast<DeclRefExpr>(Right)->getDecl();
  case Stmt::MemberExprClass:
    return cast<MemberExpr>(Left)->getMemberDecl() ==
           cast<MemberExpr>(Right)->getMemberDecl();
  case Stmt::CXXFoldExprClass:
    return cast<CXXFoldExpr>(Left)->getOperator() ==
           cast<CXXFoldExpr>(Right)->getOperator();
  case Stmt::CXXFunctionalCastExprClass:
  case Stmt::CStyleCastExprClass:
    return cast<ExplicitCastExpr>(Left)->getTypeAsWritten() ==
           cast<ExplicitCastExpr>(Right)->getTypeAsWritten();
  case Stmt::CallExprClass:
  case Stmt::ImplicitCastExprClass:
  case Stmt::ArraySubscriptExprClass:
    return true;
  case Stmt::UnaryOperatorClass:
    if (cast<UnaryOperator>(Left)->isIncrementDecrementOp())
      return false;
    return cast<UnaryOperator>(Left)->getOpcode() ==
           cast<UnaryOperator>(Right)->getOpcode();
  case Stmt::BinaryOperatorClass:
    return cast<BinaryOperator>(Left)->getOpcode() ==
           cast<BinaryOperator>(Right)->getOpcode();
  case Stmt::UnaryExprOrTypeTraitExprClass:
    const auto *LeftUnaryExpr =
        cast<UnaryExprOrTypeTraitExpr>(Left);
    const auto *RightUnaryExpr =
        cast<UnaryExprOrTypeTraitExpr>(Right);
    if (LeftUnaryExpr->isArgumentType() && RightUnaryExpr->isArgumentType())
      return LeftUnaryExpr->getArgumentType() ==
             RightUnaryExpr->getArgumentType();
    if (!LeftUnaryExpr->isArgumentType() && !RightUnaryExpr->isArgumentType())
      return areEquivalentExpr(LeftUnaryExpr->getArgumentExpr(),
                               RightUnaryExpr->getArgumentExpr());

    return false;
  }
}

// For a given expression 'x', returns whether the ranges covered by the
// relational operators are equivalent (i.e.  x <= 4 is equivalent to x < 5).
static bool areEquivalentRanges(BinaryOperatorKind OpcodeLHS,
                                const APSInt &ValueLHS,
                                BinaryOperatorKind OpcodeRHS,
                                const APSInt &ValueRHS) {
  assert(APSInt::compareValues(ValueLHS, ValueRHS) <= 0 &&
         "Values must be ordered");
  // Handle the case where constants are the same: x <= 4  <==>  x <= 4.
  if (APSInt::compareValues(ValueLHS, ValueRHS) == 0)
    return OpcodeLHS == OpcodeRHS;

  // Handle the case where constants are off by one: x <= 4  <==>  x < 5.
  APSInt ValueLhsPlus1;
  return ((OpcodeLHS == BO_LE && OpcodeRHS == BO_LT) ||
          (OpcodeLHS == BO_GT && OpcodeRHS == BO_GE)) &&
         incrementWithoutOverflow(ValueLHS, ValueLhsPlus1) &&
         APSInt::compareValues(ValueLhsPlus1, ValueRHS) == 0;
}

// For a given expression 'x', returns whether the ranges covered by the
// relational operators are fully disjoint (i.e. x < 4  and  x > 7).
static bool areExclusiveRanges(BinaryOperatorKind OpcodeLHS,
                               const APSInt &ValueLHS,
                               BinaryOperatorKind OpcodeRHS,
                               const APSInt &ValueRHS) {
  assert(APSInt::compareValues(ValueLHS, ValueRHS) <= 0 &&
         "Values must be ordered");

  // Handle cases where the constants are the same.
  if (APSInt::compareValues(ValueLHS, ValueRHS) == 0) {
    switch (OpcodeLHS) {
    case BO_EQ:
      return OpcodeRHS == BO_NE || OpcodeRHS == BO_GT || OpcodeRHS == BO_LT;
    case BO_NE:
      return OpcodeRHS == BO_EQ;
    case BO_LE:
      return OpcodeRHS == BO_GT;
    case BO_GE:
      return OpcodeRHS == BO_LT;
    case BO_LT:
      return OpcodeRHS == BO_EQ || OpcodeRHS == BO_GT || OpcodeRHS == BO_GE;
    case BO_GT:
      return OpcodeRHS == BO_EQ || OpcodeRHS == BO_LT || OpcodeRHS == BO_LE;
    default:
      return false;
    }
  }

  // Handle cases where the constants are different.
  if ((OpcodeLHS == BO_EQ || OpcodeLHS == BO_LT || OpcodeLHS == BO_LE) &&
      (OpcodeRHS == BO_EQ || OpcodeRHS == BO_GT || OpcodeRHS == BO_GE))
    return true;

  // Handle the case where constants are off by one: x > 5 && x < 6.
  APSInt ValueLhsPlus1;
  if (OpcodeLHS == BO_GT && OpcodeRHS == BO_LT &&
      incrementWithoutOverflow(ValueLHS, ValueLhsPlus1) &&
      APSInt::compareValues(ValueLhsPlus1, ValueRHS) == 0)
    return true;

  return false;
}

// Returns whether the ranges covered by the union of both relational
// expressions cover the whole domain (i.e. x < 10  and  x > 0).
static bool rangesFullyCoverDomain(BinaryOperatorKind OpcodeLHS,
                                   const APSInt &ValueLHS,
                                   BinaryOperatorKind OpcodeRHS,
                                   const APSInt &ValueRHS) {
  assert(APSInt::compareValues(ValueLHS, ValueRHS) <= 0 &&
         "Values must be ordered");

  // Handle cases where the constants are the same:  x < 5 || x >= 5.
  if (APSInt::compareValues(ValueLHS, ValueRHS) == 0) {
    switch (OpcodeLHS) {
    case BO_EQ:
      return OpcodeRHS == BO_NE;
    case BO_NE:
      return OpcodeRHS == BO_EQ;
    case BO_LE:
      return OpcodeRHS == BO_GT || OpcodeRHS == BO_GE;
    case BO_LT:
      return OpcodeRHS == BO_GE;
    case BO_GE:
      return OpcodeRHS == BO_LT || OpcodeRHS == BO_LE;
    case BO_GT:
      return OpcodeRHS == BO_LE;
    default:
      return false;
    }
  }

  // Handle the case where constants are off by one: x <= 4 || x >= 5.
  APSInt ValueLhsPlus1;
  if (OpcodeLHS == BO_LE && OpcodeRHS == BO_GE &&
      incrementWithoutOverflow(ValueLHS, ValueLhsPlus1) &&
      APSInt::compareValues(ValueLhsPlus1, ValueRHS) == 0)
    return true;

  // Handle cases where the constants are different: x > 4 || x <= 7.
  if ((OpcodeLHS == BO_GT || OpcodeLHS == BO_GE) &&
      (OpcodeRHS == BO_LT || OpcodeRHS == BO_LE))
    return true;

  // Handle cases where constants are different but both ops are !=, like:
  // x != 5 || x != 10
  if (OpcodeLHS == BO_NE && OpcodeRHS == BO_NE)
    return true;

  return false;
}

static bool rangeSubsumesRange(BinaryOperatorKind OpcodeLHS,
                               const APSInt &ValueLHS,
                               BinaryOperatorKind OpcodeRHS,
                               const APSInt &ValueRHS) {
  int Comparison = APSInt::compareValues(ValueLHS, ValueRHS);
  switch (OpcodeLHS) {
  case BO_EQ:
    return OpcodeRHS == BO_EQ && Comparison == 0;
  case BO_NE:
    return (OpcodeRHS == BO_NE && Comparison == 0) ||
           (OpcodeRHS == BO_EQ && Comparison != 0) ||
           (OpcodeRHS == BO_LT && Comparison >= 0) ||
           (OpcodeRHS == BO_LE && Comparison > 0) ||
           (OpcodeRHS == BO_GT && Comparison <= 0) ||
           (OpcodeRHS == BO_GE && Comparison < 0);

  case BO_LT:
    return ((OpcodeRHS == BO_LT && Comparison >= 0) ||
            (OpcodeRHS == BO_LE && Comparison > 0) ||
            (OpcodeRHS == BO_EQ && Comparison > 0));
  case BO_GT:
    return ((OpcodeRHS == BO_GT && Comparison <= 0) ||
            (OpcodeRHS == BO_GE && Comparison < 0) ||
            (OpcodeRHS == BO_EQ && Comparison < 0));
  case BO_LE:
    return (OpcodeRHS == BO_LT || OpcodeRHS == BO_LE || OpcodeRHS == BO_EQ) &&
           Comparison >= 0;
  case BO_GE:
    return (OpcodeRHS == BO_GT || OpcodeRHS == BO_GE || OpcodeRHS == BO_EQ) &&
           Comparison <= 0;
  default:
    return false;
  }
}

static void transformSubToCanonicalAddExpr(BinaryOperatorKind &Opcode,
                                           APSInt &Value) {
  if (Opcode == BO_Sub) {
    Opcode = BO_Add;
    Value = -Value;
  }
}

// to use in the template below
static OverloadedOperatorKind getOp(const BinaryOperator *Op) {
  return BinaryOperator::getOverloadedOperator(Op->getOpcode());
}

static OverloadedOperatorKind getOp(const CXXOperatorCallExpr *Op) {
  if (Op->getNumArgs() != 2)
    return OO_None;
  return Op->getOperator();
}

static std::pair<const Expr *, const Expr *>
getOperands(const BinaryOperator *Op) {
  return {Op->getLHS()->IgnoreParenImpCasts(),
          Op->getRHS()->IgnoreParenImpCasts()};
}

static std::pair<const Expr *, const Expr *>
getOperands(const CXXOperatorCallExpr *Op) {
  return {Op->getArg(0)->IgnoreParenImpCasts(),
          Op->getArg(1)->IgnoreParenImpCasts()};
}

template <typename TExpr>
static const TExpr *checkOpKind(const Expr *TheExpr,
                                OverloadedOperatorKind OpKind) {
  const auto *AsTExpr = dyn_cast_or_null<TExpr>(TheExpr);
  if (AsTExpr && getOp(AsTExpr) == OpKind)
    return AsTExpr;

  return nullptr;
}

// returns true if a subexpression has two directly equivalent operands and
// is already handled by operands/parametersAreEquivalent
template <typename TExpr, unsigned N>
static bool collectOperands(const Expr *Part,
                            SmallVector<const Expr *, N> &AllOperands,
                            OverloadedOperatorKind OpKind) {
  if (const auto *BinOp = checkOpKind<TExpr>(Part, OpKind)) {
    const std::pair<const Expr *, const Expr *> Operands = getOperands(BinOp);
    if (areEquivalentExpr(Operands.first, Operands.second))
      return true;
    return collectOperands<TExpr>(Operands.first, AllOperands, OpKind) ||
           collectOperands<TExpr>(Operands.second, AllOperands, OpKind);
  }

  AllOperands.push_back(Part);
  return false;
}

template <typename TExpr>
static bool hasSameOperatorParent(const Expr *TheExpr,
                                  OverloadedOperatorKind OpKind,
                                  ASTContext &Context) {
  // IgnoreParenImpCasts logic in reverse: skip surrounding uninteresting nodes
  const DynTypedNodeList Parents = Context.getParents(*TheExpr);
  for (DynTypedNode DynParent : Parents) {
    if (const auto *Parent = DynParent.get<Expr>()) {
      bool Skip = isa<ParenExpr>(Parent) || isa<ImplicitCastExpr>(Parent) ||
                  isa<FullExpr>(Parent) ||
                  isa<MaterializeTemporaryExpr>(Parent);
      if (Skip && hasSameOperatorParent<TExpr>(Parent, OpKind, Context))
        return true;
      if (checkOpKind<TExpr>(Parent, OpKind))
        return true;
    }
  }

  return false;
}

template <typename TExpr>
static bool
markDuplicateOperands(const TExpr *TheExpr,
                      ast_matchers::internal::BoundNodesTreeBuilder *Builder,
                      ASTContext &Context) {
  const OverloadedOperatorKind OpKind = getOp(TheExpr);
  if (OpKind == OO_None)
    return false;
  // if there are no nested operators of the same kind, it's handled by
  // operands/parametersAreEquivalent
  const std::pair<const Expr *, const Expr *> Operands = getOperands(TheExpr);
  if (!(checkOpKind<TExpr>(Operands.first, OpKind) ||
        checkOpKind<TExpr>(Operands.second, OpKind)))
    return false;

  // if parent is the same kind of operator, it's handled by a previous call to
  // markDuplicateOperands
  if (hasSameOperatorParent<TExpr>(TheExpr, OpKind, Context))
    return false;

  SmallVector<const Expr *, 4> AllOperands;
  if (collectOperands<TExpr>(Operands.first, AllOperands, OpKind))
    return false;
  if (collectOperands<TExpr>(Operands.second, AllOperands, OpKind))
    return false;
  size_t NumOperands = AllOperands.size();
  llvm::SmallBitVector Duplicates(NumOperands);
  for (size_t I = 0; I < NumOperands; I++) {
    if (Duplicates[I])
      continue;
    bool FoundDuplicates = false;

    for (size_t J = I + 1; J < NumOperands; J++) {
      if (AllOperands[J]->HasSideEffects(Context))
        break;

      if (areEquivalentExpr(AllOperands[I], AllOperands[J])) {
        FoundDuplicates = true;
        Duplicates.set(J);
        Builder->setBinding(SmallString<11>(llvm::formatv("duplicate{0}", J)),
                            DynTypedNode::create(*AllOperands[J]));
      }
    }

    if (FoundDuplicates)
      Builder->setBinding(SmallString<11>(llvm::formatv("duplicate{0}", I)),
                          DynTypedNode::create(*AllOperands[I]));
  }

  return Duplicates.any();
}

AST_MATCHER(Expr, isIntegerConstantExpr) {
  if (Node.isInstantiationDependent())
    return false;
  return Node.isIntegerConstantExpr(Finder->getASTContext());
}

AST_MATCHER(BinaryOperator, operandsAreEquivalent) {
  return areEquivalentExpr(Node.getLHS(), Node.getRHS());
}

AST_MATCHER(BinaryOperator, nestedOperandsAreEquivalent) {
  return markDuplicateOperands(&Node, Builder, Finder->getASTContext());
}

AST_MATCHER(ConditionalOperator, expressionsAreEquivalent) {
  return areEquivalentExpr(Node.getTrueExpr(), Node.getFalseExpr());
}

AST_MATCHER(CallExpr, parametersAreEquivalent) {
  return Node.getNumArgs() == 2 &&
         areEquivalentExpr(Node.getArg(0), Node.getArg(1));
}

AST_MATCHER(CXXOperatorCallExpr, nestedParametersAreEquivalent) {
  return markDuplicateOperands(&Node, Builder, Finder->getASTContext());
}

AST_MATCHER(BinaryOperator, binaryOperatorIsInMacro) {
  return Node.getOperatorLoc().isMacroID();
}

AST_MATCHER(ConditionalOperator, conditionalOperatorIsInMacro) {
  return Node.getQuestionLoc().isMacroID() || Node.getColonLoc().isMacroID();
}

AST_MATCHER(Expr, isMacro) { return Node.getExprLoc().isMacroID(); }

AST_MATCHER_P(Expr, expandedByMacro, ArrayRef<llvm::StringLiteral>, Names) {
  const SourceManager &SM = Finder->getASTContext().getSourceManager();
  const LangOptions &LO = Finder->getASTContext().getLangOpts();
  SourceLocation Loc = Node.getExprLoc();
  while (Loc.isMacroID()) {
    StringRef MacroName = Lexer::getImmediateMacroName(Loc, SM, LO);
    if (llvm::is_contained(Names, MacroName))
      return true;
    Loc = SM.getImmediateMacroCallerLoc(Loc);
  }
  return false;
}

// Returns a matcher for integer constant expressions.
static ast_matchers::internal::Matcher<Expr>
matchIntegerConstantExpr(StringRef Id) {
  std::string CstId = (Id + "-const").str();
  return expr(isIntegerConstantExpr()).bind(CstId);
}

// Retrieves the integer expression matched by 'matchIntegerConstantExpr' with
// name 'Id' and stores it into 'ConstExpr', the value of the expression is
// stored into `Value`.
static bool retrieveIntegerConstantExpr(const MatchFinder::MatchResult &Result,
                                        StringRef Id, APSInt &Value,
                                        const Expr *&ConstExpr) {
  std::string CstId = (Id + "-const").str();
  ConstExpr = Result.Nodes.getNodeAs<Expr>(CstId);
  if (!ConstExpr)
    return false;
  Optional<llvm::APSInt> R = ConstExpr->getIntegerConstantExpr(*Result.Context);
  if (!R)
    return false;
  Value = *R;
  return true;
}

// Overloaded `retrieveIntegerConstantExpr` for compatibility.
static bool retrieveIntegerConstantExpr(const MatchFinder::MatchResult &Result,
                                        StringRef Id, APSInt &Value) {
  const Expr *ConstExpr = nullptr;
  return retrieveIntegerConstantExpr(Result, Id, Value, ConstExpr);
}

// Returns a matcher for symbolic expressions (matches every expression except
// ingeter constant expressions).
static ast_matchers::internal::Matcher<Expr> matchSymbolicExpr(StringRef Id) {
  std::string SymId = (Id + "-sym").str();
  return ignoringParenImpCasts(
      expr(unless(isIntegerConstantExpr())).bind(SymId));
}

// Retrieves the expression matched by 'matchSymbolicExpr' with name 'Id' and
// stores it into 'SymExpr'.
static bool retrieveSymbolicExpr(const MatchFinder::MatchResult &Result,
                                 StringRef Id, const Expr *&SymExpr) {
  std::string SymId = (Id + "-sym").str();
  if (const auto *Node = Result.Nodes.getNodeAs<Expr>(SymId)) {
    SymExpr = Node;
    return true;
  }
  return false;
}

// Match a binary operator between a symbolic expression and an integer constant
// expression.
static ast_matchers::internal::Matcher<Expr>
matchBinOpIntegerConstantExpr(StringRef Id) {
  const auto BinOpCstExpr =
      expr(anyOf(binaryOperator(hasAnyOperatorName("+", "|", "&"),
                                hasOperands(matchSymbolicExpr(Id),
                                            matchIntegerConstantExpr(Id))),
                 binaryOperator(hasOperatorName("-"),
                                hasLHS(matchSymbolicExpr(Id)),
                                hasRHS(matchIntegerConstantExpr(Id)))))
          .bind(Id);
  return ignoringParenImpCasts(BinOpCstExpr);
}

// Retrieves sub-expressions matched by 'matchBinOpIntegerConstantExpr' with
// name 'Id'.
static bool
retrieveBinOpIntegerConstantExpr(const MatchFinder::MatchResult &Result,
                                 StringRef Id, BinaryOperatorKind &Opcode,
                                 const Expr *&Symbol, APSInt &Value) {
  if (const auto *BinExpr = Result.Nodes.getNodeAs<BinaryOperator>(Id)) {
    Opcode = BinExpr->getOpcode();
    return retrieveSymbolicExpr(Result, Id, Symbol) &&
           retrieveIntegerConstantExpr(Result, Id, Value);
  }
  return false;
}

// Matches relational expressions: 'Expr <op> k' (i.e. x < 2, x != 3, 12 <= x).
static ast_matchers::internal::Matcher<Expr>
matchRelationalIntegerConstantExpr(StringRef Id) {
  std::string CastId = (Id + "-cast").str();
  std::string SwapId = (Id + "-swap").str();
  std::string NegateId = (Id + "-negate").str();
  std::string OverloadId = (Id + "-overload").str();
  std::string ConstId = (Id + "-const").str();

  const auto RelationalExpr = ignoringParenImpCasts(binaryOperator(
      isComparisonOperator(), expr().bind(Id),
      anyOf(allOf(hasLHS(matchSymbolicExpr(Id)),
                  hasRHS(matchIntegerConstantExpr(Id))),
            allOf(hasLHS(matchIntegerConstantExpr(Id)),
                  hasRHS(matchSymbolicExpr(Id)), expr().bind(SwapId)))));

  // A cast can be matched as a comparator to zero. (i.e. if (x) is equivalent
  // to if (x != 0)).
  const auto CastExpr =
      implicitCastExpr(hasCastKind(CK_IntegralToBoolean),
                       hasSourceExpression(matchSymbolicExpr(Id)))
          .bind(CastId);

  const auto NegateRelationalExpr =
      unaryOperator(hasOperatorName("!"),
                    hasUnaryOperand(anyOf(CastExpr, RelationalExpr)))
          .bind(NegateId);

  // Do not bind to double negation.
  const auto NegateNegateRelationalExpr =
      unaryOperator(hasOperatorName("!"),
                    hasUnaryOperand(unaryOperator(
                        hasOperatorName("!"),
                        hasUnaryOperand(anyOf(CastExpr, RelationalExpr)))));

  const auto OverloadedOperatorExpr =
      cxxOperatorCallExpr(
          hasAnyOverloadedOperatorName("==", "!=", "<", "<=", ">", ">="),
          // Filter noisy false positives.
          unless(isMacro()), unless(isInTemplateInstantiation()),
          anyOf(hasLHS(ignoringParenImpCasts(integerLiteral().bind(ConstId))),
                hasRHS(ignoringParenImpCasts(integerLiteral().bind(ConstId)))))
          .bind(OverloadId);

  return anyOf(RelationalExpr, CastExpr, NegateRelationalExpr,
               NegateNegateRelationalExpr, OverloadedOperatorExpr);
}

// Checks whether a function param is non constant reference type, and may
// be modified in the function.
static bool isNonConstReferenceType(QualType ParamType) {
  return ParamType->isReferenceType() &&
         !ParamType.getNonReferenceType().isConstQualified();
}

// Checks whether the arguments of an overloaded operator can be modified in the
// function.
// For operators that take an instance and a constant as arguments, only the
// first argument (the instance) needs to be checked, since the constant itself
// is a temporary expression. Whether the second parameter is checked is
// controlled by the parameter `ParamsToCheckCount`.
static bool
canOverloadedOperatorArgsBeModified(const CXXOperatorCallExpr *OperatorCall,
                                    bool CheckSecondParam) {
  const auto *OperatorDecl =
      dyn_cast_or_null<FunctionDecl>(OperatorCall->getCalleeDecl());
  // if we can't find the declaration, conservatively assume it can modify
  // arguments
  if (!OperatorDecl)
    return true;

  unsigned ParamCount = OperatorDecl->getNumParams();

  // Overloaded operators declared inside a class have only one param.
  // These functions must be declared const in order to not be able to modify
  // the instance of the class they are called through.
  if (ParamCount == 1 &&
      !OperatorDecl->getType()->castAs<FunctionType>()->isConst())
    return true;

  if (isNonConstReferenceType(OperatorDecl->getParamDecl(0)->getType()))
    return true;

  return CheckSecondParam && ParamCount == 2 &&
         isNonConstReferenceType(OperatorDecl->getParamDecl(1)->getType());
}

// Retrieves sub-expressions matched by 'matchRelationalIntegerConstantExpr'
// with name 'Id'.
static bool retrieveRelationalIntegerConstantExpr(
    const MatchFinder::MatchResult &Result, StringRef Id,
    const Expr *&OperandExpr, BinaryOperatorKind &Opcode, const Expr *&Symbol,
    APSInt &Value, const Expr *&ConstExpr) {
  std::string CastId = (Id + "-cast").str();
  std::string SwapId = (Id + "-swap").str();
  std::string NegateId = (Id + "-negate").str();
  std::string OverloadId = (Id + "-overload").str();

  if (const auto *Bin = Result.Nodes.getNodeAs<BinaryOperator>(Id)) {
    // Operand received with explicit comparator.
    Opcode = Bin->getOpcode();
    OperandExpr = Bin;

    if (!retrieveIntegerConstantExpr(Result, Id, Value, ConstExpr))
      return false;
  } else if (const auto *Cast = Result.Nodes.getNodeAs<CastExpr>(CastId)) {
    // Operand received with implicit comparator (cast).
    Opcode = BO_NE;
    OperandExpr = Cast;
    Value = APSInt(32, false);
  } else if (const auto *OverloadedOperatorExpr =
                 Result.Nodes.getNodeAs<CXXOperatorCallExpr>(OverloadId)) {
    if (canOverloadedOperatorArgsBeModified(OverloadedOperatorExpr, false))
      return false;

    bool IntegerConstantIsFirstArg = false;

    if (const auto *Arg = OverloadedOperatorExpr->getArg(1)) {
      if (!Arg->isValueDependent() &&
          !Arg->isIntegerConstantExpr(*Result.Context)) {
        IntegerConstantIsFirstArg = true;
        if (const auto *Arg = OverloadedOperatorExpr->getArg(0)) {
          if (!Arg->isValueDependent() &&
              !Arg->isIntegerConstantExpr(*Result.Context))
            return false;
        } else
          return false;
      }
    } else
      return false;

    Symbol = OverloadedOperatorExpr->getArg(IntegerConstantIsFirstArg ? 1 : 0);
    OperandExpr = OverloadedOperatorExpr;
    Opcode = BinaryOperator::getOverloadedOpcode(OverloadedOperatorExpr->getOperator());

    if (!retrieveIntegerConstantExpr(Result, Id, Value, ConstExpr))
      return false;

    if (!BinaryOperator::isComparisonOp(Opcode))
      return false;

    // The call site of this function expects the constant on the RHS,
    // so change the opcode accordingly.
    if (IntegerConstantIsFirstArg)
      Opcode = BinaryOperator::reverseComparisonOp(Opcode);

    return true;
  } else {
    return false;
  }

  if (!retrieveSymbolicExpr(Result, Id, Symbol))
    return false;

  if (Result.Nodes.getNodeAs<Expr>(SwapId))
    Opcode = BinaryOperator::reverseComparisonOp(Opcode);
  if (Result.Nodes.getNodeAs<Expr>(NegateId))
    Opcode = BinaryOperator::negateComparisonOp(Opcode);
  return true;
}

// Checks for expressions like (X == 4) && (Y != 9)
static bool areSidesBinaryConstExpressions(const BinaryOperator *&BinOp, const ASTContext *AstCtx) {
  const auto *LhsBinOp = dyn_cast<BinaryOperator>(BinOp->getLHS());
  const auto *RhsBinOp = dyn_cast<BinaryOperator>(BinOp->getRHS());

  if (!LhsBinOp || !RhsBinOp)
    return false;

  auto IsIntegerConstantExpr = [AstCtx](const Expr *E) {
    return !E->isValueDependent() && E->isIntegerConstantExpr(*AstCtx);
  };

  if ((IsIntegerConstantExpr(LhsBinOp->getLHS()) ||
       IsIntegerConstantExpr(LhsBinOp->getRHS())) &&
      (IsIntegerConstantExpr(RhsBinOp->getLHS()) ||
       IsIntegerConstantExpr(RhsBinOp->getRHS())))
    return true;
  return false;
}

// Retrieves integer constant subexpressions from binary operator expressions
// that have two equivalent sides.
// E.g.: from (X == 5) && (X == 5) retrieves 5 and 5.
static bool retrieveConstExprFromBothSides(const BinaryOperator *&BinOp,
                                           BinaryOperatorKind &MainOpcode,
                                           BinaryOperatorKind &SideOpcode,
                                           const Expr *&LhsConst,
                                           const Expr *&RhsConst,
                                           const ASTContext *AstCtx) {
  assert(areSidesBinaryConstExpressions(BinOp, AstCtx) &&
         "Both sides of binary operator must be constant expressions!");

  MainOpcode = BinOp->getOpcode();

  const auto *BinOpLhs = cast<BinaryOperator>(BinOp->getLHS());
  const auto *BinOpRhs = cast<BinaryOperator>(BinOp->getRHS());

  auto IsIntegerConstantExpr = [AstCtx](const Expr *E) {
    return !E->isValueDependent() && E->isIntegerConstantExpr(*AstCtx);
  };

  LhsConst = IsIntegerConstantExpr(BinOpLhs->getLHS()) ? BinOpLhs->getLHS()
                                                       : BinOpLhs->getRHS();
  RhsConst = IsIntegerConstantExpr(BinOpRhs->getLHS()) ? BinOpRhs->getLHS()
                                                       : BinOpRhs->getRHS();

  if (!LhsConst || !RhsConst)
    return false;

  assert(BinOpLhs->getOpcode() == BinOpRhs->getOpcode() &&
         "Sides of the binary operator must be equivalent expressions!");

  SideOpcode = BinOpLhs->getOpcode();

  return true;
}

static bool isSameRawIdentifierToken(const Token &T1, const Token &T2,
                        const SourceManager &SM) {
  if (T1.getKind() != T2.getKind())
    return false;
  if (T1.isNot(tok::raw_identifier))
    return true;
  if (T1.getLength() != T2.getLength())
    return false;
  return StringRef(SM.getCharacterData(T1.getLocation()), T1.getLength()) ==
         StringRef(SM.getCharacterData(T2.getLocation()), T2.getLength());
}

bool isTokAtEndOfExpr(SourceRange ExprSR, Token T, const SourceManager &SM) {
  return SM.getExpansionLoc(ExprSR.getEnd()) == T.getLocation();
}

/// Returns true if both LhsExpr and RhsExpr are
/// macro expressions and they are expanded
/// from different macros.
static bool areExprsFromDifferentMacros(const Expr *LhsExpr,
                                        const Expr *RhsExpr,
                                        const ASTContext *AstCtx) {
  if (!LhsExpr || !RhsExpr)
    return false;
  SourceRange Lsr = LhsExpr->getSourceRange();
  SourceRange Rsr = RhsExpr->getSourceRange();
  if (!Lsr.getBegin().isMacroID() || !Rsr.getBegin().isMacroID())
    return false;

  const SourceManager &SM = AstCtx->getSourceManager();
  const LangOptions &LO = AstCtx->getLangOpts();

  std::pair<FileID, unsigned> LsrLocInfo =
      SM.getDecomposedLoc(SM.getExpansionLoc(Lsr.getBegin()));
  std::pair<FileID, unsigned> RsrLocInfo =
      SM.getDecomposedLoc(SM.getExpansionLoc(Rsr.getBegin()));
  llvm::MemoryBufferRef MB = SM.getBufferOrFake(LsrLocInfo.first);

  const char *LTokenPos = MB.getBufferStart() + LsrLocInfo.second;
  const char *RTokenPos = MB.getBufferStart() + RsrLocInfo.second;
  Lexer LRawLex(SM.getLocForStartOfFile(LsrLocInfo.first), LO,
                MB.getBufferStart(), LTokenPos, MB.getBufferEnd());
  Lexer RRawLex(SM.getLocForStartOfFile(RsrLocInfo.first), LO,
                MB.getBufferStart(), RTokenPos, MB.getBufferEnd());

  Token LTok, RTok;
  do { // Compare the expressions token-by-token.
    LRawLex.LexFromRawLexer(LTok);
    RRawLex.LexFromRawLexer(RTok);
  } while (!LTok.is(tok::eof) && !RTok.is(tok::eof) &&
           isSameRawIdentifierToken(LTok, RTok, SM) &&
           !isTokAtEndOfExpr(Lsr, LTok, SM) &&
           !isTokAtEndOfExpr(Rsr, RTok, SM));
  return (!isTokAtEndOfExpr(Lsr, LTok, SM) ||
          !isTokAtEndOfExpr(Rsr, RTok, SM)) ||
         !isSameRawIdentifierToken(LTok, RTok, SM);
}

static bool areExprsMacroAndNonMacro(const Expr *&LhsExpr,
                                     const Expr *&RhsExpr) {
  if (!LhsExpr || !RhsExpr)
    return false;

  SourceLocation LhsLoc = LhsExpr->getExprLoc();
  SourceLocation RhsLoc = RhsExpr->getExprLoc();

  return LhsLoc.isMacroID() != RhsLoc.isMacroID();
}
} // namespace

void RedundantExpressionCheck::registerMatchers(MatchFinder *Finder) {
  const auto AnyLiteralExpr = ignoringParenImpCasts(
      anyOf(cxxBoolLiteral(), characterLiteral(), integerLiteral()));

  const auto BannedIntegerLiteral =
      integerLiteral(expandedByMacro(KnownBannedMacroNames));

  // Binary with equivalent operands, like (X != 2 && X != 2).
  Finder->addMatcher(
      traverse(TK_AsIs,
               binaryOperator(
                   anyOf(isComparisonOperator(),
                         hasAnyOperatorName("-", "/", "%", "|", "&", "^", "&&",
                                            "||", "=")),
                   operandsAreEquivalent(),
                   // Filter noisy false positives.
                   unless(isInTemplateInstantiation()),
                   unless(binaryOperatorIsInMacro()),
                   unless(hasType(realFloatingPointType())),
                   unless(hasEitherOperand(hasType(realFloatingPointType()))),
                   unless(hasLHS(AnyLiteralExpr)),
                   unless(hasDescendant(BannedIntegerLiteral)))
                   .bind("binary")),
      this);

  // Logical or bitwise operator with equivalent nested operands, like (X && Y
  // && X) or (X && (Y && X))
  Finder->addMatcher(
      binaryOperator(hasAnyOperatorName("|", "&", "||", "&&", "^"),
                     nestedOperandsAreEquivalent(),
                     // Filter noisy false positives.
                     unless(isInTemplateInstantiation()),
                     unless(binaryOperatorIsInMacro()),
                     // TODO: if the banned macros are themselves duplicated
                     unless(hasDescendant(BannedIntegerLiteral)))
          .bind("nested-duplicates"),
      this);

  // Conditional (ternary) operator with equivalent operands, like (Y ? X : X).
  Finder->addMatcher(
      traverse(TK_AsIs,
               conditionalOperator(expressionsAreEquivalent(),
                                   // Filter noisy false positives.
                                   unless(conditionalOperatorIsInMacro()),
                                   unless(isInTemplateInstantiation()))
                   .bind("cond")),
      this);

  // Overloaded operators with equivalent operands.
  Finder->addMatcher(
      traverse(TK_AsIs,
               cxxOperatorCallExpr(
                   hasAnyOverloadedOperatorName("-", "/", "%", "|", "&", "^",
                                                "==", "!=", "<", "<=", ">",
                                                ">=", "&&", "||", "="),
                   parametersAreEquivalent(),
                   // Filter noisy false positives.
                   unless(isMacro()), unless(isInTemplateInstantiation()))
                   .bind("call")),
      this);

  // Overloaded operators with equivalent operands.
  Finder->addMatcher(
      cxxOperatorCallExpr(
          hasAnyOverloadedOperatorName("|", "&", "||", "&&", "^"),
          nestedParametersAreEquivalent(), argumentCountIs(2),
          // Filter noisy false positives.
          unless(isMacro()), unless(isInTemplateInstantiation()))
          .bind("nested-duplicates"),
      this);

  // Match expressions like: !(1 | 2 | 3)
  Finder->addMatcher(
      traverse(TK_AsIs,
               implicitCastExpr(
                   hasImplicitDestinationType(isInteger()),
                   has(unaryOperator(
                           hasOperatorName("!"),
                           hasUnaryOperand(ignoringParenImpCasts(binaryOperator(
                               hasAnyOperatorName("|", "&"),
                               hasLHS(anyOf(
                                   binaryOperator(hasAnyOperatorName("|", "&")),
                                   integerLiteral())),
                               hasRHS(integerLiteral())))))
                           .bind("logical-bitwise-confusion")))),
      this);

  // Match expressions like: (X << 8) & 0xFF
  Finder->addMatcher(
      traverse(TK_AsIs,
               binaryOperator(
                   hasOperatorName("&"),
                   hasOperands(ignoringParenImpCasts(binaryOperator(
                                   hasOperatorName("<<"),
                                   hasRHS(ignoringParenImpCasts(
                                       integerLiteral().bind("shift-const"))))),
                               ignoringParenImpCasts(
                                   integerLiteral().bind("and-const"))))
                   .bind("left-right-shift-confusion")),
      this);

  // Match common expressions and apply more checks to find redundant
  // sub-expressions.
  //   a) Expr <op> K1 == K2
  //   b) Expr <op> K1 == Expr
  //   c) Expr <op> K1 == Expr <op> K2
  // see: 'checkArithmeticExpr' and 'checkBitwiseExpr'
  const auto BinOpCstLeft = matchBinOpIntegerConstantExpr("lhs");
  const auto BinOpCstRight = matchBinOpIntegerConstantExpr("rhs");
  const auto CstRight = matchIntegerConstantExpr("rhs");
  const auto SymRight = matchSymbolicExpr("rhs");

  // Match expressions like: x <op> 0xFF == 0xF00.
  Finder->addMatcher(
      traverse(TK_AsIs, binaryOperator(isComparisonOperator(),
                                       hasOperands(BinOpCstLeft, CstRight))
                            .bind("binop-const-compare-to-const")),
      this);

  // Match expressions like: x <op> 0xFF == x.
  Finder->addMatcher(
      traverse(
          TK_AsIs,
          binaryOperator(isComparisonOperator(),
                         anyOf(allOf(hasLHS(BinOpCstLeft), hasRHS(SymRight)),
                               allOf(hasLHS(SymRight), hasRHS(BinOpCstLeft))))
              .bind("binop-const-compare-to-sym")),
      this);

  // Match expressions like: x <op> 10 == x <op> 12.
  Finder->addMatcher(
      traverse(TK_AsIs,
               binaryOperator(isComparisonOperator(), hasLHS(BinOpCstLeft),
                              hasRHS(BinOpCstRight),
                              // Already reported as redundant.
                              unless(operandsAreEquivalent()))
                   .bind("binop-const-compare-to-binop-const")),
      this);

  // Match relational expressions combined with logical operators and find
  // redundant sub-expressions.
  // see: 'checkRelationalExpr'

  // Match expressions like: x < 2 && x > 2.
  const auto ComparisonLeft = matchRelationalIntegerConstantExpr("lhs");
  const auto ComparisonRight = matchRelationalIntegerConstantExpr("rhs");
  Finder->addMatcher(
      traverse(TK_AsIs,
               binaryOperator(hasAnyOperatorName("||", "&&"),
                              hasLHS(ComparisonLeft), hasRHS(ComparisonRight),
                              // Already reported as redundant.
                              unless(operandsAreEquivalent()))
                   .bind("comparisons-of-symbol-and-const")),
      this);
}

void RedundantExpressionCheck::checkArithmeticExpr(
    const MatchFinder::MatchResult &Result) {
  APSInt LhsValue, RhsValue;
  const Expr *LhsSymbol = nullptr, *RhsSymbol = nullptr;
  BinaryOperatorKind LhsOpcode, RhsOpcode;

  if (const auto *ComparisonOperator = Result.Nodes.getNodeAs<BinaryOperator>(
          "binop-const-compare-to-sym")) {
    BinaryOperatorKind Opcode = ComparisonOperator->getOpcode();
    if (!retrieveBinOpIntegerConstantExpr(Result, "lhs", LhsOpcode, LhsSymbol,
                                          LhsValue) ||
        !retrieveSymbolicExpr(Result, "rhs", RhsSymbol) ||
        !areEquivalentExpr(LhsSymbol, RhsSymbol))
      return;

    // Check expressions: x + k == x  or  x - k == x.
    if (LhsOpcode == BO_Add || LhsOpcode == BO_Sub) {
      if ((LhsValue != 0 && Opcode == BO_EQ) ||
          (LhsValue == 0 && Opcode == BO_NE))
        diag(ComparisonOperator->getOperatorLoc(),
             "logical expression is always false");
      else if ((LhsValue == 0 && Opcode == BO_EQ) ||
               (LhsValue != 0 && Opcode == BO_NE))
        diag(ComparisonOperator->getOperatorLoc(),
             "logical expression is always true");
    }
  } else if (const auto *ComparisonOperator =
                 Result.Nodes.getNodeAs<BinaryOperator>(
                     "binop-const-compare-to-binop-const")) {
    BinaryOperatorKind Opcode = ComparisonOperator->getOpcode();

    if (!retrieveBinOpIntegerConstantExpr(Result, "lhs", LhsOpcode, LhsSymbol,
                                          LhsValue) ||
        !retrieveBinOpIntegerConstantExpr(Result, "rhs", RhsOpcode, RhsSymbol,
                                          RhsValue) ||
        !areEquivalentExpr(LhsSymbol, RhsSymbol))
      return;

    transformSubToCanonicalAddExpr(LhsOpcode, LhsValue);
    transformSubToCanonicalAddExpr(RhsOpcode, RhsValue);

    // Check expressions: x + 1 == x + 2  or  x + 1 != x + 2.
    if (LhsOpcode == BO_Add && RhsOpcode == BO_Add) {
      if ((Opcode == BO_EQ && APSInt::compareValues(LhsValue, RhsValue) == 0) ||
          (Opcode == BO_NE && APSInt::compareValues(LhsValue, RhsValue) != 0)) {
        diag(ComparisonOperator->getOperatorLoc(),
             "logical expression is always true");
      } else if ((Opcode == BO_EQ &&
                  APSInt::compareValues(LhsValue, RhsValue) != 0) ||
                 (Opcode == BO_NE &&
                  APSInt::compareValues(LhsValue, RhsValue) == 0)) {
        diag(ComparisonOperator->getOperatorLoc(),
             "logical expression is always false");
      }
    }
  }
}

static bool exprEvaluatesToZero(BinaryOperatorKind Opcode, APSInt Value) {
  return (Opcode == BO_And || Opcode == BO_AndAssign) && Value == 0;
}

static bool exprEvaluatesToBitwiseNegatedZero(BinaryOperatorKind Opcode,
                                              APSInt Value) {
  return (Opcode == BO_Or || Opcode == BO_OrAssign) && ~Value == 0;
}

static bool exprEvaluatesToSymbolic(BinaryOperatorKind Opcode, APSInt Value) {
  return ((Opcode == BO_Or || Opcode == BO_OrAssign) && Value == 0) ||
         ((Opcode == BO_And || Opcode == BO_AndAssign) && ~Value == 0);
}


void RedundantExpressionCheck::checkBitwiseExpr(
    const MatchFinder::MatchResult &Result) {
  if (const auto *ComparisonOperator = Result.Nodes.getNodeAs<BinaryOperator>(
          "binop-const-compare-to-const")) {
    BinaryOperatorKind Opcode = ComparisonOperator->getOpcode();

    APSInt LhsValue, RhsValue;
    const Expr *LhsSymbol = nullptr;
    BinaryOperatorKind LhsOpcode;
    if (!retrieveBinOpIntegerConstantExpr(Result, "lhs", LhsOpcode, LhsSymbol,
                                          LhsValue) ||
        !retrieveIntegerConstantExpr(Result, "rhs", RhsValue))
      return;

    uint64_t LhsConstant = LhsValue.getZExtValue();
    uint64_t RhsConstant = RhsValue.getZExtValue();
    SourceLocation Loc = ComparisonOperator->getOperatorLoc();

    // Check expression: x & k1 == k2  (i.e. x & 0xFF == 0xF00)
    if (LhsOpcode == BO_And && (LhsConstant & RhsConstant) != RhsConstant) {
      if (Opcode == BO_EQ)
        diag(Loc, "logical expression is always false");
      else if (Opcode == BO_NE)
        diag(Loc, "logical expression is always true");
    }

    // Check expression: x | k1 == k2  (i.e. x | 0xFF == 0xF00)
    if (LhsOpcode == BO_Or && (LhsConstant | RhsConstant) != RhsConstant) {
      if (Opcode == BO_EQ)
        diag(Loc, "logical expression is always false");
      else if (Opcode == BO_NE)
        diag(Loc, "logical expression is always true");
    }
  } else if (const auto *IneffectiveOperator =
                 Result.Nodes.getNodeAs<BinaryOperator>(
                     "ineffective-bitwise")) {
    APSInt Value;
    const Expr *Sym = nullptr, *ConstExpr = nullptr;

    if (!retrieveSymbolicExpr(Result, "ineffective-bitwise", Sym) ||
        !retrieveIntegerConstantExpr(Result, "ineffective-bitwise", Value,
                                     ConstExpr))
      return;

    if((Value != 0 && ~Value != 0) || Sym->getExprLoc().isMacroID())
        return;

    SourceLocation Loc = IneffectiveOperator->getOperatorLoc();

    BinaryOperatorKind Opcode = IneffectiveOperator->getOpcode();
    if (exprEvaluatesToZero(Opcode, Value)) {
      diag(Loc, "expression always evaluates to 0");
    } else if (exprEvaluatesToBitwiseNegatedZero(Opcode, Value)) {
      SourceRange ConstExprRange(ConstExpr->getBeginLoc(),
                                 ConstExpr->getEndLoc());
      StringRef ConstExprText = Lexer::getSourceText(
          CharSourceRange::getTokenRange(ConstExprRange), *Result.SourceManager,
          Result.Context->getLangOpts());

      diag(Loc, "expression always evaluates to '%0'") << ConstExprText;

    } else if (exprEvaluatesToSymbolic(Opcode, Value)) {
      SourceRange SymExprRange(Sym->getBeginLoc(), Sym->getEndLoc());

      StringRef ExprText = Lexer::getSourceText(
          CharSourceRange::getTokenRange(SymExprRange), *Result.SourceManager,
          Result.Context->getLangOpts());

      diag(Loc, "expression always evaluates to '%0'") << ExprText;
    }
  }
}

void RedundantExpressionCheck::checkRelationalExpr(
    const MatchFinder::MatchResult &Result) {
  if (const auto *ComparisonOperator = Result.Nodes.getNodeAs<BinaryOperator>(
          "comparisons-of-symbol-and-const")) {
    // Matched expressions are: (x <op> k1) <REL> (x <op> k2).
    // E.g.: (X < 2) && (X > 4)
    BinaryOperatorKind Opcode = ComparisonOperator->getOpcode();

    const Expr *LhsExpr = nullptr, *RhsExpr = nullptr;
    const Expr *LhsSymbol = nullptr, *RhsSymbol = nullptr;
    const Expr *LhsConst = nullptr, *RhsConst = nullptr;
    BinaryOperatorKind LhsOpcode, RhsOpcode;
    APSInt LhsValue, RhsValue;

    if (!retrieveRelationalIntegerConstantExpr(
            Result, "lhs", LhsExpr, LhsOpcode, LhsSymbol, LhsValue, LhsConst) ||
        !retrieveRelationalIntegerConstantExpr(
            Result, "rhs", RhsExpr, RhsOpcode, RhsSymbol, RhsValue, RhsConst) ||
        !areEquivalentExpr(LhsSymbol, RhsSymbol))
      return;

    // Bring expr to a canonical form: smallest constant must be on the left.
    if (APSInt::compareValues(LhsValue, RhsValue) > 0) {
      std::swap(LhsExpr, RhsExpr);
      std::swap(LhsValue, RhsValue);
      std::swap(LhsSymbol, RhsSymbol);
      std::swap(LhsOpcode, RhsOpcode);
    }

    // Constants come from two different macros, or one of them is a macro.
    if (areExprsFromDifferentMacros(LhsConst, RhsConst, Result.Context) ||
        areExprsMacroAndNonMacro(LhsConst, RhsConst))
      return;

    if ((Opcode == BO_LAnd || Opcode == BO_LOr) &&
        areEquivalentRanges(LhsOpcode, LhsValue, RhsOpcode, RhsValue)) {
      diag(ComparisonOperator->getOperatorLoc(),
           "equivalent expression on both sides of logical operator");
      return;
    }

    if (Opcode == BO_LAnd) {
      if (areExclusiveRanges(LhsOpcode, LhsValue, RhsOpcode, RhsValue)) {
        diag(ComparisonOperator->getOperatorLoc(),
             "logical expression is always false");
      } else if (rangeSubsumesRange(LhsOpcode, LhsValue, RhsOpcode, RhsValue)) {
        diag(LhsExpr->getExprLoc(), "expression is redundant");
      } else if (rangeSubsumesRange(RhsOpcode, RhsValue, LhsOpcode, LhsValue)) {
        diag(RhsExpr->getExprLoc(), "expression is redundant");
      }
    }

    if (Opcode == BO_LOr) {
      if (rangesFullyCoverDomain(LhsOpcode, LhsValue, RhsOpcode, RhsValue)) {
        diag(ComparisonOperator->getOperatorLoc(),
             "logical expression is always true");
      } else if (rangeSubsumesRange(LhsOpcode, LhsValue, RhsOpcode, RhsValue)) {
        diag(RhsExpr->getExprLoc(), "expression is redundant");
      } else if (rangeSubsumesRange(RhsOpcode, RhsValue, LhsOpcode, LhsValue)) {
        diag(LhsExpr->getExprLoc(), "expression is redundant");
      }
    }
  }
}

void RedundantExpressionCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *BinOp = Result.Nodes.getNodeAs<BinaryOperator>("binary")) {
    // If the expression's constants are macros, check whether they are
    // intentional.
    if (areSidesBinaryConstExpressions(BinOp, Result.Context)) {
      const Expr *LhsConst = nullptr, *RhsConst = nullptr;
      BinaryOperatorKind MainOpcode, SideOpcode;

      if (!retrieveConstExprFromBothSides(BinOp, MainOpcode, SideOpcode,
                                          LhsConst, RhsConst, Result.Context))
        return;

      if (areExprsFromDifferentMacros(LhsConst, RhsConst, Result.Context) ||
          areExprsMacroAndNonMacro(LhsConst, RhsConst))
        return;
    }

    diag(BinOp->getOperatorLoc(), "both sides of operator are equivalent");
  }

  if (const auto *CondOp =
          Result.Nodes.getNodeAs<ConditionalOperator>("cond")) {
    const Expr *TrueExpr = CondOp->getTrueExpr();
    const Expr *FalseExpr = CondOp->getFalseExpr();

    if (areExprsFromDifferentMacros(TrueExpr, FalseExpr, Result.Context) ||
        areExprsMacroAndNonMacro(TrueExpr, FalseExpr))
      return;
    diag(CondOp->getColonLoc(),
         "'true' and 'false' expressions are equivalent");
  }

  if (const auto *Call = Result.Nodes.getNodeAs<CXXOperatorCallExpr>("call")) {
    if (canOverloadedOperatorArgsBeModified(Call, true))
      return;

    diag(Call->getOperatorLoc(),
         "both sides of overloaded operator are equivalent");
  }

  if (const auto *Op = Result.Nodes.getNodeAs<Expr>("nested-duplicates")) {
    const auto *Call = dyn_cast<CXXOperatorCallExpr>(Op);
    if (Call && canOverloadedOperatorArgsBeModified(Call, true))
      return;

    StringRef Message =
        Call ? "overloaded operator has equivalent nested operands"
             : "operator has equivalent nested operands";

    const auto Diag = diag(Op->getExprLoc(), Message);
    for (const auto &KeyValue : Result.Nodes.getMap()) {
      if (StringRef(KeyValue.first).startswith("duplicate"))
        Diag << KeyValue.second.getSourceRange();
    }
  }

  if (const auto *NegateOperator =
          Result.Nodes.getNodeAs<UnaryOperator>("logical-bitwise-confusion")) {
    SourceLocation OperatorLoc = NegateOperator->getOperatorLoc();

    auto Diag =
        diag(OperatorLoc,
             "ineffective logical negation operator used; did you mean '~'?");
    SourceLocation LogicalNotLocation = OperatorLoc.getLocWithOffset(1);

    if (!LogicalNotLocation.isMacroID())
      Diag << FixItHint::CreateReplacement(
          CharSourceRange::getCharRange(OperatorLoc, LogicalNotLocation), "~");
  }

  if (const auto *BinaryAndExpr = Result.Nodes.getNodeAs<BinaryOperator>(
          "left-right-shift-confusion")) {
    const auto *ShiftingConst = Result.Nodes.getNodeAs<Expr>("shift-const");
    assert(ShiftingConst && "Expr* 'ShiftingConst' is nullptr!");
    Optional<llvm::APSInt> ShiftingValue =
        ShiftingConst->getIntegerConstantExpr(*Result.Context);

    if (!ShiftingValue)
      return;

    const auto *AndConst = Result.Nodes.getNodeAs<Expr>("and-const");
    assert(AndConst && "Expr* 'AndCont' is nullptr!");
    Optional<llvm::APSInt> AndValue =
        AndConst->getIntegerConstantExpr(*Result.Context);
    if (!AndValue)
      return;

    // If ShiftingConst is shifted left with more bits than the position of the
    // leftmost 1 in the bit representation of AndValue, AndConstant is
    // ineffective.
    if (AndValue->getActiveBits() > *ShiftingValue)
      return;

    auto Diag = diag(BinaryAndExpr->getOperatorLoc(),
                     "ineffective bitwise and operation");
  }

  // Check for the following bound expressions:
  // - "binop-const-compare-to-sym",
  // - "binop-const-compare-to-binop-const",
  // Produced message:
  // -> "logical expression is always false/true"
  checkArithmeticExpr(Result);

  // Check for the following bound expression:
  // - "binop-const-compare-to-const",
  // - "ineffective-bitwise"
  // Produced message:
  // -> "logical expression is always false/true"
  // -> "expression always evaluates to ..."
  checkBitwiseExpr(Result);

  // Check for te following bound expression:
  // - "comparisons-of-symbol-and-const",
  // Produced messages:
  // -> "equivalent expression on both sides of logical operator",
  // -> "logical expression is always false/true"
  // -> "expression is redundant"
  checkRelationalExpr(Result);
}

} // namespace misc
} // namespace tidy
} // namespace clang
