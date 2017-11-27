//===--- RedundantExpressionCheck.cpp - clang-tidy-------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "llvm/Support/Casting.h"
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
} // namespace

static const llvm::StringSet<> KnownBannedMacroNames = {"EAGAIN", "EWOULDBLOCK",
                                                        "SIGCLD", "SIGCHLD"};

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
    if (!areEquivalentExpr(dyn_cast<Expr>(*LeftIter),
                           dyn_cast<Expr>(*RightIter)))
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
  APSInt ValueLHS_plus1;
  return ((OpcodeLHS == BO_LE && OpcodeRHS == BO_LT) ||
          (OpcodeLHS == BO_GT && OpcodeRHS == BO_GE)) &&
         incrementWithoutOverflow(ValueLHS, ValueLHS_plus1) &&
         APSInt::compareValues(ValueLHS_plus1, ValueRHS) == 0;
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
  APSInt ValueLHS_plus1;
  if (OpcodeLHS == BO_GT && OpcodeRHS == BO_LT &&
      incrementWithoutOverflow(ValueLHS, ValueLHS_plus1) &&
      APSInt::compareValues(ValueLHS_plus1, ValueRHS) == 0)
    return true;

  return false;
}

// Returns whether the ranges covered by the union of both relational
// expressions covers the whole domain (i.e. x < 10  and  x > 0).
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
  APSInt ValueLHS_plus1;
  if (OpcodeLHS == BO_LE && OpcodeRHS == BO_GE &&
      incrementWithoutOverflow(ValueLHS, ValueLHS_plus1) &&
      APSInt::compareValues(ValueLHS_plus1, ValueRHS) == 0)
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

AST_MATCHER(Expr, isIntegerConstantExpr) {
  if (Node.isInstantiationDependent())
    return false;
  return Node.isIntegerConstantExpr(Finder->getASTContext());
}

AST_MATCHER(BinaryOperator, operandsAreEquivalent) {
  return areEquivalentExpr(Node.getLHS(), Node.getRHS());
}

AST_MATCHER(ConditionalOperator, expressionsAreEquivalent) {
  return areEquivalentExpr(Node.getTrueExpr(), Node.getFalseExpr());
}

AST_MATCHER(CallExpr, parametersAreEquivalent) {
  return Node.getNumArgs() == 2 &&
         areEquivalentExpr(Node.getArg(0), Node.getArg(1));
}

AST_MATCHER(BinaryOperator, binaryOperatorIsInMacro) {
  return Node.getOperatorLoc().isMacroID();
}

AST_MATCHER(ConditionalOperator, conditionalOperatorIsInMacro) {
  return Node.getQuestionLoc().isMacroID() || Node.getColonLoc().isMacroID();
}

AST_MATCHER(Expr, isMacro) { return Node.getExprLoc().isMacroID(); }

AST_MATCHER_P(Expr, expandedByMacro, llvm::StringSet<>, Names) {
  const SourceManager &SM = Finder->getASTContext().getSourceManager();
  const LangOptions &LO = Finder->getASTContext().getLangOpts();
  SourceLocation Loc = Node.getExprLoc();
  while (Loc.isMacroID()) {
    StringRef MacroName = Lexer::getImmediateMacroName(Loc, SM, LO);
    if (Names.count(MacroName))
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
  return ConstExpr && ConstExpr->isIntegerConstantExpr(Value, *Result.Context);
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
      expr(
          anyOf(binaryOperator(anyOf(hasOperatorName("+"), hasOperatorName("|"),
                                     hasOperatorName("&")),
                               hasEitherOperand(matchSymbolicExpr(Id)),
                               hasEitherOperand(matchIntegerConstantExpr(Id))),
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
          anyOf(hasOverloadedOperatorName("=="),
                hasOverloadedOperatorName("!="), hasOverloadedOperatorName("<"),
                hasOverloadedOperatorName("<="), hasOverloadedOperatorName(">"),
                hasOverloadedOperatorName(">=")),
          // Filter noisy false positives.
          unless(isMacro()), unless(isInTemplateInstantiation()))
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
canOverloadedOperatorArgsBeModified(const FunctionDecl *OperatorDecl,
                                    bool checkSecondParam) {
  unsigned ParamCount = OperatorDecl->getNumParams();

  // Overloaded operators declared inside a class have only one param.
  // These functions must be declared const in order to not be able to modify
  // the instance of the class they are called through.
  if (ParamCount == 1 &&
      !OperatorDecl->getType()->getAs<FunctionType>()->isConst())
    return true;

  if (isNonConstReferenceType(OperatorDecl->getParamDecl(0)->getType()))
    return true;

  return checkSecondParam && ParamCount == 2 &&
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
    const auto *OverloadedFunctionDecl = dyn_cast_or_null<FunctionDecl>(OverloadedOperatorExpr->getCalleeDecl());
    if (!OverloadedFunctionDecl)
      return false;

    if (canOverloadedOperatorArgsBeModified(OverloadedFunctionDecl, false))
      return false;

    if (!OverloadedOperatorExpr->getArg(1)->isIntegerConstantExpr(
            Value, *Result.Context))
      return false;

    Symbol = OverloadedOperatorExpr->getArg(0);
    OperandExpr = OverloadedOperatorExpr;
    Opcode = BinaryOperator::getOverloadedOpcode(OverloadedOperatorExpr->getOperator());

    return BinaryOperator::isComparisonOp(Opcode);
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

  if ((LhsBinOp->getLHS()->isIntegerConstantExpr(*AstCtx) ||
       LhsBinOp->getRHS()->isIntegerConstantExpr(*AstCtx)) &&
      (RhsBinOp->getLHS()->isIntegerConstantExpr(*AstCtx) ||
       RhsBinOp->getRHS()->isIntegerConstantExpr(*AstCtx)))
    return true;
  return false;
}

// Retrieves integer constant subexpressions from binary operator expressions
// that have two equivalent sides
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

  LhsConst = BinOpLhs->getLHS()->isIntegerConstantExpr(*AstCtx)
                 ? BinOpLhs->getLHS()
                 : BinOpLhs->getRHS();
  RhsConst = BinOpRhs->getLHS()->isIntegerConstantExpr(*AstCtx)
                 ? BinOpRhs->getLHS()
                 : BinOpRhs->getRHS();

  if (!LhsConst || !RhsConst)
    return false;

  assert(BinOpLhs->getOpcode() == BinOpRhs->getOpcode() &&
         "Sides of the binary operator must be equivalent expressions!");

  SideOpcode = BinOpLhs->getOpcode();

  return true;
}

static bool areExprsFromDifferentMacros(const Expr *LhsExpr,
                                        const Expr *RhsExpr,
                                        const ASTContext *AstCtx) {
  if (!LhsExpr || !RhsExpr)
    return false;

  SourceLocation LhsLoc = LhsExpr->getExprLoc();
  SourceLocation RhsLoc = RhsExpr->getExprLoc();

  if (!LhsLoc.isMacroID() || !RhsLoc.isMacroID())
    return false;

  const SourceManager &SM = AstCtx->getSourceManager();
  const LangOptions &LO = AstCtx->getLangOpts();

  return !(Lexer::getImmediateMacroName(LhsLoc, SM, LO) ==
          Lexer::getImmediateMacroName(RhsLoc, SM, LO));
}

static bool areExprsMacroAndNonMacro(const Expr *&LhsExpr,
                                     const Expr *&RhsExpr) {
  if (!LhsExpr || !RhsExpr)
    return false;

  SourceLocation LhsLoc = LhsExpr->getExprLoc();
  SourceLocation RhsLoc = RhsExpr->getExprLoc();

  return LhsLoc.isMacroID() != RhsLoc.isMacroID();
}

void RedundantExpressionCheck::registerMatchers(MatchFinder *Finder) {
  const auto AnyLiteralExpr = ignoringParenImpCasts(
      anyOf(cxxBoolLiteral(), characterLiteral(), integerLiteral()));

  const auto BannedIntegerLiteral =
      integerLiteral(expandedByMacro(KnownBannedMacroNames));

  // Binary with equivalent operands, like (X != 2 && X != 2).
  Finder->addMatcher(
      binaryOperator(anyOf(hasOperatorName("-"), hasOperatorName("/"),
                           hasOperatorName("%"), hasOperatorName("|"),
                           hasOperatorName("&"), hasOperatorName("^"),
                           matchers::isComparisonOperator(),
                           hasOperatorName("&&"), hasOperatorName("||"),
                           hasOperatorName("=")),
                     operandsAreEquivalent(),
                     // Filter noisy false positives.
                     unless(isInTemplateInstantiation()),
                     unless(binaryOperatorIsInMacro()),
                     unless(hasType(realFloatingPointType())),
                     unless(hasEitherOperand(hasType(realFloatingPointType()))),
                     unless(hasLHS(AnyLiteralExpr)),
                     unless(hasDescendant(BannedIntegerLiteral)))
          .bind("binary"),
      this);

  // Conditional (trenary) operator with equivalent operands, like (Y ? X : X).
  Finder->addMatcher(conditionalOperator(expressionsAreEquivalent(),
                                         // Filter noisy false positives.
                                         unless(conditionalOperatorIsInMacro()),
                                         unless(isInTemplateInstantiation()))
                         .bind("cond"),
                     this);

  // Overloaded operators with equivalent operands.
  Finder->addMatcher(
      cxxOperatorCallExpr(
          anyOf(
              hasOverloadedOperatorName("-"), hasOverloadedOperatorName("/"),
              hasOverloadedOperatorName("%"), hasOverloadedOperatorName("|"),
              hasOverloadedOperatorName("&"), hasOverloadedOperatorName("^"),
              hasOverloadedOperatorName("=="), hasOverloadedOperatorName("!="),
              hasOverloadedOperatorName("<"), hasOverloadedOperatorName("<="),
              hasOverloadedOperatorName(">"), hasOverloadedOperatorName(">="),
              hasOverloadedOperatorName("&&"), hasOverloadedOperatorName("||"),
              hasOverloadedOperatorName("=")),
          parametersAreEquivalent(),
          // Filter noisy false positives.
          unless(isMacro()), unless(isInTemplateInstantiation()))
          .bind("call"),
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
  Finder->addMatcher(binaryOperator(isComparisonOperator(),
                                    hasEitherOperand(BinOpCstLeft),
                                    hasEitherOperand(CstRight))
                         .bind("binop-const-compare-to-const"),
                     this);

  // Match expressions like: x <op> 0xFF == x.
  Finder->addMatcher(
      binaryOperator(isComparisonOperator(),
                     anyOf(allOf(hasLHS(BinOpCstLeft), hasRHS(SymRight)),
                           allOf(hasLHS(SymRight), hasRHS(BinOpCstLeft))))
          .bind("binop-const-compare-to-sym"),
      this);

  // Match expressions like: x <op> 10 == x <op> 12.
  Finder->addMatcher(binaryOperator(isComparisonOperator(),
                                    hasLHS(BinOpCstLeft), hasRHS(BinOpCstRight),
                                    // Already reported as redundant.
                                    unless(operandsAreEquivalent()))
                         .bind("binop-const-compare-to-binop-const"),
                     this);

  // Match relational expressions combined with logical operators and find
  // redundant sub-expressions.
  // see: 'checkRelationalExpr'

  // Match expressions like: x < 2 && x > 2.
  const auto ComparisonLeft = matchRelationalIntegerConstantExpr("lhs");
  const auto ComparisonRight = matchRelationalIntegerConstantExpr("rhs");
  Finder->addMatcher(
      binaryOperator(anyOf(hasOperatorName("||"), hasOperatorName("&&")),
                     hasLHS(ComparisonLeft), hasRHS(ComparisonRight),
                     // Already reported as redundant.
                     unless(operandsAreEquivalent()))
          .bind("comparisons-of-symbol-and-const"),
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
    const auto *OverloadedFunctionDecl = dyn_cast_or_null<FunctionDecl>(Call->getCalleeDecl());
    if (!OverloadedFunctionDecl)
      return;

    if (canOverloadedOperatorArgsBeModified(OverloadedFunctionDecl, true))
      return;

    diag(Call->getOperatorLoc(),
         "both sides of overloaded operator are equivalent");
  }

  // Check for the following bound expressions:
  // - "binop-const-compare-to-sym",
  // - "binop-const-compare-to-binop-const",
  // Produced message:
  // -> "logical expression is always false/true"
  checkArithmeticExpr(Result);

  // Check for the following bound expression:
  // - "binop-const-compare-to-const",
  // Produced message:
  // -> "logical expression is always false/true"
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
