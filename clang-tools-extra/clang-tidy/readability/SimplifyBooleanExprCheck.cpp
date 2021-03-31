//===-- SimplifyBooleanExprCheck.cpp - clang-tidy -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SimplifyBooleanExprCheck.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Lex/Lexer.h"

#include <cassert>
#include <string>
#include <utility>

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace readability {

namespace {

StringRef getText(const MatchFinder::MatchResult &Result, SourceRange Range) {
  return Lexer::getSourceText(CharSourceRange::getTokenRange(Range),
                              *Result.SourceManager,
                              Result.Context->getLangOpts());
}

template <typename T>
StringRef getText(const MatchFinder::MatchResult &Result, T &Node) {
  return getText(Result, Node.getSourceRange());
}

const char ConditionThenStmtId[] = "if-bool-yields-then";
const char ConditionElseStmtId[] = "if-bool-yields-else";
const char TernaryId[] = "ternary-bool-yields-condition";
const char TernaryNegatedId[] = "ternary-bool-yields-not-condition";
const char IfReturnsBoolId[] = "if-return";
const char IfReturnsNotBoolId[] = "if-not-return";
const char ThenLiteralId[] = "then-literal";
const char IfAssignVariableId[] = "if-assign-lvalue";
const char IfAssignLocId[] = "if-assign-loc";
const char IfAssignBoolId[] = "if-assign";
const char IfAssignNotBoolId[] = "if-assign-not";
const char IfAssignVarId[] = "if-assign-var";
const char CompoundReturnId[] = "compound-return";
const char CompoundBoolId[] = "compound-bool";
const char CompoundNotBoolId[] = "compound-bool-not";

const char IfStmtId[] = "if";

const char SimplifyOperatorDiagnostic[] =
    "redundant boolean literal supplied to boolean operator";
const char SimplifyConditionDiagnostic[] =
    "redundant boolean literal in if statement condition";
const char SimplifyConditionalReturnDiagnostic[] =
    "redundant boolean literal in conditional return statement";

const Expr *getBoolLiteral(const MatchFinder::MatchResult &Result,
                           StringRef Id) {
  if (const Expr *Literal = Result.Nodes.getNodeAs<CXXBoolLiteralExpr>(Id))
    return Literal->getBeginLoc().isMacroID() ? nullptr : Literal;
  if (const auto *Negated = Result.Nodes.getNodeAs<UnaryOperator>(Id)) {
    if (Negated->getOpcode() == UO_LNot &&
        isa<CXXBoolLiteralExpr>(Negated->getSubExpr()))
      return Negated->getBeginLoc().isMacroID() ? nullptr : Negated;
  }
  return nullptr;
}

internal::BindableMatcher<Stmt> literalOrNegatedBool(bool Value) {
  return expr(
      anyOf(cxxBoolLiteral(equals(Value)),
            unaryOperator(hasUnaryOperand(cxxBoolLiteral(equals(!Value))),
                          hasOperatorName("!"))));
}

internal::Matcher<Stmt> returnsBool(bool Value, StringRef Id = "ignored") {
  auto SimpleReturnsBool = returnStmt(has(literalOrNegatedBool(Value).bind(Id)))
                               .bind("returns-bool");
  return anyOf(SimpleReturnsBool,
               compoundStmt(statementCountIs(1), has(SimpleReturnsBool)));
}

bool needsParensAfterUnaryNegation(const Expr *E) {
  E = E->IgnoreImpCasts();
  if (isa<BinaryOperator>(E) || isa<ConditionalOperator>(E))
    return true;

  if (const auto *Op = dyn_cast<CXXOperatorCallExpr>(E))
    return Op->getNumArgs() == 2 && Op->getOperator() != OO_Call &&
           Op->getOperator() != OO_Subscript;

  return false;
}

std::pair<BinaryOperatorKind, BinaryOperatorKind> Opposites[] = {
    {BO_LT, BO_GE}, {BO_GT, BO_LE}, {BO_EQ, BO_NE}};

StringRef negatedOperator(const BinaryOperator *BinOp) {
  const BinaryOperatorKind Opcode = BinOp->getOpcode();
  for (auto NegatableOp : Opposites) {
    if (Opcode == NegatableOp.first)
      return BinOp->getOpcodeStr(NegatableOp.second);
    if (Opcode == NegatableOp.second)
      return BinOp->getOpcodeStr(NegatableOp.first);
  }
  return StringRef();
}

std::pair<OverloadedOperatorKind, StringRef> OperatorNames[] = {
    {OO_EqualEqual, "=="},   {OO_ExclaimEqual, "!="}, {OO_Less, "<"},
    {OO_GreaterEqual, ">="}, {OO_Greater, ">"},       {OO_LessEqual, "<="}};

StringRef getOperatorName(OverloadedOperatorKind OpKind) {
  for (auto Name : OperatorNames) {
    if (Name.first == OpKind)
      return Name.second;
  }

  return StringRef();
}

std::pair<OverloadedOperatorKind, OverloadedOperatorKind> OppositeOverloads[] =
    {{OO_EqualEqual, OO_ExclaimEqual},
     {OO_Less, OO_GreaterEqual},
     {OO_Greater, OO_LessEqual}};

StringRef negatedOperator(const CXXOperatorCallExpr *OpCall) {
  const OverloadedOperatorKind Opcode = OpCall->getOperator();
  for (auto NegatableOp : OppositeOverloads) {
    if (Opcode == NegatableOp.first)
      return getOperatorName(NegatableOp.second);
    if (Opcode == NegatableOp.second)
      return getOperatorName(NegatableOp.first);
  }
  return StringRef();
}

std::string asBool(StringRef Text, bool NeedsStaticCast) {
  if (NeedsStaticCast)
    return ("static_cast<bool>(" + Text + ")").str();

  return std::string(Text);
}

bool needsNullPtrComparison(const Expr *E) {
  if (const auto *ImpCast = dyn_cast<ImplicitCastExpr>(E))
    return ImpCast->getCastKind() == CK_PointerToBoolean ||
           ImpCast->getCastKind() == CK_MemberPointerToBoolean;

  return false;
}

bool needsZeroComparison(const Expr *E) {
  if (const auto *ImpCast = dyn_cast<ImplicitCastExpr>(E))
    return ImpCast->getCastKind() == CK_IntegralToBoolean;

  return false;
}

bool needsStaticCast(const Expr *E) {
  if (const auto *ImpCast = dyn_cast<ImplicitCastExpr>(E)) {
    if (ImpCast->getCastKind() == CK_UserDefinedConversion &&
        ImpCast->getSubExpr()->getType()->isBooleanType()) {
      if (const auto *MemCall =
              dyn_cast<CXXMemberCallExpr>(ImpCast->getSubExpr())) {
        if (const auto *MemDecl =
                dyn_cast<CXXConversionDecl>(MemCall->getMethodDecl())) {
          if (MemDecl->isExplicit())
            return true;
        }
      }
    }
  }

  E = E->IgnoreImpCasts();
  return !E->getType()->isBooleanType();
}

std::string compareExpressionToConstant(const MatchFinder::MatchResult &Result,
                                        const Expr *E, bool Negated,
                                        const char *Constant) {
  E = E->IgnoreImpCasts();
  const std::string ExprText =
      (isa<BinaryOperator>(E) ? ("(" + getText(Result, *E) + ")")
                              : getText(Result, *E))
          .str();
  return ExprText + " " + (Negated ? "!=" : "==") + " " + Constant;
}

std::string compareExpressionToNullPtr(const MatchFinder::MatchResult &Result,
                                       const Expr *E, bool Negated) {
  const char *NullPtr =
      Result.Context->getLangOpts().CPlusPlus11 ? "nullptr" : "NULL";
  return compareExpressionToConstant(Result, E, Negated, NullPtr);
}

std::string compareExpressionToZero(const MatchFinder::MatchResult &Result,
                                    const Expr *E, bool Negated) {
  return compareExpressionToConstant(Result, E, Negated, "0");
}

std::string replacementExpression(const MatchFinder::MatchResult &Result,
                                  bool Negated, const Expr *E) {
  E = E->IgnoreParenBaseCasts();
  if (const auto *EC = dyn_cast<ExprWithCleanups>(E))
    E = EC->getSubExpr();

  const bool NeedsStaticCast = needsStaticCast(E);
  if (Negated) {
    if (const auto *UnOp = dyn_cast<UnaryOperator>(E)) {
      if (UnOp->getOpcode() == UO_LNot) {
        if (needsNullPtrComparison(UnOp->getSubExpr()))
          return compareExpressionToNullPtr(Result, UnOp->getSubExpr(), true);

        if (needsZeroComparison(UnOp->getSubExpr()))
          return compareExpressionToZero(Result, UnOp->getSubExpr(), true);

        return replacementExpression(Result, false, UnOp->getSubExpr());
      }
    }

    if (needsNullPtrComparison(E))
      return compareExpressionToNullPtr(Result, E, false);

    if (needsZeroComparison(E))
      return compareExpressionToZero(Result, E, false);

    StringRef NegatedOperator;
    const Expr *LHS = nullptr;
    const Expr *RHS = nullptr;
    if (const auto *BinOp = dyn_cast<BinaryOperator>(E)) {
      NegatedOperator = negatedOperator(BinOp);
      LHS = BinOp->getLHS();
      RHS = BinOp->getRHS();
    } else if (const auto *OpExpr = dyn_cast<CXXOperatorCallExpr>(E)) {
      if (OpExpr->getNumArgs() == 2) {
        NegatedOperator = negatedOperator(OpExpr);
        LHS = OpExpr->getArg(0);
        RHS = OpExpr->getArg(1);
      }
    }
    if (!NegatedOperator.empty() && LHS && RHS)
      return (asBool((getText(Result, *LHS) + " " + NegatedOperator + " " +
                      getText(Result, *RHS))
                         .str(),
                     NeedsStaticCast));

    StringRef Text = getText(Result, *E);
    if (!NeedsStaticCast && needsParensAfterUnaryNegation(E))
      return ("!(" + Text + ")").str();

    if (needsNullPtrComparison(E))
      return compareExpressionToNullPtr(Result, E, false);

    if (needsZeroComparison(E))
      return compareExpressionToZero(Result, E, false);

    return ("!" + asBool(Text, NeedsStaticCast));
  }

  if (const auto *UnOp = dyn_cast<UnaryOperator>(E)) {
    if (UnOp->getOpcode() == UO_LNot) {
      if (needsNullPtrComparison(UnOp->getSubExpr()))
        return compareExpressionToNullPtr(Result, UnOp->getSubExpr(), false);

      if (needsZeroComparison(UnOp->getSubExpr()))
        return compareExpressionToZero(Result, UnOp->getSubExpr(), false);
    }
  }

  if (needsNullPtrComparison(E))
    return compareExpressionToNullPtr(Result, E, true);

  if (needsZeroComparison(E))
    return compareExpressionToZero(Result, E, true);

  return asBool(getText(Result, *E), NeedsStaticCast);
}

const Expr *stmtReturnsBool(const ReturnStmt *Ret, bool Negated) {
  if (const auto *Bool = dyn_cast<CXXBoolLiteralExpr>(Ret->getRetValue())) {
    if (Bool->getValue() == !Negated)
      return Bool;
  }
  if (const auto *Unary = dyn_cast<UnaryOperator>(Ret->getRetValue())) {
    if (Unary->getOpcode() == UO_LNot) {
      if (const auto *Bool =
              dyn_cast<CXXBoolLiteralExpr>(Unary->getSubExpr())) {
        if (Bool->getValue() == Negated)
          return Bool;
      }
    }
  }

  return nullptr;
}

const Expr *stmtReturnsBool(const IfStmt *IfRet, bool Negated) {
  if (IfRet->getElse() != nullptr)
    return nullptr;

  if (const auto *Ret = dyn_cast<ReturnStmt>(IfRet->getThen()))
    return stmtReturnsBool(Ret, Negated);

  if (const auto *Compound = dyn_cast<CompoundStmt>(IfRet->getThen())) {
    if (Compound->size() == 1) {
      if (const auto *CompoundRet = dyn_cast<ReturnStmt>(Compound->body_back()))
        return stmtReturnsBool(CompoundRet, Negated);
    }
  }

  return nullptr;
}

bool containsDiscardedTokens(const MatchFinder::MatchResult &Result,
                             CharSourceRange CharRange) {
  std::string ReplacementText =
      Lexer::getSourceText(CharRange, *Result.SourceManager,
                           Result.Context->getLangOpts())
          .str();
  Lexer Lex(CharRange.getBegin(), Result.Context->getLangOpts(),
            ReplacementText.data(), ReplacementText.data(),
            ReplacementText.data() + ReplacementText.size());
  Lex.SetCommentRetentionState(true);

  Token Tok;
  while (!Lex.LexFromRawLexer(Tok)) {
    if (Tok.is(tok::TokenKind::comment) || Tok.is(tok::TokenKind::hash))
      return true;
  }

  return false;
}

} // namespace

class SimplifyBooleanExprCheck::Visitor : public RecursiveASTVisitor<Visitor> {
 public:
  Visitor(SimplifyBooleanExprCheck *Check,
          const MatchFinder::MatchResult &Result)
      : Check(Check), Result(Result) {}

  bool VisitBinaryOperator(BinaryOperator *Op) {
    Check->reportBinOp(Result, Op);
    return true;
  }

 private:
  SimplifyBooleanExprCheck *Check;
  const MatchFinder::MatchResult &Result;
};

SimplifyBooleanExprCheck::SimplifyBooleanExprCheck(StringRef Name,
                                                   ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      ChainedConditionalReturn(Options.get("ChainedConditionalReturn", false)),
      ChainedConditionalAssignment(
          Options.get("ChainedConditionalAssignment", false)) {}

bool containsBoolLiteral(const Expr *E) {
  if (!E)
    return false;
  E = E->IgnoreParenImpCasts();
  if (isa<CXXBoolLiteralExpr>(E))
    return true;
  if (const auto *BinOp = dyn_cast<BinaryOperator>(E))
    return containsBoolLiteral(BinOp->getLHS()) ||
           containsBoolLiteral(BinOp->getRHS());
  if (const auto *UnaryOp = dyn_cast<UnaryOperator>(E))
    return containsBoolLiteral(UnaryOp->getSubExpr());
  return false;
}

void SimplifyBooleanExprCheck::reportBinOp(
    const MatchFinder::MatchResult &Result, const BinaryOperator *Op) {
  const auto *LHS = Op->getLHS()->IgnoreParenImpCasts();
  const auto *RHS = Op->getRHS()->IgnoreParenImpCasts();

  const CXXBoolLiteralExpr *Bool;
  const Expr *Other = nullptr;
  if ((Bool = dyn_cast<CXXBoolLiteralExpr>(LHS)))
    Other = RHS;
  else if ((Bool = dyn_cast<CXXBoolLiteralExpr>(RHS)))
    Other = LHS;
  else
    return;

  if (Bool->getBeginLoc().isMacroID())
    return;

  // FIXME: why do we need this?
  if (!isa<CXXBoolLiteralExpr>(Other) && containsBoolLiteral(Other))
    return;

  bool BoolValue = Bool->getValue();

  auto ReplaceWithExpression = [this, &Result, LHS, RHS,
                                Bool](const Expr *ReplaceWith, bool Negated) {
    std::string Replacement =
        replacementExpression(Result, Negated, ReplaceWith);
    SourceRange Range(LHS->getBeginLoc(), RHS->getEndLoc());
    issueDiag(Result, Bool->getBeginLoc(), SimplifyOperatorDiagnostic, Range,
              Replacement);
  };

  switch (Op->getOpcode()) {
    case BO_LAnd:
      if (BoolValue) {
        // expr && true -> expr
        ReplaceWithExpression(Other, /*Negated=*/false);
      } else {
        // expr && false -> false
        ReplaceWithExpression(Bool, /*Negated=*/false);
      }
      break;
    case BO_LOr:
      if (BoolValue) {
        // expr || true -> true
        ReplaceWithExpression(Bool, /*Negated=*/false);
      } else {
        // expr || false -> expr
        ReplaceWithExpression(Other, /*Negated=*/false);
      }
      break;
    case BO_EQ:
      // expr == true -> expr, expr == false -> !expr
      ReplaceWithExpression(Other, /*Negated=*/!BoolValue);
      break;
    case BO_NE:
      // expr != true -> !expr, expr != false -> expr
      ReplaceWithExpression(Other, /*Negated=*/BoolValue);
      break;
    default:
      break;
  }
}

void SimplifyBooleanExprCheck::matchBoolCondition(MatchFinder *Finder,
                                                  bool Value,
                                                  StringRef BooleanId) {
  Finder->addMatcher(
      ifStmt(hasCondition(literalOrNegatedBool(Value).bind(BooleanId)))
          .bind(IfStmtId),
      this);
}

void SimplifyBooleanExprCheck::matchTernaryResult(MatchFinder *Finder,
                                                  bool Value,
                                                  StringRef TernaryId) {
  Finder->addMatcher(
      conditionalOperator(hasTrueExpression(literalOrNegatedBool(Value)),
                          hasFalseExpression(literalOrNegatedBool(!Value)))
          .bind(TernaryId),
      this);
}

void SimplifyBooleanExprCheck::matchIfReturnsBool(MatchFinder *Finder,
                                                  bool Value, StringRef Id) {
  if (ChainedConditionalReturn)
    Finder->addMatcher(ifStmt(hasThen(returnsBool(Value, ThenLiteralId)),
                              hasElse(returnsBool(!Value)))
                           .bind(Id),
                       this);
  else
    Finder->addMatcher(ifStmt(unless(hasParent(ifStmt())),
                              hasThen(returnsBool(Value, ThenLiteralId)),
                              hasElse(returnsBool(!Value)))
                           .bind(Id),
                       this);
}

void SimplifyBooleanExprCheck::matchIfAssignsBool(MatchFinder *Finder,
                                                  bool Value, StringRef Id) {
  auto VarAssign = declRefExpr(hasDeclaration(decl().bind(IfAssignVarId)));
  auto VarRef = declRefExpr(hasDeclaration(equalsBoundNode(IfAssignVarId)));
  auto MemAssign = memberExpr(hasDeclaration(decl().bind(IfAssignVarId)));
  auto MemRef = memberExpr(hasDeclaration(equalsBoundNode(IfAssignVarId)));
  auto SimpleThen =
      binaryOperator(hasOperatorName("="), hasLHS(anyOf(VarAssign, MemAssign)),
                     hasLHS(expr().bind(IfAssignVariableId)),
                     hasRHS(literalOrNegatedBool(Value).bind(IfAssignLocId)));
  auto Then = anyOf(SimpleThen, compoundStmt(statementCountIs(1),
                                             hasAnySubstatement(SimpleThen)));
  auto SimpleElse =
      binaryOperator(hasOperatorName("="), hasLHS(anyOf(VarRef, MemRef)),
                     hasRHS(literalOrNegatedBool(!Value)));
  auto Else = anyOf(SimpleElse, compoundStmt(statementCountIs(1),
                                             hasAnySubstatement(SimpleElse)));
  if (ChainedConditionalAssignment)
    Finder->addMatcher(ifStmt(hasThen(Then), hasElse(Else)).bind(Id), this);
  else
    Finder->addMatcher(
        ifStmt(unless(hasParent(ifStmt())), hasThen(Then), hasElse(Else))
            .bind(Id),
        this);
}

void SimplifyBooleanExprCheck::matchCompoundIfReturnsBool(MatchFinder *Finder,
                                                          bool Value,
                                                          StringRef Id) {
  Finder->addMatcher(
      compoundStmt(
          hasAnySubstatement(
              ifStmt(hasThen(returnsBool(Value)), unless(hasElse(stmt())))),
          hasAnySubstatement(returnStmt(has(literalOrNegatedBool(!Value)))
                                 .bind(CompoundReturnId)))
          .bind(Id),
      this);
}

void SimplifyBooleanExprCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "ChainedConditionalReturn", ChainedConditionalReturn);
  Options.store(Opts, "ChainedConditionalAssignment",
                ChainedConditionalAssignment);
}

void SimplifyBooleanExprCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(translationUnitDecl().bind("top"), this);

  matchBoolCondition(Finder, true, ConditionThenStmtId);
  matchBoolCondition(Finder, false, ConditionElseStmtId);

  matchTernaryResult(Finder, true, TernaryId);
  matchTernaryResult(Finder, false, TernaryNegatedId);

  matchIfReturnsBool(Finder, true, IfReturnsBoolId);
  matchIfReturnsBool(Finder, false, IfReturnsNotBoolId);

  matchIfAssignsBool(Finder, true, IfAssignBoolId);
  matchIfAssignsBool(Finder, false, IfAssignNotBoolId);

  matchCompoundIfReturnsBool(Finder, true, CompoundBoolId);
  matchCompoundIfReturnsBool(Finder, false, CompoundNotBoolId);
}

void SimplifyBooleanExprCheck::check(const MatchFinder::MatchResult &Result) {
  if (Result.Nodes.getNodeAs<TranslationUnitDecl>("top"))
    Visitor(this, Result).TraverseAST(*Result.Context);
  else if (const Expr *TrueConditionRemoved =
               getBoolLiteral(Result, ConditionThenStmtId))
    replaceWithThenStatement(Result, TrueConditionRemoved);
  else if (const Expr *FalseConditionRemoved =
               getBoolLiteral(Result, ConditionElseStmtId))
    replaceWithElseStatement(Result, FalseConditionRemoved);
  else if (const auto *Ternary =
               Result.Nodes.getNodeAs<ConditionalOperator>(TernaryId))
    replaceWithCondition(Result, Ternary);
  else if (const auto *TernaryNegated =
               Result.Nodes.getNodeAs<ConditionalOperator>(TernaryNegatedId))
    replaceWithCondition(Result, TernaryNegated, true);
  else if (const auto *If = Result.Nodes.getNodeAs<IfStmt>(IfReturnsBoolId))
    replaceWithReturnCondition(Result, If);
  else if (const auto *IfNot =
               Result.Nodes.getNodeAs<IfStmt>(IfReturnsNotBoolId))
    replaceWithReturnCondition(Result, IfNot, true);
  else if (const auto *IfAssign =
               Result.Nodes.getNodeAs<IfStmt>(IfAssignBoolId))
    replaceWithAssignment(Result, IfAssign);
  else if (const auto *IfAssignNot =
               Result.Nodes.getNodeAs<IfStmt>(IfAssignNotBoolId))
    replaceWithAssignment(Result, IfAssignNot, true);
  else if (const auto *Compound =
               Result.Nodes.getNodeAs<CompoundStmt>(CompoundBoolId))
    replaceCompoundReturnWithCondition(Result, Compound);
  else if (const auto *Compound =
               Result.Nodes.getNodeAs<CompoundStmt>(CompoundNotBoolId))
    replaceCompoundReturnWithCondition(Result, Compound, true);
}

void SimplifyBooleanExprCheck::issueDiag(
    const ast_matchers::MatchFinder::MatchResult &Result, SourceLocation Loc,
    StringRef Description, SourceRange ReplacementRange,
    StringRef Replacement) {
  CharSourceRange CharRange =
      Lexer::makeFileCharRange(CharSourceRange::getTokenRange(ReplacementRange),
                               *Result.SourceManager, getLangOpts());

  DiagnosticBuilder Diag = diag(Loc, Description);
  if (!containsDiscardedTokens(Result, CharRange))
    Diag << FixItHint::CreateReplacement(CharRange, Replacement);
}

void SimplifyBooleanExprCheck::replaceWithThenStatement(
    const MatchFinder::MatchResult &Result, const Expr *TrueConditionRemoved) {
  const auto *IfStatement = Result.Nodes.getNodeAs<IfStmt>(IfStmtId);
  issueDiag(Result, TrueConditionRemoved->getBeginLoc(),
            SimplifyConditionDiagnostic, IfStatement->getSourceRange(),
            getText(Result, *IfStatement->getThen()));
}

void SimplifyBooleanExprCheck::replaceWithElseStatement(
    const MatchFinder::MatchResult &Result, const Expr *FalseConditionRemoved) {
  const auto *IfStatement = Result.Nodes.getNodeAs<IfStmt>(IfStmtId);
  const Stmt *ElseStatement = IfStatement->getElse();
  issueDiag(Result, FalseConditionRemoved->getBeginLoc(),
            SimplifyConditionDiagnostic, IfStatement->getSourceRange(),
            ElseStatement ? getText(Result, *ElseStatement) : "");
}

void SimplifyBooleanExprCheck::replaceWithCondition(
    const MatchFinder::MatchResult &Result, const ConditionalOperator *Ternary,
    bool Negated) {
  std::string Replacement =
      replacementExpression(Result, Negated, Ternary->getCond());
  issueDiag(Result, Ternary->getTrueExpr()->getBeginLoc(),
            "redundant boolean literal in ternary expression result",
            Ternary->getSourceRange(), Replacement);
}

void SimplifyBooleanExprCheck::replaceWithReturnCondition(
    const MatchFinder::MatchResult &Result, const IfStmt *If, bool Negated) {
  StringRef Terminator = isa<CompoundStmt>(If->getElse()) ? ";" : "";
  std::string Condition = replacementExpression(Result, Negated, If->getCond());
  std::string Replacement = ("return " + Condition + Terminator).str();
  SourceLocation Start =
      Result.Nodes.getNodeAs<CXXBoolLiteralExpr>(ThenLiteralId)->getBeginLoc();
  issueDiag(Result, Start, SimplifyConditionalReturnDiagnostic,
            If->getSourceRange(), Replacement);
}

void SimplifyBooleanExprCheck::replaceCompoundReturnWithCondition(
    const MatchFinder::MatchResult &Result, const CompoundStmt *Compound,
    bool Negated) {
  const auto *Ret = Result.Nodes.getNodeAs<ReturnStmt>(CompoundReturnId);

  // The body shouldn't be empty because the matcher ensures that it must
  // contain at least two statements:
  // 1) A `return` statement returning a boolean literal `false` or `true`
  // 2) An `if` statement with no `else` clause that consists of a single
  //    `return` statement returning the opposite boolean literal `true` or
  //    `false`.
  assert(Compound->size() >= 2);
  const IfStmt *BeforeIf = nullptr;
  CompoundStmt::const_body_iterator Current = Compound->body_begin();
  CompoundStmt::const_body_iterator After = Compound->body_begin();
  for (++After; After != Compound->body_end() && *Current != Ret;
       ++Current, ++After) {
    if (const auto *If = dyn_cast<IfStmt>(*Current)) {
      if (const Expr *Lit = stmtReturnsBool(If, Negated)) {
        if (*After == Ret) {
          if (!ChainedConditionalReturn && BeforeIf)
            continue;

          const Expr *Condition = If->getCond();
          std::string Replacement =
              "return " + replacementExpression(Result, Negated, Condition);
          issueDiag(
              Result, Lit->getBeginLoc(), SimplifyConditionalReturnDiagnostic,
              SourceRange(If->getBeginLoc(), Ret->getEndLoc()), Replacement);
          return;
        }

        BeforeIf = If;
      }
    } else {
      BeforeIf = nullptr;
    }
  }
}

void SimplifyBooleanExprCheck::replaceWithAssignment(
    const MatchFinder::MatchResult &Result, const IfStmt *IfAssign,
    bool Negated) {
  SourceRange Range = IfAssign->getSourceRange();
  StringRef VariableName =
      getText(Result, *Result.Nodes.getNodeAs<Expr>(IfAssignVariableId));
  StringRef Terminator = isa<CompoundStmt>(IfAssign->getElse()) ? ";" : "";
  std::string Condition =
      replacementExpression(Result, Negated, IfAssign->getCond());
  std::string Replacement =
      (VariableName + " = " + Condition + Terminator).str();
  SourceLocation Location =
      Result.Nodes.getNodeAs<CXXBoolLiteralExpr>(IfAssignLocId)->getBeginLoc();
  issueDiag(Result, Location,
            "redundant boolean literal in conditional assignment", Range,
            Replacement);
}

} // namespace readability
} // namespace tidy
} // namespace clang
