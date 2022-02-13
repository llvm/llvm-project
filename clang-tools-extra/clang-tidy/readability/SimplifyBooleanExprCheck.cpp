//===-- SimplifyBooleanExprCheck.cpp - clang-tidy -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SimplifyBooleanExprCheck.h"
#include "SimplifyBooleanExprMatchers.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Lex/Lexer.h"

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

} // namespace

static constexpr char ConditionThenStmtId[] = "if-bool-yields-then";
static constexpr char ConditionElseStmtId[] = "if-bool-yields-else";
static constexpr char TernaryId[] = "ternary-bool-yields-condition";
static constexpr char TernaryNegatedId[] = "ternary-bool-yields-not-condition";
static constexpr char IfReturnsBoolId[] = "if-return";
static constexpr char IfReturnsNotBoolId[] = "if-not-return";
static constexpr char ThenLiteralId[] = "then-literal";
static constexpr char IfAssignVariableId[] = "if-assign-lvalue";
static constexpr char IfAssignLocId[] = "if-assign-loc";
static constexpr char IfAssignBoolId[] = "if-assign";
static constexpr char IfAssignNotBoolId[] = "if-assign-not";
static constexpr char IfAssignVarId[] = "if-assign-var";
static constexpr char CompoundReturnId[] = "compound-return";
static constexpr char CompoundIfId[] = "compound-if";
static constexpr char CompoundBoolId[] = "compound-bool";
static constexpr char CompoundNotBoolId[] = "compound-bool-not";
static constexpr char CaseId[] = "case";
static constexpr char CaseCompoundBoolId[] = "case-compound-bool";
static constexpr char CaseCompoundNotBoolId[] = "case-compound-bool-not";
static constexpr char DefaultId[] = "default";
static constexpr char DefaultCompoundBoolId[] = "default-compound-bool";
static constexpr char DefaultCompoundNotBoolId[] = "default-compound-bool-not";
static constexpr char LabelId[] = "label";
static constexpr char LabelCompoundBoolId[] = "label-compound-bool";
static constexpr char LabelCompoundNotBoolId[] = "label-compound-bool-not";
static constexpr char IfStmtId[] = "if";

static constexpr char SimplifyOperatorDiagnostic[] =
    "redundant boolean literal supplied to boolean operator";
static constexpr char SimplifyConditionDiagnostic[] =
    "redundant boolean literal in if statement condition";
static constexpr char SimplifyConditionalReturnDiagnostic[] =
    "redundant boolean literal in conditional return statement";

static const Expr *getBoolLiteral(const MatchFinder::MatchResult &Result,
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

static internal::BindableMatcher<Stmt> literalOrNegatedBool(bool Value) {
  return expr(
      anyOf(cxxBoolLiteral(equals(Value)),
            unaryOperator(hasUnaryOperand(cxxBoolLiteral(equals(!Value))),
                          hasOperatorName("!"))));
}

static internal::Matcher<Stmt> returnsBool(bool Value,
                                           StringRef Id = "ignored") {
  auto SimpleReturnsBool = returnStmt(has(literalOrNegatedBool(Value).bind(Id)))
                               .bind("returns-bool");
  return anyOf(SimpleReturnsBool,
               compoundStmt(statementCountIs(1), has(SimpleReturnsBool)));
}

static bool needsParensAfterUnaryNegation(const Expr *E) {
  E = E->IgnoreImpCasts();
  if (isa<BinaryOperator>(E) || isa<ConditionalOperator>(E))
    return true;

  if (const auto *Op = dyn_cast<CXXOperatorCallExpr>(E))
    return Op->getNumArgs() == 2 && Op->getOperator() != OO_Call &&
           Op->getOperator() != OO_Subscript;

  return false;
}

static std::pair<BinaryOperatorKind, BinaryOperatorKind> Opposites[] = {
    {BO_LT, BO_GE}, {BO_GT, BO_LE}, {BO_EQ, BO_NE}};

static StringRef negatedOperator(const BinaryOperator *BinOp) {
  const BinaryOperatorKind Opcode = BinOp->getOpcode();
  for (auto NegatableOp : Opposites) {
    if (Opcode == NegatableOp.first)
      return BinOp->getOpcodeStr(NegatableOp.second);
    if (Opcode == NegatableOp.second)
      return BinOp->getOpcodeStr(NegatableOp.first);
  }
  return {};
}

static std::pair<OverloadedOperatorKind, StringRef> OperatorNames[] = {
    {OO_EqualEqual, "=="},   {OO_ExclaimEqual, "!="}, {OO_Less, "<"},
    {OO_GreaterEqual, ">="}, {OO_Greater, ">"},       {OO_LessEqual, "<="}};

static StringRef getOperatorName(OverloadedOperatorKind OpKind) {
  for (auto Name : OperatorNames) {
    if (Name.first == OpKind)
      return Name.second;
  }

  return {};
}

static std::pair<OverloadedOperatorKind, OverloadedOperatorKind>
    OppositeOverloads[] = {{OO_EqualEqual, OO_ExclaimEqual},
                           {OO_Less, OO_GreaterEqual},
                           {OO_Greater, OO_LessEqual}};

static StringRef negatedOperator(const CXXOperatorCallExpr *OpCall) {
  const OverloadedOperatorKind Opcode = OpCall->getOperator();
  for (auto NegatableOp : OppositeOverloads) {
    if (Opcode == NegatableOp.first)
      return getOperatorName(NegatableOp.second);
    if (Opcode == NegatableOp.second)
      return getOperatorName(NegatableOp.first);
  }
  return {};
}

static std::string asBool(StringRef Text, bool NeedsStaticCast) {
  if (NeedsStaticCast)
    return ("static_cast<bool>(" + Text + ")").str();

  return std::string(Text);
}

static bool needsNullPtrComparison(const Expr *E) {
  if (const auto *ImpCast = dyn_cast<ImplicitCastExpr>(E))
    return ImpCast->getCastKind() == CK_PointerToBoolean ||
           ImpCast->getCastKind() == CK_MemberPointerToBoolean;

  return false;
}

static bool needsZeroComparison(const Expr *E) {
  if (const auto *ImpCast = dyn_cast<ImplicitCastExpr>(E))
    return ImpCast->getCastKind() == CK_IntegralToBoolean;

  return false;
}

static bool needsStaticCast(const Expr *E) {
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

static std::string
compareExpressionToConstant(const MatchFinder::MatchResult &Result,
                            const Expr *E, bool Negated, const char *Constant) {
  E = E->IgnoreImpCasts();
  const std::string ExprText =
      (isa<BinaryOperator>(E) ? ("(" + getText(Result, *E) + ")")
                              : getText(Result, *E))
          .str();
  return ExprText + " " + (Negated ? "!=" : "==") + " " + Constant;
}

static std::string
compareExpressionToNullPtr(const MatchFinder::MatchResult &Result,
                           const Expr *E, bool Negated) {
  const char *NullPtr =
      Result.Context->getLangOpts().CPlusPlus11 ? "nullptr" : "NULL";
  return compareExpressionToConstant(Result, E, Negated, NullPtr);
}

static std::string
compareExpressionToZero(const MatchFinder::MatchResult &Result, const Expr *E,
                        bool Negated) {
  return compareExpressionToConstant(Result, E, Negated, "0");
}

static std::string replacementExpression(const MatchFinder::MatchResult &Result,
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

static const Expr *stmtReturnsBool(const ReturnStmt *Ret, bool Negated) {
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

static const Expr *stmtReturnsBool(const IfStmt *IfRet, bool Negated) {
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

static bool containsDiscardedTokens(const MatchFinder::MatchResult &Result,
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

class SimplifyBooleanExprCheck::Visitor : public RecursiveASTVisitor<Visitor> {
public:
  Visitor(SimplifyBooleanExprCheck *Check,
          const MatchFinder::MatchResult &Result)
      : Check(Check), Result(Result) {}

  bool VisitBinaryOperator(const BinaryOperator *Op) const {
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

static bool containsBoolLiteral(const Expr *E) {
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
  const Expr *Other;
  if ((Bool = dyn_cast<CXXBoolLiteralExpr>(LHS)) != nullptr)
    Other = RHS;
  else if ((Bool = dyn_cast<CXXBoolLiteralExpr>(RHS)) != nullptr)
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
    if (BoolValue)
      // expr && true -> expr
      ReplaceWithExpression(Other, /*Negated=*/false);
    else
      // expr && false -> false
      ReplaceWithExpression(Bool, /*Negated=*/false);
    break;
  case BO_LOr:
    if (BoolValue)
      // expr || true -> true
      ReplaceWithExpression(Bool, /*Negated=*/false);
    else
      // expr || false -> expr
      ReplaceWithExpression(Other, /*Negated=*/false);
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
                                                  bool Value, StringRef Id) {
  Finder->addMatcher(
      conditionalOperator(hasTrueExpression(literalOrNegatedBool(Value)),
                          hasFalseExpression(literalOrNegatedBool(!Value)))
          .bind(Id),
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

static internal::Matcher<Stmt> ifReturnValue(bool Value) {
  return ifStmt(hasThen(returnsBool(Value)), unless(hasElse(stmt())))
      .bind(CompoundIfId);
}

static internal::Matcher<Stmt> returnNotValue(bool Value) {
  return returnStmt(has(literalOrNegatedBool(!Value))).bind(CompoundReturnId);
}

void SimplifyBooleanExprCheck::matchCompoundIfReturnsBool(MatchFinder *Finder,
                                                          bool Value,
                                                          StringRef Id) {
  if (ChainedConditionalReturn)
    Finder->addMatcher(
        compoundStmt(hasSubstatementSequence(ifReturnValue(Value),
                                             returnNotValue(Value)))
            .bind(Id),
        this);
  else
    Finder->addMatcher(
        compoundStmt(hasSubstatementSequence(ifStmt(hasThen(returnsBool(Value)),
                                                    unless(hasElse(stmt())),
                                                    unless(hasParent(ifStmt())))
                                                 .bind(CompoundIfId),
                                             returnNotValue(Value)))
            .bind(Id),
        this);
}

void SimplifyBooleanExprCheck::matchCaseIfReturnsBool(MatchFinder *Finder,
                                                      bool Value,
                                                      StringRef Id) {
  internal::Matcher<Stmt> CaseStmt =
      caseStmt(hasSubstatement(ifReturnValue(Value))).bind(CaseId);
  internal::Matcher<Stmt> CompoundStmt =
      compoundStmt(hasSubstatementSequence(CaseStmt, returnNotValue(Value)))
          .bind(Id);
  Finder->addMatcher(switchStmt(has(CompoundStmt)), this);
}

void SimplifyBooleanExprCheck::matchDefaultIfReturnsBool(MatchFinder *Finder,
                                                         bool Value,
                                                         StringRef Id) {
  internal::Matcher<Stmt> DefaultStmt =
      defaultStmt(hasSubstatement(ifReturnValue(Value))).bind(DefaultId);
  internal::Matcher<Stmt> CompoundStmt =
      compoundStmt(hasSubstatementSequence(DefaultStmt, returnNotValue(Value)))
          .bind(Id);
  Finder->addMatcher(switchStmt(has(CompoundStmt)), this);
}

void SimplifyBooleanExprCheck::matchLabelIfReturnsBool(MatchFinder *Finder,
                                                       bool Value,
                                                       StringRef Id) {
  internal::Matcher<Stmt> LabelStmt =
      labelStmt(hasSubstatement(ifReturnValue(Value))).bind(LabelId);
  internal::Matcher<Stmt> CompoundStmt =
      compoundStmt(hasSubstatementSequence(LabelStmt, returnNotValue(Value)))
          .bind(Id);
  Finder->addMatcher(CompoundStmt, this);
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

  matchCaseIfReturnsBool(Finder, true, CaseCompoundBoolId);
  matchCaseIfReturnsBool(Finder, false, CaseCompoundNotBoolId);

  matchDefaultIfReturnsBool(Finder, true, DefaultCompoundBoolId);
  matchDefaultIfReturnsBool(Finder, false, DefaultCompoundNotBoolId);

  matchLabelIfReturnsBool(Finder, true, LabelCompoundBoolId);
  matchLabelIfReturnsBool(Finder, false, LabelCompoundNotBoolId);
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
    replaceWithCondition(Result, Ternary, false);
  else if (const auto *TernaryNegated =
               Result.Nodes.getNodeAs<ConditionalOperator>(TernaryNegatedId))
    replaceWithCondition(Result, TernaryNegated, true);
  else if (const auto *If = Result.Nodes.getNodeAs<IfStmt>(IfReturnsBoolId))
    replaceWithReturnCondition(Result, If, false);
  else if (const auto *IfNot =
               Result.Nodes.getNodeAs<IfStmt>(IfReturnsNotBoolId))
    replaceWithReturnCondition(Result, IfNot, true);
  else if (const auto *IfAssign =
               Result.Nodes.getNodeAs<IfStmt>(IfAssignBoolId))
    replaceWithAssignment(Result, IfAssign, false);
  else if (const auto *IfAssignNot =
               Result.Nodes.getNodeAs<IfStmt>(IfAssignNotBoolId))
    replaceWithAssignment(Result, IfAssignNot, true);
  else if (const auto *Compound =
               Result.Nodes.getNodeAs<CompoundStmt>(CompoundBoolId))
    replaceCompoundReturnWithCondition(Result, Compound, false);
  else if (const auto *CompoundNot =
               Result.Nodes.getNodeAs<CompoundStmt>(CompoundNotBoolId))
    replaceCompoundReturnWithCondition(Result, CompoundNot, true);
  else if (Result.Nodes.getNodeAs<CompoundStmt>(CaseCompoundBoolId))
    replaceCaseCompoundReturnWithCondition(Result, false);
  else if (Result.Nodes.getNodeAs<CompoundStmt>(CaseCompoundNotBoolId))
    replaceCaseCompoundReturnWithCondition(Result, true);
  else if (Result.Nodes.getNodeAs<CompoundStmt>(DefaultCompoundBoolId))
    replaceDefaultCompoundReturnWithCondition(Result, false);
  else if (Result.Nodes.getNodeAs<CompoundStmt>(DefaultCompoundNotBoolId))
    replaceDefaultCompoundReturnWithCondition(Result, true);
  else if (Result.Nodes.getNodeAs<CompoundStmt>(LabelCompoundBoolId))
    replaceLabelCompoundReturnWithCondition(Result, false);
  else if (Result.Nodes.getNodeAs<CompoundStmt>(LabelCompoundNotBoolId))
    replaceLabelCompoundReturnWithCondition(Result, true);
  else if (const auto TU = Result.Nodes.getNodeAs<Decl>("top"))
    Visitor(this, Result).TraverseDecl(const_cast<Decl *>(TU));
}

void SimplifyBooleanExprCheck::issueDiag(const MatchFinder::MatchResult &Result,
                                         SourceLocation Loc,
                                         StringRef Description,
                                         SourceRange ReplacementRange,
                                         StringRef Replacement) {
  CharSourceRange CharRange =
      Lexer::makeFileCharRange(CharSourceRange::getTokenRange(ReplacementRange),
                               *Result.SourceManager, getLangOpts());

  DiagnosticBuilder Diag = diag(Loc, Description);
  if (!containsDiscardedTokens(Result, CharRange))
    Diag << FixItHint::CreateReplacement(CharRange, Replacement);
}

void SimplifyBooleanExprCheck::replaceWithThenStatement(
    const MatchFinder::MatchResult &Result, const Expr *BoolLiteral) {
  const auto *IfStatement = Result.Nodes.getNodeAs<IfStmt>(IfStmtId);
  issueDiag(Result, BoolLiteral->getBeginLoc(), SimplifyConditionDiagnostic,
            IfStatement->getSourceRange(),
            getText(Result, *IfStatement->getThen()));
}

void SimplifyBooleanExprCheck::replaceWithElseStatement(
    const MatchFinder::MatchResult &Result, const Expr *BoolLiteral) {
  const auto *IfStatement = Result.Nodes.getNodeAs<IfStmt>(IfStmtId);
  const Stmt *ElseStatement = IfStatement->getElse();
  issueDiag(Result, BoolLiteral->getBeginLoc(), SimplifyConditionDiagnostic,
            IfStatement->getSourceRange(),
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

  // Scan through the CompoundStmt to look for a chained-if construct.
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

          std::string Replacement =
              "return " + replacementExpression(Result, Negated, If->getCond());
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

void SimplifyBooleanExprCheck::replaceCompoundReturnWithCondition(
    const MatchFinder::MatchResult &Result, bool Negated, const IfStmt *If) {
  const auto *Lit = stmtReturnsBool(If, Negated);
  const auto *Ret = Result.Nodes.getNodeAs<ReturnStmt>(CompoundReturnId);
  const std::string Replacement =
      "return " + replacementExpression(Result, Negated, If->getCond());
  issueDiag(Result, Lit->getBeginLoc(), SimplifyConditionalReturnDiagnostic,
            SourceRange(If->getBeginLoc(), Ret->getEndLoc()), Replacement);
}

void SimplifyBooleanExprCheck::replaceCaseCompoundReturnWithCondition(
    const MatchFinder::MatchResult &Result, bool Negated) {
  const auto *CaseDefault = Result.Nodes.getNodeAs<CaseStmt>(CaseId);
  const auto *If = cast<IfStmt>(CaseDefault->getSubStmt());
  replaceCompoundReturnWithCondition(Result, Negated, If);
}

void SimplifyBooleanExprCheck::replaceDefaultCompoundReturnWithCondition(
    const MatchFinder::MatchResult &Result, bool Negated) {
  const SwitchCase *CaseDefault =
      Result.Nodes.getNodeAs<DefaultStmt>(DefaultId);
  const auto *If = cast<IfStmt>(CaseDefault->getSubStmt());
  replaceCompoundReturnWithCondition(Result, Negated, If);
}

void SimplifyBooleanExprCheck::replaceLabelCompoundReturnWithCondition(
    const MatchFinder::MatchResult &Result, bool Negated) {
  const auto *Label = Result.Nodes.getNodeAs<LabelStmt>(LabelId);
  const auto *If = cast<IfStmt>(Label->getSubStmt());
  replaceCompoundReturnWithCondition(Result, Negated, If);
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
