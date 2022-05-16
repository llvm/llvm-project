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

#include <string>
#include <utility>

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace readability {

namespace {

StringRef getText(const ASTContext &Context, SourceRange Range) {
  return Lexer::getSourceText(CharSourceRange::getTokenRange(Range),
                              Context.getSourceManager(),
                              Context.getLangOpts());
}

template <typename T> StringRef getText(const ASTContext &Context, T &Node) {
  return getText(Context, Node.getSourceRange());
}

} // namespace

static constexpr char SimplifyOperatorDiagnostic[] =
    "redundant boolean literal supplied to boolean operator";
static constexpr char SimplifyConditionDiagnostic[] =
    "redundant boolean literal in if statement condition";
static constexpr char SimplifyConditionalReturnDiagnostic[] =
    "redundant boolean literal in conditional return statement";

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

static std::string compareExpressionToConstant(const ASTContext &Context,
                                               const Expr *E, bool Negated,
                                               const char *Constant) {
  E = E->IgnoreImpCasts();
  const std::string ExprText =
      (isa<BinaryOperator>(E) ? ("(" + getText(Context, *E) + ")")
                              : getText(Context, *E))
          .str();
  return ExprText + " " + (Negated ? "!=" : "==") + " " + Constant;
}

static std::string compareExpressionToNullPtr(const ASTContext &Context,
                                              const Expr *E, bool Negated) {
  const char *NullPtr = Context.getLangOpts().CPlusPlus11 ? "nullptr" : "NULL";
  return compareExpressionToConstant(Context, E, Negated, NullPtr);
}

static std::string compareExpressionToZero(const ASTContext &Context,
                                           const Expr *E, bool Negated) {
  return compareExpressionToConstant(Context, E, Negated, "0");
}

static std::string replacementExpression(const ASTContext &Context,
                                         bool Negated, const Expr *E) {
  E = E->IgnoreParenBaseCasts();
  if (const auto *EC = dyn_cast<ExprWithCleanups>(E))
    E = EC->getSubExpr();

  const bool NeedsStaticCast = needsStaticCast(E);
  if (Negated) {
    if (const auto *UnOp = dyn_cast<UnaryOperator>(E)) {
      if (UnOp->getOpcode() == UO_LNot) {
        if (needsNullPtrComparison(UnOp->getSubExpr()))
          return compareExpressionToNullPtr(Context, UnOp->getSubExpr(), true);

        if (needsZeroComparison(UnOp->getSubExpr()))
          return compareExpressionToZero(Context, UnOp->getSubExpr(), true);

        return replacementExpression(Context, false, UnOp->getSubExpr());
      }
    }

    if (needsNullPtrComparison(E))
      return compareExpressionToNullPtr(Context, E, false);

    if (needsZeroComparison(E))
      return compareExpressionToZero(Context, E, false);

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
      return (asBool((getText(Context, *LHS) + " " + NegatedOperator + " " +
                      getText(Context, *RHS))
                         .str(),
                     NeedsStaticCast));

    StringRef Text = getText(Context, *E);
    if (!NeedsStaticCast && needsParensAfterUnaryNegation(E))
      return ("!(" + Text + ")").str();

    if (needsNullPtrComparison(E))
      return compareExpressionToNullPtr(Context, E, false);

    if (needsZeroComparison(E))
      return compareExpressionToZero(Context, E, false);

    return ("!" + asBool(Text, NeedsStaticCast));
  }

  if (const auto *UnOp = dyn_cast<UnaryOperator>(E)) {
    if (UnOp->getOpcode() == UO_LNot) {
      if (needsNullPtrComparison(UnOp->getSubExpr()))
        return compareExpressionToNullPtr(Context, UnOp->getSubExpr(), false);

      if (needsZeroComparison(UnOp->getSubExpr()))
        return compareExpressionToZero(Context, UnOp->getSubExpr(), false);
    }
  }

  if (needsNullPtrComparison(E))
    return compareExpressionToNullPtr(Context, E, true);

  if (needsZeroComparison(E))
    return compareExpressionToZero(Context, E, true);

  return asBool(getText(Context, *E), NeedsStaticCast);
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

static bool containsDiscardedTokens(const ASTContext &Context,
                                    CharSourceRange CharRange) {
  std::string ReplacementText =
      Lexer::getSourceText(CharRange, Context.getSourceManager(),
                           Context.getLangOpts())
          .str();
  Lexer Lex(CharRange.getBegin(), Context.getLangOpts(), ReplacementText.data(),
            ReplacementText.data(),
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
  Visitor(SimplifyBooleanExprCheck *Check, ASTContext &Context)
      : Check(Check), Context(Context) {}

  bool traverse() { return TraverseAST(Context); }

  static bool shouldIgnore(Stmt *S) {
    switch (S->getStmtClass()) {
    case Stmt::ImplicitCastExprClass:
    case Stmt::MaterializeTemporaryExprClass:
    case Stmt::CXXBindTemporaryExprClass:
      return true;
    default:
      return false;
    }
  }

  bool dataTraverseStmtPre(Stmt *S) {
    if (S && !shouldIgnore(S))
      StmtStack.push_back(S);
    return true;
  }

  bool dataTraverseStmtPost(Stmt *S) {
    if (S && !shouldIgnore(S)) {
      assert(StmtStack.back() == S);
      StmtStack.pop_back();
    }
    return true;
  }

  bool VisitBinaryOperator(const BinaryOperator *Op) const {
    Check->reportBinOp(Context, Op);
    return true;
  }

  // Extracts a bool if an expression is (true|false|!true|!false);
  static Optional<bool> getAsBoolLiteral(const Expr *E, bool FilterMacro) {
    if (const auto *Bool = dyn_cast<CXXBoolLiteralExpr>(E)) {
      if (FilterMacro && Bool->getBeginLoc().isMacroID())
        return llvm::None;
      return Bool->getValue();
    }
    if (const auto *UnaryOp = dyn_cast<UnaryOperator>(E)) {
      if (FilterMacro && UnaryOp->getBeginLoc().isMacroID())
        return None;
      if (UnaryOp->getOpcode() == UO_LNot)
        if (Optional<bool> Res = getAsBoolLiteral(
                UnaryOp->getSubExpr()->IgnoreImplicit(), FilterMacro))
          return !*Res;
    }
    return llvm::None;
  }

  template <typename Node> struct NodeAndBool {
    const Node *Item = nullptr;
    bool Bool = false;

    operator bool() const { return Item != nullptr; }
  };

  using ExprAndBool = NodeAndBool<Expr>;
  using DeclAndBool = NodeAndBool<Decl>;

  /// Detect's return (true|false|!true|!false);
  static ExprAndBool parseReturnLiteralBool(const Stmt *S) {
    const auto *RS = dyn_cast<ReturnStmt>(S);
    if (!RS || !RS->getRetValue())
      return {};
    if (Optional<bool> Ret =
            getAsBoolLiteral(RS->getRetValue()->IgnoreImplicit(), false)) {
      return {RS->getRetValue(), *Ret};
    }
    return {};
  }

  /// If \p S is not a \c CompoundStmt, applies F on \p S, otherwise if there is
  /// only 1 statement in the \c CompoundStmt, applies F on that single
  /// statement.
  template <typename Functor>
  static auto checkSingleStatement(Stmt *S, Functor F) -> decltype(F(S)) {
    if (auto *CS = dyn_cast<CompoundStmt>(S)) {
      if (CS->size() == 1)
        return F(CS->body_front());
      return {};
    }
    return F(S);
  }

  Stmt *parent() const {
    return StmtStack.size() < 2 ? nullptr : StmtStack[StmtStack.size() - 2];
  }

  bool VisitIfStmt(IfStmt *If) {
    /*
     * if (true) ThenStmt(); -> ThenStmt();
     * if (false) ThenStmt(); -> <Empty>;
     * if (false) ThenStmt(); else ElseStmt() -> ElseStmt();
     */
    Expr *Cond = If->getCond()->IgnoreImplicit();
    if (Optional<bool> Bool = getAsBoolLiteral(Cond, true)) {
      if (*Bool)
        Check->replaceWithThenStatement(Context, If, Cond);
      else
        Check->replaceWithElseStatement(Context, If, Cond);
    }

    if (If->getElse()) {
      /*
       * if (Cond) return true; else return false; -> return Cond;
       * if (Cond) return false; else return true; -> return !Cond;
       */
      if (ExprAndBool ThenReturnBool =
              checkSingleStatement(If->getThen(), parseReturnLiteralBool)) {
        ExprAndBool ElseReturnBool =
            checkSingleStatement(If->getElse(), parseReturnLiteralBool);
        if (ElseReturnBool && ThenReturnBool.Bool != ElseReturnBool.Bool) {
          if (Check->ChainedConditionalReturn ||
              !isa_and_nonnull<IfStmt>(parent())) {
            Check->replaceWithReturnCondition(Context, If, ThenReturnBool.Item,
                                              ElseReturnBool.Bool);
          }
        }
      } else {
        /*
         * if (Cond) A = true; else A = false; -> A = Cond;
         * if (Cond) A = false; else A = true; -> A = !Cond;
         */
        Expr *Var = nullptr;
        SourceLocation Loc;
        auto VarBoolAssignmentMatcher = [&Var,
                                         &Loc](const Stmt *S) -> DeclAndBool {
          const auto *BO = dyn_cast<BinaryOperator>(S);
          if (!BO || BO->getOpcode() != BO_Assign)
            return {};
          Optional<bool> RightasBool =
              getAsBoolLiteral(BO->getRHS()->IgnoreImplicit(), false);
          if (!RightasBool)
            return {};
          Expr *IgnImp = BO->getLHS()->IgnoreImplicit();
          if (!Var) {
            // We only need to track these for the Then branch.
            Loc = BO->getRHS()->getBeginLoc();
            Var = IgnImp;
          }
          if (auto *DRE = dyn_cast<DeclRefExpr>(IgnImp))
            return {DRE->getDecl(), *RightasBool};
          if (auto *ME = dyn_cast<MemberExpr>(IgnImp))
            return {ME->getMemberDecl(), *RightasBool};
          return {};
        };
        if (DeclAndBool ThenAssignment =
                checkSingleStatement(If->getThen(), VarBoolAssignmentMatcher)) {
          DeclAndBool ElseAssignment =
              checkSingleStatement(If->getElse(), VarBoolAssignmentMatcher);
          if (ElseAssignment.Item == ThenAssignment.Item &&
              ElseAssignment.Bool != ThenAssignment.Bool) {
            if (Check->ChainedConditionalAssignment ||
                !isa_and_nonnull<IfStmt>(parent())) {
              Check->replaceWithAssignment(Context, If, Var, Loc,
                                           ElseAssignment.Bool);
            }
          }
        }
      }
    }
    return true;
  }

  bool VisitConditionalOperator(ConditionalOperator *Cond) {
    /*
     * Condition ? true : false; -> Condition
     * Condition ? false : true; -> !Condition;
     */
    if (Optional<bool> Then =
            getAsBoolLiteral(Cond->getTrueExpr()->IgnoreImplicit(), false)) {
      if (Optional<bool> Else =
              getAsBoolLiteral(Cond->getFalseExpr()->IgnoreImplicit(), false)) {
        if (*Then != *Else)
          Check->replaceWithCondition(Context, Cond, *Else);
      }
    }
    return true;
  }

  bool VisitCompoundStmt(CompoundStmt *CS) {
    if (CS->size() < 2)
      return true;
    bool CurIf = false, PrevIf = false;
    for (auto First = CS->body_begin(), Second = std::next(First),
              End = CS->body_end();
         Second != End; ++Second, ++First) {
      PrevIf = CurIf;
      CurIf = isa<IfStmt>(*First);
      ExprAndBool TrailingReturnBool = parseReturnLiteralBool(*Second);
      if (!TrailingReturnBool)
        continue;

      if (CurIf) {
        /*
         * if (Cond) return true; return false; -> return Cond;
         * if (Cond) return false; return true; -> return !Cond;
         */
        auto *If = cast<IfStmt>(*First);
        ExprAndBool ThenReturnBool =
            checkSingleStatement(If->getThen(), parseReturnLiteralBool);
        if (ThenReturnBool && ThenReturnBool.Bool != TrailingReturnBool.Bool) {
          if (Check->ChainedConditionalReturn ||
              (!PrevIf && If->getElse() == nullptr)) {
            Check->replaceCompoundReturnWithCondition(
                Context, cast<ReturnStmt>(*Second), TrailingReturnBool.Bool,
                If);
          }
        }
      } else if (isa<LabelStmt, CaseStmt, DefaultStmt>(*First)) {
        /*
         * (case X|label_X|default): if (Cond) return BoolLiteral;
         *                           return !BoolLiteral
         */
        Stmt *SubStmt =
            isa<LabelStmt>(*First)  ? cast<LabelStmt>(*First)->getSubStmt()
            : isa<CaseStmt>(*First) ? cast<CaseStmt>(*First)->getSubStmt()
                                    : cast<DefaultStmt>(*First)->getSubStmt();
        auto *SubIf = dyn_cast<IfStmt>(SubStmt);
        if (SubIf && !SubIf->getElse()) {
          ExprAndBool ThenReturnBool =
              checkSingleStatement(SubIf->getThen(), parseReturnLiteralBool);
          if (ThenReturnBool &&
              ThenReturnBool.Bool != TrailingReturnBool.Bool) {
            Check->replaceCompoundReturnWithCondition(
                Context, cast<ReturnStmt>(*Second), TrailingReturnBool.Bool,
                SubIf);
          }
        }
      }
    }
    return true;
  }

private:
  SimplifyBooleanExprCheck *Check;
  SmallVector<Stmt *, 32> StmtStack;
  ASTContext &Context;
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

void SimplifyBooleanExprCheck::reportBinOp(const ASTContext &Context,
                                           const BinaryOperator *Op) {
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

  auto ReplaceWithExpression = [this, &Context, LHS, RHS,
                                Bool](const Expr *ReplaceWith, bool Negated) {
    std::string Replacement =
        replacementExpression(Context, Negated, ReplaceWith);
    SourceRange Range(LHS->getBeginLoc(), RHS->getEndLoc());
    issueDiag(Context, Bool->getBeginLoc(), SimplifyOperatorDiagnostic, Range,
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

void SimplifyBooleanExprCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "ChainedConditionalReturn", ChainedConditionalReturn);
  Options.store(Opts, "ChainedConditionalAssignment",
                ChainedConditionalAssignment);
}

void SimplifyBooleanExprCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(translationUnitDecl(), this);
}

void SimplifyBooleanExprCheck::check(const MatchFinder::MatchResult &Result) {
  Visitor(this, *Result.Context).traverse();
}

void SimplifyBooleanExprCheck::issueDiag(const ASTContext &Context,
                                         SourceLocation Loc,
                                         StringRef Description,
                                         SourceRange ReplacementRange,
                                         StringRef Replacement) {
  CharSourceRange CharRange =
      Lexer::makeFileCharRange(CharSourceRange::getTokenRange(ReplacementRange),
                               Context.getSourceManager(), getLangOpts());

  DiagnosticBuilder Diag = diag(Loc, Description);
  if (!containsDiscardedTokens(Context, CharRange))
    Diag << FixItHint::CreateReplacement(CharRange, Replacement);
}

void SimplifyBooleanExprCheck::replaceWithThenStatement(
    const ASTContext &Context, const IfStmt *IfStatement,
    const Expr *BoolLiteral) {
  issueDiag(Context, BoolLiteral->getBeginLoc(), SimplifyConditionDiagnostic,
            IfStatement->getSourceRange(),
            getText(Context, *IfStatement->getThen()));
}

void SimplifyBooleanExprCheck::replaceWithElseStatement(
    const ASTContext &Context, const IfStmt *IfStatement,
    const Expr *BoolLiteral) {
  const Stmt *ElseStatement = IfStatement->getElse();
  issueDiag(Context, BoolLiteral->getBeginLoc(), SimplifyConditionDiagnostic,
            IfStatement->getSourceRange(),
            ElseStatement ? getText(Context, *ElseStatement) : "");
}

void SimplifyBooleanExprCheck::replaceWithCondition(
    const ASTContext &Context, const ConditionalOperator *Ternary,
    bool Negated) {
  std::string Replacement =
      replacementExpression(Context, Negated, Ternary->getCond());
  issueDiag(Context, Ternary->getTrueExpr()->getBeginLoc(),
            "redundant boolean literal in ternary expression result",
            Ternary->getSourceRange(), Replacement);
}

void SimplifyBooleanExprCheck::replaceWithReturnCondition(
    const ASTContext &Context, const IfStmt *If, const Expr *BoolLiteral,
    bool Negated) {
  StringRef Terminator = isa<CompoundStmt>(If->getElse()) ? ";" : "";
  std::string Condition =
      replacementExpression(Context, Negated, If->getCond());
  std::string Replacement = ("return " + Condition + Terminator).str();
  SourceLocation Start = BoolLiteral->getBeginLoc();
  issueDiag(Context, Start, SimplifyConditionalReturnDiagnostic,
            If->getSourceRange(), Replacement);
}

void SimplifyBooleanExprCheck::replaceCompoundReturnWithCondition(
    const ASTContext &Context, const ReturnStmt *Ret, bool Negated,
    const IfStmt *If) {
  const auto *Lit = stmtReturnsBool(If, Negated);
  const std::string Replacement =
      "return " + replacementExpression(Context, Negated, If->getCond());
  issueDiag(Context, Lit->getBeginLoc(), SimplifyConditionalReturnDiagnostic,
            SourceRange(If->getBeginLoc(), Ret->getEndLoc()), Replacement);
}

void SimplifyBooleanExprCheck::replaceWithAssignment(const ASTContext &Context,
                                                     const IfStmt *IfAssign,
                                                     const Expr *Var,
                                                     SourceLocation Loc,
                                                     bool Negated) {
  SourceRange Range = IfAssign->getSourceRange();
  StringRef VariableName = getText(Context, *Var);
  StringRef Terminator = isa<CompoundStmt>(IfAssign->getElse()) ? ";" : "";
  std::string Condition =
      replacementExpression(Context, Negated, IfAssign->getCond());
  std::string Replacement =
      (VariableName + " = " + Condition + Terminator).str();
  issueDiag(Context, Loc, "redundant boolean literal in conditional assignment",
            Range, Replacement);
}

} // namespace readability
} // namespace tidy
} // namespace clang
