//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RedundantBranchConditionCheck.h"
#include "../utils/Aliasing.h"
#include "../utils/LexerUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/AST/StmtCXX.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Analysis/Analyses/ExprMutationAnalyzer.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;
using clang::tidy::utils::hasPtrOrReferenceInFunc;

namespace clang::tidy::bugprone {

static const char CondVarStr[] = "cond_var";
static const char OuterIfStr[] = "outer_if";
static const char InnerIfStr[] = "inner_if";
static const char OuterIfVarStr[] = "outer_if_var";
static const char InnerIfVarStr[] = "inner_if_var";
static const char FuncStr[] = "func";

/// Returns whether `Var` is changed in range (`PrevS`..`NextS`).
static bool isChangedBefore(const Stmt *S, const Stmt *NextS, const Stmt *PrevS,
                            const VarDecl *Var, ASTContext *Context) {
  ExprMutationAnalyzer MutAn(*S, *Context);
  const auto &SM = Context->getSourceManager();
  const Stmt *MutS = MutAn.findMutation(Var);
  return MutS &&
         SM.isBeforeInTranslationUnit(PrevS->getEndLoc(),
                                      MutS->getBeginLoc()) &&
         SM.isBeforeInTranslationUnit(MutS->getEndLoc(), NextS->getBeginLoc());
}

/// Returns the outermost loop that encloses `S` and is itself enclosed by
/// `Outer`, or null if there is no such loop. The walk passes through
/// declarations, such as a variable initialized by a lambda, but stops at the
/// enclosing function.
static const Stmt *getOutermostLoopBetween(const Stmt *S, const Stmt *Outer,
                                           ASTContext *Context) {
  const Stmt *Loop = nullptr;
  // getParents() returns only the direct parents of a node, usually exactly
  // one, so the walk calls it once per level.
  DynTypedNodeList Parents = Context->getParents(*S);
  while (!Parents.empty()) {
    const DynTypedNode Parent = Parents[0];
    if (Parent.get<FunctionDecl>())
      break;
    if (const auto *ParentStmt = Parent.get<Stmt>()) {
      if (ParentStmt == Outer)
        break;
      if (isa<ForStmt, WhileStmt, DoStmt, CXXForRangeStmt>(ParentStmt))
        Loop = ParentStmt;
    }
    Parents = Context->getParents(Parent);
  }
  return Loop;
}

namespace {
// ast matcher to get all declaration whose  all ancestor operator are matched
// in AncestorOperatorMatcher.
AST_MATCHER_P2(Expr, getAllDeclWithAncestor,
               ast_matchers::internal::Matcher<clang::Expr>,
               AncestorOperatorMatcher, ast_matchers::internal::Matcher<Expr>,
               InnerMatcher) {
  const auto *RootExpr = Node.IgnoreParenImpCasts();
  const ast_matchers::internal::BoundNodesTreeBuilder InitialBuildTree(
      *Builder);
  if (!RootExpr)
    return false;
  llvm::SmallVector<const Expr *, 16> ExprStack;
  ExprStack.push_back(RootExpr);
  bool Matched = false;
  while (!ExprStack.empty()) {
    const auto *TopExpr = ExprStack.back();
    ExprStack.pop_back();
    if (!TopExpr)
      continue;
    const auto *TopExprIgParen = TopExpr->IgnoreParenImpCasts();
    if (const auto *BO = dyn_cast<BinaryOperator>(TopExprIgParen)) {
      ast_matchers::internal::BoundNodesTreeBuilder Useless;
      if (!AncestorOperatorMatcher.matches(*BO, Finder, &Useless))
        continue;
      ExprStack.push_back(BO->getLHS());
      ExprStack.push_back(BO->getRHS());

    } else if (const auto *UO = dyn_cast<UnaryOperator>(TopExprIgParen)) {
      ast_matchers::internal::BoundNodesTreeBuilder Useless;
      if (!AncestorOperatorMatcher.matches(*UO, Finder, &Useless))
        continue;
      ExprStack.push_back(UO->getSubExpr());
    }
    if (isa<DeclRefExpr>(TopExprIgParen)) {
      ast_matchers::internal::BoundNodesTreeBuilder Branch(InitialBuildTree);
      if (!InnerMatcher.matches(*TopExprIgParen, Finder, &Branch))
        continue;

      Builder->addMatch(Branch);
      Matched = true;
    }
  }
  return Matched;
}
} // namespace

void RedundantBranchConditionCheck::registerMatchers(MatchFinder *Finder) {
  const auto ImmutableVar =
      varDecl(anyOf(parmVarDecl(), hasLocalStorage()), hasType(isInteger()),
              unless(hasType(isVolatileQualified())))
          .bind(CondVarStr);
  Finder->addMatcher(
      ifStmt(
          hasCondition(getAllDeclWithAncestor(
              binaryOperator(hasOperatorName("&&")),
              declRefExpr(hasDeclaration(ImmutableVar)).bind(OuterIfVarStr))),
          hasThen(hasDescendant(
              ifStmt(hasCondition(getAllDeclWithAncestor(
                         anyOf(binaryOperator(hasAnyOperatorName("&&", "||")),
                               unaryOperator(hasOperatorName("!"))),
                         declRefExpr(hasDeclaration(
                                         varDecl(equalsBoundNode(CondVarStr))))
                             .bind(InnerIfVarStr))))
                  .bind(InnerIfStr))),
          forFunction(functionDecl().bind(FuncStr)))
          .bind(OuterIfStr),
      this);
}

namespace {

enum class SimpliFiedExprKind {
  True,
  False,
  NonDeductible, // those expr where are only variable are cannot be minimize
  FixUnavailable // when known variable came from macro we should only emit
                 // warning as cannot be fixable without expanding macro.
};
class ExprState {
public:
  SimpliFiedExprKind ExprKind;
  llvm::SmallVector<FixItHint, 8> Fixits;
  FixItHint ToremoveTF = {}; // use for remove true/false section in case the
  int HasSideEffect = false; // applied only for true and false
  void includeRemoveTFFixIt() {
    std::swap(ToremoveTF,
              Fixits[0]); // NOTE: in getFixItsinExpr the property always
    // maintained that False or True replacement remain at start of
    // SmallVector. if edit in future do not alter the property.
  }
};
} // namespace
// function to check if a expr has desendent relation with rootExpr.
static bool isExprChild(const Expr *RootExpr, const Expr *InnerExpr,
                        ASTContext &Ctx) {
  if (!RootExpr || !InnerExpr || RootExpr == InnerExpr)
    return false;
  const auto MatchResult =
      match(expr(hasDescendant(expr(equalsNode(InnerExpr)))), *RootExpr, Ctx);
  return !MatchResult.empty();
}
// to get all removals that can be done.
static ExprState getFixItsinExpr(const Expr *ExprNode,
                                 const DeclRefExpr *KnownVarExpr,
                                 ASTContext &Ctx) {
  const Expr *ExprRemImpPar =
      ExprNode->IgnoreParenImpCasts(); // we should not simplify parenthesis as
                                       // our check only task is to check for
                                       // redundant branch.
  const SourceManager &SM = Ctx.getSourceManager();
  const LangOptions &LangOpts = Ctx.getLangOpts();
  const llvm::StringRef LangTrueStr = LangOpts.CPlusPlus ? "true" : "1";
  const llvm::StringRef LangFalseStr = LangOpts.CPlusPlus ? "false" : "0";
  // if we find macro we check if it contain are target Expr.if True we can't
  // deduce it so unfixable.
  if (ExprRemImpPar->getExprLoc().isMacroID()) {
    if (isExprChild(ExprNode, KnownVarExpr, Ctx))
      return ExprState{SimpliFiedExprKind::FixUnavailable, {}};
    return ExprState{SimpliFiedExprKind::NonDeductible, {}};
  }
  // to check whether Declaration Expr is The one Expr that we bound to InnerIf
  // NOTE: we are Matching at node/Expr level do not replace with CondVarDecl as
  // branching assume only one Expr(KnownVarExpr) is processed.eq if(onFire){
  // if(onFire && onFire){}} as only on is bind in each warning we must only
  // process one(inner onFire)  not both.
  if (const auto *DeclExpr = dyn_cast<DeclRefExpr>(ExprRemImpPar)) {
    if (DeclExpr == KnownVarExpr) {
      const SourceLocation DeclExprEnd =
          Lexer::getLocForEndOfToken(DeclExpr->getEndLoc(), 0, SM, LangOpts);
      return ExprState{SimpliFiedExprKind::True,
                       {FixItHint::CreateReplacement(
                           CharSourceRange::getCharRange(
                               DeclExpr->getBeginLoc(), DeclExprEnd),
                           LangTrueStr)},
                       FixItHint::CreateRemoval(CharSourceRange::getCharRange(
                           DeclExpr->getBeginLoc(), DeclExprEnd))};
    }
    return ExprState{SimpliFiedExprKind::NonDeductible, {}};
  }

  if (const auto *UO = dyn_cast<UnaryOperator>(ExprRemImpPar)) {
    if (UO->getOpcode() != UO_LNot)
      return ExprState{SimpliFiedExprKind::NonDeductible, {}};
    auto InnerExpr = getFixItsinExpr(UO->getSubExpr(), KnownVarExpr, Ctx);
    const SourceLocation UOExprEnd =
        Lexer::getLocForEndOfToken(UO->getEndLoc(), 0, SM, LangOpts);
    if (InnerExpr.ExprKind == SimpliFiedExprKind::FixUnavailable)
      return InnerExpr;
    if (InnerExpr.ExprKind == SimpliFiedExprKind::True &&
        !InnerExpr.HasSideEffect) {
      return ExprState{
          SimpliFiedExprKind::False,
          {FixItHint::CreateReplacement(
              CharSourceRange::getCharRange(UO->getBeginLoc(), UOExprEnd),
              LangFalseStr)},
          FixItHint::CreateRemoval(
              CharSourceRange::getCharRange(UO->getBeginLoc(), UOExprEnd))};
    }
    if (InnerExpr.ExprKind == SimpliFiedExprKind::False &&
        !InnerExpr.HasSideEffect) {
      return ExprState{
          SimpliFiedExprKind::True,
          {FixItHint::CreateReplacement(
              CharSourceRange::getCharRange(UO->getBeginLoc(), UOExprEnd),
              LangTrueStr)},
          FixItHint::CreateRemoval(
              CharSourceRange::getCharRange(UO->getBeginLoc(), UOExprEnd))};
    }
    return ExprState{SimpliFiedExprKind::NonDeductible,
                     std::move(InnerExpr.Fixits)};
  }

  if (const auto *BO = dyn_cast<BinaryOperator>(ExprRemImpPar)) {
    if (!BO->isLogicalOp())
      return ExprState{SimpliFiedExprKind::NonDeductible, {}};

    ExprState LHS = getFixItsinExpr(BO->getLHS(), KnownVarExpr, Ctx);
    ExprState RHS = getFixItsinExpr(BO->getRHS(), KnownVarExpr, Ctx);

    if (LHS.ExprKind == SimpliFiedExprKind::FixUnavailable ||
        RHS.ExprKind == SimpliFiedExprKind::FixUnavailable) {
      return ExprState{SimpliFiedExprKind::FixUnavailable, {}};
    }
    const SourceLocation BOEnd =
        Lexer::getLocForEndOfToken(BO->getEndLoc(), 0, SM, LangOpts);
    const CharSourceRange RHSremRange =
        CharSourceRange::getCharRange(BO->getOperatorLoc(), BOEnd);
    const CharSourceRange LHSremRange = CharSourceRange::getCharRange(
        BO->getBeginLoc(), BO->getOperatorLoc().getLocWithOffset(2));

    if (BO->getOpcode() == BO_LAnd) {
      // false && RHS(SDE) ,false && RHS(NOSDE)
      if (LHS.ExprKind == SimpliFiedExprKind::False) {
        LHS.Fixits.push_back(FixItHint::CreateRemoval(RHSremRange));
        return LHS;
      }
      // LHS(NOSDE) && false ( where LHS do not have sideEffect )
      if (RHS.ExprKind == SimpliFiedExprKind::False &&
          !BO->getLHS()->HasSideEffects(Ctx)) {
        RHS.Fixits.push_back(FixItHint::CreateRemoval(LHSremRange));
        return RHS;
      }
      // LHS(SDE) && false ( where LHS do have sideEffect )
      if (RHS.ExprKind == SimpliFiedExprKind::False &&
          BO->getLHS()->HasSideEffects(Ctx)) {
        if (!RHS.HasSideEffect) {
          RHS.ToremoveTF = FixItHint::CreateRemoval(RHSremRange);
          RHS.HasSideEffect = true;
        }
        return RHS;
      }
      // true && RHS(NOSDE) , true && RHS(SDE)
      if (LHS.ExprKind == SimpliFiedExprKind::True) {
        if (LHS.HasSideEffect) {
          RHS.Fixits.append(LHS.Fixits.begin(), LHS.Fixits.end());
          return RHS;
        }
        RHS.Fixits.push_back(FixItHint::CreateRemoval(LHSremRange));
        return RHS;
      }
      // LHS(NOSDE) && true , LHS(SDE) && true
      if (RHS.ExprKind == SimpliFiedExprKind::True) {
        if (RHS.HasSideEffect) {
          LHS.Fixits.append(RHS.Fixits.begin(), RHS.Fixits.end());
          return LHS;
        }
        LHS.Fixits.push_back(FixItHint::CreateRemoval(RHSremRange));
        return LHS;
      }
    } else {
      // true || RHS(NOSDE) , true || RHS(SDE)
      if (LHS.ExprKind == SimpliFiedExprKind::True) {
        LHS.Fixits.push_back(FixItHint::CreateRemoval(RHSremRange));
        return LHS;
      }
      // LHS(NOSDE) || true
      if (RHS.ExprKind == SimpliFiedExprKind::True &&
          !BO->getLHS()->HasSideEffects(Ctx)) {
        RHS.Fixits.push_back(FixItHint::CreateRemoval(LHSremRange));
        return RHS;
      }
      // LHS(SDE) || true
      if (RHS.ExprKind == SimpliFiedExprKind::True &&
          BO->getLHS()->HasSideEffects(Ctx)) {
        if (!RHS.HasSideEffect) {
          RHS.ToremoveTF = FixItHint::CreateRemoval(RHSremRange);
          RHS.HasSideEffect = true;
        }
        return RHS;
      }
      // false || RHS(NOSDE) , false || RHS(SDE)
      if (LHS.ExprKind == SimpliFiedExprKind::False) {
        if (LHS.HasSideEffect) {
          RHS.Fixits.append(LHS.Fixits.begin(), LHS.Fixits.end());
          return RHS;
        }
        RHS.Fixits.push_back(FixItHint::CreateRemoval(LHSremRange));
        return RHS;
      }
      // LHS(NOSDE) || false , LHS(SDE) || false
      if (RHS.ExprKind == SimpliFiedExprKind::False) {
        if (RHS.HasSideEffect) {
          LHS.Fixits.append(RHS.Fixits.begin(), RHS.Fixits.end());
          return LHS;
        }
        LHS.Fixits.push_back(FixItHint::CreateRemoval(RHSremRange));
        return LHS;
      }
      // Note: only one time there is only one False/true in one side as we only
      // process specific innerIf bind and expr such as true or false token we
      // assume them as dynamic to do not show unrelated changes
    }
    LHS.Fixits.append(RHS.Fixits.begin(), RHS.Fixits.end());
    return LHS;
  }
  return ExprState{SimpliFiedExprKind::NonDeductible, {}};
}
void RedundantBranchConditionCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *OuterIf = Result.Nodes.getNodeAs<IfStmt>(OuterIfStr);
  const auto *InnerIf = Result.Nodes.getNodeAs<IfStmt>(InnerIfStr);
  const auto *CondVar = Result.Nodes.getNodeAs<VarDecl>(CondVarStr);
  const auto *Func = Result.Nodes.getNodeAs<FunctionDecl>(FuncStr);
  const DeclRefExpr *OuterIfVar = nullptr, *InnerIfVar = nullptr;
  if (const auto *Inner = Result.Nodes.getNodeAs<DeclRefExpr>(InnerIfVarStr))
    InnerIfVar = Inner;
  else
    return;
  if (const auto *Outer = Result.Nodes.getNodeAs<DeclRefExpr>(OuterIfVarStr))
    OuterIfVar = Outer;
  else
    return;

  if (OuterIfVar && InnerIfVar) {
    if (isChangedBefore(OuterIf->getThen(), InnerIfVar, OuterIfVar, CondVar,
                        Result.Context))
      return;

    if (isChangedBefore(OuterIf->getCond(), InnerIfVar, OuterIfVar, CondVar,
                        Result.Context))
      return;
  }

  // Inside a loop, a mutation anywhere in the loop runs before the inner
  // condition is evaluated again, even if it comes later in the source.
  const Stmt *Loop = getOutermostLoopBetween(InnerIf, OuterIf, Result.Context);
  if (Loop &&
      ExprMutationAnalyzer(*Loop, *Result.Context).findMutation(CondVar))
    return;

  // If the variable has an alias then it can be changed by that alias as well.
  // FIXME: could potentially support tracking pointers and references in the
  // future to improve catching true positives through aliases.
  if (hasPtrOrReferenceInFunc(Func, CondVar))
    return;

  const auto Diag = diag(InnerIf->getBeginLoc(), "redundant condition %0")
                    << CondVar;
  auto FixItResult =
      getFixItsinExpr(InnerIf->getCond(), InnerIfVar, *Result.Context);
  if (FixItResult.ExprKind == SimpliFiedExprKind::FixUnavailable)
    return;
  if (FixItResult.ExprKind == SimpliFiedExprKind::NonDeductible) {
    for (const auto &FixIts : FixItResult.Fixits)
      Diag << FixIts;
    return;
  }
  bool IsTransformValid = true;
  llvm::SmallVector<FixItHint, 8> IfBodyFixIts;
  const SourceManager &SM = (*Result.Context).getSourceManager();
  const LangOptions &LangOpts = (*Result.Context).getLangOpts();
  const SourceLocation IfBegin = InnerIf->getBeginLoc();
  if (!FixItResult.HasSideEffect) {
    const auto IfBeginToRParenRg =
        CharSourceRange::getTokenRange(IfBegin, InnerIf->getRParenLoc());
    if (IfBeginToRParenRg.isValid())
      IfBodyFixIts.push_back(FixItHint::CreateRemoval(IfBeginToRParenRg));
    else
      IsTransformValid = false;

  } else {
    const auto IfBeginRg = CharSourceRange::getTokenRange(IfBegin);
    const auto IfLPRg = CharSourceRange::getTokenRange(InnerIf->getLParenLoc());
    const auto IfRPRg = CharSourceRange::getTokenRange(InnerIf->getRParenLoc());
    if (IfBeginRg.isValid() && IfLPRg.isValid() && IfRPRg.isValid()) {
      IfBodyFixIts.push_back(FixItHint::CreateRemoval(IfBeginRg));
      IfBodyFixIts.push_back(FixItHint::CreateRemoval(IfLPRg));
      IfBodyFixIts.push_back(FixItHint::CreateRemoval(IfRPRg));
      IfBodyFixIts.push_back(
          FixItHint::CreateInsertion(InnerIf->getRParenLoc(), ";"));
    } else {
      IsTransformValid = false;
    }
  }

  if (FixItResult.ExprKind == SimpliFiedExprKind::True) {
    // remove braces
    const Stmt *IfBody = InnerIf->getThen();
    if (const auto *CS = dyn_cast<CompoundStmt>(IfBody)) {
      const auto IfLBRg = CharSourceRange::getTokenRange(CS->getLBracLoc());
      const auto IfRBRg = CharSourceRange::getTokenRange(CS->getRBracLoc());
      if (IfLBRg.isValid() && IfRBRg.isValid()) {
        IfBodyFixIts.push_back(FixItHint::CreateRemoval(IfLBRg));
        IfBodyFixIts.push_back(FixItHint::CreateRemoval(IfRBRg));
      } else {
        IsTransformValid = false;
      }
    }
    // remove else part as not used.
    const Stmt *ElseBody = InnerIf->getElse();
    if (ElseBody) {
      CharSourceRange ElseRg;
      if (const auto *CS = dyn_cast<CompoundStmt>(ElseBody)) {
        ElseRg = CharSourceRange::getTokenRange(
            InnerIf->getElseLoc(), // this to get else 'keyword' location
            CS->getRBracLoc());
      } else {
        const auto ElseEndLocWithSemi = Lexer::findLocationAfterToken(
            ElseBody->getEndLoc(), tok::semi, SM, LangOpts, true);
        ElseRg = CharSourceRange::getTokenRange(InnerIf->getElseLoc(),
                                                ElseEndLocWithSemi);
      }
      if (ElseRg.isValid())
        IfBodyFixIts.push_back(FixItHint::CreateRemoval(ElseRg));
      else
        IsTransformValid = false;
    }
  }
  if (FixItResult.ExprKind == SimpliFiedExprKind::False) {
    // remove if body
    const Stmt *IfBody = InnerIf->getThen();
    CharSourceRange IfBodyRg;
    if (const auto *CS = dyn_cast<CompoundStmt>(IfBody)) {
      IfBodyRg =
          CharSourceRange::getTokenRange(CS->getLBracLoc(), CS->getRBracLoc());
    } else { // remove extra semicolon on removing singleton expression.
      const auto IfEndLocWithSemi = Lexer::findLocationAfterToken(
          IfBody->getEndLoc(), tok::semi, SM, LangOpts, true);
      IfBodyRg = CharSourceRange::getTokenRange(IfBody->getBeginLoc(),
                                                IfEndLocWithSemi);
    }
    if (IfBodyRg.isValid())
      IfBodyFixIts.push_back(FixItHint::CreateRemoval(IfBodyRg));
    else
      IsTransformValid = false;

    const Stmt *ElseBody = InnerIf->getElse();
    if (ElseBody) {
      // remove else word as not needed
      const SourceManager &SM = (*Result.Context).getSourceManager();
      const LangOptions &LangOpts = (*Result.Context).getLangOpts();
      const auto ElseTokRg = Lexer::makeFileCharRange(
          CharSourceRange::getTokenRange(InnerIf->getElseLoc()), SM, LangOpts);
      if (ElseTokRg.isValid())
        IfBodyFixIts.push_back(FixItHint::CreateRemoval(ElseTokRg));
      else
        IsTransformValid = false;
      // remove braces from else block
      if (const auto *CS = dyn_cast<CompoundStmt>(ElseBody)) {
        const auto ElseLBRg = CharSourceRange::getTokenRange(CS->getLBracLoc());
        const auto ElseRBRg = CharSourceRange::getTokenRange(CS->getRBracLoc());
        if (ElseLBRg.isValid() && ElseRBRg.isValid()) {
          IfBodyFixIts.push_back(FixItHint::CreateRemoval(ElseLBRg));
          IfBodyFixIts.push_back(FixItHint::CreateRemoval(ElseRBRg));
        } else {
          IsTransformValid = false;
        }
      }
    }
  }
  if (IsTransformValid) {
    FixItResult.includeRemoveTFFixIt();
    for (const auto &FixIts : IfBodyFixIts)
      Diag << FixIts;
  }
  for (const auto &FixIts : FixItResult.Fixits)
    Diag << FixIts;
}

} // namespace clang::tidy::bugprone
