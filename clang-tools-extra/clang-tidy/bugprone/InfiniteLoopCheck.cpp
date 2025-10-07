//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InfiniteLoopCheck.h"
#include "../utils/Aliasing.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Analysis/Analyses/ExprMutationAnalyzer.h"
#include "clang/Analysis/CallGraph.h"
#include "llvm/ADT/SCCIterator.h"

using namespace clang::ast_matchers;
using clang::ast_matchers::internal::Matcher;
using clang::tidy::utils::hasPtrOrReferenceInFunc;

namespace clang::tidy::bugprone {

namespace {
/// matches a Decl if it has a  "no return" attribute of any kind
AST_MATCHER(Decl, declHasNoReturnAttr) {
  return Node.hasAttr<NoReturnAttr>() || Node.hasAttr<CXX11NoReturnAttr>() ||
         Node.hasAttr<C11NoReturnAttr>();
}

/// matches a FunctionType if the type includes the GNU no return attribute
AST_MATCHER(FunctionType, typeHasNoReturnAttr) {
  return Node.getNoReturnAttr();
}
} // namespace

static Matcher<Stmt> loopEndingStmt(Matcher<Stmt> Internal) {
  Matcher<QualType> IsNoReturnFunType =
      ignoringParens(functionType(typeHasNoReturnAttr()));
  Matcher<Decl> IsNoReturnDecl =
      anyOf(declHasNoReturnAttr(), functionDecl(hasType(IsNoReturnFunType)),
            varDecl(hasType(blockPointerType(pointee(IsNoReturnFunType)))));

  return stmt(anyOf(
      mapAnyOf(breakStmt, returnStmt, gotoStmt, cxxThrowExpr).with(Internal),
      callExpr(Internal,
               callee(mapAnyOf(functionDecl, /* block callee */ varDecl)
                          .with(IsNoReturnDecl))),
      objcMessageExpr(Internal, callee(IsNoReturnDecl))));
}

/// Return whether `Var` was changed in `LoopStmt`.
static bool isChanged(const Stmt *LoopStmt, const ValueDecl *Var,
                      ASTContext *Context) {
  if (const auto *ForLoop = dyn_cast<ForStmt>(LoopStmt))
    return (ForLoop->getInc() &&
            ExprMutationAnalyzer(*ForLoop->getInc(), *Context)
                .isMutated(Var)) ||
           (ForLoop->getBody() &&
            ExprMutationAnalyzer(*ForLoop->getBody(), *Context)
                .isMutated(Var)) ||
           (ForLoop->getCond() &&
            ExprMutationAnalyzer(*ForLoop->getCond(), *Context).isMutated(Var));

  return ExprMutationAnalyzer(*LoopStmt, *Context).isMutated(Var);
}

static bool isVarPossiblyChanged(const Decl *Func, const Stmt *LoopStmt,
                                 const ValueDecl *VD, ASTContext *Context) {
  const VarDecl *Var = nullptr;
  if (const auto *VarD = dyn_cast<VarDecl>(VD)) {
    Var = VarD;
  } else if (const auto *BD = dyn_cast<BindingDecl>(VD)) {
    if (const auto *DD = dyn_cast<DecompositionDecl>(BD->getDecomposedDecl()))
      Var = DD;
  }

  if (!Var)
    return false;

  if (!Var->isLocalVarDeclOrParm() || Var->getType().isVolatileQualified())
    return true;

  if (!VD->getType().getTypePtr()->isIntegerType())
    return true;

  return hasPtrOrReferenceInFunc(Func, VD) || isChanged(LoopStmt, VD, Context);
  // FIXME: Track references.
}

/// Return whether `Cond` is a variable that is possibly changed in `LoopStmt`.
static bool isVarThatIsPossiblyChanged(const Decl *Func, const Stmt *LoopStmt,
                                       const Stmt *Cond, ASTContext *Context) {
  if (const auto *DRE = dyn_cast<DeclRefExpr>(Cond)) {
    if (const auto *VD = dyn_cast<ValueDecl>(DRE->getDecl()))
      return isVarPossiblyChanged(Func, LoopStmt, VD, Context);
  } else if (isa<MemberExpr, CallExpr, ObjCIvarRefExpr, ObjCPropertyRefExpr,
                 ObjCMessageExpr>(Cond)) {
    // FIXME: Handle MemberExpr.
    return true;
  } else if (const auto *CE = dyn_cast<CastExpr>(Cond)) {
    QualType T = CE->getType();
    while (true) {
      if (T.isVolatileQualified())
        return true;

      if (!T->isAnyPointerType() && !T->isReferenceType())
        break;

      T = T->getPointeeType();
    }
  }

  return false;
}

/// Return whether at least one variable of `Cond` changed in `LoopStmt`.
static bool isAtLeastOneCondVarChanged(const Decl *Func, const Stmt *LoopStmt,
                                       const Stmt *Cond, ASTContext *Context) {
  if (isVarThatIsPossiblyChanged(Func, LoopStmt, Cond, Context))
    return true;

  for (const Stmt *Child : Cond->children()) {
    if (!Child)
      continue;

    if (isAtLeastOneCondVarChanged(Func, LoopStmt, Child, Context))
      return true;
  }
  return false;
}

/// Return the variable names in `Cond`.
static std::string getCondVarNames(const Stmt *Cond) {
  if (const auto *DRE = dyn_cast<DeclRefExpr>(Cond)) {
    if (const auto *Var = dyn_cast<VarDecl>(DRE->getDecl()))
      return std::string(Var->getName());

    if (const auto *BD = dyn_cast<BindingDecl>(DRE->getDecl())) {
      return std::string(BD->getName());
    }
  }

  std::string Result;
  for (const Stmt *Child : Cond->children()) {
    if (!Child)
      continue;

    std::string NewNames = getCondVarNames(Child);
    if (!Result.empty() && !NewNames.empty())
      Result += ", ";
    Result += NewNames;
  }
  return Result;
}

static bool isKnownToHaveValue(const Expr &Cond, const ASTContext &Ctx,
                               bool ExpectedValue) {
  if (Cond.isValueDependent()) {
    if (const auto *BinOp = dyn_cast<BinaryOperator>(&Cond)) {
      // Conjunctions (disjunctions) can still be handled if at least one
      // conjunct (disjunct) is known to be false (true).
      if (!ExpectedValue && BinOp->getOpcode() == BO_LAnd)
        return isKnownToHaveValue(*BinOp->getLHS(), Ctx, false) ||
               isKnownToHaveValue(*BinOp->getRHS(), Ctx, false);
      if (ExpectedValue && BinOp->getOpcode() == BO_LOr)
        return isKnownToHaveValue(*BinOp->getLHS(), Ctx, true) ||
               isKnownToHaveValue(*BinOp->getRHS(), Ctx, true);
      if (BinOp->getOpcode() == BO_Comma)
        return isKnownToHaveValue(*BinOp->getRHS(), Ctx, ExpectedValue);
    } else if (const auto *UnOp = dyn_cast<UnaryOperator>(&Cond)) {
      if (UnOp->getOpcode() == UO_LNot)
        return isKnownToHaveValue(*UnOp->getSubExpr(), Ctx, !ExpectedValue);
    } else if (const auto *Paren = dyn_cast<ParenExpr>(&Cond))
      return isKnownToHaveValue(*Paren->getSubExpr(), Ctx, ExpectedValue);
    else if (const auto *ImplCast = dyn_cast<ImplicitCastExpr>(&Cond))
      return isKnownToHaveValue(*ImplCast->getSubExpr(), Ctx, ExpectedValue);
    return false;
  }
  bool Result = false;
  if (Cond.EvaluateAsBooleanCondition(Result, Ctx))
    return Result == ExpectedValue;
  return false;
}

/// populates the set `Callees` with all function (and objc method) declarations
/// called in `StmtNode` if all visited call sites have resolved call targets.
///
/// \return true iff all `CallExprs` visited have callees; false otherwise
///         indicating there is an unresolved indirect call.
static bool populateCallees(const Stmt *StmtNode,
                            llvm::SmallPtrSet<const Decl *, 16> &Callees) {
  if (const auto *Call = dyn_cast<CallExpr>(StmtNode)) {
    const Decl *Callee = Call->getDirectCallee();

    if (!Callee)
      return false; // unresolved call
    Callees.insert(Callee->getCanonicalDecl());
  }
  if (const auto *Call = dyn_cast<ObjCMessageExpr>(StmtNode)) {
    const Decl *Callee = Call->getMethodDecl();

    if (!Callee)
      return false; // unresolved call
    Callees.insert(Callee->getCanonicalDecl());
  }
  for (const Stmt *Child : StmtNode->children())
    if (Child && !populateCallees(Child, Callees))
      return false;
  return true;
}

/// returns true iff `SCC` contains `Func` and its' function set overlaps with
/// `Callees`
static bool overlap(ArrayRef<CallGraphNode *> SCC,
                    const llvm::SmallPtrSet<const Decl *, 16> &Callees,
                    const Decl *Func) {
  bool ContainsFunc = false, Overlap = false;

  for (const CallGraphNode *GNode : SCC) {
    const Decl *CanDecl = GNode->getDecl()->getCanonicalDecl();

    ContainsFunc = ContainsFunc || (CanDecl == Func);
    Overlap = Overlap || Callees.contains(CanDecl);
    if (ContainsFunc && Overlap)
      return true;
  }
  return false;
}

/// returns true iff `Cond` involves at least one static local variable.
static bool hasStaticLocalVariable(const Stmt *Cond) {
  if (const auto *DRE = dyn_cast<DeclRefExpr>(Cond)) {
    if (const auto *VD = dyn_cast<VarDecl>(DRE->getDecl()))
      if (VD->isStaticLocal())
        return true;

    if (const auto *BD = dyn_cast<BindingDecl>(DRE->getDecl()))
      if (const auto *DD = dyn_cast<DecompositionDecl>(BD->getDecomposedDecl()))
        if (DD->isStaticLocal())
          return true;
  }

  for (const Stmt *Child : Cond->children())
    if (Child && hasStaticLocalVariable(Child))
      return true;
  return false;
}

/// Tests if the loop condition `Cond` involves static local variables and
/// the enclosing function `Func` is recursive.
///
///  \code
///    void f() {
///       static int i = 10;
///       i--;
///       while (i >= 0) f();
///    }
///  \endcode
///  The example above is NOT an infinite loop.
static bool hasRecursionOverStaticLoopCondVariables(const Expr *Cond,
                                                    const Stmt *LoopStmt,
                                                    const Decl *Func,
                                                    const ASTContext *Ctx) {
  if (!hasStaticLocalVariable(Cond))
    return false;

  llvm::SmallPtrSet<const Decl *, 16> CalleesInLoop;

  if (!populateCallees(LoopStmt, CalleesInLoop)) {
    // If there are unresolved indirect calls, we assume there could
    // be recursion so to avoid false alarm.
    return true;
  }
  if (CalleesInLoop.empty())
    return false;

  TranslationUnitDecl *TUDecl = Ctx->getTranslationUnitDecl();
  CallGraph CG;

  CG.addToCallGraph(TUDecl);
  // For each `SCC` containing `Func`, if functions in the `SCC`
  // overlap with `CalleesInLoop`, there is a recursive call in `LoopStmt`.
  for (llvm::scc_iterator<CallGraph *> SCCI = llvm::scc_begin(&CG),
                                       SCCE = llvm::scc_end(&CG);
       SCCI != SCCE; ++SCCI) {
    if (!SCCI.hasCycle()) // We only care about cycles, not standalone nodes.
      continue;
    // `SCC`s are mutually disjoint, so there will be no redundancy in
    // comparing `SCC` with the callee set one by one.
    if (overlap(*SCCI, CalleesInLoop, Func->getCanonicalDecl()))
      return true;
  }
  return false;
}

void InfiniteLoopCheck::registerMatchers(MatchFinder *Finder) {
  const auto LoopCondition = allOf(
      hasCondition(expr(forCallable(decl().bind("func"))).bind("condition")),
      unless(hasBody(hasDescendant(
          loopEndingStmt(forCallable(equalsBoundNode("func")))))));

  Finder->addMatcher(mapAnyOf(whileStmt, doStmt, forStmt)
                         .with(LoopCondition)
                         .bind("loop-stmt"),
                     this);
}

void InfiniteLoopCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Cond = Result.Nodes.getNodeAs<Expr>("condition");
  const auto *LoopStmt = Result.Nodes.getNodeAs<Stmt>("loop-stmt");
  const auto *Func = Result.Nodes.getNodeAs<Decl>("func");

  if (isKnownToHaveValue(*Cond, *Result.Context, false))
    return;

  bool ShouldHaveConditionVariables = true;
  if (const auto *While = dyn_cast<WhileStmt>(LoopStmt)) {
    if (const VarDecl *LoopVarDecl = While->getConditionVariable()) {
      if (const Expr *Init = LoopVarDecl->getInit()) {
        ShouldHaveConditionVariables = false;
        Cond = Init;
      }
    }
  }

  if (ExprMutationAnalyzer::isUnevaluated(LoopStmt, *Result.Context))
    return;

  if (isAtLeastOneCondVarChanged(Func, LoopStmt, Cond, Result.Context))
    return;
  if (hasRecursionOverStaticLoopCondVariables(Cond, LoopStmt, Func,
                                              Result.Context))
    return;

  std::string CondVarNames = getCondVarNames(Cond);
  if (ShouldHaveConditionVariables && CondVarNames.empty())
    return;

  if (CondVarNames.empty()) {
    diag(LoopStmt->getBeginLoc(),
         "this loop is infinite; it does not check any variables in the"
         " condition");
  } else {
    diag(LoopStmt->getBeginLoc(),
         "this loop is infinite; none of its condition variables (%0)"
         " are updated in the loop body")
        << CondVarNames;
  }
}

} // namespace clang::tidy::bugprone
