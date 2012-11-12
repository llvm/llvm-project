// MallocOverflowSecurityChecker.cpp - Check for malloc overflows -*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This checker detects a common memory allocation security flaw.
// Suppose 'unsigned int n' comes from an untrusted source. If the
// code looks like 'malloc (n * 4)', and an attacker can make 'n' be
// say MAX_UINT/4+2, then instead of allocating the correct 'n' 4-byte
// elements, this will actually allocate only two because of overflow.
// Then when the rest of the program attempts to store values past the
// second element, these values will actually overwrite other items in
// the heap, probably allowing the attacker to execute arbitrary code.
//
//===----------------------------------------------------------------------===//

#include "ClangSACheckers.h"
#include "clang/AST/EvaluatedExprVisitor.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/AnalysisManager.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "llvm/ADT/SmallVector.h"

using namespace clang;
using namespace ento;

namespace {
struct MallocOverflowCheck {
  const BinaryOperator *mulop;
  const Expr *variable;

  MallocOverflowCheck (const BinaryOperator *m, const Expr *v) 
    : mulop(m), variable (v)
  {}
};

class MallocOverflowSecurityChecker : public Checker<check::ASTCodeBody> {
public:
  void checkASTCodeBody(const Decl *D, AnalysisManager &mgr,
                        BugReporter &BR) const;

  void CheckMallocArgument(
    llvm::SmallVectorImpl<MallocOverflowCheck> &PossibleMallocOverflows,
    const Expr *TheArgument, ASTContext &Context) const;

  void OutputPossibleOverflows(
    llvm::SmallVectorImpl<MallocOverflowCheck> &PossibleMallocOverflows,
    const Decl *D, BugReporter &BR, AnalysisManager &mgr) const;

};
} // end anonymous namespace

void MallocOverflowSecurityChecker::CheckMallocArgument(
  llvm::SmallVectorImpl<MallocOverflowCheck> &PossibleMallocOverflows,
  const Expr *TheArgument,
  ASTContext &Context) const {

  /* Look for a linear combination with a single variable, and at least
   one multiplication.
   Reject anything that applies to the variable: an explicit cast,
   conditional expression, an operation that could reduce the range
   of the result, or anything too complicated :-).  */
  const Expr * e = TheArgument;
  const BinaryOperator * mulop = NULL;

  for (;;) {
    e = e->IgnoreParenImpCasts();
    if (isa<BinaryOperator>(e)) {
      const BinaryOperator * binop = dyn_cast<BinaryOperator>(e);
      BinaryOperatorKind opc = binop->getOpcode();
      // TODO: ignore multiplications by 1, reject if multiplied by 0.
      if (mulop == NULL && opc == BO_Mul)
        mulop = binop;
      if (opc != BO_Mul && opc != BO_Add && opc != BO_Sub && opc != BO_Shl)
        return;

      const Expr *lhs = binop->getLHS();
      const Expr *rhs = binop->getRHS();
      if (rhs->isEvaluatable(Context))
        e = lhs;
      else if ((opc == BO_Add || opc == BO_Mul)
               && lhs->isEvaluatable(Context))
        e = rhs;
      else
        return;
    }
    else if (isa<DeclRefExpr>(e) || isa<MemberExpr>(e))
      break;
    else
      return;
  }

  if (mulop == NULL)
    return;

  //  We've found the right structure of malloc argument, now save
  // the data so when the body of the function is completely available
  // we can check for comparisons.

  // TODO: Could push this into the innermost scope where 'e' is
  // defined, rather than the whole function.
  PossibleMallocOverflows.push_back(MallocOverflowCheck(mulop, e));
}

namespace {
// A worker class for OutputPossibleOverflows.
class CheckOverflowOps :
  public EvaluatedExprVisitor<CheckOverflowOps> {
public:
  typedef llvm::SmallVectorImpl<MallocOverflowCheck> theVecType;

private:
    theVecType &toScanFor;
    ASTContext &Context;

    bool isIntZeroExpr(const Expr *E) const {
      if (!E->getType()->isIntegralOrEnumerationType())
        return false;
      llvm::APSInt Result;
      if (E->EvaluateAsInt(Result, Context))
        return Result == 0;
      return false;
    }

    void CheckExpr(const Expr *E_p) {
      const Expr *E = E_p->IgnoreParenImpCasts();

      theVecType::iterator i = toScanFor.end();
      theVecType::iterator e = toScanFor.begin();

      if (const DeclRefExpr *DR = dyn_cast<DeclRefExpr>(E)) {
        const Decl * EdreD = DR->getDecl();
        while (i != e) {
          --i;
          if (const DeclRefExpr *DR_i = dyn_cast<DeclRefExpr>(i->variable)) {
            if (DR_i->getDecl() == EdreD)
              i = toScanFor.erase(i);
          }
        }
      }
      else if (isa<MemberExpr>(E)) {
        // No points-to analysis, just look at the member
        const Decl * EmeMD = dyn_cast<MemberExpr>(E)->getMemberDecl();
        while (i != e) {
          --i;
          if (isa<MemberExpr>(i->variable)) {
            if (dyn_cast<MemberExpr>(i->variable)->getMemberDecl() == EmeMD)
              i = toScanFor.erase (i);
          }
        }
      }
    }

  public:
    void VisitBinaryOperator(BinaryOperator *E) {
      if (E->isComparisonOp()) {
        const Expr * lhs = E->getLHS();
        const Expr * rhs = E->getRHS();
        // Ignore comparisons against zero, since they generally don't
        // protect against an overflow.
        if (!isIntZeroExpr(lhs) && ! isIntZeroExpr(rhs)) {
          CheckExpr(lhs);
          CheckExpr(rhs);
        }
      }
      EvaluatedExprVisitor<CheckOverflowOps>::VisitBinaryOperator(E);
    }

    /* We specifically ignore loop conditions, because they're typically
     not error checks.  */
    void VisitWhileStmt(WhileStmt *S) {
      return this->Visit(S->getBody());
    }
    void VisitForStmt(ForStmt *S) {
      return this->Visit(S->getBody());
    }
    void VisitDoStmt(DoStmt *S) {
      return this->Visit(S->getBody());
    }

    CheckOverflowOps(theVecType &v, ASTContext &ctx)
    : EvaluatedExprVisitor<CheckOverflowOps>(ctx),
      toScanFor(v), Context(ctx)
    { }
  };
}

// OutputPossibleOverflows - We've found a possible overflow earlier,
// now check whether Body might contain a comparison which might be
// preventing the overflow.
// This doesn't do flow analysis, range analysis, or points-to analysis; it's
// just a dumb "is there a comparison" scan.  The aim here is to
// detect the most blatent cases of overflow and educate the
// programmer.
void MallocOverflowSecurityChecker::OutputPossibleOverflows(
  llvm::SmallVectorImpl<MallocOverflowCheck> &PossibleMallocOverflows,
  const Decl *D, BugReporter &BR, AnalysisManager &mgr) const {
  // By far the most common case: nothing to check.
  if (PossibleMallocOverflows.empty())
    return;

  // Delete any possible overflows which have a comparison.
  CheckOverflowOps c(PossibleMallocOverflows, BR.getContext());
  c.Visit(mgr.getAnalysisDeclContext(D)->getBody());

  // Output warnings for all overflows that are left.
  for (CheckOverflowOps::theVecType::iterator
       i = PossibleMallocOverflows.begin(),
       e = PossibleMallocOverflows.end();
       i != e;
       ++i) {
    SourceRange R = i->mulop->getSourceRange();
    BR.EmitBasicReport(D, "malloc() size overflow", categories::UnixAPI,
      "the computation of the size of the memory allocation may overflow",
      PathDiagnosticLocation::createOperatorLoc(i->mulop,
                                                BR.getSourceManager()), &R, 1);
  }
}

void MallocOverflowSecurityChecker::checkASTCodeBody(const Decl *D,
                                             AnalysisManager &mgr,
                                             BugReporter &BR) const {

  CFG *cfg = mgr.getCFG(D);
  if (!cfg)
    return;

  // A list of variables referenced in possibly overflowing malloc operands.
  llvm::SmallVector<MallocOverflowCheck, 2> PossibleMallocOverflows;

  for (CFG::iterator it = cfg->begin(), ei = cfg->end(); it != ei; ++it) {
    CFGBlock *block = *it;
    for (CFGBlock::iterator bi = block->begin(), be = block->end();
         bi != be; ++bi) {
      if (const CFGStmt *CS = bi->getAs<CFGStmt>()) {
        if (const CallExpr *TheCall = dyn_cast<CallExpr>(CS->getStmt())) {
          // Get the callee.
          const FunctionDecl *FD = TheCall->getDirectCallee();

          if (!FD)
            return;

          // Get the name of the callee. If it's a builtin, strip off the prefix.
          IdentifierInfo *FnInfo = FD->getIdentifier();
          if (!FnInfo)
            return;

          if (FnInfo->isStr ("malloc") || FnInfo->isStr ("_MALLOC")) {
            if (TheCall->getNumArgs() == 1)
              CheckMallocArgument(PossibleMallocOverflows, TheCall->getArg(0),
                                  mgr.getASTContext());
          }
        }
      }
    }
  }

  OutputPossibleOverflows(PossibleMallocOverflows, D, BR, mgr);
}

void ento::registerMallocOverflowSecurityChecker(CheckerManager &mgr) {
  mgr.registerChecker<MallocOverflowSecurityChecker>();
}
