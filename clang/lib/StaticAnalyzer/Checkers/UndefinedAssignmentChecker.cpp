//===--- UndefinedAssignmentChecker.h ---------------------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines UndefinedAssignmentChecker, a builtin check in ExprEngine that
// checks for assigning undefined values.
//
//===----------------------------------------------------------------------===//

#include "ClangSACheckers.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"

using namespace clang;
using namespace ento;

namespace {
class UndefinedAssignmentChecker
  : public Checker<check::Bind> {
  mutable OwningPtr<BugType> BT;

public:
  void checkBind(SVal location, SVal val, const Stmt *S,
                 CheckerContext &C) const;
};
}

void UndefinedAssignmentChecker::checkBind(SVal location, SVal val,
                                           const Stmt *StoreE,
                                           CheckerContext &C) const {
  if (!val.isUndef())
    return;

  ExplodedNode *N = C.generateSink();

  if (!N)
    return;

  const char *str = "Assigned value is garbage or undefined";

  if (!BT)
    BT.reset(new BuiltinBug(str));

  // Generate a report for this bug.
  const Expr *ex = 0;

  while (StoreE) {
    if (const BinaryOperator *B = dyn_cast<BinaryOperator>(StoreE)) {
      if (B->isCompoundAssignmentOp()) {
        ProgramStateRef state = C.getState();
        if (state->getSVal(B->getLHS(), C.getLocationContext()).isUndef()) {
          str = "The left expression of the compound assignment is an "
                "uninitialized value. The computed value will also be garbage";
          ex = B->getLHS();
          break;
        }
      }

      ex = B->getRHS();
      break;
    }

    if (const DeclStmt *DS = dyn_cast<DeclStmt>(StoreE)) {
      const VarDecl *VD = dyn_cast<VarDecl>(DS->getSingleDecl());
      ex = VD->getInit();
    }

    break;
  }

  BugReport *R = new BugReport(*BT, str, N);
  if (ex) {
    R->addRange(ex->getSourceRange());
    bugreporter::trackNullOrUndefValue(N, ex, *R);
  }
  C.emitReport(R);
}

void ento::registerUndefinedAssignmentChecker(CheckerManager &mgr) {
  mgr.registerChecker<UndefinedAssignmentChecker>();
}
