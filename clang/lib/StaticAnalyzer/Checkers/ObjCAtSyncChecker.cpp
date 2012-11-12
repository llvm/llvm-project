//== ObjCAtSyncChecker.cpp - nil mutex checker for @synchronized -*- C++ -*--=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines ObjCAtSyncChecker, a builtin check that checks for null pointers
// used as mutexes for @synchronized.
//
//===----------------------------------------------------------------------===//

#include "ClangSACheckers.h"
#include "clang/AST/StmtObjC.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h"

using namespace clang;
using namespace ento;

namespace {
class ObjCAtSyncChecker
    : public Checker< check::PreStmt<ObjCAtSynchronizedStmt> > {
  mutable OwningPtr<BuiltinBug> BT_null;
  mutable OwningPtr<BuiltinBug> BT_undef;

public:
  void checkPreStmt(const ObjCAtSynchronizedStmt *S, CheckerContext &C) const;
};
} // end anonymous namespace

void ObjCAtSyncChecker::checkPreStmt(const ObjCAtSynchronizedStmt *S,
                                     CheckerContext &C) const {

  const Expr *Ex = S->getSynchExpr();
  ProgramStateRef state = C.getState();
  SVal V = state->getSVal(Ex, C.getLocationContext());

  // Uninitialized value used for the mutex?
  if (isa<UndefinedVal>(V)) {
    if (ExplodedNode *N = C.generateSink()) {
      if (!BT_undef)
        BT_undef.reset(new BuiltinBug("Uninitialized value used as mutex "
                                  "for @synchronized"));
      BugReport *report =
        new BugReport(*BT_undef, BT_undef->getDescription(), N);
      bugreporter::trackNullOrUndefValue(N, Ex, *report);
      C.emitReport(report);
    }
    return;
  }

  if (V.isUnknown())
    return;

  // Check for null mutexes.
  ProgramStateRef notNullState, nullState;
  llvm::tie(notNullState, nullState) = state->assume(cast<DefinedSVal>(V));

  if (nullState) {
    if (!notNullState) {
      // Generate an error node.  This isn't a sink since
      // a null mutex just means no synchronization occurs.
      if (ExplodedNode *N = C.addTransition(nullState)) {
        if (!BT_null)
          BT_null.reset(new BuiltinBug("Nil value used as mutex for @synchronized() "
                                   "(no synchronization will occur)"));
        BugReport *report =
          new BugReport(*BT_null, BT_null->getDescription(), N);
        bugreporter::trackNullOrUndefValue(N, Ex, *report);

        C.emitReport(report);
        return;
      }
    }
    // Don't add a transition for 'nullState'.  If the value is
    // under-constrained to be null or non-null, assume it is non-null
    // afterwards.
  }

  if (notNullState)
    C.addTransition(notNullState);
}

void ento::registerObjCAtSyncChecker(CheckerManager &mgr) {
  if (mgr.getLangOpts().ObjC2)
    mgr.registerChecker<ObjCAtSyncChecker>();
}
