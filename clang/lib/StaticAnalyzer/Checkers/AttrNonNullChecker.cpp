//===--- AttrNonNullChecker.h - Undefined arguments checker ----*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines AttrNonNullChecker, a builtin check in ExprEngine that 
// performs checks for arguments declared to have nonnull attribute.
//
//===----------------------------------------------------------------------===//

#include "ClangSACheckers.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"

using namespace clang;
using namespace ento;

namespace {
class AttrNonNullChecker
  : public Checker< check::PreCall > {
  mutable OwningPtr<BugType> BT;
public:

  void checkPreCall(const CallEvent &Call, CheckerContext &C) const;
};
} // end anonymous namespace

void AttrNonNullChecker::checkPreCall(const CallEvent &Call,
                                      CheckerContext &C) const {
  const Decl *FD = Call.getDecl();
  if (!FD)
    return;

  const NonNullAttr *Att = FD->getAttr<NonNullAttr>();
  if (!Att)
    return;

  ProgramStateRef state = C.getState();

  // Iterate through the arguments of CE and check them for null.
  for (unsigned idx = 0, count = Call.getNumArgs(); idx != count; ++idx) {
    if (!Att->isNonNull(idx))
      continue;

    SVal V = Call.getArgSVal(idx);
    DefinedSVal *DV = dyn_cast<DefinedSVal>(&V);

    // If the value is unknown or undefined, we can't perform this check.
    if (!DV)
      continue;

    if (!isa<Loc>(*DV)) {
      // If the argument is a union type, we want to handle a potential
      // transparent_union GCC extension.
      const Expr *ArgE = Call.getArgExpr(idx);
      if (!ArgE)
        continue;

      QualType T = ArgE->getType();
      const RecordType *UT = T->getAsUnionType();
      if (!UT || !UT->getDecl()->hasAttr<TransparentUnionAttr>())
        continue;

      if (nonloc::CompoundVal *CSV = dyn_cast<nonloc::CompoundVal>(DV)) {
        nonloc::CompoundVal::iterator CSV_I = CSV->begin();
        assert(CSV_I != CSV->end());
        V = *CSV_I;
        DV = dyn_cast<DefinedSVal>(&V);
        assert(++CSV_I == CSV->end());
        if (!DV)
          continue;        
      } else {
        // FIXME: Handle LazyCompoundVals?
        continue;
      }
    }

    ConstraintManager &CM = C.getConstraintManager();
    ProgramStateRef stateNotNull, stateNull;
    llvm::tie(stateNotNull, stateNull) = CM.assumeDual(state, *DV);

    if (stateNull && !stateNotNull) {
      // Generate an error node.  Check for a null node in case
      // we cache out.
      if (ExplodedNode *errorNode = C.generateSink(stateNull)) {

        // Lazily allocate the BugType object if it hasn't already been
        // created. Ownership is transferred to the BugReporter object once
        // the BugReport is passed to 'EmitWarning'.
        if (!BT)
          BT.reset(new BugType("Argument with 'nonnull' attribute passed null",
                               "API"));

        BugReport *R =
          new BugReport(*BT, "Null pointer passed as an argument to a "
                             "'nonnull' parameter", errorNode);

        // Highlight the range of the argument that was null.
        R->addRange(Call.getArgSourceRange(idx));
        if (const Expr *ArgE = Call.getArgExpr(idx))
          bugreporter::trackNullOrUndefValue(errorNode, ArgE, *R);
        // Emit the bug report.
        C.emitReport(R);
      }

      // Always return.  Either we cached out or we just emitted an error.
      return;
    }

    // If a pointer value passed the check we should assume that it is
    // indeed not null from this point forward.
    assert(stateNotNull);
    state = stateNotNull;
  }

  // If we reach here all of the arguments passed the nonnull check.
  // If 'state' has been updated generated a new node.
  C.addTransition(state);
}

void ento::registerAttrNonNullChecker(CheckerManager &mgr) {
  mgr.registerChecker<AttrNonNullChecker>();
}
