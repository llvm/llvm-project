// UndefCapturedBlockVarChecker.cpp - Uninitialized captured vars -*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This checker detects blocks that capture uninitialized values.
//
//===----------------------------------------------------------------------===//

#include "ClangSACheckers.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace ento;

namespace {
class UndefCapturedBlockVarChecker
  : public Checker< check::PostStmt<BlockExpr> > {
 mutable OwningPtr<BugType> BT;

public:
  void checkPostStmt(const BlockExpr *BE, CheckerContext &C) const;
};
} // end anonymous namespace

static const DeclRefExpr *FindBlockDeclRefExpr(const Stmt *S,
                                               const VarDecl *VD) {
  if (const DeclRefExpr *BR = dyn_cast<DeclRefExpr>(S))
    if (BR->getDecl() == VD)
      return BR;

  for (Stmt::const_child_iterator I = S->child_begin(), E = S->child_end();
       I!=E; ++I)
    if (const Stmt *child = *I) {
      const DeclRefExpr *BR = FindBlockDeclRefExpr(child, VD);
      if (BR)
        return BR;
    }

  return NULL;
}

void
UndefCapturedBlockVarChecker::checkPostStmt(const BlockExpr *BE,
                                            CheckerContext &C) const {
  if (!BE->getBlockDecl()->hasCaptures())
    return;

  ProgramStateRef state = C.getState();
  const BlockDataRegion *R =
    cast<BlockDataRegion>(state->getSVal(BE,
                                         C.getLocationContext()).getAsRegion());

  BlockDataRegion::referenced_vars_iterator I = R->referenced_vars_begin(),
                                            E = R->referenced_vars_end();

  for (; I != E; ++I) {
    // This VarRegion is the region associated with the block; we need
    // the one associated with the encompassing context.
    const VarRegion *VR = *I;
    const VarDecl *VD = VR->getDecl();

    if (VD->getAttr<BlocksAttr>() || !VD->hasLocalStorage())
      continue;

    // Get the VarRegion associated with VD in the local stack frame.
    const LocationContext *LC = C.getLocationContext();
    VR = C.getSValBuilder().getRegionManager().getVarRegion(VD, LC);
    SVal VRVal = state->getSVal(VR);

    if (VRVal.isUndef())
      if (ExplodedNode *N = C.generateSink()) {
        if (!BT)
          BT.reset(new BuiltinBug("uninitialized variable captured by block"));

        // Generate a bug report.
        SmallString<128> buf;
        llvm::raw_svector_ostream os(buf);

        os << "Variable '" << VD->getName() 
           << "' is uninitialized when captured by block";

        BugReport *R = new BugReport(*BT, os.str(), N);
        if (const Expr *Ex = FindBlockDeclRefExpr(BE->getBody(), VD))
          R->addRange(Ex->getSourceRange());
        R->addVisitor(new FindLastStoreBRVisitor(VRVal, VR));
        R->disablePathPruning();
        // need location of block
        C.emitReport(R);
      }
  }
}

void ento::registerUndefCapturedBlockVarChecker(CheckerManager &mgr) {
  mgr.registerChecker<UndefCapturedBlockVarChecker>();
}
