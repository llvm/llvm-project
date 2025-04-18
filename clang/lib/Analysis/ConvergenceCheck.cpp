//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Analyses/ConvergenceCheck.h"
#include "clang/AST/ParentMap.h"
#include "clang/AST/Stmt.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Sema/SemaInternal.h"
#include "llvm/ADT/DepthFirstIterator.h"

using namespace clang;
using namespace llvm;

static void errorJumpIntoNoConvergent(Sema &S, Stmt *From, Stmt *Parent) {
  S.Diag(Parent->getBeginLoc(), diag::err_jump_into_noconvergent);
  S.Diag(From->getBeginLoc(), diag::note_goto_affects_convergence);
}

static void warnGotoCycle(Sema &S, Stmt *From, Stmt *Parent) {
  S.Diag(Parent->getBeginLoc(),
         diag::warn_cycle_created_by_goto_affects_convergence);
  S.Diag(From->getBeginLoc(), diag::note_goto_affects_convergence);
}

static void warnJumpIntoLoop(Sema &S, Stmt *From, Stmt *Loop) {
  S.Diag(Loop->getBeginLoc(), diag::warn_loop_side_entry_affects_convergence);
  S.Diag(From->getBeginLoc(), diag::note_goto_affects_convergence);
}

static void checkConvergenceOnGoto(Sema &S, GotoStmt *From, ParentMap &PM,
                                   bool GenerateWarnings, bool GenerateTokens) {
  Stmt *To = From->getLabel()->getStmt();

  unsigned ToDepth = PM.getParentDepth(To) + 1;
  unsigned FromDepth = PM.getParentDepth(From) + 1;
  Stmt *ExpandedTo = To;
  Stmt *ExpandedFrom = From;
  while (ToDepth > FromDepth) {
    std::tie(ExpandedTo, ToDepth) = PM.lookup(ExpandedTo);
  }
  while (FromDepth > ToDepth) {
    std::tie(ExpandedFrom, FromDepth) = PM.lookup(ExpandedFrom);
  }

  // Special case: the goto statement is a descendant of the label statement.
  if (GenerateWarnings && ExpandedFrom == ExpandedTo) {
    assert(ExpandedTo == To);
    warnGotoCycle(S, From, To);
    return;
  }

  Stmt *ParentFrom = PM.getParent(ExpandedFrom);
  Stmt *ParentTo = PM.getParent(ExpandedTo);
  while (ParentFrom != ParentTo) {
    assert(ParentFrom && ParentTo);
    ExpandedFrom = ParentFrom;
    ParentFrom = PM.getParent(ExpandedFrom);
    ExpandedTo = ParentTo;
    ParentTo = PM.getParent(ExpandedTo);
  }

  SmallVector<Stmt *> Loops;
  for (Stmt *I = To; I != ParentFrom; I = PM.getParent(I)) {
    if (GenerateTokens)
      if (const auto *AS = dyn_cast<AttributedStmt>(I))
        if (hasSpecificAttr<NoConvergentAttr>(AS->getAttrs()))
          errorJumpIntoNoConvergent(S, From, I);
    // Can't jump into a ranged-for, so we don't need to look for it here.
    if (GenerateWarnings && isa<ForStmt, WhileStmt, DoStmt>(I))
      Loops.push_back(I);
  }

  if (!GenerateWarnings)
    return;

  for (Stmt *I : reverse(Loops))
    warnJumpIntoLoop(S, From, I);

  bool ToFoundFirst = false;
  for (Stmt *Child : ParentFrom->children()) {
    if (Child == ExpandedFrom)
      break;
    if (Child == ExpandedTo) {
      ToFoundFirst = true;
      break;
    }
  }

  if (ToFoundFirst) {
    warnGotoCycle(S, From, To);
  }
}

static void warnSwitchIntoLoop(Sema &S, Stmt *Case, Stmt *Loop) {
  S.Diag(Loop->getBeginLoc(), diag::warn_loop_side_entry_affects_convergence);
  S.Diag(Case->getBeginLoc(), diag::note_switch_case_affects_convergence);
}

static void checkConvergenceForSwitch(Sema &S, SwitchStmt *Switch,
                                      ParentMap &PM, bool GenerateWarnings,
                                      bool GenerateTokens) {
  for (SwitchCase *Case = Switch->getSwitchCaseList(); Case;
       Case = Case->getNextSwitchCase()) {
    SmallVector<Stmt *> Loops;
    for (Stmt *I = Case; I != Switch; I = PM.getParent(I)) {
      if (GenerateTokens)
        if (const auto *AS = dyn_cast<AttributedStmt>(I))
          if (hasSpecificAttr<NoConvergentAttr>(AS->getAttrs()))
            errorJumpIntoNoConvergent(S, Switch, I);
      // Can't jump into a ranged-for, so we don't need to look for it here.
      if (GenerateWarnings && isa<ForStmt, WhileStmt, DoStmt>(I))
        Loops.push_back(I);
    }
    if (GenerateWarnings) {
      for (Stmt *I : reverse(Loops))
        warnSwitchIntoLoop(S, Case, I);
    }
  }
}

void clang::analyzeForConvergence(Sema &S, AnalysisDeclContext &AC,
                                  bool GenerateWarnings, bool GenerateTokens) {
  // Iterating over the CFG helps trim unreachable blocks, and locates Goto
  // statements faster than iterating over the whole body.
  CFG *cfg = AC.getCFG();
  assert(cfg);
  ParentMap &PM = AC.getParentMap();
  for (CFGBlock *BI : *cfg) {
    Stmt *Term = BI->getTerminatorStmt();
    if (GotoStmt *Goto = dyn_cast_or_null<GotoStmt>(Term)) {
      checkConvergenceOnGoto(S, Goto, PM, GenerateWarnings, GenerateTokens);
    } else if (SwitchStmt *Switch = dyn_cast_or_null<SwitchStmt>(Term)) {
      checkConvergenceForSwitch(S, Switch, PM, GenerateWarnings,
                                GenerateTokens);
    }
  }
}
