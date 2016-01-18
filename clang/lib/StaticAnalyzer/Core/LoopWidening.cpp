//===--- LoopWidening.cpp - Instruction class definition --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// This file contains functions which are used to widen loops. A loop may be
/// widened to approximate the exit state(s), without analyzing every
/// iteration. The widening is done by invalidating anything which might be
/// modified by the body of the loop.
///
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/PathSensitive/LoopWidening.h"

using namespace clang;
using namespace ento;

/// Return the loops condition Stmt or NULL if LoopStmt is not a loop
static const Expr *getLoopCondition(const Stmt *LoopStmt) {
  switch (LoopStmt->getStmtClass()) {
  default:
    return nullptr;
  case Stmt::ForStmtClass:
    return cast<ForStmt>(LoopStmt)->getCond();
  case Stmt::WhileStmtClass:
    return cast<WhileStmt>(LoopStmt)->getCond();
  case Stmt::DoStmtClass:
    return cast<DoStmt>(LoopStmt)->getCond();
  }
}

namespace clang {
namespace ento {

ProgramStateRef getWidenedLoopState(ProgramStateRef PrevState,
                                    const LocationContext *LCtx,
                                    unsigned BlockCount, const Stmt *LoopStmt) {

  assert(isa<ForStmt>(LoopStmt) || isa<WhileStmt>(LoopStmt) ||
         isa<DoStmt>(LoopStmt));

  // Invalidate values in the current state.
  // TODO Make this more conservative by only invalidating values that might
  //      be modified by the body of the loop.
  // TODO Nested loops are currently widened as a result of the invalidation
  //      being so inprecise. When the invalidation is improved, the handling
  //      of nested loops will also need to be improved.
  const StackFrameContext *STC = LCtx->getCurrentStackFrame();
  MemRegionManager &MRMgr = PrevState->getStateManager().getRegionManager();
  const MemRegion *Regions[] = {MRMgr.getStackLocalsRegion(STC),
                                MRMgr.getStackArgumentsRegion(STC),
                                MRMgr.getGlobalsRegion()};
  RegionAndSymbolInvalidationTraits ITraits;
  for (auto *Region : Regions) {
    ITraits.setTrait(Region,
                     RegionAndSymbolInvalidationTraits::TK_EntireMemSpace);
  }
  return PrevState->invalidateRegions(Regions, getLoopCondition(LoopStmt),
                                      BlockCount, LCtx, true, nullptr, nullptr,
                                      &ITraits);
}

} // end namespace ento
} // end namespace clang
