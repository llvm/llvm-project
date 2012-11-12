//==- UnreachableCodeChecker.cpp - Generalized dead code checker -*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This file implements a generalized unreachable code checker using a
// path-sensitive analysis. We mark any path visited, and then walk the CFG as a
// post-analysis to determine what was never visited.
//
// A similar flow-sensitive only check exists in Analysis/ReachableCode.cpp
//===----------------------------------------------------------------------===//

#include "ClangSACheckers.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExplodedGraph.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SVals.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerHelpers.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/AST/ParentMap.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/SmallSet.h"

// The number of CFGBlock pointers we want to reserve memory for. This is used
// once for each function we analyze.
#define DEFAULT_CFGBLOCKS 256

using namespace clang;
using namespace ento;

namespace {
class UnreachableCodeChecker : public Checker<check::EndAnalysis> {
public:
  void checkEndAnalysis(ExplodedGraph &G, BugReporter &B,
                        ExprEngine &Eng) const;
private:
  typedef llvm::SmallSet<unsigned, DEFAULT_CFGBLOCKS> CFGBlocksSet;

  static inline const Stmt *getUnreachableStmt(const CFGBlock *CB);
  static void FindUnreachableEntryPoints(const CFGBlock *CB,
                                         CFGBlocksSet &reachable,
                                         CFGBlocksSet &visited);
  static bool isInvalidPath(const CFGBlock *CB, const ParentMap &PM);
  static inline bool isEmptyCFGBlock(const CFGBlock *CB);
};
}

void UnreachableCodeChecker::checkEndAnalysis(ExplodedGraph &G,
                                              BugReporter &B,
                                              ExprEngine &Eng) const {
  CFGBlocksSet reachable, visited;
  
  if (Eng.hasWorkRemaining())
    return;

  const Decl *D = 0;
  CFG *C = 0;
  ParentMap *PM = 0;
  const LocationContext *LC = 0;
  // Iterate over ExplodedGraph
  for (ExplodedGraph::node_iterator I = G.nodes_begin(), E = G.nodes_end();
      I != E; ++I) {
    const ProgramPoint &P = I->getLocation();
    LC = P.getLocationContext();

    if (!D)
      D = LC->getAnalysisDeclContext()->getDecl();
    // Save the CFG if we don't have it already
    if (!C)
      C = LC->getAnalysisDeclContext()->getUnoptimizedCFG();
    if (!PM)
      PM = &LC->getParentMap();

    if (const BlockEntrance *BE = dyn_cast<BlockEntrance>(&P)) {
      const CFGBlock *CB = BE->getBlock();
      reachable.insert(CB->getBlockID());
    }
  }

  // Bail out if we didn't get the CFG or the ParentMap.
  if (!D || !C || !PM)
    return;
  
  // Don't do anything for template instantiations.  Proving that code
  // in a template instantiation is unreachable means proving that it is
  // unreachable in all instantiations.
  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D))
    if (FD->isTemplateInstantiation())
      return;

  // Find CFGBlocks that were not covered by any node
  for (CFG::const_iterator I = C->begin(), E = C->end(); I != E; ++I) {
    const CFGBlock *CB = *I;
    // Check if the block is unreachable
    if (reachable.count(CB->getBlockID()))
      continue;

    // Check if the block is empty (an artificial block)
    if (isEmptyCFGBlock(CB))
      continue;

    // Find the entry points for this block
    if (!visited.count(CB->getBlockID()))
      FindUnreachableEntryPoints(CB, reachable, visited);

    // This block may have been pruned; check if we still want to report it
    if (reachable.count(CB->getBlockID()))
      continue;

    // Check for false positives
    if (CB->size() > 0 && isInvalidPath(CB, *PM))
      continue;

    // It is good practice to always have a "default" label in a "switch", even
    // if we should never get there. It can be used to detect errors, for
    // instance. Unreachable code directly under a "default" label is therefore
    // likely to be a false positive.
    if (const Stmt *label = CB->getLabel())
      if (label->getStmtClass() == Stmt::DefaultStmtClass)
        continue;

    // Special case for __builtin_unreachable.
    // FIXME: This should be extended to include other unreachable markers,
    // such as llvm_unreachable.
    if (!CB->empty()) {
      bool foundUnreachable = false;
      for (CFGBlock::const_iterator ci = CB->begin(), ce = CB->end();
           ci != ce; ++ci) {
        if (const CFGStmt *S = (*ci).getAs<CFGStmt>())
          if (const CallExpr *CE = dyn_cast<CallExpr>(S->getStmt())) {
            if (CE->isBuiltinCall() == Builtin::BI__builtin_unreachable) {
              foundUnreachable = true;
              break;
            }
          }
      }
      if (foundUnreachable)
        continue;
    }

    // We found a block that wasn't covered - find the statement to report
    SourceRange SR;
    PathDiagnosticLocation DL;
    SourceLocation SL;
    if (const Stmt *S = getUnreachableStmt(CB)) {
      SR = S->getSourceRange();
      DL = PathDiagnosticLocation::createBegin(S, B.getSourceManager(), LC);
      SL = DL.asLocation();
      if (SR.isInvalid() || !SL.isValid())
        continue;
    }
    else
      continue;

    // Check if the SourceLocation is in a system header
    const SourceManager &SM = B.getSourceManager();
    if (SM.isInSystemHeader(SL) || SM.isInExternCSystemHeader(SL))
      continue;

    B.EmitBasicReport(D, "Unreachable code", "Dead code",
                      "This statement is never executed", DL, SR);
  }
}

// Recursively finds the entry point(s) for this dead CFGBlock.
void UnreachableCodeChecker::FindUnreachableEntryPoints(const CFGBlock *CB,
                                                        CFGBlocksSet &reachable,
                                                        CFGBlocksSet &visited) {
  visited.insert(CB->getBlockID());

  for (CFGBlock::const_pred_iterator I = CB->pred_begin(), E = CB->pred_end();
      I != E; ++I) {
    if (!reachable.count((*I)->getBlockID())) {
      // If we find an unreachable predecessor, mark this block as reachable so
      // we don't report this block
      reachable.insert(CB->getBlockID());
      if (!visited.count((*I)->getBlockID()))
        // If we haven't previously visited the unreachable predecessor, recurse
        FindUnreachableEntryPoints(*I, reachable, visited);
    }
  }
}

// Find the Stmt* in a CFGBlock for reporting a warning
const Stmt *UnreachableCodeChecker::getUnreachableStmt(const CFGBlock *CB) {
  for (CFGBlock::const_iterator I = CB->begin(), E = CB->end(); I != E; ++I) {
    if (const CFGStmt *S = I->getAs<CFGStmt>())
      return S->getStmt();
  }
  if (const Stmt *S = CB->getTerminator())
    return S;
  else
    return 0;
}

// Determines if the path to this CFGBlock contained an element that infers this
// block is a false positive. We assume that FindUnreachableEntryPoints has
// already marked only the entry points to any dead code, so we need only to
// find the condition that led to this block (the predecessor of this block.)
// There will never be more than one predecessor.
bool UnreachableCodeChecker::isInvalidPath(const CFGBlock *CB,
                                           const ParentMap &PM) {
  // We only expect a predecessor size of 0 or 1. If it is >1, then an external
  // condition has broken our assumption (for example, a sink being placed by
  // another check). In these cases, we choose not to report.
  if (CB->pred_size() > 1)
    return true;

  // If there are no predecessors, then this block is trivially unreachable
  if (CB->pred_size() == 0)
    return false;

  const CFGBlock *pred = *CB->pred_begin();

  // Get the predecessor block's terminator conditon
  const Stmt *cond = pred->getTerminatorCondition();

  //assert(cond && "CFGBlock's predecessor has a terminator condition");
  // The previous assertion is invalid in some cases (eg do/while). Leaving
  // reporting of these situations on at the moment to help triage these cases.
  if (!cond)
    return false;

  // Run each of the checks on the conditions
  if (containsMacro(cond) || containsEnum(cond)
      || containsStaticLocal(cond) || containsBuiltinOffsetOf(cond)
      || containsStmt<UnaryExprOrTypeTraitExpr>(cond))
    return true;

  return false;
}

// Returns true if the given CFGBlock is empty
bool UnreachableCodeChecker::isEmptyCFGBlock(const CFGBlock *CB) {
  return CB->getLabel() == 0       // No labels
      && CB->size() == 0           // No statements
      && CB->getTerminator() == 0; // No terminator
}

void ento::registerUnreachableCodeChecker(CheckerManager &mgr) {
  mgr.registerChecker<UnreachableCodeChecker>();
}
