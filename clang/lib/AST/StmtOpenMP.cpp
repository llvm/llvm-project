//===--- StmtOpenMP.cpp - Classes for OpenMP directives -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the subclesses of Stmt class declared in StmtOpenMP.h
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/StmtOpenMP.h"

using namespace clang;
using namespace llvm::omp;

void OMPExecutableDirective::setClauses(ArrayRef<OMPClause *> Clauses) {
  assert(Clauses.size() == getNumClauses() &&
         "Number of clauses is not the same as the preallocated buffer");
  std::copy(Clauses.begin(), Clauses.end(), getClauses().begin());
}

bool OMPExecutableDirective::isStandaloneDirective() const {
  // Special case: 'omp target enter data', 'omp target exit data',
  // 'omp target update' are stand-alone directives, but for implementation
  // reasons they have empty synthetic structured block, to simplify codegen.
  if (isa<OMPTargetEnterDataDirective>(this) ||
      isa<OMPTargetExitDataDirective>(this) ||
      isa<OMPTargetUpdateDirective>(this))
    return true;
  return !hasAssociatedStmt() || !getAssociatedStmt();
}

const Stmt *OMPExecutableDirective::getStructuredBlock() const {
  assert(!isStandaloneDirective() &&
         "Standalone Executable Directives don't have Structured Blocks.");
  if (auto *LD = dyn_cast<OMPLoopDirective>(this))
    return LD->getBody();
  return getInnermostCapturedStmt()->getCapturedStmt();
}

Stmt *OMPLoopDirective::tryToFindNextInnerLoop(Stmt *CurStmt,
                                               bool TryImperfectlyNestedLoops) {
  Stmt *OrigStmt = CurStmt;
  CurStmt = CurStmt->IgnoreContainers();
  // Additional work for imperfectly nested loops, introduced in OpenMP 5.0.
  if (TryImperfectlyNestedLoops) {
    if (auto *CS = dyn_cast<CompoundStmt>(CurStmt)) {
      CurStmt = nullptr;
      SmallVector<CompoundStmt *, 4> Statements(1, CS);
      SmallVector<CompoundStmt *, 4> NextStatements;
      while (!Statements.empty()) {
        CS = Statements.pop_back_val();
        if (!CS)
          continue;
        for (Stmt *S : CS->body()) {
          if (!S)
            continue;
          if (isa<ForStmt>(S) || isa<CXXForRangeStmt>(S)) {
            // Only single loop construct is allowed.
            if (CurStmt) {
              CurStmt = OrigStmt;
              break;
            }
            CurStmt = S;
            continue;
          }
          S = S->IgnoreContainers();
          if (auto *InnerCS = dyn_cast_or_null<CompoundStmt>(S))
            NextStatements.push_back(InnerCS);
        }
        if (Statements.empty()) {
          // Found single inner loop or multiple loops - exit.
          if (CurStmt)
            break;
          Statements.swap(NextStatements);
        }
      }
      if (!CurStmt)
        CurStmt = OrigStmt;
    }
  }
  return CurStmt;
}

Stmt *OMPLoopDirective::getBody() {
  // This relies on the loop form is already checked by Sema.
  Stmt *Body =
      getInnermostCapturedStmt()->getCapturedStmt()->IgnoreContainers();
  if (auto *For = dyn_cast<ForStmt>(Body)) {
    Body = For->getBody();
  } else {
    assert(isa<CXXForRangeStmt>(Body) &&
           "Expected canonical for loop or range-based for loop.");
    Body = cast<CXXForRangeStmt>(Body)->getBody();
  }
  for (unsigned Cnt = 1; Cnt < CollapsedNum; ++Cnt) {
    Body = tryToFindNextInnerLoop(Body, /*TryImperfectlyNestedLoops=*/true);
    if (auto *For = dyn_cast<ForStmt>(Body)) {
      Body = For->getBody();
    } else {
      assert(isa<CXXForRangeStmt>(Body) &&
             "Expected canonical for loop or range-based for loop.");
      Body = cast<CXXForRangeStmt>(Body)->getBody();
    }
  }
  return Body;
}

void OMPLoopDirective::setCounters(ArrayRef<Expr *> A) {
  assert(A.size() == getCollapsedNumber() &&
         "Number of loop counters is not the same as the collapsed number");
  std::copy(A.begin(), A.end(), getCounters().begin());
}

void OMPLoopDirective::setPrivateCounters(ArrayRef<Expr *> A) {
  assert(A.size() == getCollapsedNumber() && "Number of loop private counters "
                                             "is not the same as the collapsed "
                                             "number");
  std::copy(A.begin(), A.end(), getPrivateCounters().begin());
}

void OMPLoopDirective::setInits(ArrayRef<Expr *> A) {
  assert(A.size() == getCollapsedNumber() &&
         "Number of counter inits is not the same as the collapsed number");
  std::copy(A.begin(), A.end(), getInits().begin());
}

void OMPLoopDirective::setUpdates(ArrayRef<Expr *> A) {
  assert(A.size() == getCollapsedNumber() &&
         "Number of counter updates is not the same as the collapsed number");
  std::copy(A.begin(), A.end(), getUpdates().begin());
}

void OMPLoopDirective::setFinals(ArrayRef<Expr *> A) {
  assert(A.size() == getCollapsedNumber() &&
         "Number of counter finals is not the same as the collapsed number");
  std::copy(A.begin(), A.end(), getFinals().begin());
}

void OMPLoopDirective::setDependentCounters(ArrayRef<Expr *> A) {
  assert(
      A.size() == getCollapsedNumber() &&
      "Number of dependent counters is not the same as the collapsed number");
  llvm::copy(A, getDependentCounters().begin());
}

void OMPLoopDirective::setDependentInits(ArrayRef<Expr *> A) {
  assert(A.size() == getCollapsedNumber() &&
         "Number of dependent inits is not the same as the collapsed number");
  llvm::copy(A, getDependentInits().begin());
}

void OMPLoopDirective::setFinalsConditions(ArrayRef<Expr *> A) {
  assert(A.size() == getCollapsedNumber() &&
         "Number of finals conditions is not the same as the collapsed number");
  llvm::copy(A, getFinalsConditions().begin());
}

OMPParallelDirective *OMPParallelDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt, Expr *TaskRedRef,
    bool HasCancel) {
  unsigned Size =
      llvm::alignTo(sizeof(OMPParallelDirective), alignof(OMPClause *));
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * Clauses.size() + sizeof(Stmt *));
  OMPParallelDirective *Dir =
      new (Mem) OMPParallelDirective(StartLoc, EndLoc, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setTaskReductionRefExpr(TaskRedRef);
  Dir->setHasCancel(HasCancel);
  return Dir;
}

OMPParallelDirective *OMPParallelDirective::CreateEmpty(const ASTContext &C,
                                                        unsigned NumClauses,
                                                        EmptyShell) {
  unsigned Size =
      llvm::alignTo(sizeof(OMPParallelDirective), alignof(OMPClause *));
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * NumClauses + sizeof(Stmt *));
  return new (Mem) OMPParallelDirective(NumClauses);
}

OMPSimdDirective *
OMPSimdDirective::Create(const ASTContext &C, SourceLocation StartLoc,
                         SourceLocation EndLoc, unsigned CollapsedNum,
                         ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt,
                         const HelperExprs &Exprs) {
  unsigned Size = llvm::alignTo(sizeof(OMPSimdDirective), alignof(OMPClause *));
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * Clauses.size() +
                 sizeof(Stmt *) * numLoopChildren(CollapsedNum, OMPD_simd));
  OMPSimdDirective *Dir = new (Mem)
      OMPSimdDirective(StartLoc, EndLoc, CollapsedNum, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setIterationVariable(Exprs.IterationVarRef);
  Dir->setLastIteration(Exprs.LastIteration);
  Dir->setCalcLastIteration(Exprs.CalcLastIteration);
  Dir->setPreCond(Exprs.PreCond);
  Dir->setCond(Exprs.Cond);
  Dir->setInit(Exprs.Init);
  Dir->setInc(Exprs.Inc);
  Dir->setCounters(Exprs.Counters);
  Dir->setPrivateCounters(Exprs.PrivateCounters);
  Dir->setInits(Exprs.Inits);
  Dir->setUpdates(Exprs.Updates);
  Dir->setFinals(Exprs.Finals);
  Dir->setDependentCounters(Exprs.DependentCounters);
  Dir->setDependentInits(Exprs.DependentInits);
  Dir->setFinalsConditions(Exprs.FinalsConditions);
  Dir->setPreInits(Exprs.PreInits);
  return Dir;
}

OMPSimdDirective *OMPSimdDirective::CreateEmpty(const ASTContext &C,
                                                unsigned NumClauses,
                                                unsigned CollapsedNum,
                                                EmptyShell) {
  unsigned Size = llvm::alignTo(sizeof(OMPSimdDirective), alignof(OMPClause *));
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * NumClauses +
                 sizeof(Stmt *) * numLoopChildren(CollapsedNum, OMPD_simd));
  return new (Mem) OMPSimdDirective(CollapsedNum, NumClauses);
}

OMPForDirective *OMPForDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt,
    const HelperExprs &Exprs, Expr *TaskRedRef, bool HasCancel) {
  unsigned Size = llvm::alignTo(sizeof(OMPForDirective), alignof(OMPClause *));
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * Clauses.size() +
                 sizeof(Stmt *) * numLoopChildren(CollapsedNum, OMPD_for));
  OMPForDirective *Dir =
      new (Mem) OMPForDirective(StartLoc, EndLoc, CollapsedNum, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setIterationVariable(Exprs.IterationVarRef);
  Dir->setLastIteration(Exprs.LastIteration);
  Dir->setCalcLastIteration(Exprs.CalcLastIteration);
  Dir->setPreCond(Exprs.PreCond);
  Dir->setCond(Exprs.Cond);
  Dir->setInit(Exprs.Init);
  Dir->setInc(Exprs.Inc);
  Dir->setIsLastIterVariable(Exprs.IL);
  Dir->setLowerBoundVariable(Exprs.LB);
  Dir->setUpperBoundVariable(Exprs.UB);
  Dir->setStrideVariable(Exprs.ST);
  Dir->setEnsureUpperBound(Exprs.EUB);
  Dir->setNextLowerBound(Exprs.NLB);
  Dir->setNextUpperBound(Exprs.NUB);
  Dir->setNumIterations(Exprs.NumIterations);
  Dir->setCounters(Exprs.Counters);
  Dir->setPrivateCounters(Exprs.PrivateCounters);
  Dir->setInits(Exprs.Inits);
  Dir->setUpdates(Exprs.Updates);
  Dir->setFinals(Exprs.Finals);
  Dir->setDependentCounters(Exprs.DependentCounters);
  Dir->setDependentInits(Exprs.DependentInits);
  Dir->setFinalsConditions(Exprs.FinalsConditions);
  Dir->setPreInits(Exprs.PreInits);
  Dir->setTaskReductionRefExpr(TaskRedRef);
  Dir->setHasCancel(HasCancel);
  return Dir;
}

OMPForDirective *OMPForDirective::CreateEmpty(const ASTContext &C,
                                              unsigned NumClauses,
                                              unsigned CollapsedNum,
                                              EmptyShell) {
  unsigned Size = llvm::alignTo(sizeof(OMPForDirective), alignof(OMPClause *));
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * NumClauses +
                 sizeof(Stmt *) * numLoopChildren(CollapsedNum, OMPD_for));
  return new (Mem) OMPForDirective(CollapsedNum, NumClauses);
}

OMPForSimdDirective *
OMPForSimdDirective::Create(const ASTContext &C, SourceLocation StartLoc,
                            SourceLocation EndLoc, unsigned CollapsedNum,
                            ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt,
                            const HelperExprs &Exprs) {
  unsigned Size =
      llvm::alignTo(sizeof(OMPForSimdDirective), alignof(OMPClause *));
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * Clauses.size() +
                 sizeof(Stmt *) * numLoopChildren(CollapsedNum, OMPD_for_simd));
  OMPForSimdDirective *Dir = new (Mem)
      OMPForSimdDirective(StartLoc, EndLoc, CollapsedNum, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setIterationVariable(Exprs.IterationVarRef);
  Dir->setLastIteration(Exprs.LastIteration);
  Dir->setCalcLastIteration(Exprs.CalcLastIteration);
  Dir->setPreCond(Exprs.PreCond);
  Dir->setCond(Exprs.Cond);
  Dir->setInit(Exprs.Init);
  Dir->setInc(Exprs.Inc);
  Dir->setIsLastIterVariable(Exprs.IL);
  Dir->setLowerBoundVariable(Exprs.LB);
  Dir->setUpperBoundVariable(Exprs.UB);
  Dir->setStrideVariable(Exprs.ST);
  Dir->setEnsureUpperBound(Exprs.EUB);
  Dir->setNextLowerBound(Exprs.NLB);
  Dir->setNextUpperBound(Exprs.NUB);
  Dir->setNumIterations(Exprs.NumIterations);
  Dir->setCounters(Exprs.Counters);
  Dir->setPrivateCounters(Exprs.PrivateCounters);
  Dir->setInits(Exprs.Inits);
  Dir->setUpdates(Exprs.Updates);
  Dir->setFinals(Exprs.Finals);
  Dir->setDependentCounters(Exprs.DependentCounters);
  Dir->setDependentInits(Exprs.DependentInits);
  Dir->setFinalsConditions(Exprs.FinalsConditions);
  Dir->setPreInits(Exprs.PreInits);
  return Dir;
}

OMPForSimdDirective *OMPForSimdDirective::CreateEmpty(const ASTContext &C,
                                                      unsigned NumClauses,
                                                      unsigned CollapsedNum,
                                                      EmptyShell) {
  unsigned Size =
      llvm::alignTo(sizeof(OMPForSimdDirective), alignof(OMPClause *));
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * NumClauses +
                 sizeof(Stmt *) * numLoopChildren(CollapsedNum, OMPD_for_simd));
  return new (Mem) OMPForSimdDirective(CollapsedNum, NumClauses);
}

OMPSectionsDirective *OMPSectionsDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt, Expr *TaskRedRef,
    bool HasCancel) {
  unsigned Size =
      llvm::alignTo(sizeof(OMPSectionsDirective), alignof(OMPClause *));
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * Clauses.size() + sizeof(Stmt *));
  OMPSectionsDirective *Dir =
      new (Mem) OMPSectionsDirective(StartLoc, EndLoc, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setTaskReductionRefExpr(TaskRedRef);
  Dir->setHasCancel(HasCancel);
  return Dir;
}

OMPSectionsDirective *OMPSectionsDirective::CreateEmpty(const ASTContext &C,
                                                        unsigned NumClauses,
                                                        EmptyShell) {
  unsigned Size =
      llvm::alignTo(sizeof(OMPSectionsDirective), alignof(OMPClause *));
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * NumClauses + sizeof(Stmt *));
  return new (Mem) OMPSectionsDirective(NumClauses);
}

OMPSectionDirective *OMPSectionDirective::Create(const ASTContext &C,
                                                 SourceLocation StartLoc,
                                                 SourceLocation EndLoc,
                                                 Stmt *AssociatedStmt,
                                                 bool HasCancel) {
  unsigned Size = llvm::alignTo(sizeof(OMPSectionDirective), alignof(Stmt *));
  void *Mem = C.Allocate(Size + sizeof(Stmt *));
  OMPSectionDirective *Dir = new (Mem) OMPSectionDirective(StartLoc, EndLoc);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setHasCancel(HasCancel);
  return Dir;
}

OMPSectionDirective *OMPSectionDirective::CreateEmpty(const ASTContext &C,
                                                      EmptyShell) {
  unsigned Size = llvm::alignTo(sizeof(OMPSectionDirective), alignof(Stmt *));
  void *Mem = C.Allocate(Size + sizeof(Stmt *));
  return new (Mem) OMPSectionDirective();
}

OMPSingleDirective *OMPSingleDirective::Create(const ASTContext &C,
                                               SourceLocation StartLoc,
                                               SourceLocation EndLoc,
                                               ArrayRef<OMPClause *> Clauses,
                                               Stmt *AssociatedStmt) {
  unsigned Size =
      llvm::alignTo(sizeof(OMPSingleDirective), alignof(OMPClause *));
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * Clauses.size() + sizeof(Stmt *));
  OMPSingleDirective *Dir =
      new (Mem) OMPSingleDirective(StartLoc, EndLoc, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  return Dir;
}

OMPSingleDirective *OMPSingleDirective::CreateEmpty(const ASTContext &C,
                                                    unsigned NumClauses,
                                                    EmptyShell) {
  unsigned Size =
      llvm::alignTo(sizeof(OMPSingleDirective), alignof(OMPClause *));
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * NumClauses + sizeof(Stmt *));
  return new (Mem) OMPSingleDirective(NumClauses);
}

OMPMasterDirective *OMPMasterDirective::Create(const ASTContext &C,
                                               SourceLocation StartLoc,
                                               SourceLocation EndLoc,
                                               Stmt *AssociatedStmt) {
  unsigned Size = llvm::alignTo(sizeof(OMPMasterDirective), alignof(Stmt *));
  void *Mem = C.Allocate(Size + sizeof(Stmt *));
  OMPMasterDirective *Dir = new (Mem) OMPMasterDirective(StartLoc, EndLoc);
  Dir->setAssociatedStmt(AssociatedStmt);
  return Dir;
}

OMPMasterDirective *OMPMasterDirective::CreateEmpty(const ASTContext &C,
                                                    EmptyShell) {
  unsigned Size = llvm::alignTo(sizeof(OMPMasterDirective), alignof(Stmt *));
  void *Mem = C.Allocate(Size + sizeof(Stmt *));
  return new (Mem) OMPMasterDirective();
}

OMPCriticalDirective *OMPCriticalDirective::Create(
    const ASTContext &C, const DeclarationNameInfo &Name,
    SourceLocation StartLoc, SourceLocation EndLoc,
    ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt) {
  unsigned Size =
      llvm::alignTo(sizeof(OMPCriticalDirective), alignof(OMPClause *));
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * Clauses.size() + sizeof(Stmt *));
  OMPCriticalDirective *Dir =
      new (Mem) OMPCriticalDirective(Name, StartLoc, EndLoc, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  return Dir;
}

OMPCriticalDirective *OMPCriticalDirective::CreateEmpty(const ASTContext &C,
                                                        unsigned NumClauses,
                                                        EmptyShell) {
  unsigned Size =
      llvm::alignTo(sizeof(OMPCriticalDirective), alignof(OMPClause *));
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * NumClauses + sizeof(Stmt *));
  return new (Mem) OMPCriticalDirective(NumClauses);
}

OMPParallelForDirective *OMPParallelForDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt,
    const HelperExprs &Exprs, Expr *TaskRedRef, bool HasCancel) {
  unsigned Size =
      llvm::alignTo(sizeof(OMPParallelForDirective), alignof(OMPClause *));
  void *Mem = C.Allocate(Size + sizeof(OMPClause *) * Clauses.size() +
                         sizeof(Stmt *) *
                             numLoopChildren(CollapsedNum, OMPD_parallel_for));
  OMPParallelForDirective *Dir = new (Mem)
      OMPParallelForDirective(StartLoc, EndLoc, CollapsedNum, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setIterationVariable(Exprs.IterationVarRef);
  Dir->setLastIteration(Exprs.LastIteration);
  Dir->setCalcLastIteration(Exprs.CalcLastIteration);
  Dir->setPreCond(Exprs.PreCond);
  Dir->setCond(Exprs.Cond);
  Dir->setInit(Exprs.Init);
  Dir->setInc(Exprs.Inc);
  Dir->setIsLastIterVariable(Exprs.IL);
  Dir->setLowerBoundVariable(Exprs.LB);
  Dir->setUpperBoundVariable(Exprs.UB);
  Dir->setStrideVariable(Exprs.ST);
  Dir->setEnsureUpperBound(Exprs.EUB);
  Dir->setNextLowerBound(Exprs.NLB);
  Dir->setNextUpperBound(Exprs.NUB);
  Dir->setNumIterations(Exprs.NumIterations);
  Dir->setCounters(Exprs.Counters);
  Dir->setPrivateCounters(Exprs.PrivateCounters);
  Dir->setInits(Exprs.Inits);
  Dir->setUpdates(Exprs.Updates);
  Dir->setFinals(Exprs.Finals);
  Dir->setDependentCounters(Exprs.DependentCounters);
  Dir->setDependentInits(Exprs.DependentInits);
  Dir->setFinalsConditions(Exprs.FinalsConditions);
  Dir->setPreInits(Exprs.PreInits);
  Dir->setTaskReductionRefExpr(TaskRedRef);
  Dir->setHasCancel(HasCancel);
  return Dir;
}

OMPParallelForDirective *
OMPParallelForDirective::CreateEmpty(const ASTContext &C, unsigned NumClauses,
                                     unsigned CollapsedNum, EmptyShell) {
  unsigned Size =
      llvm::alignTo(sizeof(OMPParallelForDirective), alignof(OMPClause *));
  void *Mem = C.Allocate(Size + sizeof(OMPClause *) * NumClauses +
                         sizeof(Stmt *) *
                             numLoopChildren(CollapsedNum, OMPD_parallel_for));
  return new (Mem) OMPParallelForDirective(CollapsedNum, NumClauses);
}

OMPParallelForSimdDirective *OMPParallelForSimdDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt,
    const HelperExprs &Exprs) {
  unsigned Size =
      llvm::alignTo(sizeof(OMPParallelForSimdDirective), alignof(OMPClause *));
  void *Mem = C.Allocate(
      Size + sizeof(OMPClause *) * Clauses.size() +
      sizeof(Stmt *) * numLoopChildren(CollapsedNum, OMPD_parallel_for_simd));
  OMPParallelForSimdDirective *Dir = new (Mem) OMPParallelForSimdDirective(
      StartLoc, EndLoc, CollapsedNum, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setIterationVariable(Exprs.IterationVarRef);
  Dir->setLastIteration(Exprs.LastIteration);
  Dir->setCalcLastIteration(Exprs.CalcLastIteration);
  Dir->setPreCond(Exprs.PreCond);
  Dir->setCond(Exprs.Cond);
  Dir->setInit(Exprs.Init);
  Dir->setInc(Exprs.Inc);
  Dir->setIsLastIterVariable(Exprs.IL);
  Dir->setLowerBoundVariable(Exprs.LB);
  Dir->setUpperBoundVariable(Exprs.UB);
  Dir->setStrideVariable(Exprs.ST);
  Dir->setEnsureUpperBound(Exprs.EUB);
  Dir->setNextLowerBound(Exprs.NLB);
  Dir->setNextUpperBound(Exprs.NUB);
  Dir->setNumIterations(Exprs.NumIterations);
  Dir->setCounters(Exprs.Counters);
  Dir->setPrivateCounters(Exprs.PrivateCounters);
  Dir->setInits(Exprs.Inits);
  Dir->setUpdates(Exprs.Updates);
  Dir->setFinals(Exprs.Finals);
  Dir->setDependentCounters(Exprs.DependentCounters);
  Dir->setDependentInits(Exprs.DependentInits);
  Dir->setFinalsConditions(Exprs.FinalsConditions);
  Dir->setPreInits(Exprs.PreInits);
  return Dir;
}

OMPParallelForSimdDirective *
OMPParallelForSimdDirective::CreateEmpty(const ASTContext &C,
                                         unsigned NumClauses,
                                         unsigned CollapsedNum, EmptyShell) {
  unsigned Size =
      llvm::alignTo(sizeof(OMPParallelForSimdDirective), alignof(OMPClause *));
  void *Mem = C.Allocate(
      Size + sizeof(OMPClause *) * NumClauses +
      sizeof(Stmt *) * numLoopChildren(CollapsedNum, OMPD_parallel_for_simd));
  return new (Mem) OMPParallelForSimdDirective(CollapsedNum, NumClauses);
}

OMPParallelMasterDirective *OMPParallelMasterDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt, Expr *TaskRedRef) {
  unsigned Size =
      llvm::alignTo(sizeof(OMPParallelMasterDirective), alignof(OMPClause *));
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * Clauses.size() + sizeof(Stmt *));
  auto *Dir =
      new (Mem) OMPParallelMasterDirective(StartLoc, EndLoc, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setTaskReductionRefExpr(TaskRedRef);
  return Dir;
}

OMPParallelMasterDirective *OMPParallelMasterDirective::CreateEmpty(const ASTContext &C,
                                                        unsigned NumClauses,
                                                        EmptyShell) {
  unsigned Size =
      llvm::alignTo(sizeof(OMPParallelMasterDirective), alignof(OMPClause *));
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * NumClauses + sizeof(Stmt *));
  return new (Mem) OMPParallelMasterDirective(NumClauses);
}

OMPParallelSectionsDirective *OMPParallelSectionsDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt, Expr *TaskRedRef,
    bool HasCancel) {
  unsigned Size =
      llvm::alignTo(sizeof(OMPParallelSectionsDirective), alignof(OMPClause *));
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * Clauses.size() + sizeof(Stmt *));
  OMPParallelSectionsDirective *Dir =
      new (Mem) OMPParallelSectionsDirective(StartLoc, EndLoc, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setTaskReductionRefExpr(TaskRedRef);
  Dir->setHasCancel(HasCancel);
  return Dir;
}

OMPParallelSectionsDirective *
OMPParallelSectionsDirective::CreateEmpty(const ASTContext &C,
                                          unsigned NumClauses, EmptyShell) {
  unsigned Size =
      llvm::alignTo(sizeof(OMPParallelSectionsDirective), alignof(OMPClause *));
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * NumClauses + sizeof(Stmt *));
  return new (Mem) OMPParallelSectionsDirective(NumClauses);
}

OMPTaskDirective *
OMPTaskDirective::Create(const ASTContext &C, SourceLocation StartLoc,
                         SourceLocation EndLoc, ArrayRef<OMPClause *> Clauses,
                         Stmt *AssociatedStmt, bool HasCancel) {
  unsigned Size = llvm::alignTo(sizeof(OMPTaskDirective), alignof(OMPClause *));
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * Clauses.size() + sizeof(Stmt *));
  OMPTaskDirective *Dir =
      new (Mem) OMPTaskDirective(StartLoc, EndLoc, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setHasCancel(HasCancel);
  return Dir;
}

OMPTaskDirective *OMPTaskDirective::CreateEmpty(const ASTContext &C,
                                                unsigned NumClauses,
                                                EmptyShell) {
  unsigned Size = llvm::alignTo(sizeof(OMPTaskDirective), alignof(OMPClause *));
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * NumClauses + sizeof(Stmt *));
  return new (Mem) OMPTaskDirective(NumClauses);
}

OMPTaskyieldDirective *OMPTaskyieldDirective::Create(const ASTContext &C,
                                                     SourceLocation StartLoc,
                                                     SourceLocation EndLoc) {
  void *Mem = C.Allocate(sizeof(OMPTaskyieldDirective));
  OMPTaskyieldDirective *Dir =
      new (Mem) OMPTaskyieldDirective(StartLoc, EndLoc);
  return Dir;
}

OMPTaskyieldDirective *OMPTaskyieldDirective::CreateEmpty(const ASTContext &C,
                                                          EmptyShell) {
  void *Mem = C.Allocate(sizeof(OMPTaskyieldDirective));
  return new (Mem) OMPTaskyieldDirective();
}

OMPBarrierDirective *OMPBarrierDirective::Create(const ASTContext &C,
                                                 SourceLocation StartLoc,
                                                 SourceLocation EndLoc) {
  void *Mem = C.Allocate(sizeof(OMPBarrierDirective));
  OMPBarrierDirective *Dir = new (Mem) OMPBarrierDirective(StartLoc, EndLoc);
  return Dir;
}

OMPBarrierDirective *OMPBarrierDirective::CreateEmpty(const ASTContext &C,
                                                      EmptyShell) {
  void *Mem = C.Allocate(sizeof(OMPBarrierDirective));
  return new (Mem) OMPBarrierDirective();
}

OMPTaskwaitDirective *OMPTaskwaitDirective::Create(const ASTContext &C,
                                                   SourceLocation StartLoc,
                                                   SourceLocation EndLoc) {
  void *Mem = C.Allocate(sizeof(OMPTaskwaitDirective));
  OMPTaskwaitDirective *Dir = new (Mem) OMPTaskwaitDirective(StartLoc, EndLoc);
  return Dir;
}

OMPTaskwaitDirective *OMPTaskwaitDirective::CreateEmpty(const ASTContext &C,
                                                        EmptyShell) {
  void *Mem = C.Allocate(sizeof(OMPTaskwaitDirective));
  return new (Mem) OMPTaskwaitDirective();
}

OMPTaskgroupDirective *OMPTaskgroupDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt, Expr *ReductionRef) {
  unsigned Size = llvm::alignTo(sizeof(OMPTaskgroupDirective) +
                                    sizeof(OMPClause *) * Clauses.size(),
                                alignof(Stmt *));
  void *Mem = C.Allocate(Size + sizeof(Stmt *) + sizeof(Expr *));
  OMPTaskgroupDirective *Dir =
      new (Mem) OMPTaskgroupDirective(StartLoc, EndLoc, Clauses.size());
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setReductionRef(ReductionRef);
  Dir->setClauses(Clauses);
  return Dir;
}

OMPTaskgroupDirective *OMPTaskgroupDirective::CreateEmpty(const ASTContext &C,
                                                          unsigned NumClauses,
                                                          EmptyShell) {
  unsigned Size = llvm::alignTo(sizeof(OMPTaskgroupDirective) +
                                    sizeof(OMPClause *) * NumClauses,
                                alignof(Stmt *));
  void *Mem = C.Allocate(Size + sizeof(Stmt *) + sizeof(Expr *));
  return new (Mem) OMPTaskgroupDirective(NumClauses);
}

OMPCancellationPointDirective *OMPCancellationPointDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    OpenMPDirectiveKind CancelRegion) {
  unsigned Size =
      llvm::alignTo(sizeof(OMPCancellationPointDirective), alignof(Stmt *));
  void *Mem = C.Allocate(Size);
  OMPCancellationPointDirective *Dir =
      new (Mem) OMPCancellationPointDirective(StartLoc, EndLoc);
  Dir->setCancelRegion(CancelRegion);
  return Dir;
}

OMPCancellationPointDirective *
OMPCancellationPointDirective::CreateEmpty(const ASTContext &C, EmptyShell) {
  unsigned Size =
      llvm::alignTo(sizeof(OMPCancellationPointDirective), alignof(Stmt *));
  void *Mem = C.Allocate(Size);
  return new (Mem) OMPCancellationPointDirective();
}

OMPCancelDirective *
OMPCancelDirective::Create(const ASTContext &C, SourceLocation StartLoc,
                           SourceLocation EndLoc, ArrayRef<OMPClause *> Clauses,
                           OpenMPDirectiveKind CancelRegion) {
  unsigned Size = llvm::alignTo(sizeof(OMPCancelDirective) +
                                    sizeof(OMPClause *) * Clauses.size(),
                                alignof(Stmt *));
  void *Mem = C.Allocate(Size);
  OMPCancelDirective *Dir =
      new (Mem) OMPCancelDirective(StartLoc, EndLoc, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setCancelRegion(CancelRegion);
  return Dir;
}

OMPCancelDirective *OMPCancelDirective::CreateEmpty(const ASTContext &C,
                                                    unsigned NumClauses,
                                                    EmptyShell) {
  unsigned Size = llvm::alignTo(sizeof(OMPCancelDirective) +
                                    sizeof(OMPClause *) * NumClauses,
                                alignof(Stmt *));
  void *Mem = C.Allocate(Size);
  return new (Mem) OMPCancelDirective(NumClauses);
}

OMPFlushDirective *OMPFlushDirective::Create(const ASTContext &C,
                                             SourceLocation StartLoc,
                                             SourceLocation EndLoc,
                                             ArrayRef<OMPClause *> Clauses) {
  unsigned Size =
      llvm::alignTo(sizeof(OMPFlushDirective), alignof(OMPClause *));
  void *Mem = C.Allocate(Size + sizeof(OMPClause *) * Clauses.size());
  OMPFlushDirective *Dir =
      new (Mem) OMPFlushDirective(StartLoc, EndLoc, Clauses.size());
  Dir->setClauses(Clauses);
  return Dir;
}

OMPFlushDirective *OMPFlushDirective::CreateEmpty(const ASTContext &C,
                                                  unsigned NumClauses,
                                                  EmptyShell) {
  unsigned Size =
      llvm::alignTo(sizeof(OMPFlushDirective), alignof(OMPClause *));
  void *Mem = C.Allocate(Size + sizeof(OMPClause *) * NumClauses);
  return new (Mem) OMPFlushDirective(NumClauses);
}

OMPDepobjDirective *OMPDepobjDirective::Create(const ASTContext &C,
                                               SourceLocation StartLoc,
                                               SourceLocation EndLoc,
                                               ArrayRef<OMPClause *> Clauses) {
  unsigned Size =
      llvm::alignTo(sizeof(OMPDepobjDirective), alignof(OMPClause *));
  void *Mem = C.Allocate(Size + sizeof(OMPClause *) * Clauses.size(),
                         alignof(OMPDepobjDirective));
  auto *Dir = new (Mem) OMPDepobjDirective(StartLoc, EndLoc, Clauses.size());
  Dir->setClauses(Clauses);
  return Dir;
}

OMPDepobjDirective *OMPDepobjDirective::CreateEmpty(const ASTContext &C,
                                                    unsigned NumClauses,
                                                    EmptyShell) {
  unsigned Size =
      llvm::alignTo(sizeof(OMPDepobjDirective), alignof(OMPClause *));
  void *Mem = C.Allocate(Size + sizeof(OMPClause *) * NumClauses,
                         alignof(OMPDepobjDirective));
  return new (Mem) OMPDepobjDirective(NumClauses);
}

OMPScanDirective *OMPScanDirective::Create(const ASTContext &C,
                                           SourceLocation StartLoc,
                                           SourceLocation EndLoc,
                                           ArrayRef<OMPClause *> Clauses) {
  unsigned Size = llvm::alignTo(sizeof(OMPScanDirective), alignof(OMPClause *));
  void *Mem = C.Allocate(Size + sizeof(OMPClause *) * Clauses.size(),
                         alignof(OMPScanDirective));
  auto *Dir = new (Mem) OMPScanDirective(StartLoc, EndLoc, Clauses.size());
  Dir->setClauses(Clauses);
  return Dir;
}

OMPScanDirective *OMPScanDirective::CreateEmpty(const ASTContext &C,
                                                unsigned NumClauses,
                                                EmptyShell) {
  unsigned Size = llvm::alignTo(sizeof(OMPScanDirective), alignof(OMPClause *));
  void *Mem = C.Allocate(Size + sizeof(OMPClause *) * NumClauses,
                         alignof(OMPScanDirective));
  return new (Mem) OMPScanDirective(NumClauses);
}

OMPOrderedDirective *OMPOrderedDirective::Create(const ASTContext &C,
                                                 SourceLocation StartLoc,
                                                 SourceLocation EndLoc,
                                                 ArrayRef<OMPClause *> Clauses,
                                                 Stmt *AssociatedStmt) {
  unsigned Size =
      llvm::alignTo(sizeof(OMPOrderedDirective), alignof(OMPClause *));
  void *Mem =
      C.Allocate(Size + sizeof(Stmt *) + sizeof(OMPClause *) * Clauses.size());
  OMPOrderedDirective *Dir =
      new (Mem) OMPOrderedDirective(StartLoc, EndLoc, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  return Dir;
}

OMPOrderedDirective *OMPOrderedDirective::CreateEmpty(const ASTContext &C,
                                                      unsigned NumClauses,
                                                      EmptyShell) {
  unsigned Size =
      llvm::alignTo(sizeof(OMPOrderedDirective), alignof(OMPClause *));
  void *Mem =
      C.Allocate(Size + sizeof(Stmt *) + sizeof(OMPClause *) * NumClauses);
  return new (Mem) OMPOrderedDirective(NumClauses);
}

OMPAtomicDirective *OMPAtomicDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt, Expr *X, Expr *V,
    Expr *E, Expr *UE, bool IsXLHSInRHSPart, bool IsPostfixUpdate) {
  unsigned Size =
      llvm::alignTo(sizeof(OMPAtomicDirective), alignof(OMPClause *));
  void *Mem = C.Allocate(Size + sizeof(OMPClause *) * Clauses.size() +
                         5 * sizeof(Stmt *));
  OMPAtomicDirective *Dir =
      new (Mem) OMPAtomicDirective(StartLoc, EndLoc, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setX(X);
  Dir->setV(V);
  Dir->setExpr(E);
  Dir->setUpdateExpr(UE);
  Dir->IsXLHSInRHSPart = IsXLHSInRHSPart;
  Dir->IsPostfixUpdate = IsPostfixUpdate;
  return Dir;
}

OMPAtomicDirective *OMPAtomicDirective::CreateEmpty(const ASTContext &C,
                                                    unsigned NumClauses,
                                                    EmptyShell) {
  unsigned Size =
      llvm::alignTo(sizeof(OMPAtomicDirective), alignof(OMPClause *));
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * NumClauses + 5 * sizeof(Stmt *));
  return new (Mem) OMPAtomicDirective(NumClauses);
}

OMPTargetDirective *OMPTargetDirective::Create(const ASTContext &C,
                                               SourceLocation StartLoc,
                                               SourceLocation EndLoc,
                                               ArrayRef<OMPClause *> Clauses,
                                               Stmt *AssociatedStmt) {
  unsigned Size =
      llvm::alignTo(sizeof(OMPTargetDirective), alignof(OMPClause *));
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * Clauses.size() + sizeof(Stmt *));
  OMPTargetDirective *Dir =
      new (Mem) OMPTargetDirective(StartLoc, EndLoc, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  return Dir;
}

OMPTargetDirective *OMPTargetDirective::CreateEmpty(const ASTContext &C,
                                                    unsigned NumClauses,
                                                    EmptyShell) {
  unsigned Size =
      llvm::alignTo(sizeof(OMPTargetDirective), alignof(OMPClause *));
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * NumClauses + sizeof(Stmt *));
  return new (Mem) OMPTargetDirective(NumClauses);
}

OMPTargetParallelDirective *OMPTargetParallelDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt, Expr *TaskRedRef,
    bool HasCancel) {
  unsigned Size =
      llvm::alignTo(sizeof(OMPTargetParallelDirective), alignof(OMPClause *));
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * Clauses.size() + sizeof(Stmt *));
  OMPTargetParallelDirective *Dir =
      new (Mem) OMPTargetParallelDirective(StartLoc, EndLoc, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setTaskReductionRefExpr(TaskRedRef);
  Dir->setHasCancel(HasCancel);
  return Dir;
}

OMPTargetParallelDirective *
OMPTargetParallelDirective::CreateEmpty(const ASTContext &C,
                                        unsigned NumClauses, EmptyShell) {
  unsigned Size =
      llvm::alignTo(sizeof(OMPTargetParallelDirective), alignof(OMPClause *));
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * NumClauses + sizeof(Stmt *));
  return new (Mem) OMPTargetParallelDirective(NumClauses);
}

OMPTargetParallelForDirective *OMPTargetParallelForDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt,
    const HelperExprs &Exprs, Expr *TaskRedRef, bool HasCancel) {
  unsigned Size = llvm::alignTo(sizeof(OMPTargetParallelForDirective),
                                alignof(OMPClause *));
  void *Mem = C.Allocate(
      Size + sizeof(OMPClause *) * Clauses.size() +
      sizeof(Stmt *) * numLoopChildren(CollapsedNum, OMPD_target_parallel_for));
  OMPTargetParallelForDirective *Dir = new (Mem) OMPTargetParallelForDirective(
      StartLoc, EndLoc, CollapsedNum, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setIterationVariable(Exprs.IterationVarRef);
  Dir->setLastIteration(Exprs.LastIteration);
  Dir->setCalcLastIteration(Exprs.CalcLastIteration);
  Dir->setPreCond(Exprs.PreCond);
  Dir->setCond(Exprs.Cond);
  Dir->setInit(Exprs.Init);
  Dir->setInc(Exprs.Inc);
  Dir->setIsLastIterVariable(Exprs.IL);
  Dir->setLowerBoundVariable(Exprs.LB);
  Dir->setUpperBoundVariable(Exprs.UB);
  Dir->setStrideVariable(Exprs.ST);
  Dir->setEnsureUpperBound(Exprs.EUB);
  Dir->setNextLowerBound(Exprs.NLB);
  Dir->setNextUpperBound(Exprs.NUB);
  Dir->setNumIterations(Exprs.NumIterations);
  Dir->setCounters(Exprs.Counters);
  Dir->setPrivateCounters(Exprs.PrivateCounters);
  Dir->setInits(Exprs.Inits);
  Dir->setUpdates(Exprs.Updates);
  Dir->setFinals(Exprs.Finals);
  Dir->setDependentCounters(Exprs.DependentCounters);
  Dir->setDependentInits(Exprs.DependentInits);
  Dir->setFinalsConditions(Exprs.FinalsConditions);
  Dir->setPreInits(Exprs.PreInits);
  Dir->setTaskReductionRefExpr(TaskRedRef);
  Dir->setHasCancel(HasCancel);
  return Dir;
}

OMPTargetParallelForDirective *
OMPTargetParallelForDirective::CreateEmpty(const ASTContext &C,
                                           unsigned NumClauses,
                                           unsigned CollapsedNum, EmptyShell) {
  unsigned Size = llvm::alignTo(sizeof(OMPTargetParallelForDirective),
                                alignof(OMPClause *));
  void *Mem = C.Allocate(
      Size + sizeof(OMPClause *) * NumClauses +
      sizeof(Stmt *) * numLoopChildren(CollapsedNum, OMPD_target_parallel_for));
  return new (Mem) OMPTargetParallelForDirective(CollapsedNum, NumClauses);
}

OMPTargetDataDirective *OMPTargetDataDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt) {
  void *Mem = C.Allocate(
      llvm::alignTo(sizeof(OMPTargetDataDirective), alignof(OMPClause *)) +
      sizeof(OMPClause *) * Clauses.size() + sizeof(Stmt *));
  OMPTargetDataDirective *Dir =
      new (Mem) OMPTargetDataDirective(StartLoc, EndLoc, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  return Dir;
}

OMPTargetDataDirective *OMPTargetDataDirective::CreateEmpty(const ASTContext &C,
                                                            unsigned N,
                                                            EmptyShell) {
  void *Mem = C.Allocate(
      llvm::alignTo(sizeof(OMPTargetDataDirective), alignof(OMPClause *)) +
      sizeof(OMPClause *) * N + sizeof(Stmt *));
  return new (Mem) OMPTargetDataDirective(N);
}

OMPTargetEnterDataDirective *OMPTargetEnterDataDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt) {
  void *Mem = C.Allocate(
      llvm::alignTo(sizeof(OMPTargetEnterDataDirective), alignof(OMPClause *)) +
      sizeof(OMPClause *) * Clauses.size() + sizeof(Stmt *));
  OMPTargetEnterDataDirective *Dir =
      new (Mem) OMPTargetEnterDataDirective(StartLoc, EndLoc, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  return Dir;
}

OMPTargetEnterDataDirective *
OMPTargetEnterDataDirective::CreateEmpty(const ASTContext &C, unsigned N,
                                         EmptyShell) {
  void *Mem = C.Allocate(
      llvm::alignTo(sizeof(OMPTargetEnterDataDirective), alignof(OMPClause *)) +
      sizeof(OMPClause *) * N + sizeof(Stmt *));
  return new (Mem) OMPTargetEnterDataDirective(N);
}

OMPTargetExitDataDirective *OMPTargetExitDataDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt) {
  void *Mem = C.Allocate(
      llvm::alignTo(sizeof(OMPTargetExitDataDirective), alignof(OMPClause *)) +
      sizeof(OMPClause *) * Clauses.size() + sizeof(Stmt *));
  OMPTargetExitDataDirective *Dir =
      new (Mem) OMPTargetExitDataDirective(StartLoc, EndLoc, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  return Dir;
}

OMPTargetExitDataDirective *
OMPTargetExitDataDirective::CreateEmpty(const ASTContext &C, unsigned N,
                                        EmptyShell) {
  void *Mem = C.Allocate(
      llvm::alignTo(sizeof(OMPTargetExitDataDirective), alignof(OMPClause *)) +
      sizeof(OMPClause *) * N + sizeof(Stmt *));
  return new (Mem) OMPTargetExitDataDirective(N);
}

OMPTeamsDirective *OMPTeamsDirective::Create(const ASTContext &C,
                                             SourceLocation StartLoc,
                                             SourceLocation EndLoc,
                                             ArrayRef<OMPClause *> Clauses,
                                             Stmt *AssociatedStmt) {
  unsigned Size =
      llvm::alignTo(sizeof(OMPTeamsDirective), alignof(OMPClause *));
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * Clauses.size() + sizeof(Stmt *));
  OMPTeamsDirective *Dir =
      new (Mem) OMPTeamsDirective(StartLoc, EndLoc, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  return Dir;
}

OMPTeamsDirective *OMPTeamsDirective::CreateEmpty(const ASTContext &C,
                                                  unsigned NumClauses,
                                                  EmptyShell) {
  unsigned Size =
      llvm::alignTo(sizeof(OMPTeamsDirective), alignof(OMPClause *));
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * NumClauses + sizeof(Stmt *));
  return new (Mem) OMPTeamsDirective(NumClauses);
}

OMPTaskLoopDirective *OMPTaskLoopDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt,
    const HelperExprs &Exprs, bool HasCancel) {
  unsigned Size =
      llvm::alignTo(sizeof(OMPTaskLoopDirective), alignof(OMPClause *));
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * Clauses.size() +
                 sizeof(Stmt *) * numLoopChildren(CollapsedNum, OMPD_taskloop));
  OMPTaskLoopDirective *Dir = new (Mem)
      OMPTaskLoopDirective(StartLoc, EndLoc, CollapsedNum, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setIterationVariable(Exprs.IterationVarRef);
  Dir->setLastIteration(Exprs.LastIteration);
  Dir->setCalcLastIteration(Exprs.CalcLastIteration);
  Dir->setPreCond(Exprs.PreCond);
  Dir->setCond(Exprs.Cond);
  Dir->setInit(Exprs.Init);
  Dir->setInc(Exprs.Inc);
  Dir->setIsLastIterVariable(Exprs.IL);
  Dir->setLowerBoundVariable(Exprs.LB);
  Dir->setUpperBoundVariable(Exprs.UB);
  Dir->setStrideVariable(Exprs.ST);
  Dir->setEnsureUpperBound(Exprs.EUB);
  Dir->setNextLowerBound(Exprs.NLB);
  Dir->setNextUpperBound(Exprs.NUB);
  Dir->setNumIterations(Exprs.NumIterations);
  Dir->setCounters(Exprs.Counters);
  Dir->setPrivateCounters(Exprs.PrivateCounters);
  Dir->setInits(Exprs.Inits);
  Dir->setUpdates(Exprs.Updates);
  Dir->setFinals(Exprs.Finals);
  Dir->setDependentCounters(Exprs.DependentCounters);
  Dir->setDependentInits(Exprs.DependentInits);
  Dir->setFinalsConditions(Exprs.FinalsConditions);
  Dir->setPreInits(Exprs.PreInits);
  Dir->setHasCancel(HasCancel);
  return Dir;
}

OMPTaskLoopDirective *OMPTaskLoopDirective::CreateEmpty(const ASTContext &C,
                                                        unsigned NumClauses,
                                                        unsigned CollapsedNum,
                                                        EmptyShell) {
  unsigned Size =
      llvm::alignTo(sizeof(OMPTaskLoopDirective), alignof(OMPClause *));
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * NumClauses +
                 sizeof(Stmt *) * numLoopChildren(CollapsedNum, OMPD_taskloop));
  return new (Mem) OMPTaskLoopDirective(CollapsedNum, NumClauses);
}

OMPTaskLoopSimdDirective *OMPTaskLoopSimdDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt,
    const HelperExprs &Exprs) {
  unsigned Size =
      llvm::alignTo(sizeof(OMPTaskLoopSimdDirective), alignof(OMPClause *));
  void *Mem = C.Allocate(Size + sizeof(OMPClause *) * Clauses.size() +
                         sizeof(Stmt *) *
                             numLoopChildren(CollapsedNum, OMPD_taskloop_simd));
  OMPTaskLoopSimdDirective *Dir = new (Mem)
      OMPTaskLoopSimdDirective(StartLoc, EndLoc, CollapsedNum, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setIterationVariable(Exprs.IterationVarRef);
  Dir->setLastIteration(Exprs.LastIteration);
  Dir->setCalcLastIteration(Exprs.CalcLastIteration);
  Dir->setPreCond(Exprs.PreCond);
  Dir->setCond(Exprs.Cond);
  Dir->setInit(Exprs.Init);
  Dir->setInc(Exprs.Inc);
  Dir->setIsLastIterVariable(Exprs.IL);
  Dir->setLowerBoundVariable(Exprs.LB);
  Dir->setUpperBoundVariable(Exprs.UB);
  Dir->setStrideVariable(Exprs.ST);
  Dir->setEnsureUpperBound(Exprs.EUB);
  Dir->setNextLowerBound(Exprs.NLB);
  Dir->setNextUpperBound(Exprs.NUB);
  Dir->setNumIterations(Exprs.NumIterations);
  Dir->setCounters(Exprs.Counters);
  Dir->setPrivateCounters(Exprs.PrivateCounters);
  Dir->setInits(Exprs.Inits);
  Dir->setUpdates(Exprs.Updates);
  Dir->setFinals(Exprs.Finals);
  Dir->setDependentCounters(Exprs.DependentCounters);
  Dir->setDependentInits(Exprs.DependentInits);
  Dir->setFinalsConditions(Exprs.FinalsConditions);
  Dir->setPreInits(Exprs.PreInits);
  return Dir;
}

OMPTaskLoopSimdDirective *
OMPTaskLoopSimdDirective::CreateEmpty(const ASTContext &C, unsigned NumClauses,
                                      unsigned CollapsedNum, EmptyShell) {
  unsigned Size =
      llvm::alignTo(sizeof(OMPTaskLoopSimdDirective), alignof(OMPClause *));
  void *Mem = C.Allocate(Size + sizeof(OMPClause *) * NumClauses +
                         sizeof(Stmt *) *
                             numLoopChildren(CollapsedNum, OMPD_taskloop_simd));
  return new (Mem) OMPTaskLoopSimdDirective(CollapsedNum, NumClauses);
}

OMPMasterTaskLoopDirective *OMPMasterTaskLoopDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt,
    const HelperExprs &Exprs, bool HasCancel) {
  unsigned Size =
      llvm::alignTo(sizeof(OMPMasterTaskLoopDirective), alignof(OMPClause *));
  void *Mem = C.Allocate(
      Size + sizeof(OMPClause *) * Clauses.size() +
      sizeof(Stmt *) * numLoopChildren(CollapsedNum, OMPD_master_taskloop));
  OMPMasterTaskLoopDirective *Dir = new (Mem) OMPMasterTaskLoopDirective(
      StartLoc, EndLoc, CollapsedNum, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setIterationVariable(Exprs.IterationVarRef);
  Dir->setLastIteration(Exprs.LastIteration);
  Dir->setCalcLastIteration(Exprs.CalcLastIteration);
  Dir->setPreCond(Exprs.PreCond);
  Dir->setCond(Exprs.Cond);
  Dir->setInit(Exprs.Init);
  Dir->setInc(Exprs.Inc);
  Dir->setIsLastIterVariable(Exprs.IL);
  Dir->setLowerBoundVariable(Exprs.LB);
  Dir->setUpperBoundVariable(Exprs.UB);
  Dir->setStrideVariable(Exprs.ST);
  Dir->setEnsureUpperBound(Exprs.EUB);
  Dir->setNextLowerBound(Exprs.NLB);
  Dir->setNextUpperBound(Exprs.NUB);
  Dir->setNumIterations(Exprs.NumIterations);
  Dir->setCounters(Exprs.Counters);
  Dir->setPrivateCounters(Exprs.PrivateCounters);
  Dir->setInits(Exprs.Inits);
  Dir->setUpdates(Exprs.Updates);
  Dir->setFinals(Exprs.Finals);
  Dir->setDependentCounters(Exprs.DependentCounters);
  Dir->setDependentInits(Exprs.DependentInits);
  Dir->setFinalsConditions(Exprs.FinalsConditions);
  Dir->setPreInits(Exprs.PreInits);
  Dir->setHasCancel(HasCancel);
  return Dir;
}

OMPMasterTaskLoopDirective *
OMPMasterTaskLoopDirective::CreateEmpty(const ASTContext &C,
                                        unsigned NumClauses,
                                        unsigned CollapsedNum, EmptyShell) {
  unsigned Size =
      llvm::alignTo(sizeof(OMPMasterTaskLoopDirective), alignof(OMPClause *));
  void *Mem = C.Allocate(
      Size + sizeof(OMPClause *) * NumClauses +
      sizeof(Stmt *) * numLoopChildren(CollapsedNum, OMPD_master_taskloop));
  return new (Mem) OMPMasterTaskLoopDirective(CollapsedNum, NumClauses);
}

OMPMasterTaskLoopSimdDirective *OMPMasterTaskLoopSimdDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt,
    const HelperExprs &Exprs) {
  unsigned Size = llvm::alignTo(sizeof(OMPMasterTaskLoopSimdDirective),
                                alignof(OMPClause *));
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * Clauses.size() +
                 sizeof(Stmt *) *
                     numLoopChildren(CollapsedNum, OMPD_master_taskloop_simd));
  auto *Dir = new (Mem) OMPMasterTaskLoopSimdDirective(
      StartLoc, EndLoc, CollapsedNum, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setIterationVariable(Exprs.IterationVarRef);
  Dir->setLastIteration(Exprs.LastIteration);
  Dir->setCalcLastIteration(Exprs.CalcLastIteration);
  Dir->setPreCond(Exprs.PreCond);
  Dir->setCond(Exprs.Cond);
  Dir->setInit(Exprs.Init);
  Dir->setInc(Exprs.Inc);
  Dir->setIsLastIterVariable(Exprs.IL);
  Dir->setLowerBoundVariable(Exprs.LB);
  Dir->setUpperBoundVariable(Exprs.UB);
  Dir->setStrideVariable(Exprs.ST);
  Dir->setEnsureUpperBound(Exprs.EUB);
  Dir->setNextLowerBound(Exprs.NLB);
  Dir->setNextUpperBound(Exprs.NUB);
  Dir->setNumIterations(Exprs.NumIterations);
  Dir->setCounters(Exprs.Counters);
  Dir->setPrivateCounters(Exprs.PrivateCounters);
  Dir->setInits(Exprs.Inits);
  Dir->setUpdates(Exprs.Updates);
  Dir->setFinals(Exprs.Finals);
  Dir->setDependentCounters(Exprs.DependentCounters);
  Dir->setDependentInits(Exprs.DependentInits);
  Dir->setFinalsConditions(Exprs.FinalsConditions);
  Dir->setPreInits(Exprs.PreInits);
  return Dir;
}

OMPMasterTaskLoopSimdDirective *
OMPMasterTaskLoopSimdDirective::CreateEmpty(const ASTContext &C,
                                            unsigned NumClauses,
                                            unsigned CollapsedNum, EmptyShell) {
  unsigned Size = llvm::alignTo(sizeof(OMPMasterTaskLoopSimdDirective),
                                alignof(OMPClause *));
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * NumClauses +
                 sizeof(Stmt *) *
                     numLoopChildren(CollapsedNum, OMPD_master_taskloop_simd));
  return new (Mem) OMPMasterTaskLoopSimdDirective(CollapsedNum, NumClauses);
}

OMPParallelMasterTaskLoopDirective *OMPParallelMasterTaskLoopDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt,
    const HelperExprs &Exprs, bool HasCancel) {
  unsigned Size = llvm::alignTo(sizeof(OMPParallelMasterTaskLoopDirective),
                                alignof(OMPClause *));
  void *Mem = C.Allocate(
      Size + sizeof(OMPClause *) * Clauses.size() +
      sizeof(Stmt *) *
          numLoopChildren(CollapsedNum, OMPD_parallel_master_taskloop));
  auto *Dir = new (Mem) OMPParallelMasterTaskLoopDirective(
      StartLoc, EndLoc, CollapsedNum, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setIterationVariable(Exprs.IterationVarRef);
  Dir->setLastIteration(Exprs.LastIteration);
  Dir->setCalcLastIteration(Exprs.CalcLastIteration);
  Dir->setPreCond(Exprs.PreCond);
  Dir->setCond(Exprs.Cond);
  Dir->setInit(Exprs.Init);
  Dir->setInc(Exprs.Inc);
  Dir->setIsLastIterVariable(Exprs.IL);
  Dir->setLowerBoundVariable(Exprs.LB);
  Dir->setUpperBoundVariable(Exprs.UB);
  Dir->setStrideVariable(Exprs.ST);
  Dir->setEnsureUpperBound(Exprs.EUB);
  Dir->setNextLowerBound(Exprs.NLB);
  Dir->setNextUpperBound(Exprs.NUB);
  Dir->setNumIterations(Exprs.NumIterations);
  Dir->setCounters(Exprs.Counters);
  Dir->setPrivateCounters(Exprs.PrivateCounters);
  Dir->setInits(Exprs.Inits);
  Dir->setUpdates(Exprs.Updates);
  Dir->setFinals(Exprs.Finals);
  Dir->setDependentCounters(Exprs.DependentCounters);
  Dir->setDependentInits(Exprs.DependentInits);
  Dir->setFinalsConditions(Exprs.FinalsConditions);
  Dir->setPreInits(Exprs.PreInits);
  Dir->setHasCancel(HasCancel);
  return Dir;
}

OMPParallelMasterTaskLoopDirective *
OMPParallelMasterTaskLoopDirective::CreateEmpty(const ASTContext &C,
                                                unsigned NumClauses,
                                                unsigned CollapsedNum,
                                                EmptyShell) {
  unsigned Size = llvm::alignTo(sizeof(OMPParallelMasterTaskLoopDirective),
                                alignof(OMPClause *));
  void *Mem = C.Allocate(
      Size + sizeof(OMPClause *) * NumClauses +
      sizeof(Stmt *) *
          numLoopChildren(CollapsedNum, OMPD_parallel_master_taskloop));
  return new (Mem) OMPParallelMasterTaskLoopDirective(CollapsedNum, NumClauses);
}

OMPParallelMasterTaskLoopSimdDirective *
OMPParallelMasterTaskLoopSimdDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt,
    const HelperExprs &Exprs) {
  unsigned Size = llvm::alignTo(sizeof(OMPParallelMasterTaskLoopSimdDirective),
                                alignof(OMPClause *));
  void *Mem = C.Allocate(
      Size + sizeof(OMPClause *) * Clauses.size() +
      sizeof(Stmt *) *
          numLoopChildren(CollapsedNum, OMPD_parallel_master_taskloop_simd));
  auto *Dir = new (Mem) OMPParallelMasterTaskLoopSimdDirective(
      StartLoc, EndLoc, CollapsedNum, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setIterationVariable(Exprs.IterationVarRef);
  Dir->setLastIteration(Exprs.LastIteration);
  Dir->setCalcLastIteration(Exprs.CalcLastIteration);
  Dir->setPreCond(Exprs.PreCond);
  Dir->setCond(Exprs.Cond);
  Dir->setInit(Exprs.Init);
  Dir->setInc(Exprs.Inc);
  Dir->setIsLastIterVariable(Exprs.IL);
  Dir->setLowerBoundVariable(Exprs.LB);
  Dir->setUpperBoundVariable(Exprs.UB);
  Dir->setStrideVariable(Exprs.ST);
  Dir->setEnsureUpperBound(Exprs.EUB);
  Dir->setNextLowerBound(Exprs.NLB);
  Dir->setNextUpperBound(Exprs.NUB);
  Dir->setNumIterations(Exprs.NumIterations);
  Dir->setCounters(Exprs.Counters);
  Dir->setPrivateCounters(Exprs.PrivateCounters);
  Dir->setInits(Exprs.Inits);
  Dir->setUpdates(Exprs.Updates);
  Dir->setFinals(Exprs.Finals);
  Dir->setDependentCounters(Exprs.DependentCounters);
  Dir->setDependentInits(Exprs.DependentInits);
  Dir->setFinalsConditions(Exprs.FinalsConditions);
  Dir->setPreInits(Exprs.PreInits);
  return Dir;
}

OMPParallelMasterTaskLoopSimdDirective *
OMPParallelMasterTaskLoopSimdDirective::CreateEmpty(const ASTContext &C,
                                                    unsigned NumClauses,
                                                    unsigned CollapsedNum,
                                                    EmptyShell) {
  unsigned Size = llvm::alignTo(sizeof(OMPParallelMasterTaskLoopSimdDirective),
                                alignof(OMPClause *));
  void *Mem = C.Allocate(
      Size + sizeof(OMPClause *) * NumClauses +
      sizeof(Stmt *) *
          numLoopChildren(CollapsedNum, OMPD_parallel_master_taskloop_simd));
  return new (Mem)
      OMPParallelMasterTaskLoopSimdDirective(CollapsedNum, NumClauses);
}

OMPDistributeDirective *OMPDistributeDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt,
    const HelperExprs &Exprs) {
  unsigned Size =
      llvm::alignTo(sizeof(OMPDistributeDirective), alignof(OMPClause *));
  void *Mem = C.Allocate(Size + sizeof(OMPClause *) * Clauses.size() +
                         sizeof(Stmt *) *
                             numLoopChildren(CollapsedNum, OMPD_distribute));
  OMPDistributeDirective *Dir = new (Mem)
      OMPDistributeDirective(StartLoc, EndLoc, CollapsedNum, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setIterationVariable(Exprs.IterationVarRef);
  Dir->setLastIteration(Exprs.LastIteration);
  Dir->setCalcLastIteration(Exprs.CalcLastIteration);
  Dir->setPreCond(Exprs.PreCond);
  Dir->setCond(Exprs.Cond);
  Dir->setInit(Exprs.Init);
  Dir->setInc(Exprs.Inc);
  Dir->setIsLastIterVariable(Exprs.IL);
  Dir->setLowerBoundVariable(Exprs.LB);
  Dir->setUpperBoundVariable(Exprs.UB);
  Dir->setStrideVariable(Exprs.ST);
  Dir->setEnsureUpperBound(Exprs.EUB);
  Dir->setNextLowerBound(Exprs.NLB);
  Dir->setNextUpperBound(Exprs.NUB);
  Dir->setNumIterations(Exprs.NumIterations);
  Dir->setCounters(Exprs.Counters);
  Dir->setPrivateCounters(Exprs.PrivateCounters);
  Dir->setInits(Exprs.Inits);
  Dir->setUpdates(Exprs.Updates);
  Dir->setFinals(Exprs.Finals);
  Dir->setDependentCounters(Exprs.DependentCounters);
  Dir->setDependentInits(Exprs.DependentInits);
  Dir->setFinalsConditions(Exprs.FinalsConditions);
  Dir->setPreInits(Exprs.PreInits);
  return Dir;
}

OMPDistributeDirective *
OMPDistributeDirective::CreateEmpty(const ASTContext &C, unsigned NumClauses,
                                    unsigned CollapsedNum, EmptyShell) {
  unsigned Size =
      llvm::alignTo(sizeof(OMPDistributeDirective), alignof(OMPClause *));
  void *Mem = C.Allocate(Size + sizeof(OMPClause *) * NumClauses +
                         sizeof(Stmt *) *
                             numLoopChildren(CollapsedNum, OMPD_distribute));
  return new (Mem) OMPDistributeDirective(CollapsedNum, NumClauses);
}

OMPTargetUpdateDirective *OMPTargetUpdateDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt) {
  unsigned Size =
      llvm::alignTo(sizeof(OMPTargetUpdateDirective), alignof(OMPClause *));
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * Clauses.size() + sizeof(Stmt *));
  OMPTargetUpdateDirective *Dir =
      new (Mem) OMPTargetUpdateDirective(StartLoc, EndLoc, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  return Dir;
}

OMPTargetUpdateDirective *
OMPTargetUpdateDirective::CreateEmpty(const ASTContext &C, unsigned NumClauses,
                                      EmptyShell) {
  unsigned Size =
      llvm::alignTo(sizeof(OMPTargetUpdateDirective), alignof(OMPClause *));
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * NumClauses + sizeof(Stmt *));
  return new (Mem) OMPTargetUpdateDirective(NumClauses);
}

OMPDistributeParallelForDirective *OMPDistributeParallelForDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt,
    const HelperExprs &Exprs, Expr *TaskRedRef, bool HasCancel) {
  unsigned Size = llvm::alignTo(sizeof(OMPDistributeParallelForDirective),
                                alignof(OMPClause *));
  void *Mem = C.Allocate(
      Size + sizeof(OMPClause *) * Clauses.size() +
      sizeof(Stmt *) *
          numLoopChildren(CollapsedNum, OMPD_distribute_parallel_for));
  OMPDistributeParallelForDirective *Dir =
      new (Mem) OMPDistributeParallelForDirective(StartLoc, EndLoc,
                                                  CollapsedNum, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setIterationVariable(Exprs.IterationVarRef);
  Dir->setLastIteration(Exprs.LastIteration);
  Dir->setCalcLastIteration(Exprs.CalcLastIteration);
  Dir->setPreCond(Exprs.PreCond);
  Dir->setCond(Exprs.Cond);
  Dir->setInit(Exprs.Init);
  Dir->setInc(Exprs.Inc);
  Dir->setIsLastIterVariable(Exprs.IL);
  Dir->setLowerBoundVariable(Exprs.LB);
  Dir->setUpperBoundVariable(Exprs.UB);
  Dir->setStrideVariable(Exprs.ST);
  Dir->setEnsureUpperBound(Exprs.EUB);
  Dir->setNextLowerBound(Exprs.NLB);
  Dir->setNextUpperBound(Exprs.NUB);
  Dir->setNumIterations(Exprs.NumIterations);
  Dir->setPrevLowerBoundVariable(Exprs.PrevLB);
  Dir->setPrevUpperBoundVariable(Exprs.PrevUB);
  Dir->setDistInc(Exprs.DistInc);
  Dir->setPrevEnsureUpperBound(Exprs.PrevEUB);
  Dir->setCounters(Exprs.Counters);
  Dir->setPrivateCounters(Exprs.PrivateCounters);
  Dir->setInits(Exprs.Inits);
  Dir->setUpdates(Exprs.Updates);
  Dir->setFinals(Exprs.Finals);
  Dir->setDependentCounters(Exprs.DependentCounters);
  Dir->setDependentInits(Exprs.DependentInits);
  Dir->setFinalsConditions(Exprs.FinalsConditions);
  Dir->setPreInits(Exprs.PreInits);
  Dir->setCombinedLowerBoundVariable(Exprs.DistCombinedFields.LB);
  Dir->setCombinedUpperBoundVariable(Exprs.DistCombinedFields.UB);
  Dir->setCombinedEnsureUpperBound(Exprs.DistCombinedFields.EUB);
  Dir->setCombinedInit(Exprs.DistCombinedFields.Init);
  Dir->setCombinedCond(Exprs.DistCombinedFields.Cond);
  Dir->setCombinedNextLowerBound(Exprs.DistCombinedFields.NLB);
  Dir->setCombinedNextUpperBound(Exprs.DistCombinedFields.NUB);
  Dir->setCombinedDistCond(Exprs.DistCombinedFields.DistCond);
  Dir->setCombinedParForInDistCond(Exprs.DistCombinedFields.ParForInDistCond);
  Dir->setTaskReductionRefExpr(TaskRedRef);
  Dir->HasCancel = HasCancel;
  return Dir;
}

OMPDistributeParallelForDirective *
OMPDistributeParallelForDirective::CreateEmpty(const ASTContext &C,
                                               unsigned NumClauses,
                                               unsigned CollapsedNum,
                                               EmptyShell) {
  unsigned Size = llvm::alignTo(sizeof(OMPDistributeParallelForDirective),
                                alignof(OMPClause *));
  void *Mem = C.Allocate(
      Size + sizeof(OMPClause *) * NumClauses +
      sizeof(Stmt *) *
          numLoopChildren(CollapsedNum, OMPD_distribute_parallel_for));
  return new (Mem) OMPDistributeParallelForDirective(CollapsedNum, NumClauses);
}

OMPDistributeParallelForSimdDirective *
OMPDistributeParallelForSimdDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt,
    const HelperExprs &Exprs) {
  unsigned Size = llvm::alignTo(sizeof(OMPDistributeParallelForSimdDirective),
                                alignof(OMPClause *));
  void *Mem = C.Allocate(
      Size + sizeof(OMPClause *) * Clauses.size() +
      sizeof(Stmt *) *
          numLoopChildren(CollapsedNum, OMPD_distribute_parallel_for_simd));
  OMPDistributeParallelForSimdDirective *Dir = new (Mem)
      OMPDistributeParallelForSimdDirective(StartLoc, EndLoc, CollapsedNum,
                                            Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setIterationVariable(Exprs.IterationVarRef);
  Dir->setLastIteration(Exprs.LastIteration);
  Dir->setCalcLastIteration(Exprs.CalcLastIteration);
  Dir->setPreCond(Exprs.PreCond);
  Dir->setCond(Exprs.Cond);
  Dir->setInit(Exprs.Init);
  Dir->setInc(Exprs.Inc);
  Dir->setIsLastIterVariable(Exprs.IL);
  Dir->setLowerBoundVariable(Exprs.LB);
  Dir->setUpperBoundVariable(Exprs.UB);
  Dir->setStrideVariable(Exprs.ST);
  Dir->setEnsureUpperBound(Exprs.EUB);
  Dir->setNextLowerBound(Exprs.NLB);
  Dir->setNextUpperBound(Exprs.NUB);
  Dir->setNumIterations(Exprs.NumIterations);
  Dir->setPrevLowerBoundVariable(Exprs.PrevLB);
  Dir->setPrevUpperBoundVariable(Exprs.PrevUB);
  Dir->setDistInc(Exprs.DistInc);
  Dir->setPrevEnsureUpperBound(Exprs.PrevEUB);
  Dir->setCounters(Exprs.Counters);
  Dir->setPrivateCounters(Exprs.PrivateCounters);
  Dir->setInits(Exprs.Inits);
  Dir->setUpdates(Exprs.Updates);
  Dir->setFinals(Exprs.Finals);
  Dir->setDependentCounters(Exprs.DependentCounters);
  Dir->setDependentInits(Exprs.DependentInits);
  Dir->setFinalsConditions(Exprs.FinalsConditions);
  Dir->setPreInits(Exprs.PreInits);
  Dir->setCombinedLowerBoundVariable(Exprs.DistCombinedFields.LB);
  Dir->setCombinedUpperBoundVariable(Exprs.DistCombinedFields.UB);
  Dir->setCombinedEnsureUpperBound(Exprs.DistCombinedFields.EUB);
  Dir->setCombinedInit(Exprs.DistCombinedFields.Init);
  Dir->setCombinedCond(Exprs.DistCombinedFields.Cond);
  Dir->setCombinedNextLowerBound(Exprs.DistCombinedFields.NLB);
  Dir->setCombinedNextUpperBound(Exprs.DistCombinedFields.NUB);
  Dir->setCombinedDistCond(Exprs.DistCombinedFields.DistCond);
  Dir->setCombinedParForInDistCond(Exprs.DistCombinedFields.ParForInDistCond);
  return Dir;
}

OMPDistributeParallelForSimdDirective *
OMPDistributeParallelForSimdDirective::CreateEmpty(const ASTContext &C,
                                                   unsigned NumClauses,
                                                   unsigned CollapsedNum,
                                                   EmptyShell) {
  unsigned Size = llvm::alignTo(sizeof(OMPDistributeParallelForSimdDirective),
                                alignof(OMPClause *));
  void *Mem = C.Allocate(
      Size + sizeof(OMPClause *) * NumClauses +
      sizeof(Stmt *) *
          numLoopChildren(CollapsedNum, OMPD_distribute_parallel_for_simd));
  return new (Mem)
      OMPDistributeParallelForSimdDirective(CollapsedNum, NumClauses);
}

OMPDistributeSimdDirective *OMPDistributeSimdDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt,
    const HelperExprs &Exprs) {
  unsigned Size =
      llvm::alignTo(sizeof(OMPDistributeSimdDirective), alignof(OMPClause *));
  void *Mem = C.Allocate(
      Size + sizeof(OMPClause *) * Clauses.size() +
      sizeof(Stmt *) *
          numLoopChildren(CollapsedNum, OMPD_distribute_simd));
  OMPDistributeSimdDirective *Dir = new (Mem) OMPDistributeSimdDirective(
      StartLoc, EndLoc, CollapsedNum, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setIterationVariable(Exprs.IterationVarRef);
  Dir->setLastIteration(Exprs.LastIteration);
  Dir->setCalcLastIteration(Exprs.CalcLastIteration);
  Dir->setPreCond(Exprs.PreCond);
  Dir->setCond(Exprs.Cond);
  Dir->setInit(Exprs.Init);
  Dir->setInc(Exprs.Inc);
  Dir->setIsLastIterVariable(Exprs.IL);
  Dir->setLowerBoundVariable(Exprs.LB);
  Dir->setUpperBoundVariable(Exprs.UB);
  Dir->setStrideVariable(Exprs.ST);
  Dir->setEnsureUpperBound(Exprs.EUB);
  Dir->setNextLowerBound(Exprs.NLB);
  Dir->setNextUpperBound(Exprs.NUB);
  Dir->setNumIterations(Exprs.NumIterations);
  Dir->setCounters(Exprs.Counters);
  Dir->setPrivateCounters(Exprs.PrivateCounters);
  Dir->setInits(Exprs.Inits);
  Dir->setUpdates(Exprs.Updates);
  Dir->setFinals(Exprs.Finals);
  Dir->setDependentCounters(Exprs.DependentCounters);
  Dir->setDependentInits(Exprs.DependentInits);
  Dir->setFinalsConditions(Exprs.FinalsConditions);
  Dir->setPreInits(Exprs.PreInits);
  return Dir;
}

OMPDistributeSimdDirective *
OMPDistributeSimdDirective::CreateEmpty(const ASTContext &C,
                                        unsigned NumClauses,
                                        unsigned CollapsedNum, EmptyShell) {
  unsigned Size =
      llvm::alignTo(sizeof(OMPDistributeSimdDirective), alignof(OMPClause *));
  void *Mem = C.Allocate(
      Size + sizeof(OMPClause *) * NumClauses +
      sizeof(Stmt *) *
          numLoopChildren(CollapsedNum, OMPD_distribute_simd));
  return new (Mem) OMPDistributeSimdDirective(CollapsedNum, NumClauses);
}

OMPTargetParallelForSimdDirective *OMPTargetParallelForSimdDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt,
    const HelperExprs &Exprs) {
  unsigned Size = llvm::alignTo(sizeof(OMPTargetParallelForSimdDirective),
                                alignof(OMPClause *));
  void *Mem = C.Allocate(
      Size + sizeof(OMPClause *) * Clauses.size() +
      sizeof(Stmt *) *
          numLoopChildren(CollapsedNum, OMPD_target_parallel_for_simd));
  OMPTargetParallelForSimdDirective *Dir =
      new (Mem) OMPTargetParallelForSimdDirective(StartLoc, EndLoc,
                                                  CollapsedNum, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setIterationVariable(Exprs.IterationVarRef);
  Dir->setLastIteration(Exprs.LastIteration);
  Dir->setCalcLastIteration(Exprs.CalcLastIteration);
  Dir->setPreCond(Exprs.PreCond);
  Dir->setCond(Exprs.Cond);
  Dir->setInit(Exprs.Init);
  Dir->setInc(Exprs.Inc);
  Dir->setIsLastIterVariable(Exprs.IL);
  Dir->setLowerBoundVariable(Exprs.LB);
  Dir->setUpperBoundVariable(Exprs.UB);
  Dir->setStrideVariable(Exprs.ST);
  Dir->setEnsureUpperBound(Exprs.EUB);
  Dir->setNextLowerBound(Exprs.NLB);
  Dir->setNextUpperBound(Exprs.NUB);
  Dir->setNumIterations(Exprs.NumIterations);
  Dir->setCounters(Exprs.Counters);
  Dir->setPrivateCounters(Exprs.PrivateCounters);
  Dir->setInits(Exprs.Inits);
  Dir->setUpdates(Exprs.Updates);
  Dir->setFinals(Exprs.Finals);
  Dir->setDependentCounters(Exprs.DependentCounters);
  Dir->setDependentInits(Exprs.DependentInits);
  Dir->setFinalsConditions(Exprs.FinalsConditions);
  Dir->setPreInits(Exprs.PreInits);
  return Dir;
}

OMPTargetParallelForSimdDirective *
OMPTargetParallelForSimdDirective::CreateEmpty(const ASTContext &C,
                                               unsigned NumClauses,
                                               unsigned CollapsedNum,
                                               EmptyShell) {
  unsigned Size = llvm::alignTo(sizeof(OMPTargetParallelForSimdDirective),
                                alignof(OMPClause *));
  void *Mem = C.Allocate(
      Size + sizeof(OMPClause *) * NumClauses +
      sizeof(Stmt *) *
          numLoopChildren(CollapsedNum, OMPD_target_parallel_for_simd));
  return new (Mem) OMPTargetParallelForSimdDirective(CollapsedNum, NumClauses);
}

OMPTargetSimdDirective *
OMPTargetSimdDirective::Create(const ASTContext &C, SourceLocation StartLoc,
                               SourceLocation EndLoc, unsigned CollapsedNum,
                               ArrayRef<OMPClause *> Clauses,
                               Stmt *AssociatedStmt, const HelperExprs &Exprs) {
  unsigned Size =
      llvm::alignTo(sizeof(OMPTargetSimdDirective), alignof(OMPClause *));
  void *Mem = C.Allocate(Size + sizeof(OMPClause *) * Clauses.size() +
                         sizeof(Stmt *) *
                             numLoopChildren(CollapsedNum, OMPD_target_simd));
  OMPTargetSimdDirective *Dir = new (Mem)
      OMPTargetSimdDirective(StartLoc, EndLoc, CollapsedNum, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setIterationVariable(Exprs.IterationVarRef);
  Dir->setLastIteration(Exprs.LastIteration);
  Dir->setCalcLastIteration(Exprs.CalcLastIteration);
  Dir->setPreCond(Exprs.PreCond);
  Dir->setCond(Exprs.Cond);
  Dir->setInit(Exprs.Init);
  Dir->setInc(Exprs.Inc);
  Dir->setCounters(Exprs.Counters);
  Dir->setPrivateCounters(Exprs.PrivateCounters);
  Dir->setInits(Exprs.Inits);
  Dir->setUpdates(Exprs.Updates);
  Dir->setFinals(Exprs.Finals);
  Dir->setDependentCounters(Exprs.DependentCounters);
  Dir->setDependentInits(Exprs.DependentInits);
  Dir->setFinalsConditions(Exprs.FinalsConditions);
  Dir->setPreInits(Exprs.PreInits);
  return Dir;
}

OMPTargetSimdDirective *
OMPTargetSimdDirective::CreateEmpty(const ASTContext &C, unsigned NumClauses,
                                    unsigned CollapsedNum, EmptyShell) {
  unsigned Size =
      llvm::alignTo(sizeof(OMPTargetSimdDirective), alignof(OMPClause *));
  void *Mem = C.Allocate(Size + sizeof(OMPClause *) * NumClauses +
                         sizeof(Stmt *) *
                             numLoopChildren(CollapsedNum, OMPD_target_simd));
  return new (Mem) OMPTargetSimdDirective(CollapsedNum, NumClauses);
}

OMPTeamsDistributeDirective *OMPTeamsDistributeDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt,
    const HelperExprs &Exprs) {
  unsigned Size =
      llvm::alignTo(sizeof(OMPTeamsDistributeDirective), alignof(OMPClause *));
  void *Mem = C.Allocate(
      Size + sizeof(OMPClause *) * Clauses.size() +
      sizeof(Stmt *) * numLoopChildren(CollapsedNum, OMPD_teams_distribute));
  OMPTeamsDistributeDirective *Dir = new (Mem) OMPTeamsDistributeDirective(
      StartLoc, EndLoc, CollapsedNum, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setIterationVariable(Exprs.IterationVarRef);
  Dir->setLastIteration(Exprs.LastIteration);
  Dir->setCalcLastIteration(Exprs.CalcLastIteration);
  Dir->setPreCond(Exprs.PreCond);
  Dir->setCond(Exprs.Cond);
  Dir->setInit(Exprs.Init);
  Dir->setInc(Exprs.Inc);
  Dir->setIsLastIterVariable(Exprs.IL);
  Dir->setLowerBoundVariable(Exprs.LB);
  Dir->setUpperBoundVariable(Exprs.UB);
  Dir->setStrideVariable(Exprs.ST);
  Dir->setEnsureUpperBound(Exprs.EUB);
  Dir->setNextLowerBound(Exprs.NLB);
  Dir->setNextUpperBound(Exprs.NUB);
  Dir->setNumIterations(Exprs.NumIterations);
  Dir->setCounters(Exprs.Counters);
  Dir->setPrivateCounters(Exprs.PrivateCounters);
  Dir->setInits(Exprs.Inits);
  Dir->setUpdates(Exprs.Updates);
  Dir->setFinals(Exprs.Finals);
  Dir->setDependentCounters(Exprs.DependentCounters);
  Dir->setDependentInits(Exprs.DependentInits);
  Dir->setFinalsConditions(Exprs.FinalsConditions);
  Dir->setPreInits(Exprs.PreInits);
  return Dir;
}

OMPTeamsDistributeDirective *
OMPTeamsDistributeDirective::CreateEmpty(const ASTContext &C,
                                         unsigned NumClauses,
                                         unsigned CollapsedNum, EmptyShell) {
  unsigned Size =
      llvm::alignTo(sizeof(OMPTeamsDistributeDirective), alignof(OMPClause *));
  void *Mem = C.Allocate(
      Size + sizeof(OMPClause *) * NumClauses +
      sizeof(Stmt *) * numLoopChildren(CollapsedNum, OMPD_teams_distribute));
  return new (Mem) OMPTeamsDistributeDirective(CollapsedNum, NumClauses);
}

OMPTeamsDistributeSimdDirective *OMPTeamsDistributeSimdDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt,
    const HelperExprs &Exprs) {
  unsigned Size = llvm::alignTo(sizeof(OMPTeamsDistributeSimdDirective),
                                alignof(OMPClause *));
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * Clauses.size() +
                 sizeof(Stmt *) *
                     numLoopChildren(CollapsedNum, OMPD_teams_distribute_simd));
  OMPTeamsDistributeSimdDirective *Dir =
      new (Mem) OMPTeamsDistributeSimdDirective(StartLoc, EndLoc, CollapsedNum,
                                                Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setIterationVariable(Exprs.IterationVarRef);
  Dir->setLastIteration(Exprs.LastIteration);
  Dir->setCalcLastIteration(Exprs.CalcLastIteration);
  Dir->setPreCond(Exprs.PreCond);
  Dir->setCond(Exprs.Cond);
  Dir->setInit(Exprs.Init);
  Dir->setInc(Exprs.Inc);
  Dir->setIsLastIterVariable(Exprs.IL);
  Dir->setLowerBoundVariable(Exprs.LB);
  Dir->setUpperBoundVariable(Exprs.UB);
  Dir->setStrideVariable(Exprs.ST);
  Dir->setEnsureUpperBound(Exprs.EUB);
  Dir->setNextLowerBound(Exprs.NLB);
  Dir->setNextUpperBound(Exprs.NUB);
  Dir->setNumIterations(Exprs.NumIterations);
  Dir->setCounters(Exprs.Counters);
  Dir->setPrivateCounters(Exprs.PrivateCounters);
  Dir->setInits(Exprs.Inits);
  Dir->setUpdates(Exprs.Updates);
  Dir->setFinals(Exprs.Finals);
  Dir->setDependentCounters(Exprs.DependentCounters);
  Dir->setDependentInits(Exprs.DependentInits);
  Dir->setFinalsConditions(Exprs.FinalsConditions);
  Dir->setPreInits(Exprs.PreInits);
  return Dir;
}

OMPTeamsDistributeSimdDirective *OMPTeamsDistributeSimdDirective::CreateEmpty(
    const ASTContext &C, unsigned NumClauses, unsigned CollapsedNum,
    EmptyShell) {
  unsigned Size = llvm::alignTo(sizeof(OMPTeamsDistributeSimdDirective),
                                alignof(OMPClause *));
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * NumClauses +
                 sizeof(Stmt *) *
                     numLoopChildren(CollapsedNum, OMPD_teams_distribute_simd));
  return new (Mem) OMPTeamsDistributeSimdDirective(CollapsedNum, NumClauses);
}

OMPTeamsDistributeParallelForSimdDirective *
OMPTeamsDistributeParallelForSimdDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt,
    const HelperExprs &Exprs) {
  auto Size = llvm::alignTo(sizeof(OMPTeamsDistributeParallelForSimdDirective),
                            alignof(OMPClause *));
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * Clauses.size() +
                 sizeof(Stmt *) *
                     numLoopChildren(CollapsedNum,
                                     OMPD_teams_distribute_parallel_for_simd));
  OMPTeamsDistributeParallelForSimdDirective *Dir = new (Mem)
      OMPTeamsDistributeParallelForSimdDirective(StartLoc, EndLoc, CollapsedNum,
                                                 Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setIterationVariable(Exprs.IterationVarRef);
  Dir->setLastIteration(Exprs.LastIteration);
  Dir->setCalcLastIteration(Exprs.CalcLastIteration);
  Dir->setPreCond(Exprs.PreCond);
  Dir->setCond(Exprs.Cond);
  Dir->setInit(Exprs.Init);
  Dir->setInc(Exprs.Inc);
  Dir->setIsLastIterVariable(Exprs.IL);
  Dir->setLowerBoundVariable(Exprs.LB);
  Dir->setUpperBoundVariable(Exprs.UB);
  Dir->setStrideVariable(Exprs.ST);
  Dir->setEnsureUpperBound(Exprs.EUB);
  Dir->setNextLowerBound(Exprs.NLB);
  Dir->setNextUpperBound(Exprs.NUB);
  Dir->setNumIterations(Exprs.NumIterations);
  Dir->setPrevLowerBoundVariable(Exprs.PrevLB);
  Dir->setPrevUpperBoundVariable(Exprs.PrevUB);
  Dir->setDistInc(Exprs.DistInc);
  Dir->setPrevEnsureUpperBound(Exprs.PrevEUB);
  Dir->setCounters(Exprs.Counters);
  Dir->setPrivateCounters(Exprs.PrivateCounters);
  Dir->setInits(Exprs.Inits);
  Dir->setUpdates(Exprs.Updates);
  Dir->setFinals(Exprs.Finals);
  Dir->setDependentCounters(Exprs.DependentCounters);
  Dir->setDependentInits(Exprs.DependentInits);
  Dir->setFinalsConditions(Exprs.FinalsConditions);
  Dir->setPreInits(Exprs.PreInits);
  Dir->setCombinedLowerBoundVariable(Exprs.DistCombinedFields.LB);
  Dir->setCombinedUpperBoundVariable(Exprs.DistCombinedFields.UB);
  Dir->setCombinedEnsureUpperBound(Exprs.DistCombinedFields.EUB);
  Dir->setCombinedInit(Exprs.DistCombinedFields.Init);
  Dir->setCombinedCond(Exprs.DistCombinedFields.Cond);
  Dir->setCombinedNextLowerBound(Exprs.DistCombinedFields.NLB);
  Dir->setCombinedNextUpperBound(Exprs.DistCombinedFields.NUB);
  Dir->setCombinedDistCond(Exprs.DistCombinedFields.DistCond);
  Dir->setCombinedParForInDistCond(Exprs.DistCombinedFields.ParForInDistCond);
  return Dir;
}

OMPTeamsDistributeParallelForSimdDirective *
OMPTeamsDistributeParallelForSimdDirective::CreateEmpty(const ASTContext &C,
                                                        unsigned NumClauses,
                                                        unsigned CollapsedNum,
                                                        EmptyShell) {
  auto Size = llvm::alignTo(sizeof(OMPTeamsDistributeParallelForSimdDirective),
                            alignof(OMPClause *));
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * NumClauses +
                 sizeof(Stmt *) *
                     numLoopChildren(CollapsedNum,
                                     OMPD_teams_distribute_parallel_for_simd));
  return new (Mem)
      OMPTeamsDistributeParallelForSimdDirective(CollapsedNum, NumClauses);
}

OMPTeamsDistributeParallelForDirective *
OMPTeamsDistributeParallelForDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt,
    const HelperExprs &Exprs, Expr *TaskRedRef, bool HasCancel) {
  auto Size = llvm::alignTo(sizeof(OMPTeamsDistributeParallelForDirective),
                            alignof(OMPClause *));
  void *Mem = C.Allocate(
      Size + sizeof(OMPClause *) * Clauses.size() +
      sizeof(Stmt *) *
          numLoopChildren(CollapsedNum, OMPD_teams_distribute_parallel_for));
  OMPTeamsDistributeParallelForDirective *Dir = new (Mem)
      OMPTeamsDistributeParallelForDirective(StartLoc, EndLoc, CollapsedNum,
                                             Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setIterationVariable(Exprs.IterationVarRef);
  Dir->setLastIteration(Exprs.LastIteration);
  Dir->setCalcLastIteration(Exprs.CalcLastIteration);
  Dir->setPreCond(Exprs.PreCond);
  Dir->setCond(Exprs.Cond);
  Dir->setInit(Exprs.Init);
  Dir->setInc(Exprs.Inc);
  Dir->setIsLastIterVariable(Exprs.IL);
  Dir->setLowerBoundVariable(Exprs.LB);
  Dir->setUpperBoundVariable(Exprs.UB);
  Dir->setStrideVariable(Exprs.ST);
  Dir->setEnsureUpperBound(Exprs.EUB);
  Dir->setNextLowerBound(Exprs.NLB);
  Dir->setNextUpperBound(Exprs.NUB);
  Dir->setNumIterations(Exprs.NumIterations);
  Dir->setPrevLowerBoundVariable(Exprs.PrevLB);
  Dir->setPrevUpperBoundVariable(Exprs.PrevUB);
  Dir->setDistInc(Exprs.DistInc);
  Dir->setPrevEnsureUpperBound(Exprs.PrevEUB);
  Dir->setCounters(Exprs.Counters);
  Dir->setPrivateCounters(Exprs.PrivateCounters);
  Dir->setInits(Exprs.Inits);
  Dir->setUpdates(Exprs.Updates);
  Dir->setFinals(Exprs.Finals);
  Dir->setDependentCounters(Exprs.DependentCounters);
  Dir->setDependentInits(Exprs.DependentInits);
  Dir->setFinalsConditions(Exprs.FinalsConditions);
  Dir->setPreInits(Exprs.PreInits);
  Dir->setCombinedLowerBoundVariable(Exprs.DistCombinedFields.LB);
  Dir->setCombinedUpperBoundVariable(Exprs.DistCombinedFields.UB);
  Dir->setCombinedEnsureUpperBound(Exprs.DistCombinedFields.EUB);
  Dir->setCombinedInit(Exprs.DistCombinedFields.Init);
  Dir->setCombinedCond(Exprs.DistCombinedFields.Cond);
  Dir->setCombinedNextLowerBound(Exprs.DistCombinedFields.NLB);
  Dir->setCombinedNextUpperBound(Exprs.DistCombinedFields.NUB);
  Dir->setCombinedDistCond(Exprs.DistCombinedFields.DistCond);
  Dir->setCombinedParForInDistCond(Exprs.DistCombinedFields.ParForInDistCond);
  Dir->setTaskReductionRefExpr(TaskRedRef);
  Dir->HasCancel = HasCancel;
  return Dir;
}

OMPTeamsDistributeParallelForDirective *
OMPTeamsDistributeParallelForDirective::CreateEmpty(const ASTContext &C,
                                                    unsigned NumClauses,
                                                    unsigned CollapsedNum,
                                                    EmptyShell) {
  auto Size = llvm::alignTo(sizeof(OMPTeamsDistributeParallelForDirective),
                            alignof(OMPClause *));
  void *Mem = C.Allocate(
      Size + sizeof(OMPClause *) * NumClauses +
      sizeof(Stmt *) *
          numLoopChildren(CollapsedNum, OMPD_teams_distribute_parallel_for));
  return new (Mem)
      OMPTeamsDistributeParallelForDirective(CollapsedNum, NumClauses);
}

OMPTargetTeamsDirective *OMPTargetTeamsDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt) {
  auto Size =
      llvm::alignTo(sizeof(OMPTargetTeamsDirective), alignof(OMPClause *));
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * Clauses.size() + sizeof(Stmt *));
  OMPTargetTeamsDirective *Dir =
      new (Mem) OMPTargetTeamsDirective(StartLoc, EndLoc, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  return Dir;
}

OMPTargetTeamsDirective *
OMPTargetTeamsDirective::CreateEmpty(const ASTContext &C, unsigned NumClauses,
                                     EmptyShell) {
  auto Size =
      llvm::alignTo(sizeof(OMPTargetTeamsDirective), alignof(OMPClause *));
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * NumClauses + sizeof(Stmt *));
  return new (Mem) OMPTargetTeamsDirective(NumClauses);
}

OMPTargetTeamsDistributeDirective *OMPTargetTeamsDistributeDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt,
    const HelperExprs &Exprs) {
  auto Size = llvm::alignTo(sizeof(OMPTargetTeamsDistributeDirective),
                            alignof(OMPClause *));
  void *Mem = C.Allocate(
      Size + sizeof(OMPClause *) * Clauses.size() +
      sizeof(Stmt *) *
          numLoopChildren(CollapsedNum, OMPD_target_teams_distribute));
  OMPTargetTeamsDistributeDirective *Dir =
      new (Mem) OMPTargetTeamsDistributeDirective(StartLoc, EndLoc, CollapsedNum,
                                                  Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setIterationVariable(Exprs.IterationVarRef);
  Dir->setLastIteration(Exprs.LastIteration);
  Dir->setCalcLastIteration(Exprs.CalcLastIteration);
  Dir->setPreCond(Exprs.PreCond);
  Dir->setCond(Exprs.Cond);
  Dir->setInit(Exprs.Init);
  Dir->setInc(Exprs.Inc);
  Dir->setIsLastIterVariable(Exprs.IL);
  Dir->setLowerBoundVariable(Exprs.LB);
  Dir->setUpperBoundVariable(Exprs.UB);
  Dir->setStrideVariable(Exprs.ST);
  Dir->setEnsureUpperBound(Exprs.EUB);
  Dir->setNextLowerBound(Exprs.NLB);
  Dir->setNextUpperBound(Exprs.NUB);
  Dir->setNumIterations(Exprs.NumIterations);
  Dir->setCounters(Exprs.Counters);
  Dir->setPrivateCounters(Exprs.PrivateCounters);
  Dir->setInits(Exprs.Inits);
  Dir->setUpdates(Exprs.Updates);
  Dir->setFinals(Exprs.Finals);
  Dir->setDependentCounters(Exprs.DependentCounters);
  Dir->setDependentInits(Exprs.DependentInits);
  Dir->setFinalsConditions(Exprs.FinalsConditions);
  Dir->setPreInits(Exprs.PreInits);
  return Dir;
}

OMPTargetTeamsDistributeDirective *
OMPTargetTeamsDistributeDirective::CreateEmpty(const ASTContext &C,
                                               unsigned NumClauses,
                                               unsigned CollapsedNum,
                                               EmptyShell) {
  auto Size = llvm::alignTo(sizeof(OMPTargetTeamsDistributeDirective),
                            alignof(OMPClause *));
  void *Mem = C.Allocate(
      Size + sizeof(OMPClause *) * NumClauses +
      sizeof(Stmt *) *
           numLoopChildren(CollapsedNum, OMPD_target_teams_distribute));
  return new (Mem) OMPTargetTeamsDistributeDirective(CollapsedNum, NumClauses);
}

OMPTargetTeamsDistributeParallelForDirective *
OMPTargetTeamsDistributeParallelForDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt,
    const HelperExprs &Exprs, Expr *TaskRedRef, bool HasCancel) {
  auto Size =
      llvm::alignTo(sizeof(OMPTargetTeamsDistributeParallelForDirective),
                    alignof(OMPClause *));
  void *Mem = C.Allocate(
      Size + sizeof(OMPClause *) * Clauses.size() +
      sizeof(Stmt *) *
          numLoopChildren(CollapsedNum,
                          OMPD_target_teams_distribute_parallel_for));
  OMPTargetTeamsDistributeParallelForDirective *Dir =
      new (Mem) OMPTargetTeamsDistributeParallelForDirective(
           StartLoc, EndLoc, CollapsedNum, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setIterationVariable(Exprs.IterationVarRef);
  Dir->setLastIteration(Exprs.LastIteration);
  Dir->setCalcLastIteration(Exprs.CalcLastIteration);
  Dir->setPreCond(Exprs.PreCond);
  Dir->setCond(Exprs.Cond);
  Dir->setInit(Exprs.Init);
  Dir->setInc(Exprs.Inc);
  Dir->setIsLastIterVariable(Exprs.IL);
  Dir->setLowerBoundVariable(Exprs.LB);
  Dir->setUpperBoundVariable(Exprs.UB);
  Dir->setStrideVariable(Exprs.ST);
  Dir->setEnsureUpperBound(Exprs.EUB);
  Dir->setNextLowerBound(Exprs.NLB);
  Dir->setNextUpperBound(Exprs.NUB);
  Dir->setNumIterations(Exprs.NumIterations);
  Dir->setPrevLowerBoundVariable(Exprs.PrevLB);
  Dir->setPrevUpperBoundVariable(Exprs.PrevUB);
  Dir->setDistInc(Exprs.DistInc);
  Dir->setPrevEnsureUpperBound(Exprs.PrevEUB);
  Dir->setCounters(Exprs.Counters);
  Dir->setPrivateCounters(Exprs.PrivateCounters);
  Dir->setInits(Exprs.Inits);
  Dir->setUpdates(Exprs.Updates);
  Dir->setFinals(Exprs.Finals);
  Dir->setDependentCounters(Exprs.DependentCounters);
  Dir->setDependentInits(Exprs.DependentInits);
  Dir->setFinalsConditions(Exprs.FinalsConditions);
  Dir->setPreInits(Exprs.PreInits);
  Dir->setCombinedLowerBoundVariable(Exprs.DistCombinedFields.LB);
  Dir->setCombinedUpperBoundVariable(Exprs.DistCombinedFields.UB);
  Dir->setCombinedEnsureUpperBound(Exprs.DistCombinedFields.EUB);
  Dir->setCombinedInit(Exprs.DistCombinedFields.Init);
  Dir->setCombinedCond(Exprs.DistCombinedFields.Cond);
  Dir->setCombinedNextLowerBound(Exprs.DistCombinedFields.NLB);
  Dir->setCombinedNextUpperBound(Exprs.DistCombinedFields.NUB);
  Dir->setCombinedDistCond(Exprs.DistCombinedFields.DistCond);
  Dir->setCombinedParForInDistCond(Exprs.DistCombinedFields.ParForInDistCond);
  Dir->setTaskReductionRefExpr(TaskRedRef);
  Dir->HasCancel = HasCancel;
  return Dir;
}

OMPTargetTeamsDistributeParallelForDirective *
OMPTargetTeamsDistributeParallelForDirective::CreateEmpty(const ASTContext &C,
                                                          unsigned NumClauses,
                                                          unsigned CollapsedNum,
                                                          EmptyShell) {
  auto Size =
      llvm::alignTo(sizeof(OMPTargetTeamsDistributeParallelForDirective),
                    alignof(OMPClause *));
  void *Mem = C.Allocate(
      Size + sizeof(OMPClause *) * NumClauses +
      sizeof(Stmt *) *
           numLoopChildren(CollapsedNum,
                           OMPD_target_teams_distribute_parallel_for));
  return new (Mem)
      OMPTargetTeamsDistributeParallelForDirective(CollapsedNum, NumClauses);
}

OMPTargetTeamsDistributeParallelForSimdDirective *
OMPTargetTeamsDistributeParallelForSimdDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt,
    const HelperExprs &Exprs) {
  auto Size =
      llvm::alignTo(sizeof(OMPTargetTeamsDistributeParallelForSimdDirective),
                    alignof(OMPClause *));
  void *Mem = C.Allocate(
      Size + sizeof(OMPClause *) * Clauses.size() +
      sizeof(Stmt *) *
          numLoopChildren(CollapsedNum,
                          OMPD_target_teams_distribute_parallel_for_simd));
  OMPTargetTeamsDistributeParallelForSimdDirective *Dir =
      new (Mem) OMPTargetTeamsDistributeParallelForSimdDirective(
           StartLoc, EndLoc, CollapsedNum, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setIterationVariable(Exprs.IterationVarRef);
  Dir->setLastIteration(Exprs.LastIteration);
  Dir->setCalcLastIteration(Exprs.CalcLastIteration);
  Dir->setPreCond(Exprs.PreCond);
  Dir->setCond(Exprs.Cond);
  Dir->setInit(Exprs.Init);
  Dir->setInc(Exprs.Inc);
  Dir->setIsLastIterVariable(Exprs.IL);
  Dir->setLowerBoundVariable(Exprs.LB);
  Dir->setUpperBoundVariable(Exprs.UB);
  Dir->setStrideVariable(Exprs.ST);
  Dir->setEnsureUpperBound(Exprs.EUB);
  Dir->setNextLowerBound(Exprs.NLB);
  Dir->setNextUpperBound(Exprs.NUB);
  Dir->setNumIterations(Exprs.NumIterations);
  Dir->setPrevLowerBoundVariable(Exprs.PrevLB);
  Dir->setPrevUpperBoundVariable(Exprs.PrevUB);
  Dir->setDistInc(Exprs.DistInc);
  Dir->setPrevEnsureUpperBound(Exprs.PrevEUB);
  Dir->setCounters(Exprs.Counters);
  Dir->setPrivateCounters(Exprs.PrivateCounters);
  Dir->setInits(Exprs.Inits);
  Dir->setUpdates(Exprs.Updates);
  Dir->setFinals(Exprs.Finals);
  Dir->setDependentCounters(Exprs.DependentCounters);
  Dir->setDependentInits(Exprs.DependentInits);
  Dir->setFinalsConditions(Exprs.FinalsConditions);
  Dir->setPreInits(Exprs.PreInits);
  Dir->setCombinedLowerBoundVariable(Exprs.DistCombinedFields.LB);
  Dir->setCombinedUpperBoundVariable(Exprs.DistCombinedFields.UB);
  Dir->setCombinedEnsureUpperBound(Exprs.DistCombinedFields.EUB);
  Dir->setCombinedInit(Exprs.DistCombinedFields.Init);
  Dir->setCombinedCond(Exprs.DistCombinedFields.Cond);
  Dir->setCombinedNextLowerBound(Exprs.DistCombinedFields.NLB);
  Dir->setCombinedNextUpperBound(Exprs.DistCombinedFields.NUB);
  Dir->setCombinedDistCond(Exprs.DistCombinedFields.DistCond);
  Dir->setCombinedParForInDistCond(Exprs.DistCombinedFields.ParForInDistCond);
  return Dir;
}

OMPTargetTeamsDistributeParallelForSimdDirective *
OMPTargetTeamsDistributeParallelForSimdDirective::CreateEmpty(
    const ASTContext &C, unsigned NumClauses, unsigned CollapsedNum,
    EmptyShell) {
  auto Size =
      llvm::alignTo(sizeof(OMPTargetTeamsDistributeParallelForSimdDirective),
                    alignof(OMPClause *));
  void *Mem = C.Allocate(
      Size + sizeof(OMPClause *) * NumClauses +
      sizeof(Stmt *) *
          numLoopChildren(CollapsedNum,
                          OMPD_target_teams_distribute_parallel_for_simd));
  return new (Mem) OMPTargetTeamsDistributeParallelForSimdDirective(
      CollapsedNum, NumClauses);
}

OMPTargetTeamsDistributeSimdDirective *
OMPTargetTeamsDistributeSimdDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt,
    const HelperExprs &Exprs) {
  auto Size = llvm::alignTo(sizeof(OMPTargetTeamsDistributeSimdDirective),
                            alignof(OMPClause *));
  void *Mem = C.Allocate(
      Size + sizeof(OMPClause *) * Clauses.size() +
      sizeof(Stmt *) *
          numLoopChildren(CollapsedNum, OMPD_target_teams_distribute_simd));
  OMPTargetTeamsDistributeSimdDirective *Dir = new (Mem)
      OMPTargetTeamsDistributeSimdDirective(StartLoc, EndLoc, CollapsedNum,
                                            Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setIterationVariable(Exprs.IterationVarRef);
  Dir->setLastIteration(Exprs.LastIteration);
  Dir->setCalcLastIteration(Exprs.CalcLastIteration);
  Dir->setPreCond(Exprs.PreCond);
  Dir->setCond(Exprs.Cond);
  Dir->setInit(Exprs.Init);
  Dir->setInc(Exprs.Inc);
  Dir->setIsLastIterVariable(Exprs.IL);
  Dir->setLowerBoundVariable(Exprs.LB);
  Dir->setUpperBoundVariable(Exprs.UB);
  Dir->setStrideVariable(Exprs.ST);
  Dir->setEnsureUpperBound(Exprs.EUB);
  Dir->setNextLowerBound(Exprs.NLB);
  Dir->setNextUpperBound(Exprs.NUB);
  Dir->setNumIterations(Exprs.NumIterations);
  Dir->setCounters(Exprs.Counters);
  Dir->setPrivateCounters(Exprs.PrivateCounters);
  Dir->setInits(Exprs.Inits);
  Dir->setUpdates(Exprs.Updates);
  Dir->setFinals(Exprs.Finals);
  Dir->setDependentCounters(Exprs.DependentCounters);
  Dir->setDependentInits(Exprs.DependentInits);
  Dir->setFinalsConditions(Exprs.FinalsConditions);
  Dir->setPreInits(Exprs.PreInits);
  return Dir;
}

OMPTargetTeamsDistributeSimdDirective *
OMPTargetTeamsDistributeSimdDirective::CreateEmpty(const ASTContext &C,
                                                   unsigned NumClauses,
                                                   unsigned CollapsedNum,
                                                   EmptyShell) {
  auto Size = llvm::alignTo(sizeof(OMPTargetTeamsDistributeSimdDirective),
                            alignof(OMPClause *));
  void *Mem = C.Allocate(
      Size + sizeof(OMPClause *) * NumClauses +
      sizeof(Stmt *) *
          numLoopChildren(CollapsedNum, OMPD_target_teams_distribute_simd));
  return new (Mem)
      OMPTargetTeamsDistributeSimdDirective(CollapsedNum, NumClauses);
}
