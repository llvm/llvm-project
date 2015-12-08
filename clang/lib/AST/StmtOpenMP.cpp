//===--- StmtOpenMP.cpp - Classes for OpenMP directives -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the subclesses of Stmt class declared in StmtOpenMP.h
//
//===----------------------------------------------------------------------===//

#include "clang/AST/StmtOpenMP.h"

#include "clang/AST/ASTContext.h"

using namespace clang;

void OMPExecutableDirective::setClauses(ArrayRef<OMPClause *> Clauses) {
  assert(Clauses.size() == getNumClauses() &&
         "Number of clauses is not the same as the preallocated buffer");
  std::copy(Clauses.begin(), Clauses.end(), getClauses().begin());
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

OMPParallelDirective *OMPParallelDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt, bool HasCancel) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPParallelDirective),
                                           llvm::alignOf<OMPClause *>());
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * Clauses.size() + sizeof(Stmt *));
  OMPParallelDirective *Dir =
      new (Mem) OMPParallelDirective(StartLoc, EndLoc, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setHasCancel(HasCancel);
  return Dir;
}

OMPParallelDirective *OMPParallelDirective::CreateEmpty(const ASTContext &C,
                                                        unsigned NumClauses,
                                                        EmptyShell) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPParallelDirective),
                                           llvm::alignOf<OMPClause *>());
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * NumClauses + sizeof(Stmt *));
  return new (Mem) OMPParallelDirective(NumClauses);
}

OMPSimdDirective *
OMPSimdDirective::Create(const ASTContext &C, SourceLocation StartLoc,
                         SourceLocation EndLoc, unsigned CollapsedNum,
                         ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt,
                         const HelperExprs &Exprs) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPSimdDirective),
                                           llvm::alignOf<OMPClause *>());
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
  return Dir;
}

OMPSimdDirective *OMPSimdDirective::CreateEmpty(const ASTContext &C,
                                                unsigned NumClauses,
                                                unsigned CollapsedNum,
                                                EmptyShell) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPSimdDirective),
                                           llvm::alignOf<OMPClause *>());
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * NumClauses +
                 sizeof(Stmt *) * numLoopChildren(CollapsedNum, OMPD_simd));
  return new (Mem) OMPSimdDirective(CollapsedNum, NumClauses);
}

OMPForDirective *
OMPForDirective::Create(const ASTContext &C, SourceLocation StartLoc,
                        SourceLocation EndLoc, unsigned CollapsedNum,
                        ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt,
                        const HelperExprs &Exprs, bool HasCancel) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPForDirective),
                                           llvm::alignOf<OMPClause *>());
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
  Dir->setCounters(Exprs.Counters);
  Dir->setPrivateCounters(Exprs.PrivateCounters);
  Dir->setInits(Exprs.Inits);
  Dir->setUpdates(Exprs.Updates);
  Dir->setFinals(Exprs.Finals);
  Dir->setHasCancel(HasCancel);
  return Dir;
}

OMPForDirective *OMPForDirective::CreateEmpty(const ASTContext &C,
                                              unsigned NumClauses,
                                              unsigned CollapsedNum,
                                              EmptyShell) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPForDirective),
                                           llvm::alignOf<OMPClause *>());
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
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPForSimdDirective),
                                           llvm::alignOf<OMPClause *>());
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
  Dir->setCounters(Exprs.Counters);
  Dir->setPrivateCounters(Exprs.PrivateCounters);
  Dir->setInits(Exprs.Inits);
  Dir->setUpdates(Exprs.Updates);
  Dir->setFinals(Exprs.Finals);
  return Dir;
}

OMPForSimdDirective *OMPForSimdDirective::CreateEmpty(const ASTContext &C,
                                                      unsigned NumClauses,
                                                      unsigned CollapsedNum,
                                                      EmptyShell) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPForSimdDirective),
                                           llvm::alignOf<OMPClause *>());
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * NumClauses +
                 sizeof(Stmt *) * numLoopChildren(CollapsedNum, OMPD_for_simd));
  return new (Mem) OMPForSimdDirective(CollapsedNum, NumClauses);
}

OMPSectionsDirective *OMPSectionsDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt, bool HasCancel) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPSectionsDirective),
                                           llvm::alignOf<OMPClause *>());
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * Clauses.size() + sizeof(Stmt *));
  OMPSectionsDirective *Dir =
      new (Mem) OMPSectionsDirective(StartLoc, EndLoc, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setHasCancel(HasCancel);
  return Dir;
}

OMPSectionsDirective *OMPSectionsDirective::CreateEmpty(const ASTContext &C,
                                                        unsigned NumClauses,
                                                        EmptyShell) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPSectionsDirective),
                                           llvm::alignOf<OMPClause *>());
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * NumClauses + sizeof(Stmt *));
  return new (Mem) OMPSectionsDirective(NumClauses);
}

OMPSectionDirective *OMPSectionDirective::Create(const ASTContext &C,
                                                 SourceLocation StartLoc,
                                                 SourceLocation EndLoc,
                                                 Stmt *AssociatedStmt,
                                                 bool HasCancel) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPSectionDirective),
                                           llvm::alignOf<Stmt *>());
  void *Mem = C.Allocate(Size + sizeof(Stmt *));
  OMPSectionDirective *Dir = new (Mem) OMPSectionDirective(StartLoc, EndLoc);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setHasCancel(HasCancel);
  return Dir;
}

OMPSectionDirective *OMPSectionDirective::CreateEmpty(const ASTContext &C,
                                                      EmptyShell) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPSectionDirective),
                                           llvm::alignOf<Stmt *>());
  void *Mem = C.Allocate(Size + sizeof(Stmt *));
  return new (Mem) OMPSectionDirective();
}

OMPSingleDirective *OMPSingleDirective::Create(const ASTContext &C,
                                               SourceLocation StartLoc,
                                               SourceLocation EndLoc,
                                               ArrayRef<OMPClause *> Clauses,
                                               Stmt *AssociatedStmt) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPSingleDirective),
                                           llvm::alignOf<OMPClause *>());
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
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPSingleDirective),
                                           llvm::alignOf<OMPClause *>());
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * NumClauses + sizeof(Stmt *));
  return new (Mem) OMPSingleDirective(NumClauses);
}

OMPMasterDirective *OMPMasterDirective::Create(const ASTContext &C,
                                               SourceLocation StartLoc,
                                               SourceLocation EndLoc,
                                               Stmt *AssociatedStmt) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPMasterDirective),
                                           llvm::alignOf<Stmt *>());
  void *Mem = C.Allocate(Size + sizeof(Stmt *));
  OMPMasterDirective *Dir = new (Mem) OMPMasterDirective(StartLoc, EndLoc);
  Dir->setAssociatedStmt(AssociatedStmt);
  return Dir;
}

OMPMasterDirective *OMPMasterDirective::CreateEmpty(const ASTContext &C,
                                                    EmptyShell) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPMasterDirective),
                                           llvm::alignOf<Stmt *>());
  void *Mem = C.Allocate(Size + sizeof(Stmt *));
  return new (Mem) OMPMasterDirective();
}

OMPCriticalDirective *OMPCriticalDirective::Create(
    const ASTContext &C, const DeclarationNameInfo &Name,
    SourceLocation StartLoc, SourceLocation EndLoc, Stmt *AssociatedStmt) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPCriticalDirective),
                                           llvm::alignOf<Stmt *>());
  void *Mem = C.Allocate(Size + sizeof(Stmt *));
  OMPCriticalDirective *Dir =
      new (Mem) OMPCriticalDirective(Name, StartLoc, EndLoc);
  Dir->setAssociatedStmt(AssociatedStmt);
  return Dir;
}

OMPCriticalDirective *OMPCriticalDirective::CreateEmpty(const ASTContext &C,
                                                        EmptyShell) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPCriticalDirective),
                                           llvm::alignOf<Stmt *>());
  void *Mem = C.Allocate(Size + sizeof(Stmt *));
  return new (Mem) OMPCriticalDirective();
}

OMPParallelForDirective *OMPParallelForDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt,
    const HelperExprs &Exprs, bool HasCancel) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPParallelForDirective),
                                           llvm::alignOf<OMPClause *>());
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
  Dir->setCounters(Exprs.Counters);
  Dir->setPrivateCounters(Exprs.PrivateCounters);
  Dir->setInits(Exprs.Inits);
  Dir->setUpdates(Exprs.Updates);
  Dir->setFinals(Exprs.Finals);
  Dir->setHasCancel(HasCancel);
  return Dir;
}

OMPParallelForDirective *
OMPParallelForDirective::CreateEmpty(const ASTContext &C, unsigned NumClauses,
                                     unsigned CollapsedNum, EmptyShell) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPParallelForDirective),
                                           llvm::alignOf<OMPClause *>());
  void *Mem = C.Allocate(Size + sizeof(OMPClause *) * NumClauses +
                         sizeof(Stmt *) *
                             numLoopChildren(CollapsedNum, OMPD_parallel_for));
  return new (Mem) OMPParallelForDirective(CollapsedNum, NumClauses);
}

OMPParallelForSimdDirective *OMPParallelForSimdDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt,
    const HelperExprs &Exprs) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPParallelForSimdDirective),
                                           llvm::alignOf<OMPClause *>());
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
  Dir->setCounters(Exprs.Counters);
  Dir->setPrivateCounters(Exprs.PrivateCounters);
  Dir->setInits(Exprs.Inits);
  Dir->setUpdates(Exprs.Updates);
  Dir->setFinals(Exprs.Finals);
  return Dir;
}

OMPParallelForSimdDirective *
OMPParallelForSimdDirective::CreateEmpty(const ASTContext &C,
                                         unsigned NumClauses,
                                         unsigned CollapsedNum, EmptyShell) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPParallelForSimdDirective),
                                           llvm::alignOf<OMPClause *>());
  void *Mem = C.Allocate(
      Size + sizeof(OMPClause *) * NumClauses +
      sizeof(Stmt *) * numLoopChildren(CollapsedNum, OMPD_parallel_for_simd));
  return new (Mem) OMPParallelForSimdDirective(CollapsedNum, NumClauses);
}

OMPParallelSectionsDirective *OMPParallelSectionsDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt, bool HasCancel) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPParallelSectionsDirective),
                                           llvm::alignOf<OMPClause *>());
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * Clauses.size() + sizeof(Stmt *));
  OMPParallelSectionsDirective *Dir =
      new (Mem) OMPParallelSectionsDirective(StartLoc, EndLoc, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setHasCancel(HasCancel);
  return Dir;
}

OMPParallelSectionsDirective *
OMPParallelSectionsDirective::CreateEmpty(const ASTContext &C,
                                          unsigned NumClauses, EmptyShell) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPParallelSectionsDirective),
                                           llvm::alignOf<OMPClause *>());
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * NumClauses + sizeof(Stmt *));
  return new (Mem) OMPParallelSectionsDirective(NumClauses);
}

OMPTaskDirective *
OMPTaskDirective::Create(const ASTContext &C, SourceLocation StartLoc,
                         SourceLocation EndLoc, ArrayRef<OMPClause *> Clauses,
                         Stmt *AssociatedStmt, bool HasCancel) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPTaskDirective),
                                           llvm::alignOf<OMPClause *>());
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
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPTaskDirective),
                                           llvm::alignOf<OMPClause *>());
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

OMPTaskgroupDirective *OMPTaskgroupDirective::Create(const ASTContext &C,
                                                     SourceLocation StartLoc,
                                                     SourceLocation EndLoc,
                                                     Stmt *AssociatedStmt) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPTaskgroupDirective),
                                           llvm::alignOf<Stmt *>());
  void *Mem = C.Allocate(Size + sizeof(Stmt *));
  OMPTaskgroupDirective *Dir =
      new (Mem) OMPTaskgroupDirective(StartLoc, EndLoc);
  Dir->setAssociatedStmt(AssociatedStmt);
  return Dir;
}

OMPTaskgroupDirective *OMPTaskgroupDirective::CreateEmpty(const ASTContext &C,
                                                          EmptyShell) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPTaskgroupDirective),
                                           llvm::alignOf<Stmt *>());
  void *Mem = C.Allocate(Size + sizeof(Stmt *));
  return new (Mem) OMPTaskgroupDirective();
}

OMPCancellationPointDirective *OMPCancellationPointDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    OpenMPDirectiveKind CancelRegion) {
  unsigned Size = llvm::RoundUpToAlignment(
      sizeof(OMPCancellationPointDirective), llvm::alignOf<Stmt *>());
  void *Mem = C.Allocate(Size);
  OMPCancellationPointDirective *Dir =
      new (Mem) OMPCancellationPointDirective(StartLoc, EndLoc);
  Dir->setCancelRegion(CancelRegion);
  return Dir;
}

OMPCancellationPointDirective *
OMPCancellationPointDirective::CreateEmpty(const ASTContext &C, EmptyShell) {
  unsigned Size = llvm::RoundUpToAlignment(
      sizeof(OMPCancellationPointDirective), llvm::alignOf<Stmt *>());
  void *Mem = C.Allocate(Size);
  return new (Mem) OMPCancellationPointDirective();
}

OMPCancelDirective *
OMPCancelDirective::Create(const ASTContext &C, SourceLocation StartLoc,
                           SourceLocation EndLoc, ArrayRef<OMPClause *> Clauses,
                           OpenMPDirectiveKind CancelRegion) {
  unsigned Size = llvm::RoundUpToAlignment(
      sizeof(OMPCancelDirective) + sizeof(OMPClause *) * Clauses.size(),
      llvm::alignOf<Stmt *>());
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
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPCancelDirective) +
                                               sizeof(OMPClause *) * NumClauses,
                                           llvm::alignOf<Stmt *>());
  void *Mem = C.Allocate(Size);
  return new (Mem) OMPCancelDirective(NumClauses);
}

OMPFlushDirective *OMPFlushDirective::Create(const ASTContext &C,
                                             SourceLocation StartLoc,
                                             SourceLocation EndLoc,
                                             ArrayRef<OMPClause *> Clauses) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPFlushDirective),
                                           llvm::alignOf<OMPClause *>());
  void *Mem = C.Allocate(Size + sizeof(OMPClause *) * Clauses.size());
  OMPFlushDirective *Dir =
      new (Mem) OMPFlushDirective(StartLoc, EndLoc, Clauses.size());
  Dir->setClauses(Clauses);
  return Dir;
}

OMPFlushDirective *OMPFlushDirective::CreateEmpty(const ASTContext &C,
                                                  unsigned NumClauses,
                                                  EmptyShell) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPFlushDirective),
                                           llvm::alignOf<OMPClause *>());
  void *Mem = C.Allocate(Size + sizeof(OMPClause *) * NumClauses);
  return new (Mem) OMPFlushDirective(NumClauses);
}

OMPOrderedDirective *OMPOrderedDirective::Create(const ASTContext &C,
                                                 SourceLocation StartLoc,
                                                 SourceLocation EndLoc,
                                                 ArrayRef<OMPClause *> Clauses,
                                                 Stmt *AssociatedStmt) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPOrderedDirective),
                                           llvm::alignOf<OMPClause *>());
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
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPOrderedDirective),
                                           llvm::alignOf<OMPClause *>());
  void *Mem =
      C.Allocate(Size + sizeof(Stmt *) + sizeof(OMPClause *) * NumClauses);
  return new (Mem) OMPOrderedDirective(NumClauses);
}

OMPAtomicDirective *OMPAtomicDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt, Expr *X, Expr *V,
    Expr *E, Expr *UE, bool IsXLHSInRHSPart, bool IsPostfixUpdate) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPAtomicDirective),
                                           llvm::alignOf<OMPClause *>());
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
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPAtomicDirective),
                                           llvm::alignOf<OMPClause *>());
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * NumClauses + 5 * sizeof(Stmt *));
  return new (Mem) OMPAtomicDirective(NumClauses);
}

OMPTargetDirective *OMPTargetDirective::Create(const ASTContext &C,
                                               SourceLocation StartLoc,
                                               SourceLocation EndLoc,
                                               ArrayRef<OMPClause *> Clauses,
                                               Stmt *AssociatedStmt) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPTargetDirective),
                                           llvm::alignOf<OMPClause *>());
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
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPTargetDirective),
                                           llvm::alignOf<OMPClause *>());
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * NumClauses + sizeof(Stmt *));
  return new (Mem) OMPTargetDirective(NumClauses);
}

OMPTargetDataDirective *OMPTargetDataDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt) {
  void *Mem =
      C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPTargetDataDirective),
                                          llvm::alignOf<OMPClause *>()) +
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
  void *Mem =
      C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPTargetDataDirective),
                                          llvm::alignOf<OMPClause *>()) +
                 sizeof(OMPClause *) * N + sizeof(Stmt *));
  return new (Mem) OMPTargetDataDirective(N);
}

OMPTeamsDirective *OMPTeamsDirective::Create(const ASTContext &C,
                                             SourceLocation StartLoc,
                                             SourceLocation EndLoc,
                                             ArrayRef<OMPClause *> Clauses,
                                             Stmt *AssociatedStmt) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPTeamsDirective),
                                           llvm::alignOf<OMPClause *>());
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
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPTeamsDirective),
                                           llvm::alignOf<OMPClause *>());
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * NumClauses + sizeof(Stmt *));
  return new (Mem) OMPTeamsDirective(NumClauses);
}

OMPTaskLoopDirective *OMPTaskLoopDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt,
    const HelperExprs &Exprs) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPTaskLoopDirective),
                                           llvm::alignOf<OMPClause *>());
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
  Dir->setCounters(Exprs.Counters);
  Dir->setPrivateCounters(Exprs.PrivateCounters);
  Dir->setInits(Exprs.Inits);
  Dir->setUpdates(Exprs.Updates);
  Dir->setFinals(Exprs.Finals);
  return Dir;
}

OMPTaskLoopDirective *OMPTaskLoopDirective::CreateEmpty(const ASTContext &C,
                                                        unsigned NumClauses,
                                                        unsigned CollapsedNum,
                                                        EmptyShell) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPTaskLoopDirective),
                                           llvm::alignOf<OMPClause *>());
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * NumClauses +
                 sizeof(Stmt *) * numLoopChildren(CollapsedNum, OMPD_taskloop));
  return new (Mem) OMPTaskLoopDirective(CollapsedNum, NumClauses);
}

OMPTaskLoopSimdDirective *OMPTaskLoopSimdDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt,
    const HelperExprs &Exprs) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPTaskLoopSimdDirective),
                                           llvm::alignOf<OMPClause *>());
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
  Dir->setCounters(Exprs.Counters);
  Dir->setPrivateCounters(Exprs.PrivateCounters);
  Dir->setInits(Exprs.Inits);
  Dir->setUpdates(Exprs.Updates);
  Dir->setFinals(Exprs.Finals);
  return Dir;
}

OMPTaskLoopSimdDirective *
OMPTaskLoopSimdDirective::CreateEmpty(const ASTContext &C, unsigned NumClauses,
                                      unsigned CollapsedNum, EmptyShell) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPTaskLoopSimdDirective),
                                           llvm::alignOf<OMPClause *>());
  void *Mem = C.Allocate(Size + sizeof(OMPClause *) * NumClauses +
                         sizeof(Stmt *) *
                             numLoopChildren(CollapsedNum, OMPD_taskloop_simd));
  return new (Mem) OMPTaskLoopSimdDirective(CollapsedNum, NumClauses);
}

OMPDistributeDirective *OMPDistributeDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt,
    const HelperExprs &Exprs) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPDistributeDirective),
                                           llvm::alignOf<OMPClause *>());
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
  Dir->setCounters(Exprs.Counters);
  Dir->setPrivateCounters(Exprs.PrivateCounters);
  Dir->setInits(Exprs.Inits);
  Dir->setUpdates(Exprs.Updates);
  Dir->setFinals(Exprs.Finals);
  return Dir;
}

OMPDistributeDirective *
OMPDistributeDirective::CreateEmpty(const ASTContext &C, unsigned NumClauses,
                                    unsigned CollapsedNum, EmptyShell) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPDistributeDirective),
                                           llvm::alignOf<OMPClause *>());
  void *Mem = C.Allocate(Size + sizeof(OMPClause *) * NumClauses +
                         sizeof(Stmt *) *
                             numLoopChildren(CollapsedNum, OMPD_distribute));
  return new (Mem) OMPDistributeDirective(CollapsedNum, NumClauses);
}
