//===- OpenMPClause.cpp - Classes for OpenMP clauses ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the subclesses of Stmt class declared in OpenMPClause.h
//
//===----------------------------------------------------------------------===//

#include "clang/AST/OpenMPClause.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
#include <cassert>

using namespace clang;

OMPClause::child_range OMPClause::children() {
  switch (getClauseKind()) {
  default:
    break;
#define OPENMP_CLAUSE(Name, Class)                                             \
  case OMPC_##Name:                                                            \
    return static_cast<Class *>(this)->children();
#include "clang/Basic/OpenMPKinds.def"
  }
  llvm_unreachable("unknown OMPClause");
}

OMPClauseWithPreInit *OMPClauseWithPreInit::get(OMPClause *C) {
  auto *Res = OMPClauseWithPreInit::get(const_cast<const OMPClause *>(C));
  return Res ? const_cast<OMPClauseWithPreInit *>(Res) : nullptr;
}

const OMPClauseWithPreInit *OMPClauseWithPreInit::get(const OMPClause *C) {
  switch (C->getClauseKind()) {
  case OMPC_schedule:
    return static_cast<const OMPScheduleClause *>(C);
  case OMPC_dist_schedule:
    return static_cast<const OMPDistScheduleClause *>(C);
  case OMPC_firstprivate:
    return static_cast<const OMPFirstprivateClause *>(C);
  case OMPC_lastprivate:
    return static_cast<const OMPLastprivateClause *>(C);
  case OMPC_reduction:
    return static_cast<const OMPReductionClause *>(C);
  case OMPC_task_reduction:
    return static_cast<const OMPTaskReductionClause *>(C);
  case OMPC_in_reduction:
    return static_cast<const OMPInReductionClause *>(C);
  case OMPC_linear:
    return static_cast<const OMPLinearClause *>(C);
  case OMPC_if:
    return static_cast<const OMPIfClause *>(C);
  case OMPC_num_threads:
    return static_cast<const OMPNumThreadsClause *>(C);
  case OMPC_num_teams:
    return static_cast<const OMPNumTeamsClause *>(C);
  case OMPC_thread_limit:
    return static_cast<const OMPThreadLimitClause *>(C);
  case OMPC_device:
    return static_cast<const OMPDeviceClause *>(C);
  case OMPC_default:
  case OMPC_proc_bind:
  case OMPC_final:
  case OMPC_safelen:
  case OMPC_simdlen:
  case OMPC_collapse:
  case OMPC_private:
  case OMPC_shared:
  case OMPC_aligned:
  case OMPC_copyin:
  case OMPC_copyprivate:
  case OMPC_ordered:
  case OMPC_nowait:
  case OMPC_untied:
  case OMPC_mergeable:
  case OMPC_threadprivate:
  case OMPC_flush:
  case OMPC_read:
  case OMPC_write:
  case OMPC_update:
  case OMPC_capture:
  case OMPC_seq_cst:
  case OMPC_depend:
  case OMPC_threads:
  case OMPC_simd:
  case OMPC_map:
  case OMPC_priority:
  case OMPC_grainsize:
  case OMPC_nogroup:
  case OMPC_num_tasks:
  case OMPC_hint:
  case OMPC_defaultmap:
  case OMPC_unknown:
  case OMPC_uniform:
  case OMPC_to:
  case OMPC_from:
  case OMPC_use_device_ptr:
  case OMPC_is_device_ptr:
    break;
  }

  return nullptr;
}

OMPClauseWithPostUpdate *OMPClauseWithPostUpdate::get(OMPClause *C) {
  auto *Res = OMPClauseWithPostUpdate::get(const_cast<const OMPClause *>(C));
  return Res ? const_cast<OMPClauseWithPostUpdate *>(Res) : nullptr;
}

const OMPClauseWithPostUpdate *OMPClauseWithPostUpdate::get(const OMPClause *C) {
  switch (C->getClauseKind()) {
  case OMPC_lastprivate:
    return static_cast<const OMPLastprivateClause *>(C);
  case OMPC_reduction:
    return static_cast<const OMPReductionClause *>(C);
  case OMPC_task_reduction:
    return static_cast<const OMPTaskReductionClause *>(C);
  case OMPC_in_reduction:
    return static_cast<const OMPInReductionClause *>(C);
  case OMPC_linear:
    return static_cast<const OMPLinearClause *>(C);
  case OMPC_schedule:
  case OMPC_dist_schedule:
  case OMPC_firstprivate:
  case OMPC_default:
  case OMPC_proc_bind:
  case OMPC_if:
  case OMPC_final:
  case OMPC_num_threads:
  case OMPC_safelen:
  case OMPC_simdlen:
  case OMPC_collapse:
  case OMPC_private:
  case OMPC_shared:
  case OMPC_aligned:
  case OMPC_copyin:
  case OMPC_copyprivate:
  case OMPC_ordered:
  case OMPC_nowait:
  case OMPC_untied:
  case OMPC_mergeable:
  case OMPC_threadprivate:
  case OMPC_flush:
  case OMPC_read:
  case OMPC_write:
  case OMPC_update:
  case OMPC_capture:
  case OMPC_seq_cst:
  case OMPC_depend:
  case OMPC_device:
  case OMPC_threads:
  case OMPC_simd:
  case OMPC_map:
  case OMPC_num_teams:
  case OMPC_thread_limit:
  case OMPC_priority:
  case OMPC_grainsize:
  case OMPC_nogroup:
  case OMPC_num_tasks:
  case OMPC_hint:
  case OMPC_defaultmap:
  case OMPC_unknown:
  case OMPC_uniform:
  case OMPC_to:
  case OMPC_from:
  case OMPC_use_device_ptr:
  case OMPC_is_device_ptr:
    break;
  }

  return nullptr;
}

OMPOrderedClause *OMPOrderedClause::Create(const ASTContext &C, Expr *Num,
                                           unsigned NumLoops,
                                           SourceLocation StartLoc,
                                           SourceLocation LParenLoc,
                                           SourceLocation EndLoc) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(2 * NumLoops));
  auto *Clause =
      new (Mem) OMPOrderedClause(Num, NumLoops, StartLoc, LParenLoc, EndLoc);
  for (unsigned I = 0; I < NumLoops; ++I) {
    Clause->setLoopNumIterations(I, nullptr);
    Clause->setLoopCounter(I, nullptr);
  }
  return Clause;
}

OMPOrderedClause *OMPOrderedClause::CreateEmpty(const ASTContext &C,
                                                unsigned NumLoops) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(2 * NumLoops));
  auto *Clause = new (Mem) OMPOrderedClause(NumLoops);
  for (unsigned I = 0; I < NumLoops; ++I) {
    Clause->setLoopNumIterations(I, nullptr);
    Clause->setLoopCounter(I, nullptr);
  }
  return Clause;
}

void OMPOrderedClause::setLoopNumIterations(unsigned NumLoop,
                                            Expr *NumIterations) {
  assert(NumLoop < NumberOfLoops && "out of loops number.");
  getTrailingObjects<Expr *>()[NumLoop] = NumIterations;
}

ArrayRef<Expr *> OMPOrderedClause::getLoopNumIterations() const {
  return llvm::makeArrayRef(getTrailingObjects<Expr *>(), NumberOfLoops);
}

void OMPOrderedClause::setLoopCounter(unsigned NumLoop, Expr *Counter) {
  assert(NumLoop < NumberOfLoops && "out of loops number.");
  getTrailingObjects<Expr *>()[NumberOfLoops + NumLoop] = Counter;
}

Expr *OMPOrderedClause::getLoopCunter(unsigned NumLoop) {
  assert(NumLoop < NumberOfLoops && "out of loops number.");
  return getTrailingObjects<Expr *>()[NumberOfLoops + NumLoop];
}

const Expr *OMPOrderedClause::getLoopCunter(unsigned NumLoop) const {
  assert(NumLoop < NumberOfLoops && "out of loops number.");
  return getTrailingObjects<Expr *>()[NumberOfLoops + NumLoop];
}

void OMPPrivateClause::setPrivateCopies(ArrayRef<Expr *> VL) {
  assert(VL.size() == varlist_size() &&
         "Number of private copies is not the same as the preallocated buffer");
  std::copy(VL.begin(), VL.end(), varlist_end());
}

OMPPrivateClause *
OMPPrivateClause::Create(const ASTContext &C, SourceLocation StartLoc,
                         SourceLocation LParenLoc, SourceLocation EndLoc,
                         ArrayRef<Expr *> VL, ArrayRef<Expr *> PrivateVL) {
  // Allocate space for private variables and initializer expressions.
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(2 * VL.size()));
  OMPPrivateClause *Clause =
      new (Mem) OMPPrivateClause(StartLoc, LParenLoc, EndLoc, VL.size());
  Clause->setVarRefs(VL);
  Clause->setPrivateCopies(PrivateVL);
  return Clause;
}

OMPPrivateClause *OMPPrivateClause::CreateEmpty(const ASTContext &C,
                                                unsigned N) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(2 * N));
  return new (Mem) OMPPrivateClause(N);
}

void OMPFirstprivateClause::setPrivateCopies(ArrayRef<Expr *> VL) {
  assert(VL.size() == varlist_size() &&
         "Number of private copies is not the same as the preallocated buffer");
  std::copy(VL.begin(), VL.end(), varlist_end());
}

void OMPFirstprivateClause::setInits(ArrayRef<Expr *> VL) {
  assert(VL.size() == varlist_size() &&
         "Number of inits is not the same as the preallocated buffer");
  std::copy(VL.begin(), VL.end(), getPrivateCopies().end());
}

OMPFirstprivateClause *
OMPFirstprivateClause::Create(const ASTContext &C, SourceLocation StartLoc,
                              SourceLocation LParenLoc, SourceLocation EndLoc,
                              ArrayRef<Expr *> VL, ArrayRef<Expr *> PrivateVL,
                              ArrayRef<Expr *> InitVL, Stmt *PreInit) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(3 * VL.size()));
  OMPFirstprivateClause *Clause =
      new (Mem) OMPFirstprivateClause(StartLoc, LParenLoc, EndLoc, VL.size());
  Clause->setVarRefs(VL);
  Clause->setPrivateCopies(PrivateVL);
  Clause->setInits(InitVL);
  Clause->setPreInitStmt(PreInit);
  return Clause;
}

OMPFirstprivateClause *OMPFirstprivateClause::CreateEmpty(const ASTContext &C,
                                                          unsigned N) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(3 * N));
  return new (Mem) OMPFirstprivateClause(N);
}

void OMPLastprivateClause::setPrivateCopies(ArrayRef<Expr *> PrivateCopies) {
  assert(PrivateCopies.size() == varlist_size() &&
         "Number of private copies is not the same as the preallocated buffer");
  std::copy(PrivateCopies.begin(), PrivateCopies.end(), varlist_end());
}

void OMPLastprivateClause::setSourceExprs(ArrayRef<Expr *> SrcExprs) {
  assert(SrcExprs.size() == varlist_size() && "Number of source expressions is "
                                              "not the same as the "
                                              "preallocated buffer");
  std::copy(SrcExprs.begin(), SrcExprs.end(), getPrivateCopies().end());
}

void OMPLastprivateClause::setDestinationExprs(ArrayRef<Expr *> DstExprs) {
  assert(DstExprs.size() == varlist_size() && "Number of destination "
                                              "expressions is not the same as "
                                              "the preallocated buffer");
  std::copy(DstExprs.begin(), DstExprs.end(), getSourceExprs().end());
}

void OMPLastprivateClause::setAssignmentOps(ArrayRef<Expr *> AssignmentOps) {
  assert(AssignmentOps.size() == varlist_size() &&
         "Number of assignment expressions is not the same as the preallocated "
         "buffer");
  std::copy(AssignmentOps.begin(), AssignmentOps.end(),
            getDestinationExprs().end());
}

OMPLastprivateClause *OMPLastprivateClause::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation LParenLoc,
    SourceLocation EndLoc, ArrayRef<Expr *> VL, ArrayRef<Expr *> SrcExprs,
    ArrayRef<Expr *> DstExprs, ArrayRef<Expr *> AssignmentOps, Stmt *PreInit,
    Expr *PostUpdate) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(5 * VL.size()));
  OMPLastprivateClause *Clause =
      new (Mem) OMPLastprivateClause(StartLoc, LParenLoc, EndLoc, VL.size());
  Clause->setVarRefs(VL);
  Clause->setSourceExprs(SrcExprs);
  Clause->setDestinationExprs(DstExprs);
  Clause->setAssignmentOps(AssignmentOps);
  Clause->setPreInitStmt(PreInit);
  Clause->setPostUpdateExpr(PostUpdate);
  return Clause;
}

OMPLastprivateClause *OMPLastprivateClause::CreateEmpty(const ASTContext &C,
                                                        unsigned N) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(5 * N));
  return new (Mem) OMPLastprivateClause(N);
}

OMPSharedClause *OMPSharedClause::Create(const ASTContext &C,
                                         SourceLocation StartLoc,
                                         SourceLocation LParenLoc,
                                         SourceLocation EndLoc,
                                         ArrayRef<Expr *> VL) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(VL.size()));
  OMPSharedClause *Clause =
      new (Mem) OMPSharedClause(StartLoc, LParenLoc, EndLoc, VL.size());
  Clause->setVarRefs(VL);
  return Clause;
}

OMPSharedClause *OMPSharedClause::CreateEmpty(const ASTContext &C, unsigned N) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(N));
  return new (Mem) OMPSharedClause(N);
}

void OMPLinearClause::setPrivates(ArrayRef<Expr *> PL) {
  assert(PL.size() == varlist_size() &&
         "Number of privates is not the same as the preallocated buffer");
  std::copy(PL.begin(), PL.end(), varlist_end());
}

void OMPLinearClause::setInits(ArrayRef<Expr *> IL) {
  assert(IL.size() == varlist_size() &&
         "Number of inits is not the same as the preallocated buffer");
  std::copy(IL.begin(), IL.end(), getPrivates().end());
}

void OMPLinearClause::setUpdates(ArrayRef<Expr *> UL) {
  assert(UL.size() == varlist_size() &&
         "Number of updates is not the same as the preallocated buffer");
  std::copy(UL.begin(), UL.end(), getInits().end());
}

void OMPLinearClause::setFinals(ArrayRef<Expr *> FL) {
  assert(FL.size() == varlist_size() &&
         "Number of final updates is not the same as the preallocated buffer");
  std::copy(FL.begin(), FL.end(), getUpdates().end());
}

OMPLinearClause *OMPLinearClause::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation LParenLoc,
    OpenMPLinearClauseKind Modifier, SourceLocation ModifierLoc,
    SourceLocation ColonLoc, SourceLocation EndLoc, ArrayRef<Expr *> VL,
    ArrayRef<Expr *> PL, ArrayRef<Expr *> IL, Expr *Step, Expr *CalcStep,
    Stmt *PreInit, Expr *PostUpdate) {
  // Allocate space for 4 lists (Vars, Inits, Updates, Finals) and 2 expressions
  // (Step and CalcStep).
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(5 * VL.size() + 2));
  OMPLinearClause *Clause = new (Mem) OMPLinearClause(
      StartLoc, LParenLoc, Modifier, ModifierLoc, ColonLoc, EndLoc, VL.size());
  Clause->setVarRefs(VL);
  Clause->setPrivates(PL);
  Clause->setInits(IL);
  // Fill update and final expressions with zeroes, they are provided later,
  // after the directive construction.
  std::fill(Clause->getInits().end(), Clause->getInits().end() + VL.size(),
            nullptr);
  std::fill(Clause->getUpdates().end(), Clause->getUpdates().end() + VL.size(),
            nullptr);
  Clause->setStep(Step);
  Clause->setCalcStep(CalcStep);
  Clause->setPreInitStmt(PreInit);
  Clause->setPostUpdateExpr(PostUpdate);
  return Clause;
}

OMPLinearClause *OMPLinearClause::CreateEmpty(const ASTContext &C,
                                              unsigned NumVars) {
  // Allocate space for 4 lists (Vars, Inits, Updates, Finals) and 2 expressions
  // (Step and CalcStep).
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(5 * NumVars + 2));
  return new (Mem) OMPLinearClause(NumVars);
}

OMPAlignedClause *
OMPAlignedClause::Create(const ASTContext &C, SourceLocation StartLoc,
                         SourceLocation LParenLoc, SourceLocation ColonLoc,
                         SourceLocation EndLoc, ArrayRef<Expr *> VL, Expr *A) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(VL.size() + 1));
  OMPAlignedClause *Clause = new (Mem)
      OMPAlignedClause(StartLoc, LParenLoc, ColonLoc, EndLoc, VL.size());
  Clause->setVarRefs(VL);
  Clause->setAlignment(A);
  return Clause;
}

OMPAlignedClause *OMPAlignedClause::CreateEmpty(const ASTContext &C,
                                                unsigned NumVars) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(NumVars + 1));
  return new (Mem) OMPAlignedClause(NumVars);
}

void OMPCopyinClause::setSourceExprs(ArrayRef<Expr *> SrcExprs) {
  assert(SrcExprs.size() == varlist_size() && "Number of source expressions is "
                                              "not the same as the "
                                              "preallocated buffer");
  std::copy(SrcExprs.begin(), SrcExprs.end(), varlist_end());
}

void OMPCopyinClause::setDestinationExprs(ArrayRef<Expr *> DstExprs) {
  assert(DstExprs.size() == varlist_size() && "Number of destination "
                                              "expressions is not the same as "
                                              "the preallocated buffer");
  std::copy(DstExprs.begin(), DstExprs.end(), getSourceExprs().end());
}

void OMPCopyinClause::setAssignmentOps(ArrayRef<Expr *> AssignmentOps) {
  assert(AssignmentOps.size() == varlist_size() &&
         "Number of assignment expressions is not the same as the preallocated "
         "buffer");
  std::copy(AssignmentOps.begin(), AssignmentOps.end(),
            getDestinationExprs().end());
}

OMPCopyinClause *OMPCopyinClause::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation LParenLoc,
    SourceLocation EndLoc, ArrayRef<Expr *> VL, ArrayRef<Expr *> SrcExprs,
    ArrayRef<Expr *> DstExprs, ArrayRef<Expr *> AssignmentOps) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(4 * VL.size()));
  OMPCopyinClause *Clause =
      new (Mem) OMPCopyinClause(StartLoc, LParenLoc, EndLoc, VL.size());
  Clause->setVarRefs(VL);
  Clause->setSourceExprs(SrcExprs);
  Clause->setDestinationExprs(DstExprs);
  Clause->setAssignmentOps(AssignmentOps);
  return Clause;
}

OMPCopyinClause *OMPCopyinClause::CreateEmpty(const ASTContext &C, unsigned N) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(4 * N));
  return new (Mem) OMPCopyinClause(N);
}

void OMPCopyprivateClause::setSourceExprs(ArrayRef<Expr *> SrcExprs) {
  assert(SrcExprs.size() == varlist_size() && "Number of source expressions is "
                                              "not the same as the "
                                              "preallocated buffer");
  std::copy(SrcExprs.begin(), SrcExprs.end(), varlist_end());
}

void OMPCopyprivateClause::setDestinationExprs(ArrayRef<Expr *> DstExprs) {
  assert(DstExprs.size() == varlist_size() && "Number of destination "
                                              "expressions is not the same as "
                                              "the preallocated buffer");
  std::copy(DstExprs.begin(), DstExprs.end(), getSourceExprs().end());
}

void OMPCopyprivateClause::setAssignmentOps(ArrayRef<Expr *> AssignmentOps) {
  assert(AssignmentOps.size() == varlist_size() &&
         "Number of assignment expressions is not the same as the preallocated "
         "buffer");
  std::copy(AssignmentOps.begin(), AssignmentOps.end(),
            getDestinationExprs().end());
}

OMPCopyprivateClause *OMPCopyprivateClause::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation LParenLoc,
    SourceLocation EndLoc, ArrayRef<Expr *> VL, ArrayRef<Expr *> SrcExprs,
    ArrayRef<Expr *> DstExprs, ArrayRef<Expr *> AssignmentOps) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(4 * VL.size()));
  OMPCopyprivateClause *Clause =
      new (Mem) OMPCopyprivateClause(StartLoc, LParenLoc, EndLoc, VL.size());
  Clause->setVarRefs(VL);
  Clause->setSourceExprs(SrcExprs);
  Clause->setDestinationExprs(DstExprs);
  Clause->setAssignmentOps(AssignmentOps);
  return Clause;
}

OMPCopyprivateClause *OMPCopyprivateClause::CreateEmpty(const ASTContext &C,
                                                        unsigned N) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(4 * N));
  return new (Mem) OMPCopyprivateClause(N);
}

void OMPReductionClause::setPrivates(ArrayRef<Expr *> Privates) {
  assert(Privates.size() == varlist_size() &&
         "Number of private copies is not the same as the preallocated buffer");
  std::copy(Privates.begin(), Privates.end(), varlist_end());
}

void OMPReductionClause::setLHSExprs(ArrayRef<Expr *> LHSExprs) {
  assert(
      LHSExprs.size() == varlist_size() &&
      "Number of LHS expressions is not the same as the preallocated buffer");
  std::copy(LHSExprs.begin(), LHSExprs.end(), getPrivates().end());
}

void OMPReductionClause::setRHSExprs(ArrayRef<Expr *> RHSExprs) {
  assert(
      RHSExprs.size() == varlist_size() &&
      "Number of RHS expressions is not the same as the preallocated buffer");
  std::copy(RHSExprs.begin(), RHSExprs.end(), getLHSExprs().end());
}

void OMPReductionClause::setReductionOps(ArrayRef<Expr *> ReductionOps) {
  assert(ReductionOps.size() == varlist_size() && "Number of reduction "
                                                  "expressions is not the same "
                                                  "as the preallocated buffer");
  std::copy(ReductionOps.begin(), ReductionOps.end(), getRHSExprs().end());
}

OMPReductionClause *OMPReductionClause::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation LParenLoc,
    SourceLocation EndLoc, SourceLocation ColonLoc, ArrayRef<Expr *> VL,
    NestedNameSpecifierLoc QualifierLoc, const DeclarationNameInfo &NameInfo,
    ArrayRef<Expr *> Privates, ArrayRef<Expr *> LHSExprs,
    ArrayRef<Expr *> RHSExprs, ArrayRef<Expr *> ReductionOps, Stmt *PreInit,
    Expr *PostUpdate) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(5 * VL.size()));
  OMPReductionClause *Clause = new (Mem) OMPReductionClause(
      StartLoc, LParenLoc, EndLoc, ColonLoc, VL.size(), QualifierLoc, NameInfo);
  Clause->setVarRefs(VL);
  Clause->setPrivates(Privates);
  Clause->setLHSExprs(LHSExprs);
  Clause->setRHSExprs(RHSExprs);
  Clause->setReductionOps(ReductionOps);
  Clause->setPreInitStmt(PreInit);
  Clause->setPostUpdateExpr(PostUpdate);
  return Clause;
}

OMPReductionClause *OMPReductionClause::CreateEmpty(const ASTContext &C,
                                                    unsigned N) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(5 * N));
  return new (Mem) OMPReductionClause(N);
}

void OMPTaskReductionClause::setPrivates(ArrayRef<Expr *> Privates) {
  assert(Privates.size() == varlist_size() &&
         "Number of private copies is not the same as the preallocated buffer");
  std::copy(Privates.begin(), Privates.end(), varlist_end());
}

void OMPTaskReductionClause::setLHSExprs(ArrayRef<Expr *> LHSExprs) {
  assert(
      LHSExprs.size() == varlist_size() &&
      "Number of LHS expressions is not the same as the preallocated buffer");
  std::copy(LHSExprs.begin(), LHSExprs.end(), getPrivates().end());
}

void OMPTaskReductionClause::setRHSExprs(ArrayRef<Expr *> RHSExprs) {
  assert(
      RHSExprs.size() == varlist_size() &&
      "Number of RHS expressions is not the same as the preallocated buffer");
  std::copy(RHSExprs.begin(), RHSExprs.end(), getLHSExprs().end());
}

void OMPTaskReductionClause::setReductionOps(ArrayRef<Expr *> ReductionOps) {
  assert(ReductionOps.size() == varlist_size() && "Number of task reduction "
                                                  "expressions is not the same "
                                                  "as the preallocated buffer");
  std::copy(ReductionOps.begin(), ReductionOps.end(), getRHSExprs().end());
}

OMPTaskReductionClause *OMPTaskReductionClause::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation LParenLoc,
    SourceLocation EndLoc, SourceLocation ColonLoc, ArrayRef<Expr *> VL,
    NestedNameSpecifierLoc QualifierLoc, const DeclarationNameInfo &NameInfo,
    ArrayRef<Expr *> Privates, ArrayRef<Expr *> LHSExprs,
    ArrayRef<Expr *> RHSExprs, ArrayRef<Expr *> ReductionOps, Stmt *PreInit,
    Expr *PostUpdate) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(5 * VL.size()));
  OMPTaskReductionClause *Clause = new (Mem) OMPTaskReductionClause(
      StartLoc, LParenLoc, EndLoc, ColonLoc, VL.size(), QualifierLoc, NameInfo);
  Clause->setVarRefs(VL);
  Clause->setPrivates(Privates);
  Clause->setLHSExprs(LHSExprs);
  Clause->setRHSExprs(RHSExprs);
  Clause->setReductionOps(ReductionOps);
  Clause->setPreInitStmt(PreInit);
  Clause->setPostUpdateExpr(PostUpdate);
  return Clause;
}

OMPTaskReductionClause *OMPTaskReductionClause::CreateEmpty(const ASTContext &C,
                                                            unsigned N) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(5 * N));
  return new (Mem) OMPTaskReductionClause(N);
}

void OMPInReductionClause::setPrivates(ArrayRef<Expr *> Privates) {
  assert(Privates.size() == varlist_size() &&
         "Number of private copies is not the same as the preallocated buffer");
  std::copy(Privates.begin(), Privates.end(), varlist_end());
}

void OMPInReductionClause::setLHSExprs(ArrayRef<Expr *> LHSExprs) {
  assert(
      LHSExprs.size() == varlist_size() &&
      "Number of LHS expressions is not the same as the preallocated buffer");
  std::copy(LHSExprs.begin(), LHSExprs.end(), getPrivates().end());
}

void OMPInReductionClause::setRHSExprs(ArrayRef<Expr *> RHSExprs) {
  assert(
      RHSExprs.size() == varlist_size() &&
      "Number of RHS expressions is not the same as the preallocated buffer");
  std::copy(RHSExprs.begin(), RHSExprs.end(), getLHSExprs().end());
}

void OMPInReductionClause::setReductionOps(ArrayRef<Expr *> ReductionOps) {
  assert(ReductionOps.size() == varlist_size() && "Number of in reduction "
                                                  "expressions is not the same "
                                                  "as the preallocated buffer");
  std::copy(ReductionOps.begin(), ReductionOps.end(), getRHSExprs().end());
}

void OMPInReductionClause::setTaskgroupDescriptors(
    ArrayRef<Expr *> TaskgroupDescriptors) {
  assert(TaskgroupDescriptors.size() == varlist_size() &&
         "Number of in reduction descriptors is not the same as the "
         "preallocated buffer");
  std::copy(TaskgroupDescriptors.begin(), TaskgroupDescriptors.end(),
            getReductionOps().end());
}

OMPInReductionClause *OMPInReductionClause::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation LParenLoc,
    SourceLocation EndLoc, SourceLocation ColonLoc, ArrayRef<Expr *> VL,
    NestedNameSpecifierLoc QualifierLoc, const DeclarationNameInfo &NameInfo,
    ArrayRef<Expr *> Privates, ArrayRef<Expr *> LHSExprs,
    ArrayRef<Expr *> RHSExprs, ArrayRef<Expr *> ReductionOps,
    ArrayRef<Expr *> TaskgroupDescriptors, Stmt *PreInit, Expr *PostUpdate) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(6 * VL.size()));
  OMPInReductionClause *Clause = new (Mem) OMPInReductionClause(
      StartLoc, LParenLoc, EndLoc, ColonLoc, VL.size(), QualifierLoc, NameInfo);
  Clause->setVarRefs(VL);
  Clause->setPrivates(Privates);
  Clause->setLHSExprs(LHSExprs);
  Clause->setRHSExprs(RHSExprs);
  Clause->setReductionOps(ReductionOps);
  Clause->setTaskgroupDescriptors(TaskgroupDescriptors);
  Clause->setPreInitStmt(PreInit);
  Clause->setPostUpdateExpr(PostUpdate);
  return Clause;
}

OMPInReductionClause *OMPInReductionClause::CreateEmpty(const ASTContext &C,
                                                        unsigned N) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(6 * N));
  return new (Mem) OMPInReductionClause(N);
}

OMPFlushClause *OMPFlushClause::Create(const ASTContext &C,
                                       SourceLocation StartLoc,
                                       SourceLocation LParenLoc,
                                       SourceLocation EndLoc,
                                       ArrayRef<Expr *> VL) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(VL.size() + 1));
  OMPFlushClause *Clause =
      new (Mem) OMPFlushClause(StartLoc, LParenLoc, EndLoc, VL.size());
  Clause->setVarRefs(VL);
  return Clause;
}

OMPFlushClause *OMPFlushClause::CreateEmpty(const ASTContext &C, unsigned N) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(N));
  return new (Mem) OMPFlushClause(N);
}

OMPDependClause *
OMPDependClause::Create(const ASTContext &C, SourceLocation StartLoc,
                        SourceLocation LParenLoc, SourceLocation EndLoc,
                        OpenMPDependClauseKind DepKind, SourceLocation DepLoc,
                        SourceLocation ColonLoc, ArrayRef<Expr *> VL,
                        unsigned NumLoops) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(VL.size() + NumLoops));
  OMPDependClause *Clause = new (Mem)
      OMPDependClause(StartLoc, LParenLoc, EndLoc, VL.size(), NumLoops);
  Clause->setVarRefs(VL);
  Clause->setDependencyKind(DepKind);
  Clause->setDependencyLoc(DepLoc);
  Clause->setColonLoc(ColonLoc);
  for (unsigned I = 0 ; I < NumLoops; ++I)
    Clause->setLoopData(I, nullptr);
  return Clause;
}

OMPDependClause *OMPDependClause::CreateEmpty(const ASTContext &C, unsigned N,
                                              unsigned NumLoops) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(N + NumLoops));
  return new (Mem) OMPDependClause(N, NumLoops);
}

void OMPDependClause::setLoopData(unsigned NumLoop, Expr *Cnt) {
  assert((getDependencyKind() == OMPC_DEPEND_sink ||
          getDependencyKind() == OMPC_DEPEND_source) &&
         NumLoop < NumLoops &&
         "Expected sink or source depend + loop index must be less number of "
         "loops.");
  auto It = std::next(getVarRefs().end(), NumLoop);
  *It = Cnt;
}

Expr *OMPDependClause::getLoopData(unsigned NumLoop) {
  assert((getDependencyKind() == OMPC_DEPEND_sink ||
          getDependencyKind() == OMPC_DEPEND_source) &&
         NumLoop < NumLoops &&
         "Expected sink or source depend + loop index must be less number of "
         "loops.");
  auto It = std::next(getVarRefs().end(), NumLoop);
  return *It;
}

const Expr *OMPDependClause::getLoopData(unsigned NumLoop) const {
  assert((getDependencyKind() == OMPC_DEPEND_sink ||
          getDependencyKind() == OMPC_DEPEND_source) &&
         NumLoop < NumLoops &&
         "Expected sink or source depend + loop index must be less number of "
         "loops.");
  auto It = std::next(getVarRefs().end(), NumLoop);
  return *It;
}

unsigned OMPClauseMappableExprCommon::getComponentsTotalNumber(
    MappableExprComponentListsRef ComponentLists) {
  unsigned TotalNum = 0u;
  for (auto &C : ComponentLists)
    TotalNum += C.size();
  return TotalNum;
}

unsigned OMPClauseMappableExprCommon::getUniqueDeclarationsTotalNumber(
    ArrayRef<const ValueDecl *> Declarations) {
  unsigned TotalNum = 0u;
  llvm::SmallPtrSet<const ValueDecl *, 8> Cache;
  for (const ValueDecl *D : Declarations) {
    const ValueDecl *VD = D ? cast<ValueDecl>(D->getCanonicalDecl()) : nullptr;
    if (Cache.count(VD))
      continue;
    ++TotalNum;
    Cache.insert(VD);
  }
  return TotalNum;
}

OMPMapClause *
OMPMapClause::Create(const ASTContext &C, SourceLocation StartLoc,
                     SourceLocation LParenLoc, SourceLocation EndLoc,
                     ArrayRef<Expr *> Vars, ArrayRef<ValueDecl *> Declarations,
                     MappableExprComponentListsRef ComponentLists,
                     OpenMPMapClauseKind TypeModifier, OpenMPMapClauseKind Type,
                     bool TypeIsImplicit, SourceLocation TypeLoc) {
  unsigned NumVars = Vars.size();
  unsigned NumUniqueDeclarations =
      getUniqueDeclarationsTotalNumber(Declarations);
  unsigned NumComponentLists = ComponentLists.size();
  unsigned NumComponents = getComponentsTotalNumber(ComponentLists);

  // We need to allocate:
  // NumVars x Expr* - we have an original list expression for each clause list
  // entry.
  // NumUniqueDeclarations x ValueDecl* - unique base declarations associated
  // with each component list.
  // (NumUniqueDeclarations + NumComponentLists) x unsigned - we specify the
  // number of lists for each unique declaration and the size of each component
  // list.
  // NumComponents x MappableComponent - the total of all the components in all
  // the lists.
  void *Mem = C.Allocate(
      totalSizeToAlloc<Expr *, ValueDecl *, unsigned,
                       OMPClauseMappableExprCommon::MappableComponent>(
          NumVars, NumUniqueDeclarations,
          NumUniqueDeclarations + NumComponentLists, NumComponents));
  OMPMapClause *Clause = new (Mem) OMPMapClause(
      TypeModifier, Type, TypeIsImplicit, TypeLoc, StartLoc, LParenLoc, EndLoc,
      NumVars, NumUniqueDeclarations, NumComponentLists, NumComponents);

  Clause->setVarRefs(Vars);
  Clause->setClauseInfo(Declarations, ComponentLists);
  Clause->setMapTypeModifier(TypeModifier);
  Clause->setMapType(Type);
  Clause->setMapLoc(TypeLoc);
  return Clause;
}

OMPMapClause *OMPMapClause::CreateEmpty(const ASTContext &C, unsigned NumVars,
                                        unsigned NumUniqueDeclarations,
                                        unsigned NumComponentLists,
                                        unsigned NumComponents) {
  void *Mem = C.Allocate(
      totalSizeToAlloc<Expr *, ValueDecl *, unsigned,
                       OMPClauseMappableExprCommon::MappableComponent>(
          NumVars, NumUniqueDeclarations,
          NumUniqueDeclarations + NumComponentLists, NumComponents));
  return new (Mem) OMPMapClause(NumVars, NumUniqueDeclarations,
                                NumComponentLists, NumComponents);
}

OMPToClause *OMPToClause::Create(const ASTContext &C, SourceLocation StartLoc,
                                 SourceLocation LParenLoc,
                                 SourceLocation EndLoc, ArrayRef<Expr *> Vars,
                                 ArrayRef<ValueDecl *> Declarations,
                                 MappableExprComponentListsRef ComponentLists) {
  unsigned NumVars = Vars.size();
  unsigned NumUniqueDeclarations =
      getUniqueDeclarationsTotalNumber(Declarations);
  unsigned NumComponentLists = ComponentLists.size();
  unsigned NumComponents = getComponentsTotalNumber(ComponentLists);

  // We need to allocate:
  // NumVars x Expr* - we have an original list expression for each clause list
  // entry.
  // NumUniqueDeclarations x ValueDecl* - unique base declarations associated
  // with each component list.
  // (NumUniqueDeclarations + NumComponentLists) x unsigned - we specify the
  // number of lists for each unique declaration and the size of each component
  // list.
  // NumComponents x MappableComponent - the total of all the components in all
  // the lists.
  void *Mem = C.Allocate(
      totalSizeToAlloc<Expr *, ValueDecl *, unsigned,
                       OMPClauseMappableExprCommon::MappableComponent>(
          NumVars, NumUniqueDeclarations,
          NumUniqueDeclarations + NumComponentLists, NumComponents));

  OMPToClause *Clause = new (Mem)
      OMPToClause(StartLoc, LParenLoc, EndLoc, NumVars, NumUniqueDeclarations,
                  NumComponentLists, NumComponents);

  Clause->setVarRefs(Vars);
  Clause->setClauseInfo(Declarations, ComponentLists);
  return Clause;
}

OMPToClause *OMPToClause::CreateEmpty(const ASTContext &C, unsigned NumVars,
                                      unsigned NumUniqueDeclarations,
                                      unsigned NumComponentLists,
                                      unsigned NumComponents) {
  void *Mem = C.Allocate(
      totalSizeToAlloc<Expr *, ValueDecl *, unsigned,
                       OMPClauseMappableExprCommon::MappableComponent>(
          NumVars, NumUniqueDeclarations,
          NumUniqueDeclarations + NumComponentLists, NumComponents));
  return new (Mem) OMPToClause(NumVars, NumUniqueDeclarations,
                               NumComponentLists, NumComponents);
}

OMPFromClause *
OMPFromClause::Create(const ASTContext &C, SourceLocation StartLoc,
                      SourceLocation LParenLoc, SourceLocation EndLoc,
                      ArrayRef<Expr *> Vars, ArrayRef<ValueDecl *> Declarations,
                      MappableExprComponentListsRef ComponentLists) {
  unsigned NumVars = Vars.size();
  unsigned NumUniqueDeclarations =
      getUniqueDeclarationsTotalNumber(Declarations);
  unsigned NumComponentLists = ComponentLists.size();
  unsigned NumComponents = getComponentsTotalNumber(ComponentLists);

  // We need to allocate:
  // NumVars x Expr* - we have an original list expression for each clause list
  // entry.
  // NumUniqueDeclarations x ValueDecl* - unique base declarations associated
  // with each component list.
  // (NumUniqueDeclarations + NumComponentLists) x unsigned - we specify the
  // number of lists for each unique declaration and the size of each component
  // list.
  // NumComponents x MappableComponent - the total of all the components in all
  // the lists.
  void *Mem = C.Allocate(
      totalSizeToAlloc<Expr *, ValueDecl *, unsigned,
                       OMPClauseMappableExprCommon::MappableComponent>(
          NumVars, NumUniqueDeclarations,
          NumUniqueDeclarations + NumComponentLists, NumComponents));

  OMPFromClause *Clause = new (Mem)
      OMPFromClause(StartLoc, LParenLoc, EndLoc, NumVars, NumUniqueDeclarations,
                    NumComponentLists, NumComponents);

  Clause->setVarRefs(Vars);
  Clause->setClauseInfo(Declarations, ComponentLists);
  return Clause;
}

OMPFromClause *OMPFromClause::CreateEmpty(const ASTContext &C, unsigned NumVars,
                                          unsigned NumUniqueDeclarations,
                                          unsigned NumComponentLists,
                                          unsigned NumComponents) {
  void *Mem = C.Allocate(
      totalSizeToAlloc<Expr *, ValueDecl *, unsigned,
                       OMPClauseMappableExprCommon::MappableComponent>(
          NumVars, NumUniqueDeclarations,
          NumUniqueDeclarations + NumComponentLists, NumComponents));
  return new (Mem) OMPFromClause(NumVars, NumUniqueDeclarations,
                                 NumComponentLists, NumComponents);
}

void OMPUseDevicePtrClause::setPrivateCopies(ArrayRef<Expr *> VL) {
  assert(VL.size() == varlist_size() &&
         "Number of private copies is not the same as the preallocated buffer");
  std::copy(VL.begin(), VL.end(), varlist_end());
}

void OMPUseDevicePtrClause::setInits(ArrayRef<Expr *> VL) {
  assert(VL.size() == varlist_size() &&
         "Number of inits is not the same as the preallocated buffer");
  std::copy(VL.begin(), VL.end(), getPrivateCopies().end());
}

OMPUseDevicePtrClause *OMPUseDevicePtrClause::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation LParenLoc,
    SourceLocation EndLoc, ArrayRef<Expr *> Vars, ArrayRef<Expr *> PrivateVars,
    ArrayRef<Expr *> Inits, ArrayRef<ValueDecl *> Declarations,
    MappableExprComponentListsRef ComponentLists) {
  unsigned NumVars = Vars.size();
  unsigned NumUniqueDeclarations =
      getUniqueDeclarationsTotalNumber(Declarations);
  unsigned NumComponentLists = ComponentLists.size();
  unsigned NumComponents = getComponentsTotalNumber(ComponentLists);

  // We need to allocate:
  // 3 x NumVars x Expr* - we have an original list expression for each clause
  // list entry and an equal number of private copies and inits.
  // NumUniqueDeclarations x ValueDecl* - unique base declarations associated
  // with each component list.
  // (NumUniqueDeclarations + NumComponentLists) x unsigned - we specify the
  // number of lists for each unique declaration and the size of each component
  // list.
  // NumComponents x MappableComponent - the total of all the components in all
  // the lists.
  void *Mem = C.Allocate(
      totalSizeToAlloc<Expr *, ValueDecl *, unsigned,
                       OMPClauseMappableExprCommon::MappableComponent>(
          3 * NumVars, NumUniqueDeclarations,
          NumUniqueDeclarations + NumComponentLists, NumComponents));

  OMPUseDevicePtrClause *Clause = new (Mem) OMPUseDevicePtrClause(
      StartLoc, LParenLoc, EndLoc, NumVars, NumUniqueDeclarations,
      NumComponentLists, NumComponents);

  Clause->setVarRefs(Vars);
  Clause->setPrivateCopies(PrivateVars);
  Clause->setInits(Inits);
  Clause->setClauseInfo(Declarations, ComponentLists);
  return Clause;
}

OMPUseDevicePtrClause *OMPUseDevicePtrClause::CreateEmpty(
    const ASTContext &C, unsigned NumVars, unsigned NumUniqueDeclarations,
    unsigned NumComponentLists, unsigned NumComponents) {
  void *Mem = C.Allocate(
      totalSizeToAlloc<Expr *, ValueDecl *, unsigned,
                       OMPClauseMappableExprCommon::MappableComponent>(
          3 * NumVars, NumUniqueDeclarations,
          NumUniqueDeclarations + NumComponentLists, NumComponents));
  return new (Mem) OMPUseDevicePtrClause(NumVars, NumUniqueDeclarations,
                                         NumComponentLists, NumComponents);
}

OMPIsDevicePtrClause *
OMPIsDevicePtrClause::Create(const ASTContext &C, SourceLocation StartLoc,
                             SourceLocation LParenLoc, SourceLocation EndLoc,
                             ArrayRef<Expr *> Vars,
                             ArrayRef<ValueDecl *> Declarations,
                             MappableExprComponentListsRef ComponentLists) {
  unsigned NumVars = Vars.size();
  unsigned NumUniqueDeclarations =
      getUniqueDeclarationsTotalNumber(Declarations);
  unsigned NumComponentLists = ComponentLists.size();
  unsigned NumComponents = getComponentsTotalNumber(ComponentLists);

  // We need to allocate:
  // NumVars x Expr* - we have an original list expression for each clause list
  // entry.
  // NumUniqueDeclarations x ValueDecl* - unique base declarations associated
  // with each component list.
  // (NumUniqueDeclarations + NumComponentLists) x unsigned - we specify the
  // number of lists for each unique declaration and the size of each component
  // list.
  // NumComponents x MappableComponent - the total of all the components in all
  // the lists.
  void *Mem = C.Allocate(
      totalSizeToAlloc<Expr *, ValueDecl *, unsigned,
                       OMPClauseMappableExprCommon::MappableComponent>(
          NumVars, NumUniqueDeclarations,
          NumUniqueDeclarations + NumComponentLists, NumComponents));

  OMPIsDevicePtrClause *Clause = new (Mem) OMPIsDevicePtrClause(
      StartLoc, LParenLoc, EndLoc, NumVars, NumUniqueDeclarations,
      NumComponentLists, NumComponents);

  Clause->setVarRefs(Vars);
  Clause->setClauseInfo(Declarations, ComponentLists);
  return Clause;
}

OMPIsDevicePtrClause *OMPIsDevicePtrClause::CreateEmpty(
    const ASTContext &C, unsigned NumVars, unsigned NumUniqueDeclarations,
    unsigned NumComponentLists, unsigned NumComponents) {
  void *Mem = C.Allocate(
      totalSizeToAlloc<Expr *, ValueDecl *, unsigned,
                       OMPClauseMappableExprCommon::MappableComponent>(
          NumVars, NumUniqueDeclarations,
          NumUniqueDeclarations + NumComponentLists, NumComponents));
  return new (Mem) OMPIsDevicePtrClause(NumVars, NumUniqueDeclarations,
                                        NumComponentLists, NumComponents);
}
