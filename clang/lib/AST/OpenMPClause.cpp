//===--- OpenMPClause.cpp - Classes for OpenMP clauses --------------------===//
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
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPPrivateClause),
                                                  llvm::alignOf<Expr *>()) +
                         2 * sizeof(Expr *) * VL.size());
  OMPPrivateClause *Clause =
      new (Mem) OMPPrivateClause(StartLoc, LParenLoc, EndLoc, VL.size());
  Clause->setVarRefs(VL);
  Clause->setPrivateCopies(PrivateVL);
  return Clause;
}

OMPPrivateClause *OMPPrivateClause::CreateEmpty(const ASTContext &C,
                                                unsigned N) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPPrivateClause),
                                                  llvm::alignOf<Expr *>()) +
                         2 * sizeof(Expr *) * N);
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
                              ArrayRef<Expr *> InitVL) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPFirstprivateClause),
                                                  llvm::alignOf<Expr *>()) +
                         3 * sizeof(Expr *) * VL.size());
  OMPFirstprivateClause *Clause =
      new (Mem) OMPFirstprivateClause(StartLoc, LParenLoc, EndLoc, VL.size());
  Clause->setVarRefs(VL);
  Clause->setPrivateCopies(PrivateVL);
  Clause->setInits(InitVL);
  return Clause;
}

OMPFirstprivateClause *OMPFirstprivateClause::CreateEmpty(const ASTContext &C,
                                                          unsigned N) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPFirstprivateClause),
                                                  llvm::alignOf<Expr *>()) +
                         3 * sizeof(Expr *) * N);
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
    ArrayRef<Expr *> DstExprs, ArrayRef<Expr *> AssignmentOps) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPLastprivateClause),
                                                  llvm::alignOf<Expr *>()) +
                         5 * sizeof(Expr *) * VL.size());
  OMPLastprivateClause *Clause =
      new (Mem) OMPLastprivateClause(StartLoc, LParenLoc, EndLoc, VL.size());
  Clause->setVarRefs(VL);
  Clause->setSourceExprs(SrcExprs);
  Clause->setDestinationExprs(DstExprs);
  Clause->setAssignmentOps(AssignmentOps);
  return Clause;
}

OMPLastprivateClause *OMPLastprivateClause::CreateEmpty(const ASTContext &C,
                                                        unsigned N) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPLastprivateClause),
                                                  llvm::alignOf<Expr *>()) +
                         5 * sizeof(Expr *) * N);
  return new (Mem) OMPLastprivateClause(N);
}

OMPSharedClause *OMPSharedClause::Create(const ASTContext &C,
                                         SourceLocation StartLoc,
                                         SourceLocation LParenLoc,
                                         SourceLocation EndLoc,
                                         ArrayRef<Expr *> VL) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPSharedClause),
                                                  llvm::alignOf<Expr *>()) +
                         sizeof(Expr *) * VL.size());
  OMPSharedClause *Clause =
      new (Mem) OMPSharedClause(StartLoc, LParenLoc, EndLoc, VL.size());
  Clause->setVarRefs(VL);
  return Clause;
}

OMPSharedClause *OMPSharedClause::CreateEmpty(const ASTContext &C, unsigned N) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPSharedClause),
                                                  llvm::alignOf<Expr *>()) +
                         sizeof(Expr *) * N);
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
    ArrayRef<Expr *> PL, ArrayRef<Expr *> IL, Expr *Step, Expr *CalcStep) {
  // Allocate space for 4 lists (Vars, Inits, Updates, Finals) and 2 expressions
  // (Step and CalcStep).
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPLinearClause),
                                                  llvm::alignOf<Expr *>()) +
                         (5 * VL.size() + 2) * sizeof(Expr *));
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
  return Clause;
}

OMPLinearClause *OMPLinearClause::CreateEmpty(const ASTContext &C,
                                              unsigned NumVars) {
  // Allocate space for 4 lists (Vars, Inits, Updates, Finals) and 2 expressions
  // (Step and CalcStep).
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPLinearClause),
                                                  llvm::alignOf<Expr *>()) +
                         (5 * NumVars + 2) * sizeof(Expr *));
  return new (Mem) OMPLinearClause(NumVars);
}

OMPAlignedClause *
OMPAlignedClause::Create(const ASTContext &C, SourceLocation StartLoc,
                         SourceLocation LParenLoc, SourceLocation ColonLoc,
                         SourceLocation EndLoc, ArrayRef<Expr *> VL, Expr *A) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPAlignedClause),
                                                  llvm::alignOf<Expr *>()) +
                         sizeof(Expr *) * (VL.size() + 1));
  OMPAlignedClause *Clause = new (Mem)
      OMPAlignedClause(StartLoc, LParenLoc, ColonLoc, EndLoc, VL.size());
  Clause->setVarRefs(VL);
  Clause->setAlignment(A);
  return Clause;
}

OMPAlignedClause *OMPAlignedClause::CreateEmpty(const ASTContext &C,
                                                unsigned NumVars) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPAlignedClause),
                                                  llvm::alignOf<Expr *>()) +
                         sizeof(Expr *) * (NumVars + 1));
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
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPCopyinClause),
                                                  llvm::alignOf<Expr *>()) +
                         4 * sizeof(Expr *) * VL.size());
  OMPCopyinClause *Clause =
      new (Mem) OMPCopyinClause(StartLoc, LParenLoc, EndLoc, VL.size());
  Clause->setVarRefs(VL);
  Clause->setSourceExprs(SrcExprs);
  Clause->setDestinationExprs(DstExprs);
  Clause->setAssignmentOps(AssignmentOps);
  return Clause;
}

OMPCopyinClause *OMPCopyinClause::CreateEmpty(const ASTContext &C, unsigned N) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPCopyinClause),
                                                  llvm::alignOf<Expr *>()) +
                         4 * sizeof(Expr *) * N);
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
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPCopyprivateClause),
                                                  llvm::alignOf<Expr *>()) +
                         4 * sizeof(Expr *) * VL.size());
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
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPCopyprivateClause),
                                                  llvm::alignOf<Expr *>()) +
                         4 * sizeof(Expr *) * N);
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
    ArrayRef<Expr *> RHSExprs, ArrayRef<Expr *> ReductionOps) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPReductionClause),
                                                  llvm::alignOf<Expr *>()) +
                         5 * sizeof(Expr *) * VL.size());
  OMPReductionClause *Clause = new (Mem) OMPReductionClause(
      StartLoc, LParenLoc, EndLoc, ColonLoc, VL.size(), QualifierLoc, NameInfo);
  Clause->setVarRefs(VL);
  Clause->setPrivates(Privates);
  Clause->setLHSExprs(LHSExprs);
  Clause->setRHSExprs(RHSExprs);
  Clause->setReductionOps(ReductionOps);
  return Clause;
}

OMPReductionClause *OMPReductionClause::CreateEmpty(const ASTContext &C,
                                                    unsigned N) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPReductionClause),
                                                  llvm::alignOf<Expr *>()) +
                         5 * sizeof(Expr *) * N);
  return new (Mem) OMPReductionClause(N);
}

OMPFlushClause *OMPFlushClause::Create(const ASTContext &C,
                                       SourceLocation StartLoc,
                                       SourceLocation LParenLoc,
                                       SourceLocation EndLoc,
                                       ArrayRef<Expr *> VL) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPFlushClause),
                                                  llvm::alignOf<Expr *>()) +
                         sizeof(Expr *) * VL.size());
  OMPFlushClause *Clause =
      new (Mem) OMPFlushClause(StartLoc, LParenLoc, EndLoc, VL.size());
  Clause->setVarRefs(VL);
  return Clause;
}

OMPFlushClause *OMPFlushClause::CreateEmpty(const ASTContext &C, unsigned N) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPFlushClause),
                                                  llvm::alignOf<Expr *>()) +
                         sizeof(Expr *) * N);
  return new (Mem) OMPFlushClause(N);
}

OMPDependClause *
OMPDependClause::Create(const ASTContext &C, SourceLocation StartLoc,
                        SourceLocation LParenLoc, SourceLocation EndLoc,
                        OpenMPDependClauseKind DepKind, SourceLocation DepLoc,
                        SourceLocation ColonLoc, ArrayRef<Expr *> VL) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPDependClause),
                                                  llvm::alignOf<Expr *>()) +
                         sizeof(Expr *) * VL.size());
  OMPDependClause *Clause =
      new (Mem) OMPDependClause(StartLoc, LParenLoc, EndLoc, VL.size());
  Clause->setVarRefs(VL);
  Clause->setDependencyKind(DepKind);
  Clause->setDependencyLoc(DepLoc);
  Clause->setColonLoc(ColonLoc);
  return Clause;
}

OMPDependClause *OMPDependClause::CreateEmpty(const ASTContext &C, unsigned N) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPDependClause),
                                                  llvm::alignOf<Expr *>()) +
                         sizeof(Expr *) * N);
  return new (Mem) OMPDependClause(N);
}

OMPMapClause *OMPMapClause::Create(const ASTContext &C, SourceLocation StartLoc,
                                   SourceLocation LParenLoc,
                                   SourceLocation EndLoc, ArrayRef<Expr *> VL,
                                   OpenMPMapClauseKind TypeModifier,
                                   OpenMPMapClauseKind Type,
                                   SourceLocation TypeLoc) {
  void *Mem = C.Allocate(
      llvm::RoundUpToAlignment(sizeof(OMPMapClause), llvm::alignOf<Expr *>()) +
      sizeof(Expr *) * VL.size());
  OMPMapClause *Clause = new (Mem) OMPMapClause(
      TypeModifier, Type, TypeLoc, StartLoc, LParenLoc, EndLoc, VL.size());
  Clause->setVarRefs(VL);
  Clause->setMapTypeModifier(TypeModifier);
  Clause->setMapType(Type);
  Clause->setMapLoc(TypeLoc);
  return Clause;
}

OMPMapClause *OMPMapClause::CreateEmpty(const ASTContext &C, unsigned N) {
  void *Mem = C.Allocate(
      llvm::RoundUpToAlignment(sizeof(OMPMapClause), llvm::alignOf<Expr *>()) +
      sizeof(Expr *) * N);
  return new (Mem) OMPMapClause(N);
}
