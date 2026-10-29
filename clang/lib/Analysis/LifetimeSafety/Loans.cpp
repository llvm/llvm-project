//===- Loans.cpp - Loan Implementation --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Analyses/LifetimeSafety/Loans.h"
#include "llvm/ADT/STLExtras.h"

namespace clang::lifetimes::internal {

void PathElement::dump(llvm::raw_ostream &OS) const {
  switch (K) {
  case Kind::Field:
    OS << "." << getFieldDecl()->getNameAsString();
    return;
  case Kind::Interior:
    OS << ".*";
    return;
  case Kind::Place:
    OS << "." << getPlaceName()->getName();
    return;
  }
  llvm_unreachable("unknown PathElement kind");
}

void AccessPath::dump(llvm::raw_ostream &OS) const {
  if (const clang::ValueDecl *VD = getAsValueDecl())
    OS << VD->getNameAsString();
  else if (const clang::MaterializeTemporaryExpr *MTE =
               getAsMaterializeTemporaryExpr())
    OS << "MaterializeTemporaryExpr at " << MTE;
  else if (const PlaceholderBase *PB = getAsPlaceholderBase()) {
    if (const auto *PVD = PB->getParmVarDecl())
      OS << "$" << PVD->getNameAsString();
    else if (PB->getImplicitThisParent())
      OS << "$this";
    else if (const auto *FD = PB->getFieldDecl())
      OS << "$this." << FD->getNameAsString();
  } else if (const auto *E = getAsNewAllocation())
    OS << "NewAllocation at " << E;
  else
    llvm_unreachable("access path base invalid");
  for (const auto &E : Elements)
    E.dump(OS);
  if (Truncated)
    OS << "...";
}

void Loan::dump(llvm::raw_ostream &OS) const {
  OS << getID() << " (Path: ";
  Path.dump(OS);
  OS << ")";
}

const Loan *LoanManager::getOrCreateProjectedLoan(LoanID BaseLoanID,
                                                  PathElement Element) {
  auto [It, Inserted] = ProjectedLoans.try_emplace({BaseLoanID, Element});
  if (!Inserted)
    return It->second;

  const Loan *Base = getLoan(BaseLoanID);
  const AccessPath &BasePath = Base->getAccessPath();

  const Loan *Result;
  if (BasePath.isTruncated()) {
    // Already truncated: it stands for every extension, including this one.
    Result = Base;
  } else {
    // A field can only recur in a path through a recursive type, and the
    // number of interior levels is bounded by the static nesting of owners;
    // either growing without bound means a loop over a data-dependent
    // structure. Truncate instead of growing.
    bool Truncate =
        Element.isField()
            ? llvm::is_contained(BasePath.getElements(), Element)
            : llvm::count_if(BasePath.getElements(), [](const PathElement &E) {
                return E.isInteriorOrPlace();
              }) >= MaxInteriorDepth;
    Result = createLoan(Truncate ? AccessPath::getTruncated(BasePath)
                                 : AccessPath(BasePath, Element),
                        Base->getIssueExpr());
    BaseLoans[Result->getID()] = BaseLoanID;
  }
  // `It` is still valid: `createLoan` does not touch `ProjectedLoans`.
  It->second = Result;
  return Result;
}

const PlaceholderBase *
LoanManager::getOrCreatePlaceholderBase(const ParmVarDecl *PVD) {
  llvm::FoldingSetNodeID ID;
  PlaceholderBase::Profile(ID, PVD);
  llvm::FoldingSetInsertToken InsertToken;
  if (PlaceholderBase *Existing = PlaceholderBases.lookup(ID, InsertToken))
    return Existing;

  void *Mem = LoanAllocator.Allocate<PlaceholderBase>();
  PlaceholderBase *NewPB = new (Mem) PlaceholderBase(PVD);
  PlaceholderBases.insert(NewPB, InsertToken);
  return NewPB;
}

const PlaceholderBase *
LoanManager::getOrCreatePlaceholderBase(const CXXMethodDecl *MD) {
  llvm::FoldingSetNodeID ID;
  PlaceholderBase::Profile(ID, MD);
  llvm::FoldingSetInsertToken InsertToken;
  if (PlaceholderBase *Existing = PlaceholderBases.lookup(ID, InsertToken))
    return Existing;

  void *Mem = LoanAllocator.Allocate<PlaceholderBase>();
  PlaceholderBase *NewPB = new (Mem) PlaceholderBase(MD);
  PlaceholderBases.insert(NewPB, InsertToken);
  return NewPB;
}

const PlaceholderBase *
LoanManager::getOrCreatePlaceholderBase(const FieldDecl *FD) {
  llvm::FoldingSetNodeID ID;
  PlaceholderBase::Profile(ID, FD);
  llvm::FoldingSetInsertToken InsertToken;
  if (PlaceholderBase *Existing = PlaceholderBases.lookup(ID, InsertToken))
    return Existing;

  void *Mem = LoanAllocator.Allocate<PlaceholderBase>();
  PlaceholderBase *NewPB = new (Mem) PlaceholderBase(FD);
  PlaceholderBases.insert(NewPB, InsertToken);
  return NewPB;
}
} // namespace clang::lifetimes::internal
