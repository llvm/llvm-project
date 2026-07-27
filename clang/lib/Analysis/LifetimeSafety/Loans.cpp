//===- Loans.cpp - Loan Implementation --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Analyses/LifetimeSafety/Loans.h"

namespace clang::lifetimes::internal {

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
  } else if (const auto *E = getAsNewAllocation())
    OS << "NewAllocation at " << E;
  else
    llvm_unreachable("access path base invalid");
  for (const auto &E : Elements)
    E.dump(OS);
}

void Loan::dump(llvm::raw_ostream &OS) const {
  OS << getID() << " (Path: ";
  Path.dump(OS);
  OS << ")";
}

const PlaceholderBase *
LoanManager::getOrCreatePlaceholderBase(const ParmVarDecl *PVD) {
  llvm::FoldingSetNodeID ID;
  ID.AddPointer(PVD);
  void *InsertPos = nullptr;
  if (PlaceholderBase *Existing =
          PlaceholderBases.FindNodeOrInsertPos(ID, InsertPos))
    return Existing;

  void *Mem = LoanAllocator.Allocate<PlaceholderBase>();
  PlaceholderBase *NewPB = new (Mem) PlaceholderBase(PVD);
  PlaceholderBases.InsertNode(NewPB, InsertPos);
  return NewPB;
}

const PlaceholderBase *
LoanManager::getOrCreatePlaceholderBase(const CXXMethodDecl *MD) {
  llvm::FoldingSetNodeID ID;
  ID.AddPointer(MD);
  void *InsertPos = nullptr;
  if (PlaceholderBase *Existing =
          PlaceholderBases.FindNodeOrInsertPos(ID, InsertPos))
    return Existing;

  void *Mem = LoanAllocator.Allocate<PlaceholderBase>();
  PlaceholderBase *NewPB = new (Mem) PlaceholderBase(MD);
  PlaceholderBases.InsertNode(NewPB, InsertPos);
  return NewPB;
}

Loan *LoanManager::getOrCreateExtendedLoan(LoanID BaseLoanID,
                                           PathElement Element) {
  ExtensionCacheKey Key = {BaseLoanID, Element};
  auto It = ExtensionCache.find(Key);
  if (It != ExtensionCache.end())
    return It->second;
  const auto *BaseLoan = getLoan(BaseLoanID);

  // Stop appending if Element is already in the path to prevent infinite path
  // accumulation (divergence) on recursive types or casts.
  //
  // This sound over-approximation guarantees termination at the cost of
  // precision. Conflating infinitely deep paths into a single truncated loan
  // may cause false positives via spurious invalidations, but never causes
  // false negatives.
  for (const PathElement &E : BaseLoan->getAccessPath().getElements())
    if (E == Element)
      return ExtensionCache[Key] = const_cast<Loan *>(BaseLoan);

  AccessPath ExtendedPath(BaseLoan->getAccessPath(), Element);
  Loan *NewLoan = createLoan(ExtendedPath, BaseLoan->getIssueExpr());
  BaseLoansMap[NewLoan->getID()] = BaseLoanID;
  return ExtensionCache[Key] = NewLoan;
}

std::optional<LoanID> LoanManager::getBaseLoan(LoanID ExtendedLoanID) const {
  auto It = BaseLoansMap.find(ExtendedLoanID);
  if (It != BaseLoansMap.end())
    return It->second;
  return std::nullopt;
}
} // namespace clang::lifetimes::internal
