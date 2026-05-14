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
  switch (K) {
  case Kind::ValueDecl:
    if (const clang::ValueDecl *VD = getAsValueDecl())
      OS << VD->getNameAsString();
    break;
  case Kind::MaterializeTemporary:
    if (const clang::MaterializeTemporaryExpr *MTE =
            getAsMaterializeTemporaryExpr())
      OS << "MaterializeTemporaryExpr at " << MTE;
    break;
  case Kind::PlaceholderParam:
    if (const auto *PVD = getAsPlaceholderParam())
      OS << "$" << PVD->getNameAsString();
    break;
  case Kind::PlaceholderThis:
    OS << "$this";
    break;
  case Kind::NewAllocation:
    if (const auto *E = getAsNewAllocation())
      OS << "NewAllocation at " << E;
    break;
  }
}

void Loan::dump(llvm::raw_ostream &OS) const {
  OS << getID() << " (Path: ";
  Path.dump(OS);
  OS << ")";
}
} // namespace clang::lifetimes::internal
