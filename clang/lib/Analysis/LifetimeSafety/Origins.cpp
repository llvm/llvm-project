//===- Origins.cpp - Origin Implementation -----------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Analyses/LifetimeSafety/Origins.h"

namespace clang::lifetimes::internal {

void OriginManager::dump(OriginID OID, llvm::raw_ostream &OS) const {
  OS << OID << " (";
  Origin O = getOrigin(OID);
  if (const ValueDecl *VD = O.getDecl())
    OS << "Decl: " << VD->getNameAsString();
  else if (const Expr *E = O.getExpr())
    OS << "Expr: " << E->getStmtClassName();
  else
    OS << "Unknown";
  OS << ")";
}

} // namespace clang::lifetimes::internal
