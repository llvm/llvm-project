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

Origin &OriginManager::addOrigin(OriginID ID, const clang::ValueDecl &D) {
  AllOrigins.emplace_back(ID, &D);
  return AllOrigins.back();
}

Origin &OriginManager::addOrigin(OriginID ID, const clang::Expr &E) {
  AllOrigins.emplace_back(ID, &E);
  return AllOrigins.back();
}

// TODO: Mark this method as const once we remove the call to getOrCreate.
OriginID OriginManager::get(const Expr &E) {
  auto It = ExprToOriginID.find(&E);
  if (It != ExprToOriginID.end())
    return It->second;
  // If the expression itself has no specific origin, and it's a reference
  // to a declaration, its origin is that of the declaration it refers to.
  // For pointer types, where we don't pre-emptively create an origin for the
  // DeclRefExpr itself.
  if (const auto *DRE = dyn_cast<DeclRefExpr>(&E))
    return get(*DRE->getDecl());
  // TODO: This should be an assert(It != ExprToOriginID.end()). The current
  // implementation falls back to getOrCreate to avoid crashing on
  // yet-unhandled pointer expressions, creating an empty origin for them.
  return getOrCreate(E);
}

OriginID OriginManager::get(const ValueDecl &D) {
  auto It = DeclToOriginID.find(&D);
  // TODO: This should be an assert(It != DeclToOriginID.end()). The current
  // implementation falls back to getOrCreate to avoid crashing on
  // yet-unhandled pointer expressions, creating an empty origin for them.
  if (It == DeclToOriginID.end())
    return getOrCreate(D);

  return It->second;
}

OriginID OriginManager::getOrCreate(const Expr &E) {
  auto It = ExprToOriginID.find(&E);
  if (It != ExprToOriginID.end())
    return It->second;

  OriginID NewID = getNextOriginID();
  addOrigin(NewID, E);
  ExprToOriginID[&E] = NewID;
  return NewID;
}

const Origin &OriginManager::getOrigin(OriginID ID) const {
  assert(ID.Value < AllOrigins.size());
  return AllOrigins[ID.Value];
}

OriginID OriginManager::getOrCreate(const ValueDecl &D) {
  auto It = DeclToOriginID.find(&D);
  if (It != DeclToOriginID.end())
    return It->second;
  OriginID NewID = getNextOriginID();
  addOrigin(NewID, D);
  DeclToOriginID[&D] = NewID;
  return NewID;
}

} // namespace clang::lifetimes::internal
