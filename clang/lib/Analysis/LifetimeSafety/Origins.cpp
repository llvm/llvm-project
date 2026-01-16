//===- Origins.cpp - Origin Implementation -----------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Analyses/LifetimeSafety/Origins.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/TypeBase.h"
#include "clang/Analysis/Analyses/LifetimeSafety/LifetimeAnnotations.h"
#include "clang/Analysis/Analyses/LifetimeSafety/LifetimeStats.h"
#include "llvm/ADT/StringMap.h"

namespace clang::lifetimes::internal {
namespace {
/// A utility class to traverse the function body in the analysis
/// context and collect the count of expressions with missing origins.
class MissingOriginCollector
    : public RecursiveASTVisitor<MissingOriginCollector> {
public:
  MissingOriginCollector(
      const llvm::DenseMap<const clang::Expr *, OriginList *> &ExprToOriginList,
      LifetimeSafetyStats &LSStats)
      : ExprToOriginList(ExprToOriginList), LSStats(LSStats) {}
  bool VisitExpr(Expr *E) {
    if (!hasOrigins(E))
      return true;
    // Check if we have an origin for this expression.
    if (!ExprToOriginList.contains(E)) {
      // No origin found: count this as missing origin.
      LSStats.ExprTypeToMissingOriginCount[E->getType().getTypePtr()]++;
      LSStats.ExprStmtClassToMissingOriginCount[std::string(
          E->getStmtClassName())]++;
    }
    return true;
  }

private:
  const llvm::DenseMap<const clang::Expr *, OriginList *> &ExprToOriginList;
  LifetimeSafetyStats &LSStats;
};
} // namespace

bool hasOrigins(QualType QT) {
  return QT->isPointerOrReferenceType() || isGslPointerType(QT);
}

/// Determines if an expression has origins that need to be tracked.
///
/// An expression has origins if:
/// - It's a glvalue (has addressable storage), OR
/// - Its type is pointer-like (pointer, reference, or gsl::Pointer)
///
/// Examples:
/// - `int x; x` : has origin (glvalue)
/// - `int* p; p` : has 2 origins (1 for glvalue and 1 for pointer type)
/// - `std::string_view{}` : has 1 origin (prvalue of pointer type)
/// - `42` : no origin (prvalue of non-pointer type)
/// - `x + y` : (where x, y are int) → no origin (prvalue of non-pointer type)
bool hasOrigins(const Expr *E) {
  return E->isGLValue() || hasOrigins(E->getType());
}

/// Returns true if the declaration has its own storage that can be borrowed.
///
/// References generally have no storage - they are aliases to other storage.
/// For example:
///   int x;      // has storage (can issue loans to x's storage)
///   int& r = x; // no storage (r is an alias to x's storage)
///   int* p;     // has storage (the pointer variable p itself has storage)
///
/// TODO: Handle lifetime extension. References initialized by temporaries
/// can have storage when the temporary's lifetime is extended:
///   const int& r = 42; // temporary has storage, lifetime extended
///   Foo&& f = Foo{};   // temporary has storage, lifetime extended
/// Currently, this function returns false for all reference types.
bool doesDeclHaveStorage(const ValueDecl *D) {
  return !D->getType()->isReferenceType();
}

OriginList *OriginManager::createNode(const ValueDecl *D, QualType QT) {
  OriginID NewID = getNextOriginID();
  AllOrigins.emplace_back(NewID, D, QT.getTypePtrOrNull());
  return new (ListAllocator.Allocate<OriginList>()) OriginList(NewID);
}

OriginList *OriginManager::createNode(const Expr *E, QualType QT) {
  OriginID NewID = getNextOriginID();
  AllOrigins.emplace_back(NewID, E, QT.getTypePtrOrNull());
  return new (ListAllocator.Allocate<OriginList>()) OriginList(NewID);
}

template <typename T>
OriginList *OriginManager::buildListForType(QualType QT, const T *Node) {
  assert(hasOrigins(QT) && "buildListForType called for non-pointer type");
  OriginList *Head = createNode(Node, QT);

  if (QT->isPointerOrReferenceType()) {
    QualType PointeeTy = QT->getPointeeType();
    // We recurse if the pointee type is pointer-like, to build the next
    // level in the origin tree. E.g., for T*& / View&.
    if (hasOrigins(PointeeTy))
      Head->setInnerOriginList(buildListForType(PointeeTy, Node));
  }
  return Head;
}

OriginList *OriginManager::getOrCreateList(const ValueDecl *D) {
  if (!hasOrigins(D->getType()))
    return nullptr;
  auto It = DeclToList.find(D);
  if (It != DeclToList.end())
    return It->second;
  return DeclToList[D] = buildListForType(D->getType(), D);
}

OriginList *OriginManager::getOrCreateList(const Expr *E) {
  if (auto *ParenIgnored = E->IgnoreParens(); ParenIgnored != E)
    return getOrCreateList(ParenIgnored);
  // We do not see CFG stmts for ExprWithCleanups. Simply peel them.
  if (const ExprWithCleanups *EWC = dyn_cast<ExprWithCleanups>(E))
    return getOrCreateList(EWC->getSubExpr());

  if (!hasOrigins(E))
    return nullptr;

  auto It = ExprToList.find(E);
  if (It != ExprToList.end())
    return It->second;

  QualType Type = E->getType();

  // Special handling for DeclRefExpr to share origins with the underlying decl.
  if (auto *DRE = dyn_cast<DeclRefExpr>(E)) {
    OriginList *Head = nullptr;
    // For non-reference declarations (e.g., `int* p`), the DeclRefExpr is an
    // lvalue (addressable) that can be borrowed, so we create an outer origin
    // for the lvalue itself, with the pointee being the declaration's list.
    // This models taking the address: `&p` borrows the storage of `p`, not what
    // `p` points to.
    if (doesDeclHaveStorage(DRE->getDecl())) {
      Head = createNode(DRE, QualType{});
      // This ensures origin sharing: multiple DeclRefExprs to the same
      // declaration share the same underlying origins.
      Head->setInnerOriginList(getOrCreateList(DRE->getDecl()));
    } else {
      // For reference-typed declarations (e.g., `int& r = p`) which have no
      // storage, the DeclRefExpr directly reuses the declaration's list since
      // references don't add an extra level of indirection at the expression
      // level.
      Head = getOrCreateList(DRE->getDecl());
    }
    return ExprToList[E] = Head;
  }

  // If E is an lvalue , it refers to storage. We model this storage as the
  // first level of origin list, as if it were a reference, because l-values are
  // addressable.
  if (E->isGLValue() && !Type->isReferenceType())
    Type = AST.getLValueReferenceType(Type);
  return ExprToList[E] = buildListForType(Type, E);
}

void OriginManager::dump(OriginID OID, llvm::raw_ostream &OS) const {
  OS << OID << " (";
  Origin O = getOrigin(OID);
  if (const ValueDecl *VD = O.getDecl()) {
    OS << "Decl: " << VD->getNameAsString();
  } else if (const Expr *E = O.getExpr()) {
    OS << "Expr: " << E->getStmtClassName();
    if (auto *DRE = dyn_cast<DeclRefExpr>(E)) {
      if (const ValueDecl *VD = DRE->getDecl())
        OS << ", Decl: " << VD->getNameAsString();
    }
  } else {
    OS << "Unknown";
  }
  if (O.Ty)
    OS << ", Type : " << QualType(O.Ty, 0).getAsString();
  OS << ")";
}

const Origin &OriginManager::getOrigin(OriginID ID) const {
  assert(ID.Value < AllOrigins.size());
  return AllOrigins[ID.Value];
}

void OriginManager::collectMissingOrigins(Stmt &FunctionBody,
                                          LifetimeSafetyStats &LSStats) {
  MissingOriginCollector Collector(this->ExprToList, LSStats);
  Collector.TraverseStmt(const_cast<Stmt *>(&FunctionBody));
}

} // namespace clang::lifetimes::internal
