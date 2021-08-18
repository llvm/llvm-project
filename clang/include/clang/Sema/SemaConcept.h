//===-- SemaConcept.h - Semantic Analysis for Constraints and Concepts ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
//  This file provides semantic analysis for C++ constraints and concepts.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_SEMACONCEPT_H
#define LLVM_CLANG_SEMA_SEMACONCEPT_H
#include "clang/AST/ASTConcept.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include <string>
#include <utility>

namespace clang {
class Sema;

struct AtomicConstraint {
  const Expr *ConstraintExpr;
  Optional<MutableArrayRef<TemplateArgumentLoc>> ParameterMapping;

  AtomicConstraint(Sema &S, const Expr *ConstraintExpr) :
      ConstraintExpr(ConstraintExpr) { };

  bool hasMatchingParameterMapping(ASTContext &C,
                                   const AtomicConstraint &Other) const {
    if (!ParameterMapping != !Other.ParameterMapping)
      return false;
    if (!ParameterMapping)
      return true;
    if (ParameterMapping->size() != Other.ParameterMapping->size())
      return false;

    for (unsigned I = 0, S = ParameterMapping->size(); I < S; ++I) {
      llvm::FoldingSetNodeID IDA, IDB;
      C.getCanonicalTemplateArgument((*ParameterMapping)[I].getArgument())
          .Profile(IDA, C);
      C.getCanonicalTemplateArgument((*Other.ParameterMapping)[I].getArgument())
          .Profile(IDB, C);
      if (IDA != IDB)
        return false;
    }
    return true;
  }

  bool subsumes(ASTContext &C, const AtomicConstraint &Other) const {
    // C++ [temp.constr.order] p2
    //   - an atomic constraint A subsumes another atomic constraint B
    //     if and only if the A and B are identical [...]
    //
    // C++ [temp.constr.atomic] p2
    //   Two atomic constraints are identical if they are formed from the
    //   same expression and the targets of the parameter mappings are
    //   equivalent according to the rules for expressions [...]

    // We do not actually substitute the parameter mappings into the
    // constraint expressions, therefore the constraint expressions are
    // the originals, and comparing them will suffice.
    if (ConstraintExpr != Other.ConstraintExpr)
      return false;

    // Check that the parameter lists are identical
    return hasMatchingParameterMapping(C, Other);
  }
};

/// \brief A normalized constraint, as defined in C++ [temp.constr.normal], is
/// either an atomic constraint, a conjunction of normalized constraints or a
/// disjunction of normalized constraints.
struct NormalizedConstraint {
  friend class Sema;

  enum CompoundConstraintKind { CCK_Conjunction, CCK_Disjunction };

  using CompoundConstraint = llvm::PointerIntPair<
      std::pair<NormalizedConstraint, NormalizedConstraint> *, 1,
      CompoundConstraintKind>;

  llvm::PointerUnion<AtomicConstraint *, CompoundConstraint> Constraint;

  NormalizedConstraint(AtomicConstraint *C): Constraint{C} { };
  NormalizedConstraint(ASTContext &C, NormalizedConstraint LHS,
                       NormalizedConstraint RHS, CompoundConstraintKind Kind)
      : Constraint{CompoundConstraint{
            new (C) std::pair<NormalizedConstraint, NormalizedConstraint>{
                std::move(LHS), std::move(RHS)}, Kind}} { };

  NormalizedConstraint(ASTContext &C, const NormalizedConstraint &Other) {
    if (Other.isAtomic()) {
      Constraint = new (C) AtomicConstraint(*Other.getAtomicConstraint());
    } else {
      Constraint = CompoundConstraint(
          new (C) std::pair<NormalizedConstraint, NormalizedConstraint>{
              NormalizedConstraint(C, Other.getLHS()),
              NormalizedConstraint(C, Other.getRHS())},
              Other.getCompoundKind());
    }
  }
  NormalizedConstraint(NormalizedConstraint &&Other):
      Constraint(Other.Constraint) {
    Other.Constraint = nullptr;
  }
  NormalizedConstraint &operator=(const NormalizedConstraint &Other) = delete;
  NormalizedConstraint &operator=(NormalizedConstraint &&Other) {
    if (&Other != this) {
      NormalizedConstraint Temp(std::move(Other));
      std::swap(Constraint, Temp.Constraint);
    }
    return *this;
  }

  CompoundConstraintKind getCompoundKind() const {
    assert(!isAtomic() && "getCompoundKind called on atomic constraint.");
    return Constraint.get<CompoundConstraint>().getInt();
  }

  bool isAtomic() const { return Constraint.is<AtomicConstraint *>(); }

  NormalizedConstraint &getLHS() const {
    assert(!isAtomic() && "getLHS called on atomic constraint.");
    return Constraint.get<CompoundConstraint>().getPointer()->first;
  }

  NormalizedConstraint &getRHS() const {
    assert(!isAtomic() && "getRHS called on atomic constraint.");
    return Constraint.get<CompoundConstraint>().getPointer()->second;
  }

  AtomicConstraint *getAtomicConstraint() const {
    assert(isAtomic() &&
           "getAtomicConstraint called on non-atomic constraint.");
    return Constraint.get<AtomicConstraint *>();
  }

private:
  static Optional<NormalizedConstraint>
  fromConstraintExprs(Sema &S, NamedDecl *D, ArrayRef<const Expr *> E);
  static Optional<NormalizedConstraint>
  fromConstraintExpr(Sema &S, NamedDecl *D, const Expr *E);
};

} // clang

#endif //LLVM_CLANG_SEMA_SEMACONCEPT_H
