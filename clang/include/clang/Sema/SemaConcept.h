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
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprConcepts.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/Ownership.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>
#include <utility>

namespace clang {
class Sema;
class MultiLevelTemplateArgumentList;

/// \brief A normalized constraint, as defined in C++ [temp.constr.normal], is
/// either an atomic constraint, a conjunction of normalized constraints or a
/// disjunction of normalized constraints.
struct NormalizedConstraint {

  enum class ConstraintKind : unsigned char {
    Atomic = 0,
    ConceptId,
    FoldExpanded,
    Compound,
  };

  enum CompoundConstraintKind : unsigned char {
    CCK_Conjunction,
    CCK_Disjunction
  };
  enum class FoldOperatorKind : unsigned char { And, Or };

  using OccurenceList = llvm::SmallBitVector;

protected:
  using ExprOrConcept =
      llvm::PointerUnion<const Expr *, const ConceptReference *>;

  struct AtomicConstraintBits {
    // Kind is the first member of all union members,
    // as we rely on their initial common sequence.
    LLVM_PREFERRED_TYPE(ConstraintKind)
    unsigned Kind : 5;
    unsigned Placeholder : 1;
    unsigned PackSubstitutionIndex : 26;
    // Indexes, IndexesForSubsumption, and Args are part of the common initial
    // sequences of constraints that do have a mapping.

    // Indexes of the parameters used in a constraint expression.
    OccurenceList Indexes;
    // Indexes of the parameters named directly in a constraint expression.
    // FIXME: we should try to reduce the size of this struct?
    OccurenceList IndexesForSubsumption;

    TemplateArgumentLoc *Args;
    TemplateParameterList *ParamList;
    ExprOrConcept ConstraintExpr;
    const NamedDecl *ConstraintDecl;
  };

  struct FoldExpandedConstraintBits {
    LLVM_PREFERRED_TYPE(ConstraintKind)
    unsigned Kind : 5;
    LLVM_PREFERRED_TYPE(FoldOperatorKind)
    unsigned FoldOperator : 1;
    unsigned Placeholder : 26;
    OccurenceList Indexes;
    OccurenceList IndexesForSubsumption;
    TemplateArgumentLoc *Args;
    TemplateParameterList *ParamList;
    const Expr *Pattern;
    const NamedDecl *ConstraintDecl;
    NormalizedConstraint *Constraint;
  };

  struct ConceptIdBits : AtomicConstraintBits {
    NormalizedConstraint *Sub;

    // Only used for parameter mapping.
    const ConceptSpecializationExpr *CSE;
  };

  struct CompoundConstraintBits {
    LLVM_PREFERRED_TYPE(ConstraintKind)
    unsigned Kind : 5;
    LLVM_PREFERRED_TYPE(CompoundConstraintKind)
    unsigned CCK : 1;
    NormalizedConstraint *LHS;
    NormalizedConstraint *RHS;
  };

  union {
    AtomicConstraintBits Atomic;
    FoldExpandedConstraintBits FoldExpanded;
    ConceptIdBits ConceptId;
    CompoundConstraintBits Compound;
  };

  ~NormalizedConstraint() {
    if (getKind() != ConstraintKind::Compound)
      Atomic.Indexes.llvm::SmallBitVector::~SmallBitVector();
  }

  NormalizedConstraint(const Expr *ConstraintExpr,
                       const NamedDecl *ConstraintDecl,
                       UnsignedOrNone PackIndex)
      : Atomic{llvm::to_underlying(ConstraintKind::Atomic),
               /*Placeholder=*/0,
               PackIndex.toInternalRepresentation(),
               /*Indexes=*/{},
               /*IndexesForSubsumption=*/{},
               /*Args=*/nullptr,
               /*ParamList=*/nullptr,
               ConstraintExpr,
               ConstraintDecl} {}

  NormalizedConstraint(const Expr *Pattern, FoldOperatorKind OpKind,
                       NormalizedConstraint *Constraint,
                       const NamedDecl *ConstraintDecl)
      : FoldExpanded{llvm::to_underlying(ConstraintKind::FoldExpanded),
                     llvm::to_underlying(OpKind),
                     /*Placeholder=*/0,
                     /*Indexes=*/{},
                     /*IndexesForSubsumption=*/{},
                     /*Args=*/nullptr,
                     /*ParamList=*/nullptr,
                     Pattern,
                     ConstraintDecl,
                     Constraint} {}

  NormalizedConstraint(const ConceptReference *ConceptId,
                       const NamedDecl *ConstraintDecl,
                       NormalizedConstraint *SubConstraint,
                       const ConceptSpecializationExpr *CSE,
                       UnsignedOrNone PackIndex)
      : ConceptId{{llvm::to_underlying(ConstraintKind::ConceptId),
                   /*Placeholder=*/0, PackIndex.toInternalRepresentation(),
                   /*Indexes=*/{},
                   /*IndexesForSubsumption=*/{},
                   /*Args=*/nullptr, /*ParamList=*/nullptr, ConceptId,
                   ConstraintDecl},
                  SubConstraint,
                  CSE} {}

  NormalizedConstraint(NormalizedConstraint *LHS, CompoundConstraintKind CCK,
                       NormalizedConstraint *RHS)
      : Compound{llvm::to_underlying(ConstraintKind::Compound),
                 llvm::to_underlying(CCK), LHS, RHS} {}

  bool hasParameterMapping() const {
    // compound constraints do not have a mapping
    // and Args is not part of their common initial sequence.
    return getKind() != ConstraintKind::Compound && Atomic.Args != nullptr;
  }

  const OccurenceList &mappingOccurenceList() const {
    assert(hasParameterMapping() && "This constraint has no parameter mapping");
    return Atomic.Indexes;
  }

  const OccurenceList &mappingOccurenceListForSubsumption() const {
    assert(hasParameterMapping() && "This constraint has no parameter mapping");
    return Atomic.IndexesForSubsumption;
  }

  llvm::MutableArrayRef<TemplateArgumentLoc> getParameterMapping() const {
    return {Atomic.Args, Atomic.Indexes.count()};
  }

  TemplateParameterList *getUsedTemplateParamList() const {
    return Atomic.ParamList;
  }

  void updateParameterMapping(OccurenceList Indexes,
                              OccurenceList IndexesForSubsumption,
                              llvm::MutableArrayRef<TemplateArgumentLoc> Args,
                              TemplateParameterList *ParamList) {
    assert(getKind() != ConstraintKind::Compound);
    assert(Indexes.count() == Args.size());
    assert(IndexesForSubsumption.size() == Indexes.size());
    assert((Indexes | IndexesForSubsumption) == Indexes);

    Atomic.IndexesForSubsumption = std::move(IndexesForSubsumption);
    Atomic.Indexes = std::move(Indexes);
    Atomic.Args = Args.data();
    Atomic.ParamList = ParamList;
  }

  bool hasMatchingParameterMapping(ASTContext &C,
                                   const NormalizedConstraint &Other) const {
    assert(getKind() != ConstraintKind::Compound);

    if (hasParameterMapping() != Other.hasParameterMapping())
      return false;
    if (!hasParameterMapping())
      return true;

    llvm::ArrayRef<TemplateArgumentLoc> ParameterMapping =
        getParameterMapping();
    llvm::ArrayRef<TemplateArgumentLoc> OtherParameterMapping =
        Other.getParameterMapping();

    const OccurenceList &Indexes = mappingOccurenceListForSubsumption();
    const OccurenceList &OtherIndexes =
        Other.mappingOccurenceListForSubsumption();

    if (ParameterMapping.size() != OtherParameterMapping.size())
      return false;
    for (unsigned I = 0, S = ParameterMapping.size(); I < S; ++I) {
      if (Indexes[I] != OtherIndexes[I])
        return false;
      if (!Indexes[I])
        continue;
      llvm::FoldingSetNodeID IDA, IDB;
      C.getCanonicalTemplateArgument(ParameterMapping[I].getArgument())
          .Profile(IDA, C);
      C.getCanonicalTemplateArgument(OtherParameterMapping[I].getArgument())
          .Profile(IDB, C);
      if (IDA != IDB)
        return false;
    }
    return true;
  }

public:
  ConstraintKind getKind() const {
    return static_cast<ConstraintKind>(Atomic.Kind);
  }

  SourceLocation getBeginLoc() const {
    switch (getKind()) {
    case ConstraintKind::Atomic:
      return cast<const Expr *>(Atomic.ConstraintExpr)->getBeginLoc();
    case ConstraintKind::ConceptId:
      return cast<const ConceptReference *>(Atomic.ConstraintExpr)
          ->getBeginLoc();
    case ConstraintKind::Compound:
      return Compound.LHS->getBeginLoc();
    case ConstraintKind::FoldExpanded:
      return FoldExpanded.Pattern->getBeginLoc();
    }
    llvm_unreachable("Unknown ConstraintKind enum");
  }

  SourceLocation getEndLoc() const {
    switch (getKind()) {
    case ConstraintKind::Atomic:
      return cast<const Expr *>(Atomic.ConstraintExpr)->getEndLoc();
    case ConstraintKind::ConceptId:
      return cast<const ConceptReference *>(Atomic.ConstraintExpr)->getEndLoc();
    case ConstraintKind::Compound:
      return Compound.RHS->getEndLoc();
    case ConstraintKind::FoldExpanded:
      return FoldExpanded.Pattern->getEndLoc();
    }
    llvm_unreachable("Unknown ConstraintKind enum");
  }

  SourceRange getSourceRange() const { return {getBeginLoc(), getEndLoc()}; }

private:
  friend class Sema;
  static NormalizedConstraint *
  fromAssociatedConstraints(Sema &S, const NamedDecl *D,
                            ArrayRef<AssociatedConstraint> ACs);
  static NormalizedConstraint *fromConstraintExpr(Sema &S, const NamedDecl *D,
                                                  const Expr *E,
                                                  UnsignedOrNone SubstIndex);
};

class CompoundConstraint : public NormalizedConstraint {
  using NormalizedConstraint::NormalizedConstraint;

public:
  static CompoundConstraint *Create(ASTContext &Ctx, NormalizedConstraint *LHS,
                                    CompoundConstraintKind CCK,
                                    NormalizedConstraint *RHS) {
    return new (Ctx) CompoundConstraint(LHS, CCK, RHS);
  }

  static CompoundConstraint *CreateConjunction(ASTContext &Ctx,
                                               NormalizedConstraint *LHS,
                                               NormalizedConstraint *RHS) {
    return new (Ctx) CompoundConstraint(LHS, CCK_Conjunction, RHS);
  }

  const NormalizedConstraint &getLHS() const { return *Compound.LHS; }

  NormalizedConstraint &getLHS() { return *Compound.LHS; }

  const NormalizedConstraint &getRHS() const { return *Compound.RHS; }

  NormalizedConstraint &getRHS() { return *Compound.RHS; }

  CompoundConstraintKind getCompoundKind() const {
    return static_cast<CompoundConstraintKind>(Compound.CCK);
  }
};

class NormalizedConstraintWithParamMapping : public NormalizedConstraint {
protected:
  using NormalizedConstraint::NormalizedConstraint;

public:
  using NormalizedConstraint::getParameterMapping;
  using NormalizedConstraint::getUsedTemplateParamList;
  using NormalizedConstraint::hasMatchingParameterMapping;
  using NormalizedConstraint::hasParameterMapping;
  using NormalizedConstraint::mappingOccurenceList;
  using NormalizedConstraint::mappingOccurenceListForSubsumption;
  using NormalizedConstraint::updateParameterMapping;

  const NamedDecl *getConstraintDecl() const { return Atomic.ConstraintDecl; }

  UnsignedOrNone getPackSubstitutionIndex() const {
    return UnsignedOrNone::fromInternalRepresentation(
        Atomic.PackSubstitutionIndex);
  }
};

class AtomicConstraint : public NormalizedConstraintWithParamMapping {
  using NormalizedConstraintWithParamMapping::
      NormalizedConstraintWithParamMapping;

public:
  static AtomicConstraint *Create(ASTContext &Ctx, const Expr *ConstraintExpr,
                                  const NamedDecl *ConstraintDecl,
                                  UnsignedOrNone PackIndex) {
    return new (Ctx)
        AtomicConstraint(ConstraintExpr, ConstraintDecl, PackIndex);
  }

  const Expr *getConstraintExpr() const {
    return cast<const Expr *>(Atomic.ConstraintExpr);
  }
};

class FoldExpandedConstraint : public NormalizedConstraintWithParamMapping {
  using NormalizedConstraintWithParamMapping::
      NormalizedConstraintWithParamMapping;

public:
  static FoldExpandedConstraint *Create(ASTContext &Ctx, const Expr *Pattern,
                                        const NamedDecl *ConstraintDecl,
                                        FoldOperatorKind OpKind,
                                        NormalizedConstraint *Constraint) {
    return new (Ctx)
        FoldExpandedConstraint(Pattern, OpKind, Constraint, ConstraintDecl);
  }

  using NormalizedConstraint::hasMatchingParameterMapping;

  FoldOperatorKind getFoldOperator() const {
    return static_cast<FoldOperatorKind>(FoldExpanded.FoldOperator);
  }

  const Expr *getPattern() const { return FoldExpanded.Pattern; }

  const NormalizedConstraint &getNormalizedPattern() const {
    return *FoldExpanded.Constraint;
  }

  NormalizedConstraint &getNormalizedPattern() {
    return *FoldExpanded.Constraint;
  }

  static bool AreCompatibleForSubsumption(const FoldExpandedConstraint &A,
                                          const FoldExpandedConstraint &B);
};

class ConceptIdConstraint : public NormalizedConstraintWithParamMapping {
  using NormalizedConstraintWithParamMapping::
      NormalizedConstraintWithParamMapping;

public:
  static ConceptIdConstraint *
  Create(ASTContext &Ctx, const ConceptReference *ConceptId,
         NormalizedConstraint *SubConstraint, const NamedDecl *ConstraintDecl,
         const ConceptSpecializationExpr *CSE, UnsignedOrNone PackIndex) {
    return new (Ctx) ConceptIdConstraint(ConceptId, ConstraintDecl,
                                         SubConstraint, CSE, PackIndex);
  }

  const ConceptSpecializationExpr *getConceptSpecializationExpr() const {
    return ConceptId.CSE;
  }

  const ConceptReference *getConceptId() const {
    return cast<const ConceptReference *>(ConceptId.ConstraintExpr);
  }

  const NormalizedConstraint &getNormalizedConstraint() const {
    return *ConceptId.Sub;
  }

  NormalizedConstraint &getNormalizedConstraint() { return *ConceptId.Sub; }
};

struct UnsubstitutedConstraintSatisfactionCacheResult {
  ExprResult SubstExpr;
  ConstraintSatisfaction Satisfaction;
};

/// \brief SubsumptionChecker establishes subsumption
/// between two set of constraints.
class SubsumptionChecker {
public:
  using SubsumptionCallable = llvm::function_ref<bool(
      const AtomicConstraint &, const AtomicConstraint &)>;

  SubsumptionChecker(Sema &SemaRef, SubsumptionCallable Callable = {});

  std::optional<bool> Subsumes(const NamedDecl *DP,
                               ArrayRef<AssociatedConstraint> P,
                               const NamedDecl *DQ,
                               ArrayRef<AssociatedConstraint> Q);

  bool Subsumes(const NormalizedConstraint *P, const NormalizedConstraint *Q);

private:
  Sema &SemaRef;
  SubsumptionCallable Callable;

  // Each Literal has a unique value that is enough to establish
  // its identity.
  // Some constraints (fold expended) require special subsumption
  // handling logic beyond comparing values, so we store a flag
  // to let us quickly dispatch to each kind of variable.
  struct Literal {
    enum Kind { Atomic, FoldExpanded };

    unsigned Value : 16;
    LLVM_PREFERRED_TYPE(Kind)
    unsigned Kind : 1;

    bool operator==(const Literal &Other) const { return Value == Other.Value; }
    bool operator<(const Literal &Other) const { return Value < Other.Value; }
  };
  using Clause = llvm::SmallVector<Literal>;
  using Formula = llvm::SmallVector<Clause, 5>;

  struct CNFFormula : Formula {
    static constexpr auto Kind = NormalizedConstraint::CCK_Conjunction;
    using Formula::Formula;
  };
  struct DNFFormula : Formula {
    static constexpr auto Kind = NormalizedConstraint::CCK_Disjunction;
    using Formula::Formula;
  };

  struct MappedAtomicConstraint {
    const AtomicConstraint *Constraint;
    Literal ID;
  };

  struct FoldExpendedConstraintKey {
    FoldExpandedConstraint::FoldOperatorKind Kind;
    const AtomicConstraint *Constraint;
    Literal ID;
  };

  llvm::DenseMap<const Expr *, llvm::SmallDenseMap<llvm::FoldingSetNodeID,
                                                   MappedAtomicConstraint>>
      AtomicMap;

  llvm::DenseMap<const Expr *, std::vector<FoldExpendedConstraintKey>> FoldMap;

  // A map from a literal to a corresponding associated constraint.
  // We do not have enough bits left for a pointer union here :(
  llvm::DenseMap<uint16_t, const void *> ReverseMap;

  // Fold expanded constraints ask us to recursively establish subsumption.
  // This caches the result.
  llvm::SmallDenseMap<
      std::pair<const FoldExpandedConstraint *, const FoldExpandedConstraint *>,
      bool>
      FoldSubsumptionCache;

  // Each <atomic, fold expanded constraint> is represented as a single ID.
  // This is intentionally kept small we can't handle a large number of
  // constraints anyway.
  uint16_t NextID;

  bool Subsumes(const DNFFormula &P, const CNFFormula &Q);
  bool Subsumes(Literal A, Literal B);
  bool Subsumes(const FoldExpandedConstraint *A,
                const FoldExpandedConstraint *B);
  bool DNFSubsumes(const Clause &P, const Clause &Q);

  CNFFormula CNF(const NormalizedConstraint &C);
  DNFFormula DNF(const NormalizedConstraint &C);

  template <typename FormulaType>
  FormulaType Normalize(const NormalizedConstraint &C);
  void AddUniqueClauseToFormula(Formula &F, Clause C);

  Literal find(const AtomicConstraint *);
  Literal find(const FoldExpandedConstraint *);

  uint16_t getNewLiteralId();
};

} // namespace clang

#endif // LLVM_CLANG_SEMA_SEMACONCEPT_H
