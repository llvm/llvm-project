//===-- DataflowAnalysisContext.cpp -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines a DataflowAnalysisContext class that owns objects that
//  encompass the state of a program and stores context that is used during
//  dataflow analysis.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/FlowSensitive/DataflowAnalysisContext.h"
#include "clang/AST/ExprCXX.h"
#include "clang/Analysis/FlowSensitive/DebugSupport.h"
#include "clang/Analysis/FlowSensitive/Value.h"
#include "llvm/Support/Debug.h"
#include <cassert>
#include <memory>
#include <utility>

namespace clang {
namespace dataflow {

StorageLocation &
DataflowAnalysisContext::getStableStorageLocation(QualType Type) {
  if (!Type.isNull() &&
      (Type->isStructureOrClassType() || Type->isUnionType())) {
    // FIXME: Explore options to avoid eager initialization of fields as some of
    // them might not be needed for a particular analysis.
    llvm::DenseMap<const ValueDecl *, StorageLocation *> FieldLocs;
    for (const FieldDecl *Field : getObjectFields(Type))
      FieldLocs.insert({Field, &getStableStorageLocation(Field->getType())});
    return takeOwnership(
        std::make_unique<AggregateStorageLocation>(Type, std::move(FieldLocs)));
  }
  return takeOwnership(std::make_unique<ScalarStorageLocation>(Type));
}

StorageLocation &
DataflowAnalysisContext::getStableStorageLocation(const VarDecl &D) {
  if (auto *Loc = getStorageLocation(D))
    return *Loc;
  auto &Loc = getStableStorageLocation(D.getType());
  setStorageLocation(D, Loc);
  return Loc;
}

StorageLocation &
DataflowAnalysisContext::getStableStorageLocation(const Expr &E) {
  if (auto *Loc = getStorageLocation(E))
    return *Loc;
  auto &Loc = getStableStorageLocation(E.getType());
  setStorageLocation(E, Loc);
  return Loc;
}

PointerValue &
DataflowAnalysisContext::getOrCreateNullPointerValue(QualType PointeeType) {
  auto CanonicalPointeeType =
      PointeeType.isNull() ? PointeeType : PointeeType.getCanonicalType();
  auto Res = NullPointerVals.try_emplace(CanonicalPointeeType, nullptr);
  if (Res.second) {
    auto &PointeeLoc = getStableStorageLocation(CanonicalPointeeType);
    Res.first->second =
        &takeOwnership(std::make_unique<PointerValue>(PointeeLoc));
  }
  return *Res.first->second;
}

static std::pair<BoolValue *, BoolValue *>
makeCanonicalBoolValuePair(BoolValue &LHS, BoolValue &RHS) {
  auto Res = std::make_pair(&LHS, &RHS);
  if (&RHS < &LHS)
    std::swap(Res.first, Res.second);
  return Res;
}

BoolValue &DataflowAnalysisContext::getOrCreateConjunction(BoolValue &LHS,
                                                           BoolValue &RHS) {
  if (&LHS == &RHS)
    return LHS;

  auto Res = ConjunctionVals.try_emplace(makeCanonicalBoolValuePair(LHS, RHS),
                                         nullptr);
  if (Res.second)
    Res.first->second =
        &takeOwnership(std::make_unique<ConjunctionValue>(LHS, RHS));
  return *Res.first->second;
}

BoolValue &DataflowAnalysisContext::getOrCreateDisjunction(BoolValue &LHS,
                                                           BoolValue &RHS) {
  if (&LHS == &RHS)
    return LHS;

  auto Res = DisjunctionVals.try_emplace(makeCanonicalBoolValuePair(LHS, RHS),
                                         nullptr);
  if (Res.second)
    Res.first->second =
        &takeOwnership(std::make_unique<DisjunctionValue>(LHS, RHS));
  return *Res.first->second;
}

BoolValue &DataflowAnalysisContext::getOrCreateNegation(BoolValue &Val) {
  auto Res = NegationVals.try_emplace(&Val, nullptr);
  if (Res.second)
    Res.first->second = &takeOwnership(std::make_unique<NegationValue>(Val));
  return *Res.first->second;
}

BoolValue &DataflowAnalysisContext::getOrCreateImplication(BoolValue &LHS,
                                                           BoolValue &RHS) {
  return &LHS == &RHS ? getBoolLiteralValue(true)
                      : getOrCreateDisjunction(getOrCreateNegation(LHS), RHS);
}

BoolValue &DataflowAnalysisContext::getOrCreateIff(BoolValue &LHS,
                                                   BoolValue &RHS) {
  return &LHS == &RHS
             ? getBoolLiteralValue(true)
             : getOrCreateConjunction(getOrCreateImplication(LHS, RHS),
                                      getOrCreateImplication(RHS, LHS));
}

AtomicBoolValue &DataflowAnalysisContext::makeFlowConditionToken() {
  return createAtomicBoolValue();
}

void DataflowAnalysisContext::addFlowConditionConstraint(
    AtomicBoolValue &Token, BoolValue &Constraint) {
  auto Res = FlowConditionConstraints.try_emplace(&Token, &Constraint);
  if (!Res.second) {
    Res.first->second = &getOrCreateConjunction(*Res.first->second, Constraint);
  }
}

AtomicBoolValue &
DataflowAnalysisContext::forkFlowCondition(AtomicBoolValue &Token) {
  auto &ForkToken = makeFlowConditionToken();
  FlowConditionDeps[&ForkToken].insert(&Token);
  addFlowConditionConstraint(ForkToken, Token);
  return ForkToken;
}

AtomicBoolValue &
DataflowAnalysisContext::joinFlowConditions(AtomicBoolValue &FirstToken,
                                            AtomicBoolValue &SecondToken) {
  auto &Token = makeFlowConditionToken();
  FlowConditionDeps[&Token].insert(&FirstToken);
  FlowConditionDeps[&Token].insert(&SecondToken);
  addFlowConditionConstraint(Token,
                             getOrCreateDisjunction(FirstToken, SecondToken));
  return Token;
}

Solver::Result
DataflowAnalysisContext::querySolver(llvm::DenseSet<BoolValue *> Constraints) {
  Constraints.insert(&getBoolLiteralValue(true));
  Constraints.insert(&getOrCreateNegation(getBoolLiteralValue(false)));
  return S->solve(std::move(Constraints));
}

bool DataflowAnalysisContext::flowConditionImplies(AtomicBoolValue &Token,
                                                   BoolValue &Val) {
  // Returns true if and only if truth assignment of the flow condition implies
  // that `Val` is also true. We prove whether or not this property holds by
  // reducing the problem to satisfiability checking. In other words, we attempt
  // to show that assuming `Val` is false makes the constraints induced by the
  // flow condition unsatisfiable.
  llvm::DenseSet<BoolValue *> Constraints = {&Token, &getOrCreateNegation(Val)};
  llvm::DenseSet<AtomicBoolValue *> VisitedTokens;
  addTransitiveFlowConditionConstraints(Token, Constraints, VisitedTokens);
  return isUnsatisfiable(std::move(Constraints));
}

bool DataflowAnalysisContext::flowConditionIsTautology(AtomicBoolValue &Token) {
  // Returns true if and only if we cannot prove that the flow condition can
  // ever be false.
  llvm::DenseSet<BoolValue *> Constraints = {&getOrCreateNegation(Token)};
  llvm::DenseSet<AtomicBoolValue *> VisitedTokens;
  addTransitiveFlowConditionConstraints(Token, Constraints, VisitedTokens);
  return isUnsatisfiable(std::move(Constraints));
}

bool DataflowAnalysisContext::equivalentBoolValues(BoolValue &Val1,
                                                   BoolValue &Val2) {
  llvm::DenseSet<BoolValue *> Constraints = {
      &getOrCreateNegation(getOrCreateIff(Val1, Val2))};
  return isUnsatisfiable(Constraints);
}

void DataflowAnalysisContext::addTransitiveFlowConditionConstraints(
    AtomicBoolValue &Token, llvm::DenseSet<BoolValue *> &Constraints,
    llvm::DenseSet<AtomicBoolValue *> &VisitedTokens) {
  auto Res = VisitedTokens.insert(&Token);
  if (!Res.second)
    return;

  auto ConstraintsIT = FlowConditionConstraints.find(&Token);
  if (ConstraintsIT == FlowConditionConstraints.end()) {
    Constraints.insert(&Token);
  } else {
    // Bind flow condition token via `iff` to its set of constraints:
    // FC <=> (C1 ^ C2 ^ ...), where Ci are constraints
    Constraints.insert(&getOrCreateIff(Token, *ConstraintsIT->second));
  }

  auto DepsIT = FlowConditionDeps.find(&Token);
  if (DepsIT != FlowConditionDeps.end()) {
    for (AtomicBoolValue *DepToken : DepsIT->second) {
      addTransitiveFlowConditionConstraints(*DepToken, Constraints,
                                            VisitedTokens);
    }
  }
}

BoolValue &DataflowAnalysisContext::substituteBoolValue(
    BoolValue &Val,
    llvm::DenseMap<BoolValue *, BoolValue *> &SubstitutionsCache) {
  auto IT = SubstitutionsCache.find(&Val);
  if (IT != SubstitutionsCache.end()) {
    // Return memoized result of substituting this boolean value.
    return *IT->second;
  }

  // Handle substitution on the boolean value (and its subvalues), saving the
  // result into `SubstitutionsCache`.
  BoolValue *Result;
  switch (Val.getKind()) {
  case Value::Kind::AtomicBool: {
    Result = &Val;
    break;
  }
  case Value::Kind::Negation: {
    auto &Negation = *cast<NegationValue>(&Val);
    auto &Sub = substituteBoolValue(Negation.getSubVal(), SubstitutionsCache);
    Result = &getOrCreateNegation(Sub);
    break;
  }
  case Value::Kind::Disjunction: {
    auto &Disjunct = *cast<DisjunctionValue>(&Val);
    auto &LeftSub =
        substituteBoolValue(Disjunct.getLeftSubValue(), SubstitutionsCache);
    auto &RightSub =
        substituteBoolValue(Disjunct.getRightSubValue(), SubstitutionsCache);
    Result = &getOrCreateDisjunction(LeftSub, RightSub);
    break;
  }
  case Value::Kind::Conjunction: {
    auto &Conjunct = *cast<ConjunctionValue>(&Val);
    auto &LeftSub =
        substituteBoolValue(Conjunct.getLeftSubValue(), SubstitutionsCache);
    auto &RightSub =
        substituteBoolValue(Conjunct.getRightSubValue(), SubstitutionsCache);
    Result = &getOrCreateConjunction(LeftSub, RightSub);
    break;
  }
  default:
    llvm_unreachable("Unhandled Value Kind");
  }
  SubstitutionsCache[&Val] = Result;
  return *Result;
}

BoolValue &DataflowAnalysisContext::buildAndSubstituteFlowCondition(
    AtomicBoolValue &Token,
    llvm::DenseMap<AtomicBoolValue *, BoolValue *> Substitutions) {
  assert(
      Substitutions.find(&getBoolLiteralValue(true)) == Substitutions.end() &&
      Substitutions.find(&getBoolLiteralValue(false)) == Substitutions.end() &&
      "Do not substitute true/false boolean literals");
  llvm::DenseMap<BoolValue *, BoolValue *> SubstitutionsCache(
      Substitutions.begin(), Substitutions.end());
  return buildAndSubstituteFlowConditionWithCache(Token, SubstitutionsCache);
}

BoolValue &DataflowAnalysisContext::buildAndSubstituteFlowConditionWithCache(
    AtomicBoolValue &Token,
    llvm::DenseMap<BoolValue *, BoolValue *> &SubstitutionsCache) {
  auto ConstraintsIT = FlowConditionConstraints.find(&Token);
  if (ConstraintsIT == FlowConditionConstraints.end()) {
    return getBoolLiteralValue(true);
  }
  auto DepsIT = FlowConditionDeps.find(&Token);
  if (DepsIT != FlowConditionDeps.end()) {
    for (AtomicBoolValue *DepToken : DepsIT->second) {
      auto &NewDep = buildAndSubstituteFlowConditionWithCache(
          *DepToken, SubstitutionsCache);
      SubstitutionsCache[DepToken] = &NewDep;
    }
  }
  return substituteBoolValue(*ConstraintsIT->second, SubstitutionsCache);
}

void DataflowAnalysisContext::dumpFlowCondition(AtomicBoolValue &Token) {
  llvm::DenseSet<BoolValue *> Constraints = {&Token};
  llvm::DenseSet<AtomicBoolValue *> VisitedTokens;
  addTransitiveFlowConditionConstraints(Token, Constraints, VisitedTokens);

  llvm::DenseMap<const AtomicBoolValue *, std::string> AtomNames = {
      {&getBoolLiteralValue(false), "False"},
      {&getBoolLiteralValue(true), "True"}};
  llvm::dbgs() << debugString(Constraints, AtomNames);
}

} // namespace dataflow
} // namespace clang

using namespace clang;

const Expr &clang::dataflow::ignoreCFGOmittedNodes(const Expr &E) {
  const Expr *Current = &E;
  if (auto *EWC = dyn_cast<ExprWithCleanups>(Current)) {
    Current = EWC->getSubExpr();
    assert(Current != nullptr);
  }
  Current = Current->IgnoreParens();
  assert(Current != nullptr);
  return *Current;
}

const Stmt &clang::dataflow::ignoreCFGOmittedNodes(const Stmt &S) {
  if (auto *E = dyn_cast<Expr>(&S))
    return ignoreCFGOmittedNodes(*E);
  return S;
}

// FIXME: Does not precisely handle non-virtual diamond inheritance. A single
// field decl will be modeled for all instances of the inherited field.
static void
getFieldsFromClassHierarchy(QualType Type,
                            llvm::DenseSet<const FieldDecl *> &Fields) {
  if (Type->isIncompleteType() || Type->isDependentType() ||
      !Type->isRecordType())
    return;

  for (const FieldDecl *Field : Type->getAsRecordDecl()->fields())
    Fields.insert(Field);
  if (auto *CXXRecord = Type->getAsCXXRecordDecl())
    for (const CXXBaseSpecifier &Base : CXXRecord->bases())
      getFieldsFromClassHierarchy(Base.getType(), Fields);
}

/// Gets the set of all fields in the type.
llvm::DenseSet<const FieldDecl *>
clang::dataflow::getObjectFields(QualType Type) {
  llvm::DenseSet<const FieldDecl *> Fields;
  getFieldsFromClassHierarchy(Type, Fields);
  return Fields;
}
