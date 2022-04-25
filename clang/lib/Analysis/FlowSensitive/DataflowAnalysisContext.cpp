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
#include "clang/Analysis/FlowSensitive/Value.h"
#include <cassert>
#include <memory>
#include <utility>

namespace clang {
namespace dataflow {

static std::pair<BoolValue *, BoolValue *>
makeCanonicalBoolValuePair(BoolValue &LHS, BoolValue &RHS) {
  auto Res = std::make_pair(&LHS, &RHS);
  if (&RHS < &LHS)
    std::swap(Res.first, Res.second);
  return Res;
}

BoolValue &
DataflowAnalysisContext::getOrCreateConjunctionValue(BoolValue &LHS,
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

BoolValue &
DataflowAnalysisContext::getOrCreateDisjunctionValue(BoolValue &LHS,
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

BoolValue &DataflowAnalysisContext::getOrCreateNegationValue(BoolValue &Val) {
  auto Res = NegationVals.try_emplace(&Val, nullptr);
  if (Res.second)
    Res.first->second = &takeOwnership(std::make_unique<NegationValue>(Val));
  return *Res.first->second;
}

AtomicBoolValue &DataflowAnalysisContext::makeFlowConditionToken() {
  AtomicBoolValue &Token = createAtomicBoolValue();
  FlowConditionRemainingConjuncts[&Token] = {};
  FlowConditionFirstConjuncts[&Token] = &Token;
  return Token;
}

void DataflowAnalysisContext::addFlowConditionConstraint(
    AtomicBoolValue &Token, BoolValue &Constraint) {
  FlowConditionRemainingConjuncts[&Token].insert(&getOrCreateDisjunctionValue(
      Constraint, getOrCreateNegationValue(Token)));
  FlowConditionFirstConjuncts[&Token] =
      &getOrCreateDisjunctionValue(*FlowConditionFirstConjuncts[&Token],
                                   getOrCreateNegationValue(Constraint));
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
  addFlowConditionConstraint(
      Token, getOrCreateDisjunctionValue(FirstToken, SecondToken));
  return Token;
}

bool DataflowAnalysisContext::flowConditionImplies(AtomicBoolValue &Token,
                                                   BoolValue &Val) {
  // Returns true if and only if truth assignment of the flow condition implies
  // that `Val` is also true. We prove whether or not this property holds by
  // reducing the problem to satisfiability checking. In other words, we attempt
  // to show that assuming `Val` is false makes the constraints induced by the
  // flow condition unsatisfiable.
  llvm::DenseSet<BoolValue *> Constraints = {
      &Token,
      &getBoolLiteralValue(true),
      &getOrCreateNegationValue(getBoolLiteralValue(false)),
      &getOrCreateNegationValue(Val),
  };
  llvm::DenseSet<AtomicBoolValue *> VisitedTokens;
  addTransitiveFlowConditionConstraints(Token, Constraints, VisitedTokens);
  return S->solve(std::move(Constraints)) == Solver::Result::Unsatisfiable;
}

void DataflowAnalysisContext::addTransitiveFlowConditionConstraints(
    AtomicBoolValue &Token, llvm::DenseSet<BoolValue *> &Constraints,
    llvm::DenseSet<AtomicBoolValue *> &VisitedTokens) const {
  auto Res = VisitedTokens.insert(&Token);
  if (!Res.second)
    return;

  auto FirstConjunctIT = FlowConditionFirstConjuncts.find(&Token);
  if (FirstConjunctIT != FlowConditionFirstConjuncts.end())
    Constraints.insert(FirstConjunctIT->second);
  auto RemainingConjunctsIT = FlowConditionRemainingConjuncts.find(&Token);
  if (RemainingConjunctsIT != FlowConditionRemainingConjuncts.end())
    Constraints.insert(RemainingConjunctsIT->second.begin(),
                       RemainingConjunctsIT->second.end());

  auto DepsIT = FlowConditionDeps.find(&Token);
  if (DepsIT != FlowConditionDeps.end()) {
    for (AtomicBoolValue *DepToken : DepsIT->second)
      addTransitiveFlowConditionConstraints(*DepToken, Constraints,
                                            VisitedTokens);
  }
}

} // namespace dataflow
} // namespace clang
