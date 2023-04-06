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
#include "clang/Analysis/FlowSensitive/Logger.h"
#include "clang/Analysis/FlowSensitive/Value.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include <cassert>
#include <memory>
#include <utility>

static llvm::cl::opt<std::string>
    DataflowLog("dataflow-log", llvm::cl::Hidden, llvm::cl::ValueOptional,
                llvm::cl::desc("Emit log of dataflow analysis. With no arg, "
                               "writes textual log to stderr."));

namespace clang {
namespace dataflow {

void DataflowAnalysisContext::addModeledFields(
    const llvm::DenseSet<const FieldDecl *> &Fields) {
  llvm::set_union(ModeledFields, Fields);
}

llvm::DenseSet<const FieldDecl *>
DataflowAnalysisContext::getReferencedFields(QualType Type) {
  llvm::DenseSet<const FieldDecl *> Fields = getObjectFields(Type);
  llvm::set_intersect(Fields, ModeledFields);
  return Fields;
}

StorageLocation &DataflowAnalysisContext::createStorageLocation(QualType Type) {
  if (!Type.isNull() && Type->isRecordType()) {
    llvm::DenseMap<const ValueDecl *, StorageLocation *> FieldLocs;
    // During context-sensitive analysis, a struct may be allocated in one
    // function, but its field accessed in a function lower in the stack than
    // the allocation. Since we only collect fields used in the function where
    // the allocation occurs, we can't apply that filter when performing
    // context-sensitive analysis. But, this only applies to storage locations,
    // since field access it not allowed to fail. In contrast, field *values*
    // don't need this allowance, since the API allows for uninitialized fields.
    auto Fields = Opts.ContextSensitiveOpts ? getObjectFields(Type)
                                            : getReferencedFields(Type);
    for (const FieldDecl *Field : Fields)
      FieldLocs.insert({Field, &createStorageLocation(Field->getType())});
    return create<AggregateStorageLocation>(Type, std::move(FieldLocs));
  }
  return create<ScalarStorageLocation>(Type);
}

StorageLocation &
DataflowAnalysisContext::getStableStorageLocation(const VarDecl &D) {
  if (auto *Loc = getStorageLocation(D))
    return *Loc;
  auto &Loc = createStorageLocation(D.getType());
  setStorageLocation(D, Loc);
  return Loc;
}

StorageLocation &
DataflowAnalysisContext::getStableStorageLocation(const Expr &E) {
  if (auto *Loc = getStorageLocation(E))
    return *Loc;
  auto &Loc = createStorageLocation(E.getType());
  setStorageLocation(E, Loc);
  return Loc;
}

PointerValue &
DataflowAnalysisContext::getOrCreateNullPointerValue(QualType PointeeType) {
  auto CanonicalPointeeType =
      PointeeType.isNull() ? PointeeType : PointeeType.getCanonicalType();
  auto Res = NullPointerVals.try_emplace(CanonicalPointeeType, nullptr);
  if (Res.second) {
    auto &PointeeLoc = createStorageLocation(CanonicalPointeeType);
    Res.first->second = &create<PointerValue>(PointeeLoc);
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
    Res.first->second = &create<ConjunctionValue>(LHS, RHS);
  return *Res.first->second;
}

BoolValue &DataflowAnalysisContext::getOrCreateDisjunction(BoolValue &LHS,
                                                           BoolValue &RHS) {
  if (&LHS == &RHS)
    return LHS;

  auto Res = DisjunctionVals.try_emplace(makeCanonicalBoolValuePair(LHS, RHS),
                                         nullptr);
  if (Res.second)
    Res.first->second = &create<DisjunctionValue>(LHS, RHS);
  return *Res.first->second;
}

BoolValue &DataflowAnalysisContext::getOrCreateNegation(BoolValue &Val) {
  auto Res = NegationVals.try_emplace(&Val, nullptr);
  if (Res.second)
    Res.first->second = &create<NegationValue>(Val);
  return *Res.first->second;
}

BoolValue &DataflowAnalysisContext::getOrCreateImplication(BoolValue &LHS,
                                                           BoolValue &RHS) {
  if (&LHS == &RHS)
    return getBoolLiteralValue(true);

  auto Res = ImplicationVals.try_emplace(std::make_pair(&LHS, &RHS), nullptr);
  if (Res.second)
    Res.first->second = &create<ImplicationValue>(LHS, RHS);
  return *Res.first->second;
}

BoolValue &DataflowAnalysisContext::getOrCreateIff(BoolValue &LHS,
                                                   BoolValue &RHS) {
  if (&LHS == &RHS)
    return getBoolLiteralValue(true);

  auto Res = BiconditionalVals.try_emplace(makeCanonicalBoolValuePair(LHS, RHS),
                                           nullptr);
  if (Res.second)
    Res.first->second = &create<BiconditionalValue>(LHS, RHS);
  return *Res.first->second;
}

AtomicBoolValue &DataflowAnalysisContext::makeFlowConditionToken() {
  return create<AtomicBoolValue>();
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

  auto ConstraintsIt = FlowConditionConstraints.find(&Token);
  if (ConstraintsIt == FlowConditionConstraints.end()) {
    Constraints.insert(&Token);
  } else {
    // Bind flow condition token via `iff` to its set of constraints:
    // FC <=> (C1 ^ C2 ^ ...), where Ci are constraints
    Constraints.insert(&getOrCreateIff(Token, *ConstraintsIt->second));
  }

  auto DepsIt = FlowConditionDeps.find(&Token);
  if (DepsIt != FlowConditionDeps.end()) {
    for (AtomicBoolValue *DepToken : DepsIt->second) {
      addTransitiveFlowConditionConstraints(*DepToken, Constraints,
                                            VisitedTokens);
    }
  }
}

BoolValue &DataflowAnalysisContext::substituteBoolValue(
    BoolValue &Val,
    llvm::DenseMap<BoolValue *, BoolValue *> &SubstitutionsCache) {
  auto It = SubstitutionsCache.find(&Val);
  if (It != SubstitutionsCache.end()) {
    // Return memoized result of substituting this boolean value.
    return *It->second;
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
  case Value::Kind::Implication: {
    auto &IV = *cast<ImplicationValue>(&Val);
    auto &LeftSub =
        substituteBoolValue(IV.getLeftSubValue(), SubstitutionsCache);
    auto &RightSub =
        substituteBoolValue(IV.getRightSubValue(), SubstitutionsCache);
    Result = &getOrCreateImplication(LeftSub, RightSub);
    break;
  }
  case Value::Kind::Biconditional: {
    auto &BV = *cast<BiconditionalValue>(&Val);
    auto &LeftSub =
        substituteBoolValue(BV.getLeftSubValue(), SubstitutionsCache);
    auto &RightSub =
        substituteBoolValue(BV.getRightSubValue(), SubstitutionsCache);
    Result = &getOrCreateIff(LeftSub, RightSub);
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
  assert(!Substitutions.contains(&getBoolLiteralValue(true)) &&
         !Substitutions.contains(&getBoolLiteralValue(false)) &&
         "Do not substitute true/false boolean literals");
  llvm::DenseMap<BoolValue *, BoolValue *> SubstitutionsCache(
      Substitutions.begin(), Substitutions.end());
  return buildAndSubstituteFlowConditionWithCache(Token, SubstitutionsCache);
}

BoolValue &DataflowAnalysisContext::buildAndSubstituteFlowConditionWithCache(
    AtomicBoolValue &Token,
    llvm::DenseMap<BoolValue *, BoolValue *> &SubstitutionsCache) {
  auto ConstraintsIt = FlowConditionConstraints.find(&Token);
  if (ConstraintsIt == FlowConditionConstraints.end()) {
    return getBoolLiteralValue(true);
  }
  auto DepsIt = FlowConditionDeps.find(&Token);
  if (DepsIt != FlowConditionDeps.end()) {
    for (AtomicBoolValue *DepToken : DepsIt->second) {
      auto &NewDep = buildAndSubstituteFlowConditionWithCache(
          *DepToken, SubstitutionsCache);
      SubstitutionsCache[DepToken] = &NewDep;
    }
  }
  return substituteBoolValue(*ConstraintsIt->second, SubstitutionsCache);
}

void DataflowAnalysisContext::dumpFlowCondition(AtomicBoolValue &Token,
                                                llvm::raw_ostream &OS) {
  llvm::DenseSet<BoolValue *> Constraints = {&Token};
  llvm::DenseSet<AtomicBoolValue *> VisitedTokens;
  addTransitiveFlowConditionConstraints(Token, Constraints, VisitedTokens);

  llvm::DenseMap<const AtomicBoolValue *, std::string> AtomNames = {
      {&getBoolLiteralValue(false), "False"},
      {&getBoolLiteralValue(true), "True"}};
  OS << debugString(Constraints, AtomNames);
}

const ControlFlowContext *
DataflowAnalysisContext::getControlFlowContext(const FunctionDecl *F) {
  // Canonicalize the key:
  F = F->getDefinition();
  if (F == nullptr)
    return nullptr;
  auto It = FunctionContexts.find(F);
  if (It != FunctionContexts.end())
    return &It->second;

  if (Stmt *Body = F->getBody()) {
    auto CFCtx = ControlFlowContext::build(F, *Body, F->getASTContext());
    // FIXME: Handle errors.
    assert(CFCtx);
    auto Result = FunctionContexts.insert({F, std::move(*CFCtx)});
    return &Result.first->second;
  }

  return nullptr;
}

DataflowAnalysisContext::DataflowAnalysisContext(std::unique_ptr<Solver> S,
                                                 Options Opts)
    : S(std::move(S)), TrueVal(create<AtomicBoolValue>()),
      FalseVal(create<AtomicBoolValue>()), Opts(Opts) {
  assert(this->S != nullptr);
  // If the -dataflow-log command-line flag was set, synthesize a logger.
  // This is ugly but provides a uniform method for ad-hoc debugging dataflow-
  // based tools.
  if (Opts.Log == nullptr) {
    if (DataflowLog.getNumOccurrences() > 0) {
      LogOwner = Logger::textual(llvm::errs());
      this->Opts.Log = LogOwner.get();
      // FIXME: if the flag is given a value, write an HTML log to a file.
    } else {
      this->Opts.Log = &Logger::null();
    }
  }
}

DataflowAnalysisContext::~DataflowAnalysisContext() = default;

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
