//===-- DataflowEnvironment.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines an Environment class that is used by dataflow analyses
//  that run over Control-Flow Graphs (CFGs) to keep track of the state of the
//  program at given program points.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Type.h"
#include "clang/Analysis/FlowSensitive/DataflowLattice.h"
#include "clang/Analysis/FlowSensitive/StorageLocation.h"
#include "clang/Analysis/FlowSensitive/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>
#include <memory>
#include <utility>

namespace clang {
namespace dataflow {

// FIXME: convert these to parameters of the analysis or environment. Current
// settings have been experimentaly validated, but only for a particular
// analysis.
static constexpr int MaxCompositeValueDepth = 3;
static constexpr int MaxCompositeValueSize = 1000;

/// Returns a map consisting of key-value entries that are present in both maps.
template <typename K, typename V>
llvm::DenseMap<K, V> intersectDenseMaps(const llvm::DenseMap<K, V> &Map1,
                                        const llvm::DenseMap<K, V> &Map2) {
  llvm::DenseMap<K, V> Result;
  for (auto &Entry : Map1) {
    auto It = Map2.find(Entry.first);
    if (It != Map2.end() && Entry.second == It->second)
      Result.insert({Entry.first, Entry.second});
  }
  return Result;
}

/// Returns true if and only if `Val1` is equivalent to `Val2`.
static bool equivalentValues(QualType Type, Value *Val1,
                             const Environment &Env1, Value *Val2,
                             const Environment &Env2,
                             Environment::ValueModel &Model) {
  if (Val1 == Val2)
    return true;

  if (auto *IndVal1 = dyn_cast<IndirectionValue>(Val1)) {
    auto *IndVal2 = cast<IndirectionValue>(Val2);
    assert(IndVal1->getKind() == IndVal2->getKind());
    if (&IndVal1->getPointeeLoc() == &IndVal2->getPointeeLoc())
      return true;
  }

  return Model.compareEquivalent(Type, *Val1, Env1, *Val2, Env2);
}

/// Attempts to merge distinct values `Val1` and `Val1` in `Env1` and `Env2`,
/// respectively, of the same type `Type`. Merging generally produces a single
/// value that (soundly) approximates the two inputs, although the actual
/// meaning depends on `Model`.
static Value *mergeDistinctValues(QualType Type, Value *Val1, Environment &Env1,
                                  Value *Val2, const Environment &Env2,
                                  Environment::ValueModel &Model) {
  // Join distinct boolean values preserving information about the constraints
  // in the respective path conditions. Note: this construction can, in
  // principle, result in exponential growth in the size of boolean values.
  // Potential optimizations may be worth considering. For example, represent
  // the flow condition of each environment using a bool atom and store, in
  // `DataflowAnalysisContext`, a mapping of bi-conditionals between flow
  // condition atoms and flow condition constraints. Something like:
  // \code
  //   FC1 <=> C1 ^ C2
  //   FC2 <=> C2 ^ C3 ^ C4
  //   FC3 <=> (FC1 v FC2) ^ C5
  // \code
  // Then, we can track dependencies between flow conditions (e.g. above `FC3`
  // depends on `FC1` and `FC2`) and modify `flowConditionImplies` to construct
  // a formula that includes the bi-conditionals for all flow condition atoms in
  // the transitive set, before invoking the solver.
  //
  // FIXME: Does not work for backedges, since the two (or more) paths will not
  // have mutually exclusive conditions.
  if (auto *Expr1 = dyn_cast<BoolValue>(Val1)) {
    for (BoolValue *Constraint : Env1.getFlowConditionConstraints()) {
      Expr1 = &Env1.makeAnd(*Expr1, *Constraint);
    }
    auto *Expr2 = cast<BoolValue>(Val2);
    for (BoolValue *Constraint : Env2.getFlowConditionConstraints()) {
      Expr2 = &Env1.makeAnd(*Expr2, *Constraint);
    }
    return &Env1.makeOr(*Expr1, *Expr2);
  }

  // FIXME: Consider destroying `MergedValue` immediately if `ValueModel::merge`
  // returns false to avoid storing unneeded values in `DACtx`.
  if (Value *MergedVal = Env1.createValue(Type))
    if (Model.merge(Type, *Val1, Env1, *Val2, Env2, *MergedVal, Env1))
      return MergedVal;

  return nullptr;
}

/// Initializes a global storage value.
static void initGlobalVar(const VarDecl &D, Environment &Env) {
  if (!D.hasGlobalStorage() ||
      Env.getStorageLocation(D, SkipPast::None) != nullptr)
    return;

  auto &Loc = Env.createStorageLocation(D);
  Env.setStorageLocation(D, Loc);
  if (auto *Val = Env.createValue(D.getType()))
    Env.setValue(Loc, *Val);
}

/// Initializes a global storage value.
static void initGlobalVar(const Decl &D, Environment &Env) {
  if (auto *V = dyn_cast<VarDecl>(&D))
    initGlobalVar(*V, Env);
}

/// Initializes global storage values that are declared or referenced from
/// sub-statements of `S`.
// FIXME: Add support for resetting globals after function calls to enable
// the implementation of sound analyses.
static void initGlobalVars(const Stmt &S, Environment &Env) {
  for (auto *Child : S.children()) {
    if (Child != nullptr)
      initGlobalVars(*Child, Env);
  }

  if (auto *DS = dyn_cast<DeclStmt>(&S)) {
    if (DS->isSingleDecl()) {
      initGlobalVar(*DS->getSingleDecl(), Env);
    } else {
      for (auto *D : DS->getDeclGroup())
        initGlobalVar(*D, Env);
    }
  } else if (auto *E = dyn_cast<DeclRefExpr>(&S)) {
    initGlobalVar(*E->getDecl(), Env);
  } else if (auto *E = dyn_cast<MemberExpr>(&S)) {
    initGlobalVar(*E->getMemberDecl(), Env);
  }
}

/// Returns constraints that represent the disjunction of `Constraints1` and
/// `Constraints2`.
///
/// Requirements:
///
///  The elements of `Constraints1` and `Constraints2` must not be null.
llvm::DenseSet<BoolValue *>
joinConstraints(DataflowAnalysisContext *Context,
                const llvm::DenseSet<BoolValue *> &Constraints1,
                const llvm::DenseSet<BoolValue *> &Constraints2) {
  // `(X ^ Y) v (X ^ Z)` is logically equivalent to `X ^ (Y v Z)`. Therefore, to
  // avoid unnecessarily expanding the resulting set of constraints, we will add
  // all common constraints of `Constraints1` and `Constraints2` directly and
  // add a disjunction of the constraints that are not common.

  llvm::DenseSet<BoolValue *> JoinedConstraints;

  if (Constraints1.empty() || Constraints2.empty()) {
    // Disjunction of empty set and non-empty set is represented as empty set.
    return JoinedConstraints;
  }

  BoolValue *Val1 = nullptr;
  for (BoolValue *Constraint : Constraints1) {
    if (Constraints2.contains(Constraint)) {
      // Add common constraints directly to `JoinedConstraints`.
      JoinedConstraints.insert(Constraint);
    } else if (Val1 == nullptr) {
      Val1 = Constraint;
    } else {
      Val1 = &Context->getOrCreateConjunctionValue(*Val1, *Constraint);
    }
  }

  BoolValue *Val2 = nullptr;
  for (BoolValue *Constraint : Constraints2) {
    // Common constraints are added to `JoinedConstraints` above.
    if (Constraints1.contains(Constraint)) {
      continue;
    }
    if (Val2 == nullptr) {
      Val2 = Constraint;
    } else {
      Val2 = &Context->getOrCreateConjunctionValue(*Val2, *Constraint);
    }
  }

  // An empty set of constraints (represented as a null value) is interpreted as
  // `true` and `true v X` is logically equivalent to `true` so we need to add a
  // constraint only if both `Val1` and `Val2` are not null.
  if (Val1 != nullptr && Val2 != nullptr)
    JoinedConstraints.insert(
        &Context->getOrCreateDisjunctionValue(*Val1, *Val2));

  return JoinedConstraints;
}

static void
getFieldsFromClassHierarchy(QualType Type, bool IgnorePrivateFields,
                            llvm::DenseSet<const FieldDecl *> &Fields) {
  if (Type->isIncompleteType() || Type->isDependentType() ||
      !Type->isRecordType())
    return;

  for (const FieldDecl *Field : Type->getAsRecordDecl()->fields()) {
    if (IgnorePrivateFields &&
        (Field->getAccess() == AS_private ||
         (Field->getAccess() == AS_none && Type->getAsRecordDecl()->isClass())))
      continue;
    Fields.insert(Field);
  }
  if (auto *CXXRecord = Type->getAsCXXRecordDecl()) {
    for (const CXXBaseSpecifier &Base : CXXRecord->bases()) {
      // Ignore private fields (including default access in C++ classes) in
      // base classes, because they are not visible in derived classes.
      getFieldsFromClassHierarchy(Base.getType(), /*IgnorePrivateFields=*/true,
                                  Fields);
    }
  }
}

/// Gets the set of all fields accesible from the type.
///
/// FIXME: Does not precisely handle non-virtual diamond inheritance. A single
/// field decl will be modeled for all instances of the inherited field.
static llvm::DenseSet<const FieldDecl *>
getAccessibleObjectFields(QualType Type) {
  llvm::DenseSet<const FieldDecl *> Fields;
  // Don't ignore private fields for the class itself, only its super classes.
  getFieldsFromClassHierarchy(Type, /*IgnorePrivateFields=*/false, Fields);
  return Fields;
}

Environment::Environment(DataflowAnalysisContext &DACtx,
                         const DeclContext &DeclCtx)
    : Environment(DACtx) {
  if (const auto *FuncDecl = dyn_cast<FunctionDecl>(&DeclCtx)) {
    assert(FuncDecl->getBody() != nullptr);
    initGlobalVars(*FuncDecl->getBody(), *this);
    for (const auto *ParamDecl : FuncDecl->parameters()) {
      assert(ParamDecl != nullptr);
      auto &ParamLoc = createStorageLocation(*ParamDecl);
      setStorageLocation(*ParamDecl, ParamLoc);
      if (Value *ParamVal = createValue(ParamDecl->getType()))
        setValue(ParamLoc, *ParamVal);
    }
  }

  if (const auto *MethodDecl = dyn_cast<CXXMethodDecl>(&DeclCtx)) {
    if (!MethodDecl->isStatic()) {
      QualType ThisPointeeType = MethodDecl->getThisObjectType();
      // FIXME: Add support for union types.
      if (!ThisPointeeType->isUnionType()) {
        auto &ThisPointeeLoc = createStorageLocation(ThisPointeeType);
        DACtx.setThisPointeeStorageLocation(ThisPointeeLoc);
        if (Value *ThisPointeeVal = createValue(ThisPointeeType))
          setValue(ThisPointeeLoc, *ThisPointeeVal);
      }
    }
  }
}

bool Environment::equivalentTo(const Environment &Other,
                               Environment::ValueModel &Model) const {
  assert(DACtx == Other.DACtx);

  if (DeclToLoc != Other.DeclToLoc)
    return false;

  if (ExprToLoc != Other.ExprToLoc)
    return false;

  if (MemberLocToStruct != Other.MemberLocToStruct)
    return false;

  // Compare the contents for the intersection of their domains.
  for (auto &Entry : LocToVal) {
    const StorageLocation *Loc = Entry.first;
    assert(Loc != nullptr);

    Value *Val = Entry.second;
    assert(Val != nullptr);

    auto It = Other.LocToVal.find(Loc);
    if (It == Other.LocToVal.end())
      continue;
    assert(It->second != nullptr);

    if (!equivalentValues(Loc->getType(), Val, *this, It->second, Other, Model))
      return false;
  }

  return true;
}

LatticeJoinEffect Environment::join(const Environment &Other,
                                    Environment::ValueModel &Model) {
  assert(DACtx == Other.DACtx);

  auto Effect = LatticeJoinEffect::Unchanged;

  const unsigned DeclToLocSizeBefore = DeclToLoc.size();
  DeclToLoc = intersectDenseMaps(DeclToLoc, Other.DeclToLoc);
  if (DeclToLocSizeBefore != DeclToLoc.size())
    Effect = LatticeJoinEffect::Changed;

  const unsigned ExprToLocSizeBefore = ExprToLoc.size();
  ExprToLoc = intersectDenseMaps(ExprToLoc, Other.ExprToLoc);
  if (ExprToLocSizeBefore != ExprToLoc.size())
    Effect = LatticeJoinEffect::Changed;

  const unsigned MemberLocToStructSizeBefore = MemberLocToStruct.size();
  MemberLocToStruct =
      intersectDenseMaps(MemberLocToStruct, Other.MemberLocToStruct);
  if (MemberLocToStructSizeBefore != MemberLocToStruct.size())
    Effect = LatticeJoinEffect::Changed;

  // Move `LocToVal` so that `Environment::ValueModel::merge` can safely assign
  // values to storage locations while this code iterates over the current
  // assignments.
  llvm::DenseMap<const StorageLocation *, Value *> OldLocToVal =
      std::move(LocToVal);
  for (auto &Entry : OldLocToVal) {
    const StorageLocation *Loc = Entry.first;
    assert(Loc != nullptr);

    Value *Val = Entry.second;
    assert(Val != nullptr);

    auto It = Other.LocToVal.find(Loc);
    if (It == Other.LocToVal.end())
      continue;
    assert(It->second != nullptr);

    if (Val == It->second) {
      LocToVal.insert({Loc, Val});
      continue;
    }

    if (Value *MergedVal = mergeDistinctValues(Loc->getType(), Val, *this,
                                               It->second, Other, Model))
      LocToVal.insert({Loc, MergedVal});
  }
  if (OldLocToVal.size() != LocToVal.size())
    Effect = LatticeJoinEffect::Changed;

  FlowConditionConstraints = joinConstraints(DACtx, FlowConditionConstraints,
                                             Other.FlowConditionConstraints);

  return Effect;
}

StorageLocation &Environment::createStorageLocation(QualType Type) {
  assert(!Type.isNull());
  if (Type->isStructureOrClassType() || Type->isUnionType()) {
    // FIXME: Explore options to avoid eager initialization of fields as some of
    // them might not be needed for a particular analysis.
    llvm::DenseMap<const ValueDecl *, StorageLocation *> FieldLocs;
    for (const FieldDecl *Field : getAccessibleObjectFields(Type)) {
      FieldLocs.insert({Field, &createStorageLocation(Field->getType())});
    }
    return takeOwnership(
        std::make_unique<AggregateStorageLocation>(Type, std::move(FieldLocs)));
  }
  return takeOwnership(std::make_unique<ScalarStorageLocation>(Type));
}

StorageLocation &Environment::createStorageLocation(const VarDecl &D) {
  // Evaluated declarations are always assigned the same storage locations to
  // ensure that the environment stabilizes across loop iterations. Storage
  // locations for evaluated declarations are stored in the analysis context.
  if (auto *Loc = DACtx->getStorageLocation(D))
    return *Loc;
  auto &Loc = createStorageLocation(D.getType());
  DACtx->setStorageLocation(D, Loc);
  return Loc;
}

StorageLocation &Environment::createStorageLocation(const Expr &E) {
  // Evaluated expressions are always assigned the same storage locations to
  // ensure that the environment stabilizes across loop iterations. Storage
  // locations for evaluated expressions are stored in the analysis context.
  if (auto *Loc = DACtx->getStorageLocation(E))
    return *Loc;
  auto &Loc = createStorageLocation(E.getType());
  DACtx->setStorageLocation(E, Loc);
  return Loc;
}

void Environment::setStorageLocation(const ValueDecl &D, StorageLocation &Loc) {
  assert(DeclToLoc.find(&D) == DeclToLoc.end());
  DeclToLoc[&D] = &Loc;
}

StorageLocation *Environment::getStorageLocation(const ValueDecl &D,
                                                 SkipPast SP) const {
  auto It = DeclToLoc.find(&D);
  return It == DeclToLoc.end() ? nullptr : &skip(*It->second, SP);
}

void Environment::setStorageLocation(const Expr &E, StorageLocation &Loc) {
  assert(ExprToLoc.find(&E) == ExprToLoc.end());
  ExprToLoc[&E] = &Loc;
}

StorageLocation *Environment::getStorageLocation(const Expr &E,
                                                 SkipPast SP) const {
  // FIXME: Add a test with parens.
  auto It = ExprToLoc.find(E.IgnoreParens());
  return It == ExprToLoc.end() ? nullptr : &skip(*It->second, SP);
}

StorageLocation *Environment::getThisPointeeStorageLocation() const {
  return DACtx->getThisPointeeStorageLocation();
}

void Environment::setValue(const StorageLocation &Loc, Value &Val) {
  LocToVal[&Loc] = &Val;

  if (auto *StructVal = dyn_cast<StructValue>(&Val)) {
    auto &AggregateLoc = *cast<AggregateStorageLocation>(&Loc);

    const QualType Type = AggregateLoc.getType();
    assert(Type->isStructureOrClassType());

    for (const FieldDecl *Field : getAccessibleObjectFields(Type)) {
      assert(Field != nullptr);
      StorageLocation &FieldLoc = AggregateLoc.getChild(*Field);
      MemberLocToStruct[&FieldLoc] = std::make_pair(StructVal, Field);
      if (auto *FieldVal = StructVal->getChild(*Field))
        setValue(FieldLoc, *FieldVal);
    }
  }

  auto IT = MemberLocToStruct.find(&Loc);
  if (IT != MemberLocToStruct.end()) {
    // `Loc` is the location of a struct member so we need to also update the
    // value of the member in the corresponding `StructValue`.

    assert(IT->second.first != nullptr);
    StructValue &StructVal = *IT->second.first;

    assert(IT->second.second != nullptr);
    const ValueDecl &Member = *IT->second.second;

    StructVal.setChild(Member, Val);
  }
}

Value *Environment::getValue(const StorageLocation &Loc) const {
  auto It = LocToVal.find(&Loc);
  return It == LocToVal.end() ? nullptr : It->second;
}

Value *Environment::getValue(const ValueDecl &D, SkipPast SP) const {
  auto *Loc = getStorageLocation(D, SP);
  if (Loc == nullptr)
    return nullptr;
  return getValue(*Loc);
}

Value *Environment::getValue(const Expr &E, SkipPast SP) const {
  auto *Loc = getStorageLocation(E, SP);
  if (Loc == nullptr)
    return nullptr;
  return getValue(*Loc);
}

Value *Environment::createValue(QualType Type) {
  llvm::DenseSet<QualType> Visited;
  int CreatedValuesCount = 0;
  Value *Val = createValueUnlessSelfReferential(Type, Visited, /*Depth=*/0,
                                                CreatedValuesCount);
  if (CreatedValuesCount > MaxCompositeValueSize) {
    llvm::errs() << "Attempting to initialize a huge value of type: "
                 << Type.getAsString() << "\n";
  }
  return Val;
}

Value *Environment::createValueUnlessSelfReferential(
    QualType Type, llvm::DenseSet<QualType> &Visited, int Depth,
    int &CreatedValuesCount) {
  assert(!Type.isNull());

  // Allow unlimited fields at depth 1; only cap at deeper nesting levels.
  if ((Depth > 1 && CreatedValuesCount > MaxCompositeValueSize) ||
      Depth > MaxCompositeValueDepth)
    return nullptr;

  if (Type->isBooleanType()) {
    CreatedValuesCount++;
    return &makeAtomicBoolValue();
  }

  if (Type->isIntegerType()) {
    CreatedValuesCount++;
    return &takeOwnership(std::make_unique<IntegerValue>());
  }

  if (Type->isReferenceType()) {
    CreatedValuesCount++;
    QualType PointeeType = Type->castAs<ReferenceType>()->getPointeeType();
    auto &PointeeLoc = createStorageLocation(PointeeType);

    if (!Visited.contains(PointeeType.getCanonicalType())) {
      Visited.insert(PointeeType.getCanonicalType());
      Value *PointeeVal = createValueUnlessSelfReferential(
          PointeeType, Visited, Depth, CreatedValuesCount);
      Visited.erase(PointeeType.getCanonicalType());

      if (PointeeVal != nullptr)
        setValue(PointeeLoc, *PointeeVal);
    }

    return &takeOwnership(std::make_unique<ReferenceValue>(PointeeLoc));
  }

  if (Type->isPointerType()) {
    CreatedValuesCount++;
    QualType PointeeType = Type->castAs<PointerType>()->getPointeeType();
    auto &PointeeLoc = createStorageLocation(PointeeType);

    if (!Visited.contains(PointeeType.getCanonicalType())) {
      Visited.insert(PointeeType.getCanonicalType());
      Value *PointeeVal = createValueUnlessSelfReferential(
          PointeeType, Visited, Depth, CreatedValuesCount);
      Visited.erase(PointeeType.getCanonicalType());

      if (PointeeVal != nullptr)
        setValue(PointeeLoc, *PointeeVal);
    }

    return &takeOwnership(std::make_unique<PointerValue>(PointeeLoc));
  }

  if (Type->isStructureOrClassType()) {
    CreatedValuesCount++;
    // FIXME: Initialize only fields that are accessed in the context that is
    // being analyzed.
    llvm::DenseMap<const ValueDecl *, Value *> FieldValues;
    for (const FieldDecl *Field : getAccessibleObjectFields(Type)) {
      assert(Field != nullptr);

      QualType FieldType = Field->getType();
      if (Visited.contains(FieldType.getCanonicalType()))
        continue;

      Visited.insert(FieldType.getCanonicalType());
      if (auto *FieldValue = createValueUnlessSelfReferential(
              FieldType, Visited, Depth + 1, CreatedValuesCount))
        FieldValues.insert({Field, FieldValue});
      Visited.erase(FieldType.getCanonicalType());
    }

    return &takeOwnership(
        std::make_unique<StructValue>(std::move(FieldValues)));
  }

  return nullptr;
}

StorageLocation &Environment::skip(StorageLocation &Loc, SkipPast SP) const {
  switch (SP) {
  case SkipPast::None:
    return Loc;
  case SkipPast::Reference:
    // References cannot be chained so we only need to skip past one level of
    // indirection.
    if (auto *Val = dyn_cast_or_null<ReferenceValue>(getValue(Loc)))
      return Val->getPointeeLoc();
    return Loc;
  case SkipPast::ReferenceThenPointer:
    StorageLocation &LocPastRef = skip(Loc, SkipPast::Reference);
    if (auto *Val = dyn_cast_or_null<PointerValue>(getValue(LocPastRef)))
      return Val->getPointeeLoc();
    return LocPastRef;
  }
  llvm_unreachable("bad SkipPast kind");
}

const StorageLocation &Environment::skip(const StorageLocation &Loc,
                                         SkipPast SP) const {
  return skip(*const_cast<StorageLocation *>(&Loc), SP);
}

void Environment::addToFlowCondition(BoolValue &Val) {
  FlowConditionConstraints.insert(&Val);
}

bool Environment::flowConditionImplies(BoolValue &Val) const {
  // Returns true if and only if truth assignment of the flow condition implies
  // that `Val` is also true. We prove whether or not this property holds by
  // reducing the problem to satisfiability checking. In other words, we attempt
  // to show that assuming `Val` is false makes the constraints induced by the
  // flow condition unsatisfiable.
  llvm::DenseSet<BoolValue *> Constraints = {
      &makeNot(Val), &getBoolLiteralValue(true),
      &makeNot(getBoolLiteralValue(false))};
  Constraints.insert(FlowConditionConstraints.begin(),
                     FlowConditionConstraints.end());
  return DACtx->getSolver().solve(std::move(Constraints)) ==
         Solver::Result::Unsatisfiable;
}

} // namespace dataflow
} // namespace clang
