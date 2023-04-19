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
#include "clang/Analysis/FlowSensitive/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
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

static bool compareDistinctValues(QualType Type, Value &Val1,
                                  const Environment &Env1, Value &Val2,
                                  const Environment &Env2,
                                  Environment::ValueModel &Model) {
  // Note: Potentially costly, but, for booleans, we could check whether both
  // can be proven equivalent in their respective environments.

  // FIXME: move the reference/pointers logic from `areEquivalentValues` to here
  // and implement separate, join/widen specific handling for
  // reference/pointers.
  switch (Model.compare(Type, Val1, Env1, Val2, Env2)) {
  case ComparisonResult::Same:
    return true;
  case ComparisonResult::Different:
    return false;
  case ComparisonResult::Unknown:
    switch (Val1.getKind()) {
    case Value::Kind::Integer:
    case Value::Kind::Reference:
    case Value::Kind::Pointer:
    case Value::Kind::Struct:
      // FIXME: this choice intentionally introduces unsoundness to allow
      // for convergence. Once we have widening support for the
      // reference/pointer and struct built-in models, this should be
      // `false`.
      return true;
    default:
      return false;
    }
  }
  llvm_unreachable("All cases covered in switch");
}

/// Attempts to merge distinct values `Val1` and `Val2` in `Env1` and `Env2`,
/// respectively, of the same type `Type`. Merging generally produces a single
/// value that (soundly) approximates the two inputs, although the actual
/// meaning depends on `Model`.
static Value *mergeDistinctValues(QualType Type, Value &Val1,
                                  const Environment &Env1, Value &Val2,
                                  const Environment &Env2,
                                  Environment &MergedEnv,
                                  Environment::ValueModel &Model) {
  // Join distinct boolean values preserving information about the constraints
  // in the respective path conditions.
  if (isa<BoolValue>(&Val1) && isa<BoolValue>(&Val2)) {
    // FIXME: Checking both values should be unnecessary, since they should have
    // a consistent shape.  However, right now we can end up with BoolValue's in
    // integer-typed variables due to our incorrect handling of
    // boolean-to-integer casts (we just propagate the BoolValue to the result
    // of the cast). So, a join can encounter an integer in one branch but a
    // bool in the other.
    // For example:
    // ```
    // std::optional<bool> o;
    // int x;
    // if (o.has_value())
    //   x = o.value();
    // ```
    auto *Expr1 = cast<BoolValue>(&Val1);
    auto *Expr2 = cast<BoolValue>(&Val2);
    auto &MergedVal = MergedEnv.makeAtomicBoolValue();
    MergedEnv.addToFlowCondition(MergedEnv.makeOr(
        MergedEnv.makeAnd(Env1.getFlowConditionToken(),
                          MergedEnv.makeIff(MergedVal, *Expr1)),
        MergedEnv.makeAnd(Env2.getFlowConditionToken(),
                          MergedEnv.makeIff(MergedVal, *Expr2))));
    return &MergedVal;
  }

  // FIXME: Consider destroying `MergedValue` immediately if `ValueModel::merge`
  // returns false to avoid storing unneeded values in `DACtx`.
  // FIXME: Creating the value based on the type alone creates misshapen values
  // for lvalues, since the type does not reflect the need for `ReferenceValue`.
  if (Value *MergedVal = MergedEnv.createValue(Type))
    if (Model.merge(Type, Val1, Env1, Val2, Env2, *MergedVal, MergedEnv))
      return MergedVal;

  return nullptr;
}

// When widening does not change `Current`, return value will equal `&Prev`.
static Value &widenDistinctValues(QualType Type, Value &Prev,
                                  const Environment &PrevEnv, Value &Current,
                                  Environment &CurrentEnv,
                                  Environment::ValueModel &Model) {
  // Boolean-model widening.
  if (isa<BoolValue>(&Prev)) {
    assert(isa<BoolValue>(Current));
    // Widen to Top, because we know they are different values. If previous was
    // already Top, re-use that to (implicitly) indicate that no change occured.
    if (isa<TopBoolValue>(Prev))
      return Prev;
    return CurrentEnv.makeTopBoolValue();
  }

  // FIXME: Add other built-in model widening.

  // Custom-model widening.
  if (auto *W = Model.widen(Type, Prev, PrevEnv, Current, CurrentEnv))
    return *W;

  // Default of widening is a no-op: leave the current value unchanged.
  return Current;
}

/// Initializes a global storage value.
static void insertIfGlobal(const Decl &D,
                           llvm::DenseSet<const VarDecl *> &Vars) {
  if (auto *V = dyn_cast<VarDecl>(&D))
    if (V->hasGlobalStorage())
      Vars.insert(V);
}

static void insertIfFunction(const Decl &D,
                             llvm::DenseSet<const FunctionDecl *> &Funcs) {
  if (auto *FD = dyn_cast<FunctionDecl>(&D))
    Funcs.insert(FD);
}

static void
getFieldsGlobalsAndFuncs(const Decl &D,
                         llvm::DenseSet<const FieldDecl *> &Fields,
                         llvm::DenseSet<const VarDecl *> &Vars,
                         llvm::DenseSet<const FunctionDecl *> &Funcs) {
  insertIfGlobal(D, Vars);
  insertIfFunction(D, Funcs);
  if (const auto *Decomp = dyn_cast<DecompositionDecl>(&D))
    for (const auto *B : Decomp->bindings())
      if (auto *ME = dyn_cast_or_null<MemberExpr>(B->getBinding()))
        // FIXME: should we be using `E->getFoundDecl()`?
        if (const auto *FD = dyn_cast<FieldDecl>(ME->getMemberDecl()))
          Fields.insert(FD);
}

/// Traverses `S` and inserts into `Fields`, `Vars` and `Funcs` any fields,
/// global variables and functions that are declared in or referenced from
/// sub-statements.
static void
getFieldsGlobalsAndFuncs(const Stmt &S,
                         llvm::DenseSet<const FieldDecl *> &Fields,
                         llvm::DenseSet<const VarDecl *> &Vars,
                         llvm::DenseSet<const FunctionDecl *> &Funcs) {
  for (auto *Child : S.children())
    if (Child != nullptr)
      getFieldsGlobalsAndFuncs(*Child, Fields, Vars, Funcs);

  if (auto *DS = dyn_cast<DeclStmt>(&S)) {
    if (DS->isSingleDecl())
      getFieldsGlobalsAndFuncs(*DS->getSingleDecl(), Fields, Vars, Funcs);
    else
      for (auto *D : DS->getDeclGroup())
        getFieldsGlobalsAndFuncs(*D, Fields, Vars, Funcs);
  } else if (auto *E = dyn_cast<DeclRefExpr>(&S)) {
    insertIfGlobal(*E->getDecl(), Vars);
    insertIfFunction(*E->getDecl(), Funcs);
  } else if (auto *E = dyn_cast<MemberExpr>(&S)) {
    // FIXME: should we be using `E->getFoundDecl()`?
    const ValueDecl *VD = E->getMemberDecl();
    insertIfGlobal(*VD, Vars);
    insertIfFunction(*VD, Funcs);
    if (const auto *FD = dyn_cast<FieldDecl>(VD))
      Fields.insert(FD);
  }
}

// FIXME: Add support for resetting globals after function calls to enable
// the implementation of sound analyses.
void Environment::initFieldsGlobalsAndFuncs(const FunctionDecl *FuncDecl) {
  assert(FuncDecl->getBody() != nullptr);

  llvm::DenseSet<const FieldDecl *> Fields;
  llvm::DenseSet<const VarDecl *> Vars;
  llvm::DenseSet<const FunctionDecl *> Funcs;

  // Look for global variable and field references in the
  // constructor-initializers.
  if (const auto *CtorDecl = dyn_cast<CXXConstructorDecl>(FuncDecl)) {
    for (const auto *Init : CtorDecl->inits()) {
      if (const auto *M = Init->getAnyMember())
          Fields.insert(M);
      const Expr *E = Init->getInit();
      assert(E != nullptr);
      getFieldsGlobalsAndFuncs(*E, Fields, Vars, Funcs);
    }
    // Add all fields mentioned in default member initializers.
    for (const FieldDecl *F : CtorDecl->getParent()->fields())
      if (const auto *I = F->getInClassInitializer())
          getFieldsGlobalsAndFuncs(*I, Fields, Vars, Funcs);
  }
  getFieldsGlobalsAndFuncs(*FuncDecl->getBody(), Fields, Vars, Funcs);

  // These have to be added before the lines that follow to ensure that
  // `create*` work correctly for structs.
  DACtx->addModeledFields(Fields);

  for (const VarDecl *D : Vars) {
    if (getStorageLocation(*D, SkipPast::None) != nullptr)
      continue;
    auto &Loc = createStorageLocation(*D);
    setStorageLocation(*D, Loc);
    if (auto *Val = createValue(D->getType()))
      setValue(Loc, *Val);
  }

  for (const FunctionDecl *FD : Funcs) {
    if (getStorageLocation(*FD, SkipPast::None) != nullptr)
      continue;
    auto &Loc = createStorageLocation(FD->getType());
    setStorageLocation(*FD, Loc);
  }
}

Environment::Environment(DataflowAnalysisContext &DACtx)
    : DACtx(&DACtx), FlowConditionToken(&DACtx.makeFlowConditionToken()) {}

Environment::Environment(const Environment &Other)
    : DACtx(Other.DACtx), CallStack(Other.CallStack),
      ReturnLoc(Other.ReturnLoc), ThisPointeeLoc(Other.ThisPointeeLoc),
      DeclToLoc(Other.DeclToLoc), ExprToLoc(Other.ExprToLoc),
      LocToVal(Other.LocToVal), MemberLocToStruct(Other.MemberLocToStruct),
      FlowConditionToken(&DACtx->forkFlowCondition(*Other.FlowConditionToken)) {
}

Environment &Environment::operator=(const Environment &Other) {
  Environment Copy(Other);
  *this = std::move(Copy);
  return *this;
}

Environment::Environment(DataflowAnalysisContext &DACtx,
                         const DeclContext &DeclCtx)
    : Environment(DACtx) {
  CallStack.push_back(&DeclCtx);

  if (const auto *FuncDecl = dyn_cast<FunctionDecl>(&DeclCtx)) {
    assert(FuncDecl->getBody() != nullptr);

    initFieldsGlobalsAndFuncs(FuncDecl);

    for (const auto *ParamDecl : FuncDecl->parameters()) {
      assert(ParamDecl != nullptr);
      auto &ParamLoc = createStorageLocation(*ParamDecl);
      setStorageLocation(*ParamDecl, ParamLoc);
      if (Value *ParamVal = createValue(ParamDecl->getType()))
        setValue(ParamLoc, *ParamVal);
    }

    QualType ReturnType = FuncDecl->getReturnType();
    ReturnLoc = &createStorageLocation(ReturnType);
  }

  if (const auto *MethodDecl = dyn_cast<CXXMethodDecl>(&DeclCtx)) {
    auto *Parent = MethodDecl->getParent();
    assert(Parent != nullptr);
    if (Parent->isLambda())
      MethodDecl = dyn_cast<CXXMethodDecl>(Parent->getDeclContext());

    // FIXME: Initialize the ThisPointeeLoc of lambdas too.
    if (MethodDecl && !MethodDecl->isStatic()) {
      QualType ThisPointeeType = MethodDecl->getThisObjectType();
      ThisPointeeLoc = &createStorageLocation(ThisPointeeType);
      if (Value *ThisPointeeVal = createValue(ThisPointeeType))
        setValue(*ThisPointeeLoc, *ThisPointeeVal);
    }
  }
}

bool Environment::canDescend(unsigned MaxDepth,
                             const DeclContext *Callee) const {
  return CallStack.size() <= MaxDepth && !llvm::is_contained(CallStack, Callee);
}

Environment Environment::pushCall(const CallExpr *Call) const {
  Environment Env(*this);

  // FIXME: Support references here.
  Env.ReturnLoc = getStorageLocation(*Call, SkipPast::Reference);

  if (const auto *MethodCall = dyn_cast<CXXMemberCallExpr>(Call)) {
    if (const Expr *Arg = MethodCall->getImplicitObjectArgument()) {
      if (!isa<CXXThisExpr>(Arg))
        Env.ThisPointeeLoc = getStorageLocation(*Arg, SkipPast::Reference);
      // Otherwise (when the argument is `this`), retain the current
      // environment's `ThisPointeeLoc`.
    }
  }

  Env.pushCallInternal(Call->getDirectCallee(),
                       llvm::ArrayRef(Call->getArgs(), Call->getNumArgs()));

  return Env;
}

Environment Environment::pushCall(const CXXConstructExpr *Call) const {
  Environment Env(*this);

  // FIXME: Support references here.
  Env.ReturnLoc = getStorageLocation(*Call, SkipPast::Reference);

  Env.ThisPointeeLoc = Env.ReturnLoc;

  Env.pushCallInternal(Call->getConstructor(),
                       llvm::ArrayRef(Call->getArgs(), Call->getNumArgs()));

  return Env;
}

void Environment::pushCallInternal(const FunctionDecl *FuncDecl,
                                   ArrayRef<const Expr *> Args) {
  CallStack.push_back(FuncDecl);

  initFieldsGlobalsAndFuncs(FuncDecl);

  const auto *ParamIt = FuncDecl->param_begin();

  // FIXME: Parameters don't always map to arguments 1:1; examples include
  // overloaded operators implemented as member functions, and parameter packs.
  for (unsigned ArgIndex = 0; ArgIndex < Args.size(); ++ParamIt, ++ArgIndex) {
    assert(ParamIt != FuncDecl->param_end());

    const Expr *Arg = Args[ArgIndex];
    auto *ArgLoc = getStorageLocation(*Arg, SkipPast::Reference);
    if (ArgLoc == nullptr)
      continue;

    const VarDecl *Param = *ParamIt;
    auto &Loc = createStorageLocation(*Param);
    setStorageLocation(*Param, Loc);

    QualType ParamType = Param->getType();
    if (ParamType->isReferenceType()) {
      auto &Val = create<ReferenceValue>(*ArgLoc);
      setValue(Loc, Val);
    } else if (auto *ArgVal = getValue(*ArgLoc)) {
      setValue(Loc, *ArgVal);
    } else if (Value *Val = createValue(ParamType)) {
      setValue(Loc, *Val);
    }
  }
}

void Environment::popCall(const Environment &CalleeEnv) {
  // We ignore `DACtx` because it's already the same in both. We don't want the
  // callee's `DeclCtx`, `ReturnLoc` or `ThisPointeeLoc`. We don't bring back
  // `DeclToLoc` and `ExprToLoc` because we want to be able to later analyze the
  // same callee in a different context, and `setStorageLocation` requires there
  // to not already be a storage location assigned. Conceptually, these maps
  // capture information from the local scope, so when popping that scope, we do
  // not propagate the maps.
  this->LocToVal = std::move(CalleeEnv.LocToVal);
  this->MemberLocToStruct = std::move(CalleeEnv.MemberLocToStruct);
  this->FlowConditionToken = std::move(CalleeEnv.FlowConditionToken);
}

bool Environment::equivalentTo(const Environment &Other,
                               Environment::ValueModel &Model) const {
  assert(DACtx == Other.DACtx);

  if (ReturnLoc != Other.ReturnLoc)
    return false;

  if (ThisPointeeLoc != Other.ThisPointeeLoc)
    return false;

  if (DeclToLoc != Other.DeclToLoc)
    return false;

  if (ExprToLoc != Other.ExprToLoc)
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

    if (!areEquivalentValues(*Val, *It->second) &&
        !compareDistinctValues(Loc->getType(), *Val, *this, *It->second, Other,
                               Model))
      return false;
  }

  return true;
}

LatticeJoinEffect Environment::widen(const Environment &PrevEnv,
                                     Environment::ValueModel &Model) {
  assert(DACtx == PrevEnv.DACtx);
  assert(ReturnLoc == PrevEnv.ReturnLoc);
  assert(ThisPointeeLoc == PrevEnv.ThisPointeeLoc);
  assert(CallStack == PrevEnv.CallStack);

  auto Effect = LatticeJoinEffect::Unchanged;

  // By the API, `PrevEnv` is a previous version of the environment for the same
  // block, so we have some guarantees about its shape. In particular, it will
  // be the result of a join or widen operation on previous values for this
  // block. For `DeclToLoc` and `ExprToLoc`, join guarantees that these maps are
  // subsets of the maps in `PrevEnv`. So, as long as we maintain this property
  // here, we don't need change their current values to widen.
  //
  // FIXME: `MemberLocToStruct` does not share the above property, because
  // `join` can cause the map size to increase (when we add fresh data in places
  // of conflict). Once this issue with join is resolved, re-enable the
  // assertion below or replace with something that captures the desired
  // invariant.
  assert(DeclToLoc.size() <= PrevEnv.DeclToLoc.size());
  assert(ExprToLoc.size() <= PrevEnv.ExprToLoc.size());
  // assert(MemberLocToStruct.size() <= PrevEnv.MemberLocToStruct.size());

  llvm::DenseMap<const StorageLocation *, Value *> WidenedLocToVal;
  for (auto &Entry : LocToVal) {
    const StorageLocation *Loc = Entry.first;
    assert(Loc != nullptr);

    Value *Val = Entry.second;
    assert(Val != nullptr);

    auto PrevIt = PrevEnv.LocToVal.find(Loc);
    if (PrevIt == PrevEnv.LocToVal.end())
      continue;
    assert(PrevIt->second != nullptr);

    if (areEquivalentValues(*Val, *PrevIt->second)) {
      WidenedLocToVal.insert({Loc, Val});
      continue;
    }

    Value &WidenedVal = widenDistinctValues(Loc->getType(), *PrevIt->second,
                                            PrevEnv, *Val, *this, Model);
    WidenedLocToVal.insert({Loc, &WidenedVal});
    if (&WidenedVal != PrevIt->second)
      Effect = LatticeJoinEffect::Changed;
  }
  LocToVal = std::move(WidenedLocToVal);
  // FIXME: update the equivalence calculation for `MemberLocToStruct`, once we
  // have a systematic way of soundly comparing this map.
  if (DeclToLoc.size() != PrevEnv.DeclToLoc.size() ||
      ExprToLoc.size() != PrevEnv.ExprToLoc.size() ||
      LocToVal.size() != PrevEnv.LocToVal.size() ||
      MemberLocToStruct.size() != PrevEnv.MemberLocToStruct.size())
    Effect = LatticeJoinEffect::Changed;

  return Effect;
}

LatticeJoinEffect Environment::join(const Environment &Other,
                                    Environment::ValueModel &Model) {
  assert(DACtx == Other.DACtx);
  assert(ReturnLoc == Other.ReturnLoc);
  assert(ThisPointeeLoc == Other.ThisPointeeLoc);
  assert(CallStack == Other.CallStack);

  auto Effect = LatticeJoinEffect::Unchanged;

  Environment JoinedEnv(*DACtx);

  JoinedEnv.CallStack = CallStack;
  JoinedEnv.ReturnLoc = ReturnLoc;
  JoinedEnv.ThisPointeeLoc = ThisPointeeLoc;

  JoinedEnv.DeclToLoc = intersectDenseMaps(DeclToLoc, Other.DeclToLoc);
  if (DeclToLoc.size() != JoinedEnv.DeclToLoc.size())
    Effect = LatticeJoinEffect::Changed;

  JoinedEnv.ExprToLoc = intersectDenseMaps(ExprToLoc, Other.ExprToLoc);
  if (ExprToLoc.size() != JoinedEnv.ExprToLoc.size())
    Effect = LatticeJoinEffect::Changed;

  JoinedEnv.MemberLocToStruct =
      intersectDenseMaps(MemberLocToStruct, Other.MemberLocToStruct);
  if (MemberLocToStruct.size() != JoinedEnv.MemberLocToStruct.size())
    Effect = LatticeJoinEffect::Changed;

  // FIXME: set `Effect` as needed.
  // FIXME: update join to detect backedges and simplify the flow condition
  // accordingly.
  JoinedEnv.FlowConditionToken = &DACtx->joinFlowConditions(
      *FlowConditionToken, *Other.FlowConditionToken);

  for (auto &Entry : LocToVal) {
    const StorageLocation *Loc = Entry.first;
    assert(Loc != nullptr);

    Value *Val = Entry.second;
    assert(Val != nullptr);

    auto It = Other.LocToVal.find(Loc);
    if (It == Other.LocToVal.end())
      continue;
    assert(It->second != nullptr);

    if (areEquivalentValues(*Val, *It->second)) {
      JoinedEnv.LocToVal.insert({Loc, Val});
      continue;
    }

    if (Value *MergedVal =
            mergeDistinctValues(Loc->getType(), *Val, *this, *It->second, Other,
                                JoinedEnv, Model)) {
      JoinedEnv.LocToVal.insert({Loc, MergedVal});
      Effect = LatticeJoinEffect::Changed;
    }
  }
  if (LocToVal.size() != JoinedEnv.LocToVal.size())
    Effect = LatticeJoinEffect::Changed;

  *this = std::move(JoinedEnv);

  return Effect;
}

StorageLocation &Environment::createStorageLocation(QualType Type) {
  return DACtx->createStorageLocation(Type);
}

StorageLocation &Environment::createStorageLocation(const VarDecl &D) {
  // Evaluated declarations are always assigned the same storage locations to
  // ensure that the environment stabilizes across loop iterations. Storage
  // locations for evaluated declarations are stored in the analysis context.
  return DACtx->getStableStorageLocation(D);
}

StorageLocation &Environment::createStorageLocation(const Expr &E) {
  // Evaluated expressions are always assigned the same storage locations to
  // ensure that the environment stabilizes across loop iterations. Storage
  // locations for evaluated expressions are stored in the analysis context.
  return DACtx->getStableStorageLocation(E);
}

void Environment::setStorageLocation(const ValueDecl &D, StorageLocation &Loc) {
  assert(!DeclToLoc.contains(&D));
  DeclToLoc[&D] = &Loc;
}

StorageLocation *Environment::getStorageLocation(const ValueDecl &D,
                                                 SkipPast SP) const {
  auto It = DeclToLoc.find(&D);
  return It == DeclToLoc.end() ? nullptr : &skip(*It->second, SP);
}

void Environment::setStorageLocation(const Expr &E, StorageLocation &Loc) {
  const Expr &CanonE = ignoreCFGOmittedNodes(E);
  assert(!ExprToLoc.contains(&CanonE));
  ExprToLoc[&CanonE] = &Loc;
}

StorageLocation *Environment::getStorageLocation(const Expr &E,
                                                 SkipPast SP) const {
  // FIXME: Add a test with parens.
  auto It = ExprToLoc.find(&ignoreCFGOmittedNodes(E));
  return It == ExprToLoc.end() ? nullptr : &skip(*It->second, SP);
}

StorageLocation *Environment::getThisPointeeStorageLocation() const {
  return ThisPointeeLoc;
}

StorageLocation *Environment::getReturnStorageLocation() const {
  return ReturnLoc;
}

PointerValue &Environment::getOrCreateNullPointerValue(QualType PointeeType) {
  return DACtx->getOrCreateNullPointerValue(PointeeType);
}

void Environment::setValue(const StorageLocation &Loc, Value &Val) {
  LocToVal[&Loc] = &Val;

  if (auto *StructVal = dyn_cast<StructValue>(&Val)) {
    auto &AggregateLoc = *cast<AggregateStorageLocation>(&Loc);

    const QualType Type = AggregateLoc.getType();
    assert(Type->isRecordType());

    for (const FieldDecl *Field : DACtx->getReferencedFields(Type)) {
      assert(Field != nullptr);
      StorageLocation &FieldLoc = AggregateLoc.getChild(*Field);
      MemberLocToStruct[&FieldLoc] = std::make_pair(StructVal, Field);
      if (auto *FieldVal = StructVal->getChild(*Field))
        setValue(FieldLoc, *FieldVal);
    }
  }

  auto It = MemberLocToStruct.find(&Loc);
  if (It != MemberLocToStruct.end()) {
    // `Loc` is the location of a struct member so we need to also update the
    // value of the member in the corresponding `StructValue`.

    assert(It->second.first != nullptr);
    StructValue &StructVal = *It->second.first;

    assert(It->second.second != nullptr);
    const ValueDecl &Member = *It->second.second;

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
    llvm::errs() << "Attempting to initialize a huge value of type: " << Type
                 << '\n';
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
    // FIXME: consider instead `return nullptr`, given that we do nothing useful
    // with integers, and so distinguishing them serves no purpose, but could
    // prevent convergence.
    CreatedValuesCount++;
    return &create<IntegerValue>();
  }

  if (Type->isReferenceType() || Type->isPointerType()) {
    CreatedValuesCount++;
    QualType PointeeType = Type->getPointeeType();
    auto &PointeeLoc = createStorageLocation(PointeeType);

    if (Visited.insert(PointeeType.getCanonicalType()).second) {
      Value *PointeeVal = createValueUnlessSelfReferential(
          PointeeType, Visited, Depth, CreatedValuesCount);
      Visited.erase(PointeeType.getCanonicalType());

      if (PointeeVal != nullptr)
        setValue(PointeeLoc, *PointeeVal);
    }

    if (Type->isReferenceType())
      return &create<ReferenceValue>(PointeeLoc);
    else
      return &create<PointerValue>(PointeeLoc);
  }

  if (Type->isRecordType()) {
    CreatedValuesCount++;
    llvm::DenseMap<const ValueDecl *, Value *> FieldValues;
    for (const FieldDecl *Field : DACtx->getReferencedFields(Type)) {
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

    return &create<StructValue>(std::move(FieldValues));
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
      return Val->getReferentLoc();
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
  DACtx->addFlowConditionConstraint(*FlowConditionToken, Val);
}

bool Environment::flowConditionImplies(BoolValue &Val) const {
  return DACtx->flowConditionImplies(*FlowConditionToken, Val);
}

void Environment::dump(raw_ostream &OS) const {
  // FIXME: add printing for remaining fields and allow caller to decide what
  // fields are printed.
  OS << "DeclToLoc:\n";
  for (auto [D, L] : DeclToLoc)
    OS << "  [" << D->getName() << ", " << L << "]\n";

  OS << "ExprToLoc:\n";
  for (auto [E, L] : ExprToLoc)
    OS << "  [" << E << ", " << L << "]\n";

  OS << "LocToVal:\n";
  for (auto [L, V] : LocToVal) {
    OS << "  [" << L << ", " << V << ": " << *V << "]\n";
  }

  OS << "FlowConditionToken:\n";
  DACtx->dumpFlowCondition(*FlowConditionToken, OS);
}

void Environment::dump() const {
  dump(llvm::dbgs());
}

} // namespace dataflow
} // namespace clang
