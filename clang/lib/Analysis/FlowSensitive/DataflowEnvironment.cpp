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
#include "llvm/ADT/MapVector.h"
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
    auto &Expr1 = cast<BoolValue>(Val1).formula();
    auto &Expr2 = cast<BoolValue>(Val2).formula();
    auto &A = MergedEnv.arena();
    auto &MergedVal = A.makeAtomRef(A.makeAtom());
    MergedEnv.addToFlowCondition(
        A.makeOr(A.makeAnd(A.makeAtomRef(Env1.getFlowConditionToken()),
                           A.makeEquals(MergedVal, Expr1)),
                 A.makeAnd(A.makeAtomRef(Env2.getFlowConditionToken()),
                           A.makeEquals(MergedVal, Expr2))));
    return &A.makeBoolValue(MergedVal);
  }

  Value *MergedVal = nullptr;
  if (auto *StructVal1 = dyn_cast<StructValue>(&Val1)) {
    [[maybe_unused]] auto *StructVal2 = cast<StructValue>(&Val2);

    // Values to be merged are always associated with the same location in
    // `LocToVal`. The location stored in `StructVal` should therefore also
    // be the same.
    assert(&StructVal1->getAggregateLoc() == &StructVal2->getAggregateLoc());

    // `StructVal1` and `StructVal2` may have different properties associated
    // with them. Create a new `StructValue` without any properties so that we
    // soundly approximate both values. If a particular analysis needs to merge
    // properties, it should do so in `DataflowAnalysis::merge()`.
    MergedVal = &MergedEnv.create<StructValue>(StructVal1->getAggregateLoc());
  } else {
    MergedVal = MergedEnv.createValue(Type);
  }

  // FIXME: Consider destroying `MergedValue` immediately if `ValueModel::merge`
  // returns false to avoid storing unneeded values in `DACtx`.
  // FIXME: Creating the value based on the type alone creates misshapen values
  // for lvalues, since the type does not reflect the need for `ReferenceValue`.
  // This issue will be resolved when `ReferenceValue` is eliminated as part
  // of the ongoing migration to strict handling of value categories (see
  // https://discourse.llvm.org/t/70086 for details).
  if (MergedVal)
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
getFieldsGlobalsAndFuncs(const Decl &D, FieldSet &Fields,
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
getFieldsGlobalsAndFuncs(const Stmt &S, FieldSet &Fields,
                         llvm::DenseSet<const VarDecl *> &Vars,
                         llvm::DenseSet<const FunctionDecl *> &Funcs) {
  for (auto *Child : S.children())
    if (Child != nullptr)
      getFieldsGlobalsAndFuncs(*Child, Fields, Vars, Funcs);
  if (const auto *DefaultInit = dyn_cast<CXXDefaultInitExpr>(&S))
    getFieldsGlobalsAndFuncs(*DefaultInit->getExpr(), Fields, Vars, Funcs);

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
  } else if (auto *InitList = dyn_cast<InitListExpr>(&S)) {
    if (RecordDecl *RD = InitList->getType()->getAsRecordDecl())
      for (const auto *FD : getFieldsForInitListExpr(RD))
        Fields.insert(FD);
  }
}

// FIXME: Add support for resetting globals after function calls to enable
// the implementation of sound analyses.
void Environment::initFieldsGlobalsAndFuncs(const FunctionDecl *FuncDecl) {
  assert(FuncDecl->getBody() != nullptr);

  FieldSet Fields;
  llvm::DenseSet<const VarDecl *> Vars;
  llvm::DenseSet<const FunctionDecl *> Funcs;

  // Look for global variable and field references in the
  // constructor-initializers.
  if (const auto *CtorDecl = dyn_cast<CXXConstructorDecl>(FuncDecl)) {
    for (const auto *Init : CtorDecl->inits()) {
      if (Init->isMemberInitializer()) {
        Fields.insert(Init->getMember());
      } else if (Init->isIndirectMemberInitializer()) {
        for (const auto *I : Init->getIndirectMember()->chain())
          Fields.insert(cast<FieldDecl>(I));
      }
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
    if (getStorageLocation(*D) != nullptr)
      continue;

    setStorageLocation(*D, createObject(*D));
  }

  for (const FunctionDecl *FD : Funcs) {
    if (getStorageLocation(*FD) != nullptr)
      continue;
    auto &Loc = createStorageLocation(FD->getType());
    setStorageLocation(*FD, Loc);
  }
}

Environment::Environment(DataflowAnalysisContext &DACtx)
    : DACtx(&DACtx),
      FlowConditionToken(DACtx.arena().makeFlowConditionToken()) {}

Environment Environment::fork() const {
  Environment Copy(*this);
  Copy.FlowConditionToken = DACtx->forkFlowCondition(FlowConditionToken);
  return Copy;
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
      setStorageLocation(*ParamDecl, createObject(*ParamDecl, nullptr));
    }
  }

  if (const auto *MethodDecl = dyn_cast<CXXMethodDecl>(&DeclCtx)) {
    auto *Parent = MethodDecl->getParent();
    assert(Parent != nullptr);
    if (Parent->isLambda())
      MethodDecl = dyn_cast<CXXMethodDecl>(Parent->getDeclContext());

    // FIXME: Initialize the ThisPointeeLoc of lambdas too.
    if (MethodDecl && !MethodDecl->isStatic()) {
      QualType ThisPointeeType = MethodDecl->getThisObjectType();
      ThisPointeeLoc =
          &cast<StructValue>(createValue(ThisPointeeType))->getAggregateLoc();
    }
  }
}

bool Environment::canDescend(unsigned MaxDepth,
                             const DeclContext *Callee) const {
  return CallStack.size() <= MaxDepth && !llvm::is_contained(CallStack, Callee);
}

Environment Environment::pushCall(const CallExpr *Call) const {
  Environment Env(*this);

  if (const auto *MethodCall = dyn_cast<CXXMemberCallExpr>(Call)) {
    if (const Expr *Arg = MethodCall->getImplicitObjectArgument()) {
      if (!isa<CXXThisExpr>(Arg))
        Env.ThisPointeeLoc = cast<AggregateStorageLocation>(
            getStorageLocation(*Arg, SkipPast::Reference));
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

  Env.ThisPointeeLoc = &Env.getResultObjectLocation(*Call);

  Env.pushCallInternal(Call->getConstructor(),
                       llvm::ArrayRef(Call->getArgs(), Call->getNumArgs()));

  return Env;
}

void Environment::pushCallInternal(const FunctionDecl *FuncDecl,
                                   ArrayRef<const Expr *> Args) {
  // Canonicalize to the definition of the function. This ensures that we're
  // putting arguments into the same `ParamVarDecl`s` that the callee will later
  // be retrieving them from.
  assert(FuncDecl->getDefinition() != nullptr);
  FuncDecl = FuncDecl->getDefinition();

  CallStack.push_back(FuncDecl);

  initFieldsGlobalsAndFuncs(FuncDecl);

  const auto *ParamIt = FuncDecl->param_begin();

  // FIXME: Parameters don't always map to arguments 1:1; examples include
  // overloaded operators implemented as member functions, and parameter packs.
  for (unsigned ArgIndex = 0; ArgIndex < Args.size(); ++ParamIt, ++ArgIndex) {
    assert(ParamIt != FuncDecl->param_end());
    const VarDecl *Param = *ParamIt;
    setStorageLocation(*Param, createObject(*Param, Args[ArgIndex]));
  }
}

void Environment::popCall(const CallExpr *Call, const Environment &CalleeEnv) {
  // We ignore `DACtx` because it's already the same in both. We don't want the
  // callee's `DeclCtx`, `ReturnVal`, `ReturnLoc` or `ThisPointeeLoc`. We don't
  // bring back `DeclToLoc` and `ExprToLoc` because we want to be able to later
  // analyze the same callee in a different context, and `setStorageLocation`
  // requires there to not already be a storage location assigned. Conceptually,
  // these maps capture information from the local scope, so when popping that
  // scope, we do not propagate the maps.
  this->LocToVal = std::move(CalleeEnv.LocToVal);
  this->FlowConditionToken = std::move(CalleeEnv.FlowConditionToken);

  if (Call->isGLValue()) {
    if (CalleeEnv.ReturnLoc != nullptr)
      setStorageLocationStrict(*Call, *CalleeEnv.ReturnLoc);
  } else if (!Call->getType()->isVoidType()) {
    if (CalleeEnv.ReturnVal != nullptr)
      setValueStrict(*Call, *CalleeEnv.ReturnVal);
  }
}

void Environment::popCall(const CXXConstructExpr *Call,
                          const Environment &CalleeEnv) {
  // See also comment in `popCall(const CallExpr *, const Environment &)` above.
  this->LocToVal = std::move(CalleeEnv.LocToVal);
  this->FlowConditionToken = std::move(CalleeEnv.FlowConditionToken);

  if (Value *Val = CalleeEnv.getValue(*CalleeEnv.ThisPointeeLoc)) {
    setValueStrict(*Call, *Val);
  }
}

bool Environment::equivalentTo(const Environment &Other,
                               Environment::ValueModel &Model) const {
  assert(DACtx == Other.DACtx);

  if (ReturnVal != Other.ReturnVal)
    return false;

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
  assert(ReturnVal == PrevEnv.ReturnVal);
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
  assert(DeclToLoc.size() <= PrevEnv.DeclToLoc.size());
  assert(ExprToLoc.size() <= PrevEnv.ExprToLoc.size());

  llvm::MapVector<const StorageLocation *, Value *> WidenedLocToVal;
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
  if (DeclToLoc.size() != PrevEnv.DeclToLoc.size() ||
      ExprToLoc.size() != PrevEnv.ExprToLoc.size() ||
      LocToVal.size() != PrevEnv.LocToVal.size())
    Effect = LatticeJoinEffect::Changed;

  return Effect;
}

Environment Environment::join(const Environment &EnvA, const Environment &EnvB,
                              Environment::ValueModel &Model) {
  assert(EnvA.DACtx == EnvB.DACtx);
  assert(EnvA.ThisPointeeLoc == EnvB.ThisPointeeLoc);
  assert(EnvA.CallStack == EnvB.CallStack);

  Environment JoinedEnv(*EnvA.DACtx);

  JoinedEnv.CallStack = EnvA.CallStack;
  JoinedEnv.ThisPointeeLoc = EnvA.ThisPointeeLoc;

  if (EnvA.ReturnVal == nullptr || EnvB.ReturnVal == nullptr) {
    // `ReturnVal` might not always get set -- for example if we have a return
    // statement of the form `return some_other_func()` and we decide not to
    // analyze `some_other_func()`.
    // In this case, we can't say anything about the joined return value -- we
    // don't simply want to propagate the return value that we do have, because
    // it might not be the correct one.
    // This occurs for example in the test `ContextSensitiveMutualRecursion`.
    JoinedEnv.ReturnVal = nullptr;
  } else if (areEquivalentValues(*EnvA.ReturnVal, *EnvB.ReturnVal)) {
    JoinedEnv.ReturnVal = EnvA.ReturnVal;
  } else {
    assert(!EnvA.CallStack.empty());
    // FIXME: Make `CallStack` a vector of `FunctionDecl` so we don't need this
    // cast.
    auto *Func = dyn_cast<FunctionDecl>(EnvA.CallStack.back());
    assert(Func != nullptr);
    if (Value *MergedVal =
            mergeDistinctValues(Func->getReturnType(), *EnvA.ReturnVal, EnvA,
                                *EnvB.ReturnVal, EnvB, JoinedEnv, Model))
      JoinedEnv.ReturnVal = MergedVal;
  }

  if (EnvA.ReturnLoc == EnvB.ReturnLoc)
    JoinedEnv.ReturnLoc = EnvA.ReturnLoc;
  else
    JoinedEnv.ReturnLoc = nullptr;

  // FIXME: Once we're able to remove declarations from `DeclToLoc` when their
  // lifetime ends, add an assertion that there aren't any entries in
  // `DeclToLoc` and `Other.DeclToLoc` that map the same declaration to
  // different storage locations.
  JoinedEnv.DeclToLoc = intersectDenseMaps(EnvA.DeclToLoc, EnvB.DeclToLoc);

  JoinedEnv.ExprToLoc = intersectDenseMaps(EnvA.ExprToLoc, EnvB.ExprToLoc);

  // FIXME: update join to detect backedges and simplify the flow condition
  // accordingly.
  JoinedEnv.FlowConditionToken = EnvA.DACtx->joinFlowConditions(
      EnvA.FlowConditionToken, EnvB.FlowConditionToken);

  for (auto &Entry : EnvA.LocToVal) {
    const StorageLocation *Loc = Entry.first;
    assert(Loc != nullptr);

    Value *Val = Entry.second;
    assert(Val != nullptr);

    auto It = EnvB.LocToVal.find(Loc);
    if (It == EnvB.LocToVal.end())
      continue;
    assert(It->second != nullptr);

    if (areEquivalentValues(*Val, *It->second)) {
      JoinedEnv.LocToVal.insert({Loc, Val});
      continue;
    }

    if (Value *MergedVal = mergeDistinctValues(
            Loc->getType(), *Val, EnvA, *It->second, EnvB, JoinedEnv, Model)) {
      JoinedEnv.LocToVal.insert({Loc, MergedVal});
    }
  }

  return JoinedEnv;
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
  assert(!isa_and_nonnull<ReferenceValue>(getValue(Loc)));
  DeclToLoc[&D] = &Loc;
}

StorageLocation *Environment::getStorageLocation(const ValueDecl &D) const {
  auto It = DeclToLoc.find(&D);
  if (It == DeclToLoc.end())
    return nullptr;

  StorageLocation *Loc = It->second;

  assert(!isa_and_nonnull<ReferenceValue>(getValue(*Loc)));

  return Loc;
}

void Environment::setStorageLocation(const Expr &E, StorageLocation &Loc) {
  const Expr &CanonE = ignoreCFGOmittedNodes(E);
  assert(!ExprToLoc.contains(&CanonE));
  ExprToLoc[&CanonE] = &Loc;
}

void Environment::setStorageLocationStrict(const Expr &E,
                                           StorageLocation &Loc) {
  // `DeclRefExpr`s to builtin function types aren't glvalues, for some reason,
  // but we still want to be able to associate a `StorageLocation` with them,
  // so allow these as an exception.
  assert(E.isGLValue() ||
         E.getType()->isSpecificBuiltinType(BuiltinType::BuiltinFn));
  setStorageLocation(E, Loc);
}

StorageLocation *Environment::getStorageLocation(const Expr &E,
                                                 SkipPast SP) const {
  // FIXME: Add a test with parens.
  auto It = ExprToLoc.find(&ignoreCFGOmittedNodes(E));
  return It == ExprToLoc.end() ? nullptr : &skip(*It->second, SP);
}

StorageLocation *Environment::getStorageLocationStrict(const Expr &E) const {
  // See comment in `setStorageLocationStrict()`.
  assert(E.isGLValue() ||
         E.getType()->isSpecificBuiltinType(BuiltinType::BuiltinFn));
  StorageLocation *Loc = getStorageLocation(E, SkipPast::None);

  if (Loc == nullptr)
    return nullptr;

  if (auto *RefVal = dyn_cast_or_null<ReferenceValue>(getValue(*Loc)))
    return &RefVal->getReferentLoc();

  return Loc;
}

AggregateStorageLocation *Environment::getThisPointeeStorageLocation() const {
  return ThisPointeeLoc;
}

AggregateStorageLocation &
Environment::getResultObjectLocation(const Expr &RecordPRValue) {
  assert(RecordPRValue.getType()->isRecordType());
  assert(RecordPRValue.isPRValue());

  if (StorageLocation *ExistingLoc =
          getStorageLocation(RecordPRValue, SkipPast::None))
    return *cast<AggregateStorageLocation>(ExistingLoc);
  auto &Loc = cast<AggregateStorageLocation>(
      DACtx->getStableStorageLocation(RecordPRValue));
  setStorageLocation(RecordPRValue, Loc);
  return Loc;
}

PointerValue &Environment::getOrCreateNullPointerValue(QualType PointeeType) {
  return DACtx->getOrCreateNullPointerValue(PointeeType);
}

void Environment::setValue(const StorageLocation &Loc, Value &Val) {
  assert(!isa<StructValue>(&Val) ||
         &cast<StructValue>(&Val)->getAggregateLoc() == &Loc);

  LocToVal[&Loc] = &Val;
}

void Environment::setValueStrict(const Expr &E, Value &Val) {
  assert(E.isPRValue());
  assert(!isa<ReferenceValue>(Val));

  if (auto *StructVal = dyn_cast<StructValue>(&Val)) {
    if (auto *ExistingVal = cast_or_null<StructValue>(getValueStrict(E)))
      assert(&ExistingVal->getAggregateLoc() == &StructVal->getAggregateLoc());
    if (StorageLocation *ExistingLoc = getStorageLocation(E, SkipPast::None))
      assert(ExistingLoc == &StructVal->getAggregateLoc());
    else
      setStorageLocation(E, StructVal->getAggregateLoc());
    setValue(StructVal->getAggregateLoc(), Val);
    return;
  }

  StorageLocation *Loc = getStorageLocation(E, SkipPast::None);
  if (Loc == nullptr) {
    Loc = &createStorageLocation(E);
    setStorageLocation(E, *Loc);
  }
  setValue(*Loc, Val);
}

Value *Environment::getValue(const StorageLocation &Loc) const {
  return LocToVal.lookup(&Loc);
}

Value *Environment::getValue(const ValueDecl &D) const {
  auto *Loc = getStorageLocation(D);
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

Value *Environment::getValueStrict(const Expr &E) const {
  assert(E.isPRValue());
  Value *Val = getValue(E, SkipPast::None);

  assert(Val == nullptr || !isa<ReferenceValue>(Val));

  return Val;
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
    return &arena().create<IntegerValue>();
  }

  if (Type->isReferenceType() || Type->isPointerType()) {
    CreatedValuesCount++;
    QualType PointeeType = Type->getPointeeType();
    StorageLocation &PointeeLoc =
        createLocAndMaybeValue(PointeeType, Visited, Depth, CreatedValuesCount);

    if (Type->isReferenceType())
      return &arena().create<ReferenceValue>(PointeeLoc);
    else
      return &arena().create<PointerValue>(PointeeLoc);
  }

  if (Type->isRecordType()) {
    CreatedValuesCount++;
    llvm::DenseMap<const ValueDecl *, StorageLocation *> FieldLocs;
    for (const FieldDecl *Field : DACtx->getModeledFields(Type)) {
      assert(Field != nullptr);

      QualType FieldType = Field->getType();

      FieldLocs.insert(
          {Field, &createLocAndMaybeValue(FieldType, Visited, Depth + 1,
                                          CreatedValuesCount)});
    }

    AggregateStorageLocation &Loc =
        arena().create<AggregateStorageLocation>(Type, std::move(FieldLocs));
    StructValue &StructVal = create<StructValue>(Loc);

    // As we already have a storage location for the `StructValue`, we can and
    // should associate them in the environment.
    setValue(Loc, StructVal);

    return &StructVal;
  }

  return nullptr;
}

StorageLocation &
Environment::createLocAndMaybeValue(QualType Ty,
                                    llvm::DenseSet<QualType> &Visited,
                                    int Depth, int &CreatedValuesCount) {
  if (!Visited.insert(Ty.getCanonicalType()).second)
    return createStorageLocation(Ty.getNonReferenceType());
  Value *Val = createValueUnlessSelfReferential(
      Ty.getNonReferenceType(), Visited, Depth, CreatedValuesCount);
  Visited.erase(Ty.getCanonicalType());

  Ty = Ty.getNonReferenceType();

  if (Val == nullptr)
    return createStorageLocation(Ty);

  if (Ty->isRecordType())
    return cast<StructValue>(Val)->getAggregateLoc();

  StorageLocation &Loc = createStorageLocation(Ty);
  setValue(Loc, *Val);
  return Loc;
}

StorageLocation &Environment::createObjectInternal(const VarDecl *D,
                                                   QualType Ty,
                                                   const Expr *InitExpr) {
  if (Ty->isReferenceType()) {
    // Although variables of reference type always need to be initialized, it
    // can happen that we can't see the initializer, so `InitExpr` may still
    // be null.
    if (InitExpr) {
      if (auto *InitExprLoc =
              getStorageLocation(*InitExpr, SkipPast::Reference))
        return *InitExprLoc;
    }

    // Even though we have an initializer, we might not get an
    // InitExprLoc, for example if the InitExpr is a CallExpr for which we
    // don't have a function body. In this case, we just invent a storage
    // location and value -- it's the best we can do.
    return createObjectInternal(D, Ty.getNonReferenceType(), nullptr);
  }

  Value *Val = nullptr;
  if (InitExpr)
    // In the (few) cases where an expression is intentionally
    // "uninterpreted", `InitExpr` is not associated with a value.  There are
    // two ways to handle this situation: propagate the status, so that
    // uninterpreted initializers result in uninterpreted variables, or
    // provide a default value. We choose the latter so that later refinements
    // of the variable can be used for reasoning about the surrounding code.
    // For this reason, we let this case be handled by the `createValue()`
    // call below.
    //
    // FIXME. If and when we interpret all language cases, change this to
    // assert that `InitExpr` is interpreted, rather than supplying a
    // default value (assuming we don't update the environment API to return
    // references).
    Val = getValueStrict(*InitExpr);
  if (!Val)
    Val = createValue(Ty);

  if (Ty->isRecordType())
    return cast<StructValue>(Val)->getAggregateLoc();

  StorageLocation &Loc =
      D ? createStorageLocation(*D) : createStorageLocation(Ty);

  if (Val)
    setValue(Loc, *Val);

  return Loc;
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
  }
  llvm_unreachable("bad SkipPast kind");
}

const StorageLocation &Environment::skip(const StorageLocation &Loc,
                                         SkipPast SP) const {
  return skip(*const_cast<StorageLocation *>(&Loc), SP);
}

void Environment::addToFlowCondition(const Formula &Val) {
  DACtx->addFlowConditionConstraint(FlowConditionToken, Val);
}

bool Environment::flowConditionImplies(const Formula &Val) const {
  return DACtx->flowConditionImplies(FlowConditionToken, Val);
}

void Environment::dump(raw_ostream &OS) const {
  // FIXME: add printing for remaining fields and allow caller to decide what
  // fields are printed.
  OS << "DeclToLoc:\n";
  for (auto [D, L] : DeclToLoc)
    OS << "  [" << D->getNameAsString() << ", " << L << "]\n";

  OS << "ExprToLoc:\n";
  for (auto [E, L] : ExprToLoc)
    OS << "  [" << E << ", " << L << "]\n";

  OS << "LocToVal:\n";
  for (auto [L, V] : LocToVal) {
    OS << "  [" << L << ", " << V << ": " << *V << "]\n";
  }

  OS << "FlowConditionToken:\n";
  DACtx->dumpFlowCondition(FlowConditionToken, OS);
}

void Environment::dump() const {
  dump(llvm::dbgs());
}

AggregateStorageLocation *
getImplicitObjectLocation(const CXXMemberCallExpr &MCE,
                          const Environment &Env) {
  Expr *ImplicitObject = MCE.getImplicitObjectArgument();
  if (ImplicitObject == nullptr)
    return nullptr;
  StorageLocation *Loc =
      Env.getStorageLocation(*ImplicitObject, SkipPast::Reference);
  if (Loc == nullptr)
    return nullptr;
  if (ImplicitObject->getType()->isPointerType()) {
    if (auto *Val = cast_or_null<PointerValue>(Env.getValue(*Loc)))
      return &cast<AggregateStorageLocation>(Val->getPointeeLoc());
    return nullptr;
  }
  return cast<AggregateStorageLocation>(Loc);
}

AggregateStorageLocation *getBaseObjectLocation(const MemberExpr &ME,
                                                const Environment &Env) {
  Expr *Base = ME.getBase();
  if (Base == nullptr)
    return nullptr;
  StorageLocation *Loc = Env.getStorageLocation(*Base, SkipPast::Reference);
  if (Loc == nullptr)
    return nullptr;
  if (ME.isArrow()) {
    if (auto *Val = cast_or_null<PointerValue>(Env.getValue(*Loc)))
      return &cast<AggregateStorageLocation>(Val->getPointeeLoc());
    return nullptr;
  }
  return cast<AggregateStorageLocation>(Loc);
}

std::vector<FieldDecl *> getFieldsForInitListExpr(const RecordDecl *RD) {
  // Unnamed bitfields are only used for padding and do not appear in
  // `InitListExpr`'s inits. However, those fields do appear in `RecordDecl`'s
  // field list, and we thus need to remove them before mapping inits to
  // fields to avoid mapping inits to the wrongs fields.
  std::vector<FieldDecl *> Fields;
  llvm::copy_if(
      RD->fields(), std::back_inserter(Fields),
      [](const FieldDecl *Field) { return !Field->isUnnamedBitfield(); });
  return Fields;
}

StructValue &refreshStructValue(AggregateStorageLocation &Loc,
                                Environment &Env) {
  auto &NewVal = Env.create<StructValue>(Loc);
  Env.setValue(Loc, NewVal);
  return NewVal;
}

StructValue &refreshStructValue(const Expr &Expr, Environment &Env) {
  assert(Expr.getType()->isRecordType());

  if (Expr.isPRValue()) {
    if (auto *ExistingVal =
            cast_or_null<StructValue>(Env.getValueStrict(Expr))) {
      auto &NewVal = Env.create<StructValue>(ExistingVal->getAggregateLoc());
      Env.setValueStrict(Expr, NewVal);
      return NewVal;
    }

    auto &NewVal = *cast<StructValue>(Env.createValue(Expr.getType()));
    Env.setValueStrict(Expr, NewVal);
    return NewVal;
  }

  if (auto *Loc = cast_or_null<AggregateStorageLocation>(
          Env.getStorageLocationStrict(Expr))) {
    auto &NewVal = Env.create<StructValue>(*Loc);
    Env.setValue(*Loc, NewVal);
    return NewVal;
  }

  auto &NewVal = *cast<StructValue>(Env.createValue(Expr.getType()));
  Env.setStorageLocationStrict(Expr, NewVal.getAggregateLoc());
  return NewVal;
}

} // namespace dataflow
} // namespace clang
