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
#include "llvm/Support/ErrorHandling.h"
#include <cassert>
#include <utility>

namespace clang {
namespace dataflow {

// FIXME: convert these to parameters of the analysis or environment. Current
// settings have been experimentaly validated, but only for a particular
// analysis.
static constexpr int MaxCompositeValueDepth = 3;
static constexpr int MaxCompositeValueSize = 1000;

/// Returns whether all declarations that `DeclToLoc1` and `DeclToLoc2` have in
/// common map to the same storage location in both maps.
bool declToLocConsistent(
    const llvm::DenseMap<const ValueDecl *, StorageLocation *> &DeclToLoc1,
    const llvm::DenseMap<const ValueDecl *, StorageLocation *> &DeclToLoc2) {
  for (auto &Entry : DeclToLoc1) {
    auto It = DeclToLoc2.find(Entry.first);
    if (It != DeclToLoc2.end() && Entry.second != It->second)
      return false;
  }

  return true;
}

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

// Whether to consider equivalent two values with an unknown relation.
//
// FIXME: this function is a hack enabling unsoundness to support
// convergence. Once we have widening support for the reference/pointer and
// struct built-in models, this should be unconditionally `false` (and inlined
// as such at its call sites).
static bool equateUnknownValues(Value::Kind K) {
  switch (K) {
  case Value::Kind::Integer:
  case Value::Kind::Pointer:
  case Value::Kind::Record:
    return true;
  default:
    return false;
  }
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
    return equateUnknownValues(Val1.getKind());
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
  if (auto *RecordVal1 = dyn_cast<RecordValue>(&Val1)) {
    auto *RecordVal2 = cast<RecordValue>(&Val2);

    if (&RecordVal1->getLoc() == &RecordVal2->getLoc())
      // `RecordVal1` and `RecordVal2` may have different properties associated
      // with them. Create a new `RecordValue` with the same location but
      // without any properties so that we soundly approximate both values. If a
      // particular analysis needs to merge properties, it should do so in
      // `DataflowAnalysis::merge()`.
      MergedVal = &MergedEnv.create<RecordValue>(RecordVal1->getLoc());
    else
      // If the locations for the two records are different, need to create a
      // completely new value.
      MergedVal = MergedEnv.createValue(Type);
  } else {
    MergedVal = MergedEnv.createValue(Type);
  }

  // FIXME: Consider destroying `MergedValue` immediately if `ValueModel::merge`
  // returns false to avoid storing unneeded values in `DACtx`.
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

  return equateUnknownValues(Prev.getKind()) ? Prev : Current;
}

// Returns whether the values in `Map1` and `Map2` compare equal for those
// keys that `Map1` and `Map2` have in common.
template <typename Key>
bool compareKeyToValueMaps(const llvm::MapVector<Key, Value *> &Map1,
                           const llvm::MapVector<Key, Value *> &Map2,
                           const Environment &Env1, const Environment &Env2,
                           Environment::ValueModel &Model) {
  for (auto &Entry : Map1) {
    Key K = Entry.first;
    assert(K != nullptr);

    Value *Val = Entry.second;
    assert(Val != nullptr);

    auto It = Map2.find(K);
    if (It == Map2.end())
      continue;
    assert(It->second != nullptr);

    if (!areEquivalentValues(*Val, *It->second) &&
        !compareDistinctValues(K->getType(), *Val, Env1, *It->second, Env2,
                               Model))
      return false;
  }

  return true;
}

// Perform a join on either `LocToVal` or `ExprToVal`. `Key` must be either
// `const StorageLocation *` or `const Expr *`.
template <typename Key>
llvm::MapVector<Key, Value *>
joinKeyToValueMap(const llvm::MapVector<Key, Value *> &Map1,
                  const llvm::MapVector<Key, Value *> &Map2,
                  const Environment &Env1, const Environment &Env2,
                  Environment &JoinedEnv, Environment::ValueModel &Model) {
  llvm::MapVector<Key, Value *> MergedMap;
  for (auto &Entry : Map1) {
    Key K = Entry.first;
    assert(K != nullptr);

    Value *Val = Entry.second;
    assert(Val != nullptr);

    auto It = Map2.find(K);
    if (It == Map2.end())
      continue;
    assert(It->second != nullptr);

    if (areEquivalentValues(*Val, *It->second)) {
      MergedMap.insert({K, Val});
      continue;
    }

    if (Value *MergedVal = mergeDistinctValues(
            K->getType(), *Val, Env1, *It->second, Env2, JoinedEnv, Model)) {
      MergedMap.insert({K, MergedVal});
    }
  }

  return MergedMap;
}

// Perform widening on either `LocToVal` or `ExprToVal`. `Key` must be either
// `const StorageLocation *` or `const Expr *`.
template <typename Key>
llvm::MapVector<Key, Value *>
widenKeyToValueMap(const llvm::MapVector<Key, Value *> &CurMap,
                   const llvm::MapVector<Key, Value *> &PrevMap,
                   Environment &CurEnv, const Environment &PrevEnv,
                   Environment::ValueModel &Model, LatticeJoinEffect &Effect) {
  llvm::MapVector<Key, Value *> WidenedMap;
  for (auto &Entry : CurMap) {
    Key K = Entry.first;
    assert(K != nullptr);

    Value *Val = Entry.second;
    assert(Val != nullptr);

    auto PrevIt = PrevMap.find(K);
    if (PrevIt == PrevMap.end())
      continue;
    assert(PrevIt->second != nullptr);

    if (areEquivalentValues(*Val, *PrevIt->second)) {
      WidenedMap.insert({K, Val});
      continue;
    }

    Value &WidenedVal = widenDistinctValues(K->getType(), *PrevIt->second,
                                            PrevEnv, *Val, CurEnv, Model);
    WidenedMap.insert({K, &WidenedVal});
    if (&WidenedVal != PrevIt->second)
      Effect = LatticeJoinEffect::Changed;
  }

  return WidenedMap;
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

static MemberExpr *getMemberForAccessor(const CXXMemberCallExpr &C) {
  if (!C.getMethodDecl())
    return nullptr;
  auto *Body = dyn_cast_or_null<CompoundStmt>(C.getMethodDecl()->getBody());
  if (!Body || Body->size() != 1)
    return nullptr;
  if (auto *RS = dyn_cast<ReturnStmt>(*Body->body_begin()))
    if (auto *Return = RS->getRetValue())
      return dyn_cast<MemberExpr>(Return->IgnoreParenImpCasts());
  return nullptr;
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
  } else if (const auto *C = dyn_cast<CXXMemberCallExpr>(&S)) {
    // If this is a method that returns a member variable but does nothing else,
    // model the field of the return value.
    if (MemberExpr *E = getMemberForAccessor(*C))
      if (const auto *FD = dyn_cast<FieldDecl>(E->getMemberDecl()))
        Fields.insert(FD);
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

    if (Parent->isLambda()) {
      for (auto Capture : Parent->captures()) {
        if (Capture.capturesVariable()) {
          const auto *VarDecl = Capture.getCapturedVar();
          assert(VarDecl != nullptr);
          setStorageLocation(*VarDecl, createObject(*VarDecl, nullptr));
        } else if (Capture.capturesThis()) {
          const auto *SurroundingMethodDecl =
              cast<CXXMethodDecl>(DeclCtx.getNonClosureAncestor());
          QualType ThisPointeeType =
              SurroundingMethodDecl->getFunctionObjectParameterType();
          ThisPointeeLoc =
              &cast<RecordValue>(createValue(ThisPointeeType))->getLoc();
        }
      }
    } else if (MethodDecl->isImplicitObjectMemberFunction()) {
      QualType ThisPointeeType = MethodDecl->getFunctionObjectParameterType();
      ThisPointeeLoc =
          &cast<RecordValue>(createValue(ThisPointeeType))->getLoc();
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
          Env.ThisPointeeLoc =
              cast<RecordStorageLocation>(getStorageLocation(*Arg));
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
  // We ignore some entries of `CalleeEnv`:
  // - `DACtx` because is already the same in both
  // - We don't want the callee's `DeclCtx`, `ReturnVal`, `ReturnLoc` or
  //   `ThisPointeeLoc` because they don't apply to us.
  // - `DeclToLoc`, `ExprToLoc`, and `ExprToVal` capture information from the
  //   callee's local scope, so when popping that scope, we do not propagate
  //   the maps.
  this->LocToVal = std::move(CalleeEnv.LocToVal);
  this->FlowConditionToken = std::move(CalleeEnv.FlowConditionToken);

  if (Call->isGLValue()) {
    if (CalleeEnv.ReturnLoc != nullptr)
      setStorageLocation(*Call, *CalleeEnv.ReturnLoc);
  } else if (!Call->getType()->isVoidType()) {
    if (CalleeEnv.ReturnVal != nullptr)
      setValue(*Call, *CalleeEnv.ReturnVal);
  }
}

void Environment::popCall(const CXXConstructExpr *Call,
                          const Environment &CalleeEnv) {
  // See also comment in `popCall(const CallExpr *, const Environment &)` above.
  this->LocToVal = std::move(CalleeEnv.LocToVal);
  this->FlowConditionToken = std::move(CalleeEnv.FlowConditionToken);

  if (Value *Val = CalleeEnv.getValue(*CalleeEnv.ThisPointeeLoc)) {
    setValue(*Call, *Val);
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

  if (!compareKeyToValueMaps(ExprToVal, Other.ExprToVal, *this, Other, Model))
    return false;

  if (!compareKeyToValueMaps(LocToVal, Other.LocToVal, *this, Other, Model))
    return false;

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
  // block. For `DeclToLoc`, `ExprToVal`, and `ExprToLoc`, join guarantees that
  // these maps are subsets of the maps in `PrevEnv`. So, as long as we maintain
  // this property here, we don't need change their current values to widen.
  assert(DeclToLoc.size() <= PrevEnv.DeclToLoc.size());
  assert(ExprToVal.size() <= PrevEnv.ExprToVal.size());
  assert(ExprToLoc.size() <= PrevEnv.ExprToLoc.size());

  ExprToVal = widenKeyToValueMap(ExprToVal, PrevEnv.ExprToVal, *this, PrevEnv,
                                 Model, Effect);

  LocToVal = widenKeyToValueMap(LocToVal, PrevEnv.LocToVal, *this, PrevEnv,
                                Model, Effect);
  if (DeclToLoc.size() != PrevEnv.DeclToLoc.size() ||
      ExprToLoc.size() != PrevEnv.ExprToLoc.size() ||
      ExprToVal.size() != PrevEnv.ExprToVal.size() ||
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

  assert(declToLocConsistent(EnvA.DeclToLoc, EnvB.DeclToLoc));
  JoinedEnv.DeclToLoc = intersectDenseMaps(EnvA.DeclToLoc, EnvB.DeclToLoc);

  JoinedEnv.ExprToLoc = intersectDenseMaps(EnvA.ExprToLoc, EnvB.ExprToLoc);

  // FIXME: update join to detect backedges and simplify the flow condition
  // accordingly.
  JoinedEnv.FlowConditionToken = EnvA.DACtx->joinFlowConditions(
      EnvA.FlowConditionToken, EnvB.FlowConditionToken);

  JoinedEnv.ExprToVal = joinKeyToValueMap(EnvA.ExprToVal, EnvB.ExprToVal, EnvA,
                                          EnvB, JoinedEnv, Model);

  JoinedEnv.LocToVal = joinKeyToValueMap(EnvA.LocToVal, EnvB.LocToVal, EnvA,
                                         EnvB, JoinedEnv, Model);

  return JoinedEnv;
}

StorageLocation &Environment::createStorageLocation(QualType Type) {
  return DACtx->createStorageLocation(Type);
}

StorageLocation &Environment::createStorageLocation(const ValueDecl &D) {
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

StorageLocation *Environment::getStorageLocation(const ValueDecl &D) const {
  auto It = DeclToLoc.find(&D);
  if (It == DeclToLoc.end())
    return nullptr;

  StorageLocation *Loc = It->second;

  return Loc;
}

void Environment::removeDecl(const ValueDecl &D) {
  assert(DeclToLoc.contains(&D));
  DeclToLoc.erase(&D);
}

void Environment::setStorageLocation(const Expr &E, StorageLocation &Loc) {
  // `DeclRefExpr`s to builtin function types aren't glvalues, for some reason,
  // but we still want to be able to associate a `StorageLocation` with them,
  // so allow these as an exception.
  assert(E.isGLValue() ||
         E.getType()->isSpecificBuiltinType(BuiltinType::BuiltinFn));
  setStorageLocationInternal(E, Loc);
}

StorageLocation *Environment::getStorageLocation(const Expr &E) const {
  // See comment in `setStorageLocation()`.
  assert(E.isGLValue() ||
         E.getType()->isSpecificBuiltinType(BuiltinType::BuiltinFn));
  return getStorageLocationInternal(E);
}

RecordStorageLocation *Environment::getThisPointeeStorageLocation() const {
  return ThisPointeeLoc;
}

RecordStorageLocation &
Environment::getResultObjectLocation(const Expr &RecordPRValue) {
  assert(RecordPRValue.getType()->isRecordType());
  assert(RecordPRValue.isPRValue());

  if (StorageLocation *ExistingLoc = getStorageLocationInternal(RecordPRValue))
    return *cast<RecordStorageLocation>(ExistingLoc);
  auto &Loc = cast<RecordStorageLocation>(
      DACtx->getStableStorageLocation(RecordPRValue));
  setStorageLocationInternal(RecordPRValue, Loc);
  return Loc;
}

PointerValue &Environment::getOrCreateNullPointerValue(QualType PointeeType) {
  return DACtx->getOrCreateNullPointerValue(PointeeType);
}

void Environment::setValue(const StorageLocation &Loc, Value &Val) {
  assert(!isa<RecordValue>(&Val) || &cast<RecordValue>(&Val)->getLoc() == &Loc);

  LocToVal[&Loc] = &Val;
}

void Environment::setValue(const Expr &E, Value &Val) {
  assert(E.isPRValue());
  ExprToVal[&E] = &Val;
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

Value *Environment::getValue(const Expr &E) const {
  if (E.isPRValue()) {
    auto It = ExprToVal.find(&ignoreCFGOmittedNodes(E));
    return It == ExprToVal.end() ? nullptr : It->second;
  }

  auto It = ExprToLoc.find(&ignoreCFGOmittedNodes(E));
  if (It == ExprToLoc.end())
    return nullptr;
  return getValue(*It->second);
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

void Environment::setStorageLocationInternal(const Expr &E,
                                             StorageLocation &Loc) {
  const Expr &CanonE = ignoreCFGOmittedNodes(E);
  assert(!ExprToLoc.contains(&CanonE));
  ExprToLoc[&CanonE] = &Loc;
}

StorageLocation *Environment::getStorageLocationInternal(const Expr &E) const {
  auto It = ExprToLoc.find(&ignoreCFGOmittedNodes(E));
  return It == ExprToLoc.end() ? nullptr : &*It->second;
}

Value *Environment::createValueUnlessSelfReferential(
    QualType Type, llvm::DenseSet<QualType> &Visited, int Depth,
    int &CreatedValuesCount) {
  assert(!Type.isNull());
  assert(!Type->isReferenceType());

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

  if (Type->isPointerType()) {
    CreatedValuesCount++;
    QualType PointeeType = Type->getPointeeType();
    StorageLocation &PointeeLoc =
        createLocAndMaybeValue(PointeeType, Visited, Depth, CreatedValuesCount);

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

    RecordStorageLocation &Loc =
        arena().create<RecordStorageLocation>(Type, std::move(FieldLocs));
    RecordValue &RecordVal = create<RecordValue>(Loc);

    // As we already have a storage location for the `RecordValue`, we can and
    // should associate them in the environment.
    setValue(Loc, RecordVal);

    return &RecordVal;
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
    return cast<RecordValue>(Val)->getLoc();

  StorageLocation &Loc = createStorageLocation(Ty);
  setValue(Loc, *Val);
  return Loc;
}

StorageLocation &Environment::createObjectInternal(const ValueDecl *D,
                                                   QualType Ty,
                                                   const Expr *InitExpr) {
  if (Ty->isReferenceType()) {
    // Although variables of reference type always need to be initialized, it
    // can happen that we can't see the initializer, so `InitExpr` may still
    // be null.
    if (InitExpr) {
      if (auto *InitExprLoc = getStorageLocation(*InitExpr))
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
    Val = getValue(*InitExpr);
  if (!Val)
    Val = createValue(Ty);

  if (Ty->isRecordType())
    return cast<RecordValue>(Val)->getLoc();

  StorageLocation &Loc =
      D ? createStorageLocation(*D) : createStorageLocation(Ty);

  if (Val)
    setValue(Loc, *Val);

  return Loc;
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

  OS << "ExprToVal:\n";
  for (auto [E, V] : ExprToVal)
    OS << "  [" << E << ", " << V << ": " << *V << "]\n";

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

RecordStorageLocation *getImplicitObjectLocation(const CXXMemberCallExpr &MCE,
                                                 const Environment &Env) {
  Expr *ImplicitObject = MCE.getImplicitObjectArgument();
  if (ImplicitObject == nullptr)
    return nullptr;
  if (ImplicitObject->getType()->isPointerType()) {
    if (auto *Val = cast_or_null<PointerValue>(Env.getValue(*ImplicitObject)))
      return &cast<RecordStorageLocation>(Val->getPointeeLoc());
    return nullptr;
  }
  return cast_or_null<RecordStorageLocation>(
      Env.getStorageLocation(*ImplicitObject));
}

RecordStorageLocation *getBaseObjectLocation(const MemberExpr &ME,
                                             const Environment &Env) {
  Expr *Base = ME.getBase();
  if (Base == nullptr)
    return nullptr;
  if (ME.isArrow()) {
    if (auto *Val = cast_or_null<PointerValue>(Env.getValue(*Base)))
      return &cast<RecordStorageLocation>(Val->getPointeeLoc());
    return nullptr;
  }
  return cast_or_null<RecordStorageLocation>(Env.getStorageLocation(*Base));
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

RecordValue &refreshRecordValue(RecordStorageLocation &Loc, Environment &Env) {
  auto &NewVal = Env.create<RecordValue>(Loc);
  Env.setValue(Loc, NewVal);
  return NewVal;
}

RecordValue &refreshRecordValue(const Expr &Expr, Environment &Env) {
  assert(Expr.getType()->isRecordType());

  if (Expr.isPRValue()) {
    if (auto *ExistingVal = cast_or_null<RecordValue>(Env.getValue(Expr))) {
      auto &NewVal = Env.create<RecordValue>(ExistingVal->getLoc());
      Env.setValue(Expr, NewVal);
      return NewVal;
    }

    auto &NewVal = *cast<RecordValue>(Env.createValue(Expr.getType()));
    Env.setValue(Expr, NewVal);
    return NewVal;
  }

  if (auto *Loc =
          cast_or_null<RecordStorageLocation>(Env.getStorageLocation(Expr))) {
    auto &NewVal = Env.create<RecordValue>(*Loc);
    Env.setValue(*Loc, NewVal);
    return NewVal;
  }

  auto &NewVal = *cast<RecordValue>(Env.createValue(Expr.getType()));
  Env.setStorageLocation(Expr, NewVal.getLoc());
  return NewVal;
}

} // namespace dataflow
} // namespace clang
