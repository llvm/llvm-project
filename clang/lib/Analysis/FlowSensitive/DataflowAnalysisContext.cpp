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
#include "clang/Analysis/FlowSensitive/Formula.h"
#include "clang/Analysis/FlowSensitive/Logger.h"
#include "clang/Analysis/FlowSensitive/SimplifyConstraints.h"
#include "clang/Analysis/FlowSensitive/Value.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <memory>
#include <string>
#include <utility>
#include <vector>

static llvm::cl::opt<std::string> DataflowLog(
    "dataflow-log", llvm::cl::Hidden, llvm::cl::ValueOptional,
    llvm::cl::desc("Emit log of dataflow analysis. With no arg, writes textual "
                   "log to stderr. With an arg, writes HTML logs under the "
                   "specified directory (one per analyzed function)."));

namespace clang {
namespace dataflow {

FieldSet DataflowAnalysisContext::getModeledFields(QualType Type) {
  // During context-sensitive analysis, a struct may be allocated in one
  // function, but its field accessed in a function lower in the stack than
  // the allocation. Since we only collect fields used in the function where
  // the allocation occurs, we can't apply that filter when performing
  // context-sensitive analysis. But, this only applies to storage locations,
  // since field access it not allowed to fail. In contrast, field *values*
  // don't need this allowance, since the API allows for uninitialized fields.
  if (Opts.ContextSensitiveOpts)
    return getObjectFields(Type);

  return llvm::set_intersection(getObjectFields(Type), ModeledFields);
}

/// Returns the fields of a `RecordDecl` that are initialized by an
/// `InitListExpr`, in the order in which they appear in
/// `InitListExpr::inits()`.
/// `Init->getType()` must be a record type.
static std::vector<const FieldDecl *>
getFieldsForInitListExpr(const InitListExpr *InitList) {
  const RecordDecl *RD = InitList->getType()->getAsRecordDecl();
  assert(RD != nullptr);

  std::vector<const FieldDecl *> Fields;

  if (InitList->getType()->isUnionType()) {
    Fields.push_back(InitList->getInitializedFieldInUnion());
    return Fields;
  }

  // Unnamed bitfields are only used for padding and do not appear in
  // `InitListExpr`'s inits. However, those fields do appear in `RecordDecl`'s
  // field list, and we thus need to remove them before mapping inits to
  // fields to avoid mapping inits to the wrongs fields.
  llvm::copy_if(
      RD->fields(), std::back_inserter(Fields),
      [](const FieldDecl *Field) { return !Field->isUnnamedBitfield(); });
  return Fields;
}

RecordInitListHelper::RecordInitListHelper(const InitListExpr *InitList) {
  auto *RD = InitList->getType()->getAsCXXRecordDecl();
  assert(RD != nullptr);

  std::vector<const FieldDecl *> Fields = getFieldsForInitListExpr(InitList);
  ArrayRef<Expr *> Inits = InitList->inits();

  // Unions initialized with an empty initializer list need special treatment.
  // For structs/classes initialized with an empty initializer list, Clang
  // puts `ImplicitValueInitExpr`s in `InitListExpr::inits()`, but for unions,
  // it doesn't do this -- so we create an `ImplicitValueInitExpr` ourselves.
  SmallVector<Expr *> InitsForUnion;
  if (InitList->getType()->isUnionType() && Inits.empty()) {
    assert(Fields.size() == 1);
    ImplicitValueInitForUnion.emplace(Fields.front()->getType());
    InitsForUnion.push_back(&*ImplicitValueInitForUnion);
    Inits = InitsForUnion;
  }

  size_t InitIdx = 0;

  assert(Fields.size() + RD->getNumBases() == Inits.size());
  for (const CXXBaseSpecifier &Base : RD->bases()) {
    assert(InitIdx < Inits.size());
    Expr *Init = Inits[InitIdx++];
    BaseInits.emplace_back(&Base, Init);
  }

  assert(Fields.size() == Inits.size() - InitIdx);
  for (const FieldDecl *Field : Fields) {
    assert(InitIdx < Inits.size());
    Expr *Init = Inits[InitIdx++];
    FieldInits.emplace_back(Field, Init);
  }
}

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
  // Use getCalleeDecl instead of getMethodDecl in order to handle
  // pointer-to-member calls.
  const auto *MethodDecl = dyn_cast_or_null<CXXMethodDecl>(C.getCalleeDecl());
  if (!MethodDecl)
    return nullptr;
  auto *Body = dyn_cast_or_null<CompoundStmt>(MethodDecl->getBody());
  if (!Body || Body->size() != 1)
    return nullptr;
  if (auto *RS = dyn_cast<ReturnStmt>(*Body->body_begin()))
    if (auto *Return = RS->getRetValue())
      return dyn_cast<MemberExpr>(Return->IgnoreParenImpCasts());
  return nullptr;
}

static void getReferencedDecls(const Decl &D, FieldSet &Fields,
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
static void getReferencedDecls(const Stmt &S, FieldSet &Fields,
                               llvm::DenseSet<const VarDecl *> &Vars,
                               llvm::DenseSet<const FunctionDecl *> &Funcs) {
  for (auto *Child : S.children())
    if (Child != nullptr)
      getReferencedDecls(*Child, Fields, Vars, Funcs);
  if (const auto *DefaultArg = dyn_cast<CXXDefaultArgExpr>(&S))
    getReferencedDecls(*DefaultArg->getExpr(), Fields, Vars, Funcs);
  if (const auto *DefaultInit = dyn_cast<CXXDefaultInitExpr>(&S))
    getReferencedDecls(*DefaultInit->getExpr(), Fields, Vars, Funcs);

  if (auto *DS = dyn_cast<DeclStmt>(&S)) {
    if (DS->isSingleDecl())
      getReferencedDecls(*DS->getSingleDecl(), Fields, Vars, Funcs);
    else
      for (auto *D : DS->getDeclGroup())
        getReferencedDecls(*D, Fields, Vars, Funcs);
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
    if (InitList->getType()->isRecordType())
      for (const auto *FD : getFieldsForInitListExpr(InitList))
        Fields.insert(FD);
  }
}

ReferencedDecls getReferencedDecls(const FunctionDecl &FD) {
  ReferencedDecls Result;
  // Look for global variable and field references in the
  // constructor-initializers.
  if (const auto *CtorDecl = dyn_cast<CXXConstructorDecl>(&FD)) {
    for (const auto *Init : CtorDecl->inits()) {
      if (Init->isMemberInitializer()) {
        Result.Fields.insert(Init->getMember());
      } else if (Init->isIndirectMemberInitializer()) {
        for (const auto *I : Init->getIndirectMember()->chain())
          Result.Fields.insert(cast<FieldDecl>(I));
      }
      const Expr *E = Init->getInit();
      assert(E != nullptr);
      getReferencedDecls(*E, Result.Fields, Result.Globals, Result.Functions);
    }
    // Add all fields mentioned in default member initializers.
    for (const FieldDecl *F : CtorDecl->getParent()->fields())
      if (const auto *I = F->getInClassInitializer())
        getReferencedDecls(*I, Result.Fields, Result.Globals, Result.Functions);
  }
  getReferencedDecls(*FD.getBody(), Result.Fields, Result.Globals,
                     Result.Functions);

  return Result;
}

void DataflowAnalysisContext::addModeledFields(const FieldSet &Fields) {
  ModeledFields.set_union(Fields);
}

StorageLocation &DataflowAnalysisContext::createStorageLocation(QualType Type) {
  if (!Type.isNull() && Type->isRecordType()) {
    llvm::DenseMap<const ValueDecl *, StorageLocation *> FieldLocs;
    for (const FieldDecl *Field : getModeledFields(Type))
      if (Field->getType()->isReferenceType())
        FieldLocs.insert({Field, nullptr});
      else
        FieldLocs.insert({Field, &createStorageLocation(
                                     Field->getType().getNonReferenceType())});

    RecordStorageLocation::SyntheticFieldMap SyntheticFields;
    for (const auto &Entry : getSyntheticFields(Type))
      SyntheticFields.insert(
          {Entry.getKey(),
           &createStorageLocation(Entry.getValue().getNonReferenceType())});

    return createRecordStorageLocation(Type, std::move(FieldLocs),
                                       std::move(SyntheticFields));
  }
  return arena().create<ScalarStorageLocation>(Type);
}

// Returns the keys for a given `StringMap`.
// Can't use `StringSet` as the return type as it doesn't support `operator==`.
template <typename T>
static llvm::DenseSet<llvm::StringRef> getKeys(const llvm::StringMap<T> &Map) {
  return llvm::DenseSet<llvm::StringRef>(Map.keys().begin(), Map.keys().end());
}

RecordStorageLocation &DataflowAnalysisContext::createRecordStorageLocation(
    QualType Type, RecordStorageLocation::FieldToLoc FieldLocs,
    RecordStorageLocation::SyntheticFieldMap SyntheticFields) {
  assert(Type->isRecordType());
  assert(containsSameFields(getModeledFields(Type), FieldLocs));
  assert(getKeys(getSyntheticFields(Type)) == getKeys(SyntheticFields));

  RecordStorageLocationCreated = true;
  return arena().create<RecordStorageLocation>(Type, std::move(FieldLocs),
                                               std::move(SyntheticFields));
}

StorageLocation &
DataflowAnalysisContext::getStableStorageLocation(const ValueDecl &D) {
  if (auto *Loc = DeclToLoc.lookup(&D))
    return *Loc;
  auto &Loc = createStorageLocation(D.getType().getNonReferenceType());
  DeclToLoc[&D] = &Loc;
  return Loc;
}

StorageLocation &
DataflowAnalysisContext::getStableStorageLocation(const Expr &E) {
  const Expr &CanonE = ignoreCFGOmittedNodes(E);

  if (auto *Loc = ExprToLoc.lookup(&CanonE))
    return *Loc;
  auto &Loc = createStorageLocation(CanonE.getType());
  ExprToLoc[&CanonE] = &Loc;
  return Loc;
}

PointerValue &
DataflowAnalysisContext::getOrCreateNullPointerValue(QualType PointeeType) {
  auto CanonicalPointeeType =
      PointeeType.isNull() ? PointeeType : PointeeType.getCanonicalType();
  auto Res = NullPointerVals.try_emplace(CanonicalPointeeType, nullptr);
  if (Res.second) {
    auto &PointeeLoc = createStorageLocation(CanonicalPointeeType);
    Res.first->second = &arena().create<PointerValue>(PointeeLoc);
  }
  return *Res.first->second;
}

void DataflowAnalysisContext::addInvariant(const Formula &Constraint) {
  if (Invariant == nullptr)
    Invariant = &Constraint;
  else
    Invariant = &arena().makeAnd(*Invariant, Constraint);
}

void DataflowAnalysisContext::addFlowConditionConstraint(
    Atom Token, const Formula &Constraint) {
  auto Res = FlowConditionConstraints.try_emplace(Token, &Constraint);
  if (!Res.second) {
    Res.first->second =
        &arena().makeAnd(*Res.first->second, Constraint);
  }
}

Atom DataflowAnalysisContext::forkFlowCondition(Atom Token) {
  Atom ForkToken = arena().makeFlowConditionToken();
  FlowConditionDeps[ForkToken].insert(Token);
  addFlowConditionConstraint(ForkToken, arena().makeAtomRef(Token));
  return ForkToken;
}

Atom
DataflowAnalysisContext::joinFlowConditions(Atom FirstToken,
                                            Atom SecondToken) {
  Atom Token = arena().makeFlowConditionToken();
  FlowConditionDeps[Token].insert(FirstToken);
  FlowConditionDeps[Token].insert(SecondToken);
  addFlowConditionConstraint(Token,
                             arena().makeOr(arena().makeAtomRef(FirstToken),
                                            arena().makeAtomRef(SecondToken)));
  return Token;
}

Solver::Result DataflowAnalysisContext::querySolver(
    llvm::SetVector<const Formula *> Constraints) {
  return S->solve(Constraints.getArrayRef());
}

bool DataflowAnalysisContext::flowConditionImplies(Atom Token,
                                                   const Formula &F) {
  if (F.isLiteral(true))
    return true;

  // Returns true if and only if truth assignment of the flow condition implies
  // that `F` is also true. We prove whether or not this property holds by
  // reducing the problem to satisfiability checking. In other words, we attempt
  // to show that assuming `F` is false makes the constraints induced by the
  // flow condition unsatisfiable.
  llvm::SetVector<const Formula *> Constraints;
  Constraints.insert(&arena().makeAtomRef(Token));
  Constraints.insert(&arena().makeNot(F));
  addTransitiveFlowConditionConstraints(Token, Constraints);
  return isUnsatisfiable(std::move(Constraints));
}

bool DataflowAnalysisContext::flowConditionAllows(Atom Token,
                                                  const Formula &F) {
  if (F.isLiteral(false))
    return false;

  llvm::SetVector<const Formula *> Constraints;
  Constraints.insert(&arena().makeAtomRef(Token));
  Constraints.insert(&F);
  addTransitiveFlowConditionConstraints(Token, Constraints);
  return isSatisfiable(std::move(Constraints));
}

bool DataflowAnalysisContext::equivalentFormulas(const Formula &Val1,
                                                 const Formula &Val2) {
  llvm::SetVector<const Formula *> Constraints;
  Constraints.insert(&arena().makeNot(arena().makeEquals(Val1, Val2)));
  return isUnsatisfiable(std::move(Constraints));
}

void DataflowAnalysisContext::addTransitiveFlowConditionConstraints(
    Atom Token, llvm::SetVector<const Formula *> &Constraints) {
  llvm::DenseSet<Atom> AddedTokens;
  std::vector<Atom> Remaining = {Token};

  if (Invariant)
    Constraints.insert(Invariant);
  // Define all the flow conditions that might be referenced in constraints.
  while (!Remaining.empty()) {
    auto Token = Remaining.back();
    Remaining.pop_back();
    if (!AddedTokens.insert(Token).second)
      continue;

    auto ConstraintsIt = FlowConditionConstraints.find(Token);
    if (ConstraintsIt == FlowConditionConstraints.end()) {
      Constraints.insert(&arena().makeAtomRef(Token));
    } else {
      // Bind flow condition token via `iff` to its set of constraints:
      // FC <=> (C1 ^ C2 ^ ...), where Ci are constraints
      Constraints.insert(&arena().makeEquals(arena().makeAtomRef(Token),
                                             *ConstraintsIt->second));
    }

    if (auto DepsIt = FlowConditionDeps.find(Token);
        DepsIt != FlowConditionDeps.end())
      for (Atom A : DepsIt->second)
        Remaining.push_back(A);
  }
}

static void printAtomList(const llvm::SmallVector<Atom> &Atoms,
                          llvm::raw_ostream &OS) {
  OS << "(";
  for (size_t i = 0; i < Atoms.size(); ++i) {
    OS << Atoms[i];
    if (i + 1 < Atoms.size())
      OS << ", ";
  }
  OS << ")\n";
}

void DataflowAnalysisContext::dumpFlowCondition(Atom Token,
                                                llvm::raw_ostream &OS) {
  llvm::SetVector<const Formula *> Constraints;
  Constraints.insert(&arena().makeAtomRef(Token));
  addTransitiveFlowConditionConstraints(Token, Constraints);

  OS << "Flow condition token: " << Token << "\n";
  SimplifyConstraintsInfo Info;
  llvm::SetVector<const Formula *> OriginalConstraints = Constraints;
  simplifyConstraints(Constraints, arena(), &Info);
  if (!Constraints.empty()) {
    OS << "Constraints:\n";
    for (const auto *Constraint : Constraints) {
      Constraint->print(OS);
      OS << "\n";
    }
  }
  if (!Info.TrueAtoms.empty()) {
    OS << "True atoms: ";
    printAtomList(Info.TrueAtoms, OS);
  }
  if (!Info.FalseAtoms.empty()) {
    OS << "False atoms: ";
    printAtomList(Info.FalseAtoms, OS);
  }
  if (!Info.EquivalentAtoms.empty()) {
    OS << "Equivalent atoms:\n";
    for (const llvm::SmallVector<Atom> &Class : Info.EquivalentAtoms)
      printAtomList(Class, OS);
  }

  OS << "\nFlow condition constraints before simplification:\n";
  for (const auto *Constraint : OriginalConstraints) {
    Constraint->print(OS);
    OS << "\n";
  }
}

const AdornedCFG *
DataflowAnalysisContext::getAdornedCFG(const FunctionDecl *F) {
  // Canonicalize the key:
  F = F->getDefinition();
  if (F == nullptr)
    return nullptr;
  auto It = FunctionContexts.find(F);
  if (It != FunctionContexts.end())
    return &It->second;

  if (F->doesThisDeclarationHaveABody()) {
    auto ACFG = AdornedCFG::build(*F);
    // FIXME: Handle errors.
    assert(ACFG);
    auto Result = FunctionContexts.insert({F, std::move(*ACFG)});
    return &Result.first->second;
  }

  return nullptr;
}

static std::unique_ptr<Logger> makeLoggerFromCommandLine() {
  if (DataflowLog.empty())
    return Logger::textual(llvm::errs());

  llvm::StringRef Dir = DataflowLog;
  if (auto EC = llvm::sys::fs::create_directories(Dir))
    llvm::errs() << "Failed to create log dir: " << EC.message() << "\n";
  // All analysis runs within a process will log to the same directory.
  // Share a counter so they don't all overwrite each other's 0.html.
  // (Don't share a logger, it's not threadsafe).
  static std::atomic<unsigned> Counter = {0};
  auto StreamFactory =
      [Dir(Dir.str())]() mutable -> std::unique_ptr<llvm::raw_ostream> {
    llvm::SmallString<256> File(Dir);
    llvm::sys::path::append(File,
                            std::to_string(Counter.fetch_add(1)) + ".html");
    std::error_code EC;
    auto OS = std::make_unique<llvm::raw_fd_ostream>(File, EC);
    if (EC) {
      llvm::errs() << "Failed to create log " << File << ": " << EC.message()
                   << "\n";
      return std::make_unique<llvm::raw_null_ostream>();
    }
    return OS;
  };
  return Logger::html(std::move(StreamFactory));
}

DataflowAnalysisContext::DataflowAnalysisContext(std::unique_ptr<Solver> S,
                                                 Options Opts)
    : S(std::move(S)), A(std::make_unique<Arena>()), Opts(Opts) {
  assert(this->S != nullptr);
  // If the -dataflow-log command-line flag was set, synthesize a logger.
  // This is ugly but provides a uniform method for ad-hoc debugging dataflow-
  // based tools.
  if (Opts.Log == nullptr) {
    if (DataflowLog.getNumOccurrences() > 0) {
      LogOwner = makeLoggerFromCommandLine();
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
static void getFieldsFromClassHierarchy(QualType Type,
                                        clang::dataflow::FieldSet &Fields) {
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
clang::dataflow::FieldSet clang::dataflow::getObjectFields(QualType Type) {
  FieldSet Fields;
  getFieldsFromClassHierarchy(Type, Fields);
  return Fields;
}

bool clang::dataflow::containsSameFields(
    const clang::dataflow::FieldSet &Fields,
    const clang::dataflow::RecordStorageLocation::FieldToLoc &FieldLocs) {
  if (Fields.size() != FieldLocs.size())
    return false;
  for ([[maybe_unused]] auto [Field, Loc] : FieldLocs)
    if (!Fields.contains(cast_or_null<FieldDecl>(Field)))
      return false;
  return true;
}
