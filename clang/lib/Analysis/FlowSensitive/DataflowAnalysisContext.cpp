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
    return arena().create<RecordStorageLocation>(Type, std::move(FieldLocs));
  }
  return arena().create<ScalarStorageLocation>(Type);
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
  Constraints.insert(&arena().makeLiteral(true));
  Constraints.insert(&arena().makeNot(arena().makeLiteral(false)));
  return S->solve(Constraints.getArrayRef());
}

bool DataflowAnalysisContext::flowConditionImplies(Atom Token,
                                                   const Formula &Val) {
  // Returns true if and only if truth assignment of the flow condition implies
  // that `Val` is also true. We prove whether or not this property holds by
  // reducing the problem to satisfiability checking. In other words, we attempt
  // to show that assuming `Val` is false makes the constraints induced by the
  // flow condition unsatisfiable.
  llvm::SetVector<const Formula *> Constraints;
  Constraints.insert(&arena().makeAtomRef(Token));
  Constraints.insert(&arena().makeNot(Val));
  addTransitiveFlowConditionConstraints(Token, Constraints);
  return isUnsatisfiable(std::move(Constraints));
}

bool DataflowAnalysisContext::flowConditionIsTautology(Atom Token) {
  // Returns true if and only if we cannot prove that the flow condition can
  // ever be false.
  llvm::SetVector<const Formula *> Constraints;
  Constraints.insert(&arena().makeNot(arena().makeAtomRef(Token)));
  addTransitiveFlowConditionConstraints(Token, Constraints);
  return isUnsatisfiable(std::move(Constraints));
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

void DataflowAnalysisContext::dumpFlowCondition(Atom Token,
                                                llvm::raw_ostream &OS) {
  llvm::SetVector<const Formula *> Constraints;
  Constraints.insert(&arena().makeAtomRef(Token));
  addTransitiveFlowConditionConstraints(Token, Constraints);

  // TODO: have formulas know about true/false directly instead
  Atom True = arena().makeLiteral(true).getAtom();
  Atom False = arena().makeLiteral(false).getAtom();
  Formula::AtomNames Names = {{False, "false"}, {True, "true"}};

  for (const auto *Constraint : Constraints) {
    Constraint->print(OS, &Names);
    OS << "\n";
  }
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

  if (F->hasBody()) {
    auto CFCtx = ControlFlowContext::build(*F);
    // FIXME: Handle errors.
    assert(CFCtx);
    auto Result = FunctionContexts.insert({F, std::move(*CFCtx)});
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
