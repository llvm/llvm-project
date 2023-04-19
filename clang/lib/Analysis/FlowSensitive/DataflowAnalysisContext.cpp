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
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include <cassert>
#include <memory>
#include <utility>

static llvm::cl::opt<std::string> DataflowLog(
    "dataflow-log", llvm::cl::Hidden, llvm::cl::ValueOptional,
    llvm::cl::desc("Emit log of dataflow analysis. With no arg, writes textual "
                   "log to stderr. With an arg, writes HTML logs under the "
                   "specified directory (one per analyzed function)."));

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
    return arena().create<AggregateStorageLocation>(Type, std::move(FieldLocs));
  }
  return arena().create<ScalarStorageLocation>(Type);
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
    Res.first->second = &arena().create<PointerValue>(PointeeLoc);
  }
  return *Res.first->second;
}

void DataflowAnalysisContext::addFlowConditionConstraint(
    AtomicBoolValue &Token, BoolValue &Constraint) {
  auto Res = FlowConditionConstraints.try_emplace(&Token, &Constraint);
  if (!Res.second) {
    Res.first->second =
        &arena().makeAnd(*Res.first->second, Constraint);
  }
}

AtomicBoolValue &
DataflowAnalysisContext::forkFlowCondition(AtomicBoolValue &Token) {
  auto &ForkToken = arena().makeFlowConditionToken();
  FlowConditionDeps[&ForkToken].insert(&Token);
  addFlowConditionConstraint(ForkToken, Token);
  return ForkToken;
}

AtomicBoolValue &
DataflowAnalysisContext::joinFlowConditions(AtomicBoolValue &FirstToken,
                                            AtomicBoolValue &SecondToken) {
  auto &Token = arena().makeFlowConditionToken();
  FlowConditionDeps[&Token].insert(&FirstToken);
  FlowConditionDeps[&Token].insert(&SecondToken);
  addFlowConditionConstraint(
      Token, arena().makeOr(FirstToken, SecondToken));
  return Token;
}

Solver::Result
DataflowAnalysisContext::querySolver(llvm::DenseSet<BoolValue *> Constraints) {
  Constraints.insert(&arena().makeLiteral(true));
  Constraints.insert(
      &arena().makeNot(arena().makeLiteral(false)));
  return S->solve(std::move(Constraints));
}

bool DataflowAnalysisContext::flowConditionImplies(AtomicBoolValue &Token,
                                                   BoolValue &Val) {
  // Returns true if and only if truth assignment of the flow condition implies
  // that `Val` is also true. We prove whether or not this property holds by
  // reducing the problem to satisfiability checking. In other words, we attempt
  // to show that assuming `Val` is false makes the constraints induced by the
  // flow condition unsatisfiable.
  llvm::DenseSet<BoolValue *> Constraints = {&Token,
                                             &arena().makeNot(Val)};
  llvm::DenseSet<AtomicBoolValue *> VisitedTokens;
  addTransitiveFlowConditionConstraints(Token, Constraints, VisitedTokens);
  return isUnsatisfiable(std::move(Constraints));
}

bool DataflowAnalysisContext::flowConditionIsTautology(AtomicBoolValue &Token) {
  // Returns true if and only if we cannot prove that the flow condition can
  // ever be false.
  llvm::DenseSet<BoolValue *> Constraints = {
      &arena().makeNot(Token)};
  llvm::DenseSet<AtomicBoolValue *> VisitedTokens;
  addTransitiveFlowConditionConstraints(Token, Constraints, VisitedTokens);
  return isUnsatisfiable(std::move(Constraints));
}

bool DataflowAnalysisContext::equivalentBoolValues(BoolValue &Val1,
                                                   BoolValue &Val2) {
  llvm::DenseSet<BoolValue *> Constraints = {
      &arena().makeNot(arena().makeEquals(Val1, Val2))};
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
    Constraints.insert(&arena().makeEquals(Token, *ConstraintsIt->second));
  }

  auto DepsIt = FlowConditionDeps.find(&Token);
  if (DepsIt != FlowConditionDeps.end()) {
    for (AtomicBoolValue *DepToken : DepsIt->second) {
      addTransitiveFlowConditionConstraints(*DepToken, Constraints,
                                            VisitedTokens);
    }
  }
}

void DataflowAnalysisContext::dumpFlowCondition(AtomicBoolValue &Token,
                                                llvm::raw_ostream &OS) {
  llvm::DenseSet<BoolValue *> Constraints = {&Token};
  llvm::DenseSet<AtomicBoolValue *> VisitedTokens;
  addTransitiveFlowConditionConstraints(Token, Constraints, VisitedTokens);

  llvm::DenseMap<const AtomicBoolValue *, std::string> AtomNames = {
      {&arena().makeLiteral(false), "False"},
      {&arena().makeLiteral(true), "True"}};
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
