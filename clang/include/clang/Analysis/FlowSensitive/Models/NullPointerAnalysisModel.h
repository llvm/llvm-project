//===-- NullPointerAnalysisModel.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a generic null-pointer analysis model, used for finding
// pointer null-checks after the pointer has already been dereferenced.
//
// Only a limited set of operations are currently recognized. Notably, pointer
// arithmetic, null-pointer assignments and _nullable/_nonnull attributes are
// missing as of yet.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_ANALYSIS_FLOWSENSITIVE_MODELS_NULLPOINTERANALYSISMODEL_H
#define CLANG_ANALYSIS_FLOWSENSITIVE_MODELS_NULLPOINTERANALYSISMODEL_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/FlowSensitive/CFGMatchSwitch.h"
#include "clang/Analysis/FlowSensitive/DataflowAnalysis.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/DataflowLattice.h"
#include "clang/Analysis/FlowSensitive/MapLattice.h"
#include "clang/Analysis/FlowSensitive/NoopLattice.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

namespace clang::dataflow {

class NullPointerAnalysisModel
    : public DataflowAnalysis<NullPointerAnalysisModel, NoopLattice> {
public:
  /// A transparent wrapper around the function arguments of transferBranch().
  /// Does not outlive the call to transferBranch().
  struct TransferArgs {
    bool Branch;
    Environment &Env;
  };

private:
  CFGMatchSwitch<Environment> TransferMatchSwitch;
  ASTMatchSwitch<Stmt, TransferArgs> BranchTransferMatchSwitch;

public:
  explicit NullPointerAnalysisModel(ASTContext &Context);

  static NoopLattice initialElement() { return {}; }

  static ast_matchers::StatementMatcher ptrValueMatcher();

  void transfer(const CFGElement &E, NoopLattice &State, Environment &Env);

  void transferBranch(bool Branch, const Stmt *E, NoopLattice &State,
                      Environment &Env);

  void join(QualType Type, const Value &Val1, const Environment &Env1,
            const Value &Val2, const Environment &Env2, Value &MergedVal,
            Environment &MergedEnv) override;

  ComparisonResult compare(QualType Type, const Value &Val1,
                           const Environment &Env1, const Value &Val2,
                           const Environment &Env2) override;

  std::optional<WidenResult> widen(QualType Type, Value &Prev, const Environment &PrevEnv,
               Value &Current, Environment &CurrentEnv) override;
};

class NullCheckAfterDereferenceDiagnoser {
public:
  struct DiagnoseArgs {
    llvm::DenseMap<const Value *, const Expr *> &ValToDerefLoc;
    llvm::DenseMap<SourceLocation, const Value *> &WarningLocToVal;
    const Environment &Env;
  };

  /// Returns source locations for pointers that were checked when known to be
  // null, and checked after already dereferenced, respectively.
  using ResultType =
      std::pair<std::vector<SourceLocation>, std::vector<SourceLocation>>;

  // Maps a pointer's Value to a dereference, null-assignment, etc.
  // This is used later to construct the Note tag.
  llvm::DenseMap<const Value *, const Expr *> ValToDerefLoc;
  // Maps Maps a warning's SourceLocation to its relevant Value.
  llvm::DenseMap<SourceLocation, const Value *> WarningLocToVal;

private:
  CFGMatchSwitch<DiagnoseArgs, ResultType> DiagnoseMatchSwitch;

public:
  NullCheckAfterDereferenceDiagnoser();

  ResultType diagnose(ASTContext &Ctx, const CFGElement *Elt,
                      const Environment &Env);
};

} // namespace clang::dataflow

#endif // CLANG_ANALYSIS_FLOWSENSITIVE_MODELS_NULLPOINTERANALYSISMODEL_H
