//===-- DeadCodeModel.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_ANALYSIS_FLOWSENSITIVE_MODELS_DEADCODEMODEL_H
#define CLANG_ANALYSIS_FLOWSENSITIVE_MODELS_DEADCODEMODEL_H

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

class DeadCodeModel : public DataflowAnalysis<DeadCodeModel, NoopLattice> {
public:
  explicit DeadCodeModel(ASTContext &Context)
      : DataflowAnalysis<DeadCodeModel, NoopLattice>(Context) {}

  static NoopLattice initialElement() { return {}; }

  void transfer(const CFGElement &E, NoopLattice &State, Environment &Env) {}

  void transferBranch(bool Branch, const Stmt *S, NoopLattice &State,
                      Environment &Env);
};

class DeadCodeDiagnoser {
public:
  /// Returns source locations for pointers that were checked when known to be
  // null, and checked after already dereferenced, respectively.
  enum class DiagnosticType { AlwaysTrue, AlwaysFalse };

  struct DiagnosticEntry {
    SourceLocation Location;
    DiagnosticType Type;
  };
  using ResultType = llvm::SmallVector<DiagnosticEntry>;

private:
  CFGMatchSwitch<const Environment, ResultType> DiagnoseMatchSwitch;

public:
  DeadCodeDiagnoser();

  ResultType operator()(const CFGElement &Elt, ASTContext &Ctx,
                        const TransferStateForDiagnostics<NoopLattice> &State);
};

} // namespace clang::dataflow

#endif