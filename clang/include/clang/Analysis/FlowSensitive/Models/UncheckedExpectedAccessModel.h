//===-- UncheckedExpectedAccessModel.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines a dataflow analysis that detects unsafe accesses to the
//  values of `std::expected` objects.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_ANALYSIS_FLOWSENSITIVE_MODELS_UNCHECKEDEXPECTEDACCESSMODEL_H
#define CLANG_ANALYSIS_FLOWSENSITIVE_MODELS_UNCHECKEDEXPECTEDACCESSMODEL_H

#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/FlowSensitive/CFGMatchSwitch.h"
#include "clang/Analysis/FlowSensitive/DataflowAnalysis.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/NoopLattice.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/SmallVector.h"

namespace clang {
namespace dataflow {

/// Dataflow analysis that models whether `std::expected` objects hold values.
class UncheckedExpectedAccessModel
    : public DataflowAnalysis<UncheckedExpectedAccessModel, NoopLattice> {
public:
  UncheckedExpectedAccessModel(ASTContext &Ctx, Environment &Env);

  /// Returns a matcher for calls to member functions of `std::expected`.
  static ast_matchers::StatementMatcher callToExpectedClass();

  static NoopLattice initialElement() { return {}; }

  void transfer(const CFGElement &Elt, NoopLattice &L, Environment &Env);

private:
  CFGMatchSwitch<TransferState<NoopLattice>> TransferMatchSwitch;
};

/// The part of a `std::expected` object that is accessed.
///
/// The values are used as `%select` indices in diagnostics, so they must not
/// be reordered.
enum class UncheckedExpectedAccessKind : int {
  /// `value()`, `operator*` or `operator->`, which require a value.
  Value = 0,
  /// `error()`, which requires an error.
  Error = 1,
};

/// Diagnostic information for an unchecked `std::expected` access.
struct UncheckedExpectedAccessDiagnostic {
  SourceLocation Loc;
  UncheckedExpectedAccessKind Kind;
};

/// Reports accesses to the value (or error) of `std::expected` objects that are
/// not provably preceded by a check that the object holds a value (or error).
class UncheckedExpectedAccessDiagnoser {
public:
  UncheckedExpectedAccessDiagnoser();

  llvm::SmallVector<UncheckedExpectedAccessDiagnostic>
  operator()(const CFGElement &Elt, ASTContext &Ctx,
             const TransferStateForDiagnostics<NoopLattice> &State) {
    return DiagnoseMatchSwitch(Elt, Ctx, State.Env);
  }

private:
  CFGMatchSwitch<const Environment,
                 llvm::SmallVector<UncheckedExpectedAccessDiagnostic>>
      DiagnoseMatchSwitch;
};

} // namespace dataflow
} // namespace clang

#endif // CLANG_ANALYSIS_FLOWSENSITIVE_MODELS_UNCHECKEDEXPECTEDACCESSMODEL_H
