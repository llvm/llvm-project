//===-- UncheckedOptionalAccessModel.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines a dataflow analysis that detects unsafe uses of optional
//  values.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_ANALYSIS_FLOWSENSITIVE_MODELS_UNCHECKEDOPTIONALACCESSMODEL_H
#define CLANG_ANALYSIS_FLOWSENSITIVE_MODELS_UNCHECKEDOPTIONALACCESSMODEL_H

#include "clang/AST/ASTContext.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/FlowSensitive/CFGMatchSwitch.h"
#include "clang/Analysis/FlowSensitive/DataflowAnalysis.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/NoopLattice.h"
#include "clang/Basic/SourceLocation.h"
#include <vector>

namespace clang {
namespace dataflow {

// FIXME: Explore using an allowlist-approach, where constructs supported by the
// analysis are always enabled and additional constructs are enabled through the
// `Options`.
struct UncheckedOptionalAccessModelOptions {
  /// In generating diagnostics, ignore optionals reachable through overloaded
  /// `operator*` or `operator->` (other than those of the optional type
  /// itself). The analysis does not equate the results of such calls, so it
  /// can't identify when their results are used safely (across calls),
  /// resulting in false positives in all such cases. Note: this option does not
  /// cover access through `operator[]`.
  bool IgnoreSmartPointerDereference = false;
};

/// Dataflow analysis that models whether optionals hold values or not.
///
/// Models the `std::optional`, `absl::optional`, and `base::Optional` types.
class UncheckedOptionalAccessModel
    : public DataflowAnalysis<UncheckedOptionalAccessModel, NoopLattice> {
public:
  UncheckedOptionalAccessModel(ASTContext &Ctx);

  /// Returns a matcher for the optional classes covered by this model.
  static ast_matchers::DeclarationMatcher optionalClassDecl();

  static NoopLattice initialElement() { return {}; }

  void transfer(const CFGElement &Elt, NoopLattice &L, Environment &Env);

  ComparisonResult compare(QualType Type, const Value &Val1,
                           const Environment &Env1, const Value &Val2,
                           const Environment &Env2) override;

  bool merge(QualType Type, const Value &Val1, const Environment &Env1,
             const Value &Val2, const Environment &Env2, Value &MergedVal,
             Environment &MergedEnv) override;

  Value *widen(QualType Type, Value &Prev, const Environment &PrevEnv,
               Value &Current, Environment &CurrentEnv) override;

private:
  CFGMatchSwitch<TransferState<NoopLattice>> TransferMatchSwitch;
};

class UncheckedOptionalAccessDiagnoser {
public:
  UncheckedOptionalAccessDiagnoser(
      UncheckedOptionalAccessModelOptions Options = {});

  std::vector<SourceLocation> diagnose(ASTContext &Ctx, const CFGElement *Elt,
                                       const Environment &Env);

private:
  CFGMatchSwitch<const Environment, std::vector<SourceLocation>>
      DiagnoseMatchSwitch;
};

} // namespace dataflow
} // namespace clang

#endif // CLANG_ANALYSIS_FLOWSENSITIVE_MODELS_UNCHECKEDOPTIONALACCESSMODEL_H
