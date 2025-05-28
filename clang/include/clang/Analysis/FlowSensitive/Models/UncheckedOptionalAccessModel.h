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
#include "clang/Analysis/FlowSensitive/CachedConstAccessorsLattice.h"
#include "clang/Analysis/FlowSensitive/DataflowAnalysis.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/MatchSwitch.h"
#include "clang/Analysis/FlowSensitive/NoopLattice.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include <memory>

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
  ///
  /// FIXME: we now cache and equate the result of const accessors
  /// that look like unique_ptr, have both `->` (returning a pointer type) and
  /// `*` (returning a reference type). This includes mixing `->` and
  /// `*` in a sequence of calls as long as the object is not modified. Once we
  /// are confident in this const accessor caching, we shouldn't need the
  /// IgnoreSmartPointerDereference option anymore.
  bool IgnoreSmartPointerDereference = false;
};

using UncheckedOptionalAccessLattice = CachedConstAccessorsLattice<NoopLattice>;

/// Dataflow analysis that models whether optionals hold values or not.
///
/// Models the `std::optional`, `absl::optional`, and `base::Optional` types.
class UncheckedOptionalAccessModel : public DataflowAnalysis {
public:
  using Lattice = UncheckedOptionalAccessLattice;

  UncheckedOptionalAccessModel(ASTContext &Ctx, dataflow::Environment &Env);

  /// Returns a matcher for the optional classes covered by this model.
  static ast_matchers::DeclarationMatcher optionalClassDecl();

  std::unique_ptr<DataflowLattice> initialElement() override {
    return std::make_unique<Lattice>();
  }

  void transfer(const CFGElement &Elt, DataflowLattice &L,
                Environment &Env) override;

private:
  CFGMatchSwitch<TransferState<Lattice>> TransferMatchSwitch;
};

/// Diagnostic information for an unchecked optional access.
struct UncheckedOptionalAccessDiagnostic {
  CharSourceRange Range;
};

class UncheckedOptionalAccessDiagnoser {
public:
  UncheckedOptionalAccessDiagnoser(
      UncheckedOptionalAccessModelOptions Options = {});

  llvm::SmallVector<UncheckedOptionalAccessDiagnostic>
  operator()(const CFGElement &Elt, ASTContext &Ctx,
             const TransferStateForDiagnostics<UncheckedOptionalAccessLattice>
                 &State) {
    return DiagnoseMatchSwitch(Elt, Ctx, State.Env);
  }

private:
  CFGMatchSwitch<const Environment,
                 llvm::SmallVector<UncheckedOptionalAccessDiagnostic>>
      DiagnoseMatchSwitch;
};

} // namespace dataflow
} // namespace clang

#endif // CLANG_ANALYSIS_FLOWSENSITIVE_MODELS_UNCHECKEDOPTIONALACCESSMODEL_H
