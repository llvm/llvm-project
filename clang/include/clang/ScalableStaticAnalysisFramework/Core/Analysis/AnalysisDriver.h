//===- AnalysisDriver.h ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Central orchestrator for whole-program analysis. Takes ownership of an
// LUSummary, drives all registered analyses in topological dependency order,
// and returns a WPASuite.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_ANALYSIS_ANALYSISDRIVER_H
#define LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_ANALYSIS_ANALYSISDRIVER_H

#include "clang/ScalableStaticAnalysisFramework/Core/Analysis/AnalysisRegistry.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Analysis/WPASuite.h"
#include "clang/ScalableStaticAnalysisFramework/Core/EntityLinker/LUSummary.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include <memory>
#include <vector>

namespace clang::ssaf {

/// Orchestrates whole-program analysis over an LUSummary.
///
/// Three run() patterns are supported:
///   - run() &&        -- all registered analyses; silently skips any whose
///                        entity data is absent or whose dependency was
///                        skipped. Requires an rvalue driver because this
///                        exhausts the LUSummary.
///   - run(names)      -- named subset plus transitive dependencies; returns
///                        Expected and fails if any listed name has no
///                        registered analysis or missing entity data.
///   - run<ResultTs..> -- type-safe variant of run(names).
class AnalysisDriver final {
public:
  explicit AnalysisDriver(std::unique_ptr<LUSummary> LU);

  /// Runs all registered analyses in topological dependency order.
  /// Silently skips analyses with absent entity data or skipped dependencies.
  ///
  /// Requires an rvalue driver (std::move(Driver).run()) because this
  /// exhausts all remaining LUSummary data.
  [[nodiscard]] llvm::Expected<WPASuite> run() &&;

  /// Runs only the named analyses (plus their transitive dependencies).
  ///
  /// Returns an error if any listed AnalysisName has no registered analysis
  /// or if a required SummaryAnalysis has no matching entity data in the
  /// LUSummary. The EntityIdTable is copied (not moved) so the driver remains
  /// usable for subsequent calls.
  [[nodiscard]] llvm::Expected<WPASuite>
  run(llvm::ArrayRef<AnalysisName> Names);

  /// Type-safe variant of run(names). Derives names from
  /// ResultTs::analysisName().
  template <typename... ResultTs> [[nodiscard]] llvm::Expected<WPASuite> run() {
    return run({ResultTs::analysisName()...});
  }

private:
  std::unique_ptr<LUSummary> LU;

  /// Instantiates all analyses reachable from \p Roots (plus transitive
  /// dependencies) and returns them in topological order via a single DFS.
  /// Reports an error on unregistered names or cycles.
  static llvm::Expected<std::vector<std::unique_ptr<AnalysisBase>>>
  sortTopologically(llvm::ArrayRef<AnalysisName> Roots);

  /// Executes a topologically-sorted analysis list and returns a WPASuite.
  /// \p IdTable is moved into the returned WPASuite.
  llvm::Expected<WPASuite>
  execute(EntityIdTable IdTable,
          std::vector<std::unique_ptr<AnalysisBase>> Sorted);

  llvm::Error
  executeSummaryAnalysis(std::unique_ptr<SummaryAnalysisBase> Summary,
                         WPASuite &Suite);

  llvm::Error
  executeDerivedAnalysis(std::unique_ptr<DerivedAnalysisBase> Derived,
                         WPASuite &Suite);
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_ANALYSIS_ANALYSISDRIVER_H
