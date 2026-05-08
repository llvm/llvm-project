//===- LUSummaryConsumer.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// LUSummaryConsumer constructs SummaryData objects by routing LUSummary entity
// data to the corresponding SummaryDataBuilder objects.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_SUMMARYDATA_LUSUMMARYCONSUMER_H
#define LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_SUMMARYDATA_LUSUMMARYCONSUMER_H

#include "clang/ScalableStaticAnalysisFramework/Core/EntityLinker/LUSummary.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/SummaryName.h"
#include "clang/ScalableStaticAnalysisFramework/Core/SummaryData/SummaryDataStore.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include <memory>
#include <utility>

namespace clang::ssaf {

/// Consumes a LUSummary by dispatching its entity data to registered
/// SummaryDataBuilders and returning the results in a SummaryDataStore.
///
/// Three consumption patterns are supported:
///   - \c run() &&          — processes every analysis present in the
///                            LUSummary, silently skipping any whose data is
///                            absent or whose builder is not registered.
///                            Cannot fail, so it returns \c SummaryDataStore
///                            directly. Requires an rvalue consumer because
///                            this pattern exhausts all remaining data.
///   - \c run(names)        — processes a named subset; returns
///                            \c llvm::Expected<SummaryDataStore> and fails if
///                            any requested name has no data in the LUSummary
///                            or no registered builder.
///   - \c run<DataTs...>()  — type-safe variant of \c run(names) with the
///                            same error semantics and return type.
///
/// All patterns consume the underlying LUSummary data, so each analysis can
/// only be retrieved once across all run() calls.
class LUSummaryConsumer final {
public:
  explicit LUSummaryConsumer(std::unique_ptr<LUSummary> LU)
      : LU(std::move(LU)) {}

  /// Processes all registered analyses in LUSummary and returns the results.
  /// Silently skips analyses with no data or no registered builder.
  ///
  /// Requires an rvalue consumer (call as \c std::move(Consumer).run()) because
  /// this pattern exhausts all remaining LUSummary data.
  [[nodiscard]] SummaryDataStore run() &&;

  /// Processes the named analyses and returns the results.
  ///
  /// Returns an error if any name has no data in the LUSummary or no
  /// registered builder.
  [[nodiscard]] llvm::Expected<SummaryDataStore>
  run(llvm::ArrayRef<SummaryName> Names);

  /// Processes analyses for each of the given types and returns the results.
  ///
  /// Returns an error if any type has no data in the LUSummary or no
  /// registered builder.
  template <typename... DataTs>
  [[nodiscard]] llvm::Expected<SummaryDataStore> run() {
    return run({DataTs::summaryName()...});
  }

private:
  std::unique_ptr<LUSummary> LU;

  /// Iterator into LUSummary::Data — the map from SummaryName to entity data.
  using LUDataIterator = decltype(std::declval<LUSummary &>().Data)::iterator;

  /// Core build implementation. Instantiates the registered builder for the
  /// analysis at \p It, delivers all entities, finalizes, and returns the
  /// built data. Returns an error if no builder is registered. Erases the
  /// LUSummary entry on success.
  llvm::Expected<std::unique_ptr<SummaryData>> build(LUDataIterator It);

  /// Looks up \p SN in the LUSummary and delegates to the iterator overload.
  /// Returns an error if no data for \p SN exists or no builder is registered.
  llvm::Expected<std::unique_ptr<SummaryData>> build(const SummaryName &SN);
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_SUMMARYDATA_LUSUMMARYCONSUMER_H
