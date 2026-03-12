//===- LUSummaryConsumer.h ------------------------------------------------===//
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

#ifndef LLVM_CLANG_ANALYSIS_SCALABLE_SUMMARYDATA_LUSUMMARYCONSUMER_H
#define LLVM_CLANG_ANALYSIS_SCALABLE_SUMMARYDATA_LUSUMMARYCONSUMER_H

#include "clang/Analysis/Scalable/EntityLinker/LUSummary.h"
#include "clang/Analysis/Scalable/Model/EntityId.h"
#include "clang/Analysis/Scalable/Model/SummaryName.h"
#include "clang/Analysis/Scalable/SummaryData/SummaryData.h"
#include "clang/Analysis/Scalable/SummaryData/SummaryDataTraits.h"
#include "clang/Analysis/Scalable/TUSummary/EntitySummary.h"
#include <map>
#include <memory>

namespace clang::ssaf {

/// Consumes a LUSummary by dispatching its entity data to all registered
/// SummaryDataBuilders and collecting the resulting data.
class LUSummaryConsumer final {
public:
  explicit LUSummaryConsumer(std::unique_ptr<LUSummary> LU)
      : LU(std::move(LU)) {}

  /// Instantiates a builder for each SummaryName present in the LUSummary,
  /// delivers its entities, finalizes it, and stores the resulting data. Each
  /// builder is fully processed before the next SummaryName is visited.
  /// Builders are discarded on return.
  ///
  /// \pre Must be called exactly once.
  void run();

  /// Transfers ownership of the data for \p DataT to the caller.
  ///
  /// Returns nullptr if no builder for \p DataT was registered or run() has
  /// not been called. A second call for the same DataT also returns nullptr.
  template <typename DataT> [[nodiscard]] std::unique_ptr<DataT> getData() {
    static_assert(std::is_base_of_v<SummaryData, DataT>,
                  "DataT must derive from SummaryData");
    static_assert(HasSummaryName<DataT>::value,
                  "DataT must have a static summaryName() method");

    auto It = Data.find(DataT::summaryName());
    if (It == Data.end()) {
      return nullptr;
    }
    auto *RawPtr = static_cast<DataT *>(It->second.release());
    Data.erase(It);
    return std::unique_ptr<DataT>(RawPtr);
  }

private:
  using EntityDataMap = std::map<EntityId, std::unique_ptr<EntitySummary>>;

  std::unique_ptr<LUSummary> LU;
  std::map<SummaryName, std::unique_ptr<SummaryData>> Data;
  bool WasRun = false;

  void run(const SummaryName &SN, EntityDataMap &Data);
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_ANALYSIS_SCALABLE_SUMMARYDATA_LUSUMMARYCONSUMER_H
