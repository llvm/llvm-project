//===- LUSummaryConsumer.h ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// LUSummaryConsumer constructs SummaryView objects by routing LUSummary entity
// data to the corresponding SummaryViewBuilder objects.
//
// Typical usage:
//
//   auto LU = Format->readLUSummary(Path);
//
//   LUSummaryConsumer Consumer(std::move(LU));
//   Consumer.run();
//
//   auto View = Consumer.getView<MyView>();
//   // View is std::unique_ptr<MyView>; Consumer no longer holds it.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_SCALABLE_SUMMARYVIEW_LUSUMMARYCONSUMER_H
#define LLVM_CLANG_ANALYSIS_SCALABLE_SUMMARYVIEW_LUSUMMARYCONSUMER_H

#include "clang/Analysis/Scalable/EntityLinker/LUSummary.h"
#include "clang/Analysis/Scalable/Model/EntityId.h"
#include "clang/Analysis/Scalable/Model/SummaryName.h"
#include "clang/Analysis/Scalable/SummaryView/SummaryView.h"
#include "clang/Analysis/Scalable/TUSummary/EntitySummary.h"
#include <map>
#include <memory>

namespace clang::ssaf {

/// Consumes a LUSummary by dispatching its entity data to all registered
/// SummaryViewBuilders and collecting the resulting views.
class LUSummaryConsumer final {
  std::unique_ptr<LUSummary> LU;
  std::map<SummaryName, std::unique_ptr<SummaryView>> Views;
  bool WasRun = false;

public:
  explicit LUSummaryConsumer(std::unique_ptr<LUSummary> LU)
      : LU(std::move(LU)) {}

  /// Instantiates a builder for each SummaryName present in the LUSummary,
  /// delivers its entities, finalizes it, and stores the resulting view.
  /// Each builder is fully processed (addSummary → finalize → getView) before
  /// the next SummaryName is visited. Builders are discarded on return.
  ///
  /// \pre Must be called exactly once.
  void run();

  /// Transfers ownership of the view for \p ViewT to the caller.
  ///
  /// Returns nullptr if no builder for \p ViewT was registered or run() has
  /// not been called. A second call for the same ViewT also returns nullptr.
  template <typename ViewT> [[nodiscard]] std::unique_ptr<ViewT> getView() {
    auto It = Views.find(ViewT::summaryName());
    if (It == Views.end()) {
      return nullptr;
    }
    auto *RawPtr = static_cast<ViewT *>(It->second.release());
    Views.erase(It);
    return std::unique_ptr<ViewT>(RawPtr);
  }

private:
  using EntityDataMap = std::map<EntityId, std::unique_ptr<EntitySummary>>;

  /// Processes all entities for a single SummaryName: instantiates the
  /// registered builder, delivers entities, finalizes, and stores the view.
  /// Does nothing if no builder is registered for \p SN.
  void run(const SummaryName &SN, EntityDataMap &Data);
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_ANALYSIS_SCALABLE_SUMMARYVIEW_LUSUMMARYCONSUMER_H
