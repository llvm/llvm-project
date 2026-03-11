//===- SummaryViewBuilder.h
//------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines SummaryViewBuilderBase (abstract base known to the
// registry and LUSummaryConsumer) and the typed intermediate template
// SummaryViewBuilder<ViewT, SummaryT> that concrete builders inherit from.
//
// To implement a view builder, inherit from SummaryViewBuilder:
//
//   class MyViewBuilder
//       : public SummaryViewBuilder<MyView, MyEntitySummary> {
//   public:
//     void addSummary(EntityId Id,
//                     std::unique_ptr<MyEntitySummary> Summary) override {
//       // accumulate into getView()
//     }
//     // summaryName() and getView() && provided by the intermediate.
//     // override finalize() if post-processing is needed.
//   };
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_SCALABLE_SUMMARYVIEW_SUMMARYVIEWBUILDER_H
#define LLVM_CLANG_ANALYSIS_SCALABLE_SUMMARYVIEW_SUMMARYVIEWBUILDER_H

#include "clang/Analysis/Scalable/Model/EntityId.h"
#include "clang/Analysis/Scalable/Model/SummaryName.h"
#include "clang/Analysis/Scalable/SummaryView/SummaryView.h"
#include "clang/Analysis/Scalable/TUSummary/EntitySummary.h"
#include <memory>

namespace clang::ssaf {

/// Abstract base class for all summary view builders.
///
/// Known to the registry and LUSummaryConsumer. Receives entities one at a
/// time via \c addSummary(), is finalized via \c finalize(), and transfers
/// ownership of the built view via \c getView() &&.
class SummaryViewBuilderBase {
public:
  virtual ~SummaryViewBuilderBase() = default;

  /// Returns the SummaryName this builder handles.
  virtual SummaryName summaryName() const = 0;

  /// Called once per entity belonging to this builder's analysis.
  /// Takes ownership of the summary data.
  virtual void addSummary(EntityId Id,
                          std::unique_ptr<EntitySummary> Summary) = 0;

  /// Called after all entities have been added.
  virtual void finalize() {}

  /// Transfers ownership of the built view (type-erased).
  /// Called by LUSummaryConsumer after finalize(). The rvalue ref-qualifier
  /// enforces single use — the builder cannot be accessed after this call.
  virtual std::unique_ptr<SummaryView> getView() && = 0;
};

/// Typed intermediate template that concrete builders inherit from.
///
/// Provides:
/// - \c summaryName() derived from \c ViewT::summaryName().
/// - \c getView() && which moves out the built view.
/// - A protected \c getView() accessor for use during construction.
/// - NVI dispatch: seals the base \c addSummary(EntityId, EntitySummary) as
///   \c final, downcasts, and dispatches to the typed overload.
///
/// Concrete builders only need to implement the typed
/// \c addSummary(EntityId, unique_ptr<SummaryT>) overload.
template <typename ViewT, typename SummaryT>
class SummaryViewBuilder : public SummaryViewBuilderBase {
  std::unique_ptr<ViewT> View;

protected:
  ViewT &getView() & { return *View; }

public:
  SummaryViewBuilder() : View(std::make_unique<ViewT>()) {}

  SummaryName summaryName() const override { return ViewT::summaryName(); }

  std::unique_ptr<SummaryView> getView() && override { return std::move(View); }

  /// Typed customization point — concrete builders override this.
  virtual void addSummary(EntityId Id, std::unique_ptr<SummaryT> Summary) = 0;

private:
  /// Seals the base overload, downcasts, and dispatches to the typed overload.
  void addSummary(EntityId Id, std::unique_ptr<EntitySummary> Summary) final {
    addSummary(Id, std::unique_ptr<SummaryT>(
                       static_cast<SummaryT *>(Summary.release())));
  }
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_ANALYSIS_SCALABLE_SUMMARYVIEW_SUMMARYVIEWBUILDER_H
