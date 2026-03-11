//===- SummaryViewBuilder.h -----------------------------------------------===//
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
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_SCALABLE_SUMMARYVIEW_SUMMARYVIEWBUILDER_H
#define LLVM_CLANG_ANALYSIS_SCALABLE_SUMMARYVIEW_SUMMARYVIEWBUILDER_H

#include "clang/Analysis/Scalable/Model/EntityId.h"
#include "clang/Analysis/Scalable/Model/SummaryName.h"
#include "clang/Analysis/Scalable/SummaryView/SummaryView.h"
#include "clang/Analysis/Scalable/SummaryView/SummaryViewTraits.h"
#include "clang/Analysis/Scalable/TUSummary/EntitySummary.h"
#include <memory>

namespace clang::ssaf {

class LUSummaryConsumer;

/// Abstract base class for all summary view builders.
///
/// Known to the registry and LUSummaryConsumer. Receives entities one at a
/// time via \c addSummary(), is finalized via \c finalize(), and transfers
/// ownership of the built view via \c getView().
class SummaryViewBuilderBase {
  friend class LUSummaryConsumer;

public:
  virtual ~SummaryViewBuilderBase() = default;

private:
  /// Called once per entity belonging to this builder's analysis.
  /// Takes ownership of the summary data.
  virtual void addSummary(EntityId Id,
                          std::unique_ptr<EntitySummary> Summary) = 0;

  /// Called after all entities have been added.
  virtual void finalize() {}

  /// Transfers ownership of the built view. Called by LUSummaryConsumer after
  /// finalize(). The rvalue ref-qualifier enforces single use — the builder
  /// cannot be accessed after this call.
  virtual std::unique_ptr<SummaryView> getView() && = 0;
};

/// Typed intermediate template that concrete builders inherit from.
/// Concrete builders must implement the typed
/// \c addSummary(EntityId, unique_ptr<SummaryT>) overload, and may override
/// \c finalize() for any post-processing needed after all entities are added.
template <typename ViewT, typename SummaryT>
class SummaryViewBuilder : public SummaryViewBuilderBase {
  static_assert(std::is_base_of_v<SummaryView, ViewT>,
                "ViewT must derive from SummaryView");
  static_assert(HasSummaryName<ViewT>::value,
                "ViewT must have a static summaryName() method");

  std::unique_ptr<ViewT> View;

public:
  SummaryViewBuilder() : View(std::make_unique<ViewT>()) {}

  /// Returns the SummaryName of the view this builder produces.
  /// Used by SummaryViewBuilderRegistry::Add to derive the registry entry name.
  static SummaryName summaryName() { return ViewT::summaryName(); }

  /// Typed customization point — concrete builders override this.
  virtual void addSummary(EntityId Id, std::unique_ptr<SummaryT> Summary) = 0;

protected:
  ViewT &getView() & { return *View; }

private:
  std::unique_ptr<SummaryView> getView() && override { return std::move(View); }

  /// Seals the base overload, downcasts, and dispatches to the typed overload.
  void addSummary(EntityId Id, std::unique_ptr<EntitySummary> Summary) final {
    addSummary(Id, std::unique_ptr<SummaryT>(
                       static_cast<SummaryT *>(Summary.release())));
  }
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_ANALYSIS_SCALABLE_SUMMARYVIEW_SUMMARYVIEWBUILDER_H
