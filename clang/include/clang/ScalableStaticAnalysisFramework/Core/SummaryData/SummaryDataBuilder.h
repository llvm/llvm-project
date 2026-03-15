//===- SummaryDataBuilder.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines SummaryDataBuilderBase (abstract base known to the
// registry and LUSummaryConsumer) and the typed intermediate template
// SummaryDataBuilder<DataT, SummaryT> that concrete builders inherit from.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_SUMMARYDATA_SUMMARYDATABUILDER_H
#define LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_SUMMARYDATA_SUMMARYDATABUILDER_H

#include "clang/ScalableStaticAnalysisFramework/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/SummaryName.h"
#include "clang/ScalableStaticAnalysisFramework/Core/SummaryData/SummaryData.h"
#include "clang/ScalableStaticAnalysisFramework/Core/SummaryData/SummaryDataTraits.h"
#include "clang/ScalableStaticAnalysisFramework/Core/TUSummary/EntitySummary.h"
#include <memory>

namespace clang::ssaf {

class LUSummaryConsumer;

/// Abstract base class for all summary data builders.
///
/// Known to the registry and LUSummaryConsumer. Receives entities one at a
/// time via \c addSummary(), is finalized via \c finalize(), and transfers
/// ownership of the built data via \c getData().
class SummaryDataBuilderBase {
  friend class LUSummaryConsumer;

public:
  virtual ~SummaryDataBuilderBase() = default;

private:
  /// Called once per entity belonging to this builder's analysis.
  /// Takes ownership of the summary data.
  virtual void addSummary(EntityId Id,
                          std::unique_ptr<EntitySummary> Summary) = 0;

  /// Called after all entities have been added.
  virtual void finalize() {}

  /// Transfers ownership of the built data. Called by LUSummaryConsumer after
  /// finalize(). The rvalue ref-qualifier enforces single use — the builder
  /// cannot be accessed after this call.
  virtual std::unique_ptr<SummaryData> getData() && = 0;
};

/// Typed intermediate template that concrete builders inherit from.
/// Concrete builders must implement the typed
/// \c addSummary(EntityId, unique_ptr<SummaryT>) overload, and may override
/// \c finalize() for any post-processing needed after all entities are added.
template <typename DataT, typename SummaryT>
class SummaryDataBuilder : public SummaryDataBuilderBase {
  static_assert(std::is_base_of_v<SummaryData, DataT>,
                "DataT must derive from SummaryData");
  static_assert(HasSummaryName<DataT>::value,
                "DataT must have a static summaryName() method");
  static_assert(std::is_base_of_v<EntitySummary, SummaryT>,
                "SummaryT must derive from EntitySummary");

  std::unique_ptr<DataT> Data;

public:
  SummaryDataBuilder() : Data(std::make_unique<DataT>()) {}

  /// Returns the SummaryName of the data this builder produces.
  /// Used by SummaryDataBuilderRegistry::Add to derive the registry entry name.
  static SummaryName summaryName() { return DataT::summaryName(); }

protected:
  /// Typed customization point — concrete builders override this.
  virtual void addSummary(EntityId Id, std::unique_ptr<SummaryT> Summary) = 0;

  DataT &getData() & { return *Data; }

private:
  std::unique_ptr<SummaryData> getData() && override { return std::move(Data); }

  /// Seals the base overload, downcasts, and dispatches to the typed overload.
  void addSummary(EntityId Id, std::unique_ptr<EntitySummary> Summary) final {
    addSummary(Id, std::unique_ptr<SummaryT>(
                       static_cast<SummaryT *>(Summary.release())));
  }
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_SUMMARYDATA_SUMMARYDATABUILDER_H
