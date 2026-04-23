//===- SummaryAnalysis.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines SummaryAnalysisBase (type-erased base known to AnalysisDriver) and
// the typed intermediate SummaryAnalysis<ResultT, EntitySummaryT> that
// concrete analyses inherit from.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_WHOLEPROGRAMANALYSIS_SUMMARYANALYSIS_H
#define LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_WHOLEPROGRAMANALYSIS_SUMMARYANALYSIS_H

#include "clang/ScalableStaticAnalysisFramework/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/SummaryName.h"
#include "clang/ScalableStaticAnalysisFramework/Core/TUSummary/EntitySummary.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/AnalysisBase.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/AnalysisResult.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/AnalysisTraits.h"
#include "llvm/Support/Error.h"
#include <memory>

namespace clang::ssaf {

class AnalysisDriver;
class AnalysisRegistry;

/// Type-erased base for summary analyses. Known to AnalysisDriver.
///
/// Not subclassed directly -- use SummaryAnalysis<ResultT, EntitySummaryT>.
/// A summary analysis processes per-entity EntitySummary objects from the
/// LUSummary one at a time, accumulating whole-program data into an
/// AnalysisResult.
class SummaryAnalysisBase : public AnalysisBase {
  friend class AnalysisDriver;

protected:
  SummaryAnalysisBase() : AnalysisBase(AnalysisBase::Kind::Summary) {}

public:
  /// SummaryName of the EntitySummary type this analysis consumes.
  /// Used by the driver to route entities from the LUSummary.
  virtual SummaryName getSummaryName() const = 0;

private:
  /// Called once before any add() calls. Default is a no-op.
  virtual llvm::Error initialize() { return llvm::Error::success(); }

  /// Called once per matching entity. The driver retains ownership of the
  /// summary; multiple SummaryAnalysis instances may receive the same entity.
  virtual llvm::Error add(EntityId Id, const EntitySummary &Summary) = 0;

  /// Called after all entities have been processed. Default is a no-op.
  virtual llvm::Error finalize() { return llvm::Error::success(); }
};

/// Typed intermediate that concrete summary analyses inherit from.
///
/// Concrete analyses must implement:
///   llvm::Error add(EntityId Id, const EntitySummaryT &Summary) override;
/// and may override initialize() and finalize().
///
/// The result being built is accessible via getResult() const & (read-only) and
/// getResult() & (mutable) within the analysis implementation.
template <typename ResultT, typename EntitySummaryT>
class SummaryAnalysis : public SummaryAnalysisBase {
  static_assert(std::is_base_of_v<AnalysisResult, ResultT>,
                "ResultT must derive from AnalysisResult");
  static_assert(HasAnalysisName_v<ResultT>,
                "ResultT must have a static analysisName() method");
  static_assert(std::is_base_of_v<EntitySummary, EntitySummaryT>,
                "EntitySummaryT must derive from EntitySummary");

  friend class AnalysisRegistry;
  using ResultType = ResultT;

  std::unique_ptr<ResultT> Result = std::make_unique<ResultT>();

public:
  /// Used by AnalysisRegistry::Add to derive the registry entry name.
  AnalysisName getAnalysisName() const final { return ResultT::analysisName(); }

  SummaryName getSummaryName() const final {
    return EntitySummaryT::summaryName();
  }

  const std::vector<AnalysisName> &getDependencyNames() const final {
    static const std::vector<AnalysisName> Empty;
    return Empty;
  }

  /// Called once before the first add() call. Override for initialization.
  virtual llvm::Error initialize() override { return llvm::Error::success(); }

  /// Called once per matching entity. Implement to accumulate data.
  virtual llvm::Error add(EntityId Id, const EntitySummaryT &Summary) = 0;

  /// Called after all entities have been processed.
  /// Override for post-processing.
  virtual llvm::Error finalize() override { return llvm::Error::success(); }

protected:
  /// Read-only access to the result being built.
  const ResultT &getResult() const & { return *Result; }

  /// Mutable access to the result being built.
  ResultT &getResult() & { return *Result; }

private:
  /// Seals the type-erased base overload, downcasts, and dispatches to the
  /// typed add().
  llvm::Error add(EntityId Id, const EntitySummary &Summary) final {
    return add(Id, static_cast<const EntitySummaryT &>(Summary));
  }

  /// Type-erased result extraction for the driver.
  std::unique_ptr<AnalysisResult> takeResult() && final {
    return std::move(Result);
  }
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_WHOLEPROGRAMANALYSIS_SUMMARYANALYSIS_H
