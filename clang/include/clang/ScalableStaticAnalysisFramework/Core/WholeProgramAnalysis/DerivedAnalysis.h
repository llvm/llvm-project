//===- DerivedAnalysis.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines DerivedAnalysisBase (type-erased base known to AnalysisDriver) and
// the typed intermediate DerivedAnalysis<ResultT, DepResultTs...> that
// concrete analyses inherit from.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_WHOLEPROGRAMANALYSIS_DERIVEDANALYSIS_H
#define LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_WHOLEPROGRAMANALYSIS_DERIVEDANALYSIS_H

#include "clang/ScalableStaticAnalysisFramework/Core/Support/ErrorBuilder.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/AnalysisBase.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/AnalysisName.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/AnalysisResult.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/AnalysisTraits.h"
#include "llvm/Support/Error.h"
#include <map>
#include <memory>
#include <vector>

namespace clang::ssaf {

class AnalysisDriver;
class AnalysisRegistry;

/// Type-erased base for derived analyses. Known to AnalysisDriver.
///
/// Not subclassed directly -- use DerivedAnalysis<ResultT, DepResultTs...>.
/// A derived analysis consumes previously produced AnalysisResult objects
/// and computes a new one via an initialize/step/finalize lifecycle.
class DerivedAnalysisBase : public AnalysisBase {
  friend class AnalysisDriver;

protected:
  DerivedAnalysisBase() : AnalysisBase(AnalysisBase::Kind::Derived) {}

private:
  /// Called once with the dependency results before the step() loop.
  ///
  /// \param DepResults  Immutable results of all declared dependencies, keyed
  ///                    by AnalysisName. Guaranteed to contain every name
  ///                    returned by dependencyNames().
  virtual llvm::Error initialize(
      const std::map<AnalysisName, const AnalysisResult *> &DepResults) = 0;

  /// Performs one pass.
  /// Returns true if another pass is needed; false when converged.
  virtual llvm::Expected<bool> step() = 0;

  /// Called after the step() loop converges. Default is a no-op.
  virtual llvm::Error finalize() { return llvm::Error::success(); }
};

/// Typed intermediate that concrete derived analyses inherit from.
///
/// Concrete analyses must implement:
///   llvm::Error initialize(const DepResultTs &...) override;
///   llvm::Expected<bool> step() override;
/// and may override finalize().
///
/// Dependencies are fixed for the lifetime of the analysis: initialize()
/// binds them once, step() is called until it returns false, and
/// finalize() post-processes after convergence.
template <typename ResultT, typename... DepResultTs>
class DerivedAnalysis : public DerivedAnalysisBase {
  static_assert(std::is_base_of_v<AnalysisResult, ResultT>,
                "ResultT must derive from AnalysisResult");
  static_assert(HasAnalysisName_v<ResultT>,
                "ResultT must have a static analysisName() method");
  static_assert((std::is_base_of_v<AnalysisResult, DepResultTs> && ...),
                "Every DepResultT must derive from AnalysisResult");
  static_assert((HasAnalysisName_v<DepResultTs> && ...),
                "Every DepResultT must have a static analysisName() method");

  friend class AnalysisRegistry;
  using ResultType = ResultT;

  std::unique_ptr<ResultT> Result = std::make_unique<ResultT>();

public:
  /// Used by AnalysisRegistry::Add to derive the registry entry name.
  AnalysisName analysisName() const final { return ResultT::analysisName(); }

  const std::vector<AnalysisName> &dependencyNames() const final {
    static const std::vector<AnalysisName> Names = {
        DepResultTs::analysisName()...};
    return Names;
  }

  /// Called once with the fixed dependency results before the step() loop.
  virtual llvm::Error initialize(const DepResultTs &...) = 0;

  /// Performs one step. Returns true if another step is needed; false when
  /// converged. Single-step analyses always return false.
  virtual llvm::Expected<bool> step() override = 0;

  /// Called after the step() loop converges. Override for post-processing.
  virtual llvm::Error finalize() override { return llvm::Error::success(); }

protected:
  /// Read-only access to the result being built.
  const ResultT &result() const & { return *Result; }

  /// Mutable access to the result being built.
  ResultT &result() & { return *Result; }

private:
  /// Seals the type-erased base overload, downcasts, and dispatches to the
  /// typed initialize(). All dependencies are guaranteed present by the driver.
  llvm::Error
  initialize(const std::map<AnalysisName, const AnalysisResult *> &Map) final {
    auto lookup = [&Map](const AnalysisName &Name) -> const AnalysisResult * {
      auto It = Map.find(Name);
      if (It == Map.end()) {
        ErrorBuilder::fatal("dependency '{0}' missing from DepResults map; "
                            "dependency graph is not topologically sorted",
                            Name);
      }
      return It->second;
    };
    return initialize(*static_cast<const DepResultTs *>(
        lookup(DepResultTs::analysisName()))...);
  }

  /// Type-erased result extraction for the driver.
  std::unique_ptr<AnalysisResult> result() && final {
    return std::move(Result);
  }
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_WHOLEPROGRAMANALYSIS_DERIVEDANALYSIS_H
