//===- WPASuite.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The value returned by AnalysisDriver::run(). Bundles the EntityIdTable
// with the analysis results keyed by AnalysisName.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_WHOLEPROGRAMANALYSIS_WPASUITE_H
#define LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_WHOLEPROGRAMANALYSIS_WPASUITE_H

#include "clang/ScalableStaticAnalysisFramework/Core/Model/EntityIdTable.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Support/ErrorBuilder.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/AnalysisName.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/AnalysisResult.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/AnalysisTraits.h"
#include "llvm/Support/Error.h"
#include <map>
#include <memory>

namespace clang::ssaf {

class AnalysisDriver;
class SerializationFormat;
class TestFixture;

/// Bundles the EntityIdTable (moved from the LUSummary) and the analysis
/// results produced by one AnalysisDriver::run() call, keyed by AnalysisName.
///
/// This is the natural unit of persistence: entity names and analysis results
/// are self-contained in one object.
class WPASuite {
  friend class AnalysisDriver;
  friend class SerializationFormat;
  friend class TestFixture;

  EntityIdTable IdTable;
  std::map<AnalysisName, std::unique_ptr<AnalysisResult>> Data;

  WPASuite() = default;

public:
  /// Returns the EntityIdTable that maps EntityId values to their symbolic
  /// names.
  const EntityIdTable &idTable() const { return IdTable; }

  /// Returns true if a result for \p ResultT is present.
  template <typename ResultT> [[nodiscard]] bool contains() const {
    static_assert(std::is_base_of_v<AnalysisResult, ResultT>,
                  "ResultT must derive from AnalysisResult");
    static_assert(HasAnalysisName_v<ResultT>,
                  "ResultT must have a static analysisName() method");

    return contains(ResultT::analysisName());
  }

  /// Returns true if a result for \p Name is present.
  [[nodiscard]] bool contains(AnalysisName Name) const {
    return Data.find(Name) != Data.end();
  }

  /// Returns a const reference to the result for \p ResultT, or an error if
  /// absent.
  template <typename ResultT>
  [[nodiscard]] llvm::Expected<const ResultT &> get() const {
    static_assert(std::is_base_of_v<AnalysisResult, ResultT>,
                  "ResultT must derive from AnalysisResult");
    static_assert(HasAnalysisName_v<ResultT>,
                  "ResultT must have a static analysisName() method");

    auto Result = get(ResultT::analysisName());
    if (!Result) {
      return Result.takeError();
    }
    return static_cast<const ResultT &>(*Result);
  }

  /// Returns a const reference to the result for \p Name, or an error if
  /// absent.
  [[nodiscard]] llvm::Expected<const AnalysisResult &>
  get(AnalysisName Name) const {
    auto It = Data.find(Name);
    if (It == Data.end()) {
      return ErrorBuilder::create(std::errc::invalid_argument,
                                  "no result for '{0}' in WPASuite", Name)
          .build();
    }
    return *It->second;
  }
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_WHOLEPROGRAMANALYSIS_WPASUITE_H
