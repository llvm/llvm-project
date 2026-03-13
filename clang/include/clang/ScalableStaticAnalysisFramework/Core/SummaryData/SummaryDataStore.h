//===- SummaryDataStore.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Owns a collection of SummaryData objects keyed by SummaryName.
// Produced by LUSummaryConsumer::run() variants.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_SUMMARYDATA_SUMMARYDATASTORE_H
#define LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_SUMMARYDATA_SUMMARYDATASTORE_H

#include "clang/ScalableStaticAnalysisFramework/Core/Model/SummaryName.h"
#include "clang/ScalableStaticAnalysisFramework/Core/SummaryData/SummaryData.h"
#include "clang/ScalableStaticAnalysisFramework/Core/SummaryData/SummaryDataTraits.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Support/ErrorBuilder.h"
#include "llvm/Support/Error.h"
#include <map>
#include <memory>

namespace clang::ssaf {

class LUSummaryConsumer;

/// Owns a collection of SummaryData objects keyed by SummaryName.
/// Produced by LUSummaryConsumer::run() variants.
class SummaryDataStore {
  friend class LUSummaryConsumer;

  std::map<SummaryName, std::unique_ptr<SummaryData>> Data;

public:
  /// Returns true if data for \p Name is stored.
  [[nodiscard]] bool contains(const SummaryName &Name) const {
    return Data.find(Name) != Data.end();
  }

  /// Returns true if data for \p DataT is stored.
  template <typename DataT> [[nodiscard]] bool contains() const {
    static_assert(std::is_base_of_v<SummaryData, DataT>,
                  "DataT must derive from SummaryData");
    static_assert(HasSummaryName<DataT>::value,
                  "DataT must have a static summaryName() method");

    return contains(DataT::summaryName());
  }

  /// Returns a reference to the data for \p DataT, or an error if
  /// no data for \p DataT is stored.
  template <typename DataT> [[nodiscard]] llvm::Expected<DataT &> get() {
    static_assert(std::is_base_of_v<SummaryData, DataT>,
                  "DataT must derive from SummaryData");
    static_assert(HasSummaryName<DataT>::value,
                  "DataT must have a static summaryName() method");

    auto Result = get(DataT::summaryName());
    if (!Result) {
      return Result.takeError();
    }
    return static_cast<DataT &>(*Result);
  }

  /// Returns a reference to the data for \p Name, or an error if
  /// no data for \p Name is stored.
  [[nodiscard]] llvm::Expected<SummaryData &> get(const SummaryName &Name) {
    auto It = Data.find(Name);
    if (It == Data.end()) {
      return ErrorBuilder::create(std::errc::invalid_argument,
                                  "no data for analysis '{0}' in store",
                                  Name.str())
          .build();
    }
    return *It->second;
  }

  /// Transfers ownership of the data for \p DataT to the caller, or returns
  /// an error if no data for \p DataT is stored.
  template <typename DataT>
  [[nodiscard]] llvm::Expected<std::unique_ptr<DataT>> take() {
    static_assert(std::is_base_of_v<SummaryData, DataT>,
                  "DataT must derive from SummaryData");
    static_assert(HasSummaryName<DataT>::value,
                  "DataT must have a static summaryName() method");

    auto Result = take(DataT::summaryName());
    if (!Result) {
      return Result.takeError();
    }
    return std::unique_ptr<DataT>(static_cast<DataT *>(Result->release()));
  }

  /// Transfers ownership of the data for \p Name to the caller, or returns
  /// an error if no data for \p Name is stored.
  [[nodiscard]] llvm::Expected<std::unique_ptr<SummaryData>>
  take(const SummaryName &Name) {
    auto It = Data.find(Name);
    if (It == Data.end()) {
      return ErrorBuilder::create(std::errc::invalid_argument,
                                  "no data for analysis '{0}' in store",
                                  Name.str())
          .build();
    }
    auto Ptr = std::move(It->second);
    Data.erase(It);
    return Ptr;
  }
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_SUMMARYDATA_SUMMARYDATASTORE_H
