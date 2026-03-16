//===- LUSummaryConsumer.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysisFramework/Core/SummaryData/LUSummaryConsumer.h"
#include "clang/ScalableStaticAnalysisFramework/Core/SummaryData/SummaryDataBuilderRegistry.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Support/ErrorBuilder.h"

using namespace clang;
using namespace ssaf;

llvm::Expected<std::unique_ptr<SummaryData>>
LUSummaryConsumer::build(LUDataIterator LUIt) {
  const SummaryName SN = LUIt->first;
  auto Builder = SummaryDataBuilderRegistry::instantiate(SN.str());
  if (!Builder) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                "no builder registered for analysis '{0}'",
                                SN.str())
        .build();
  }

  for (auto &[Id, Summary] : LUIt->second) {
    Builder->addSummary(Id, std::move(Summary));
  }
  Builder->finalize();
  LU->Data.erase(LUIt);

  return std::move(*Builder).getData();
}

llvm::Expected<std::unique_ptr<SummaryData>>
LUSummaryConsumer::build(const SummaryName &SN) {
  auto LUIt = LU->Data.find(SN);
  if (LUIt == LU->Data.end()) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                "no data for analysis '{0}' in LUSummary",
                                SN.str())
        .build();
  }
  return build(LUIt);
}

llvm::Expected<SummaryDataStore>
LUSummaryConsumer::run(llvm::ArrayRef<SummaryName> Names) {
  SummaryDataStore Store;
  for (const auto &SN : Names) {
    auto Result = build(SN);
    if (!Result) {
      return Result.takeError();
    }
    Store.Data.emplace(SN, std::move(*Result));
  }
  return Store;
}

SummaryDataStore LUSummaryConsumer::run() && {
  SummaryDataStore Store;
  // Advance the iterator before calling build(): build() erases the current
  // element on success, but std::map only invalidates iterators to the erased
  // element, so the pre-advanced iterator remains valid in all cases.
  auto It = LU->Data.begin();
  while (It != LU->Data.end()) {
    auto Current = It++;             // Read the comment above!
    SummaryName SN = Current->first; // copy before build() potentially erases
    auto Result = build(Current);
    if (!Result) {
      llvm::consumeError(Result.takeError());
      continue;
    }
    Store.Data.emplace(std::move(SN), std::move(*Result));
  }
  return Store;
}
