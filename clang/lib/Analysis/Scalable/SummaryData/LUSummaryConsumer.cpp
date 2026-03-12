//===- LUSummaryConsumer.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Scalable/SummaryData/LUSummaryConsumer.h"
#include "clang/Analysis/Scalable/SummaryData/SummaryDataBuilderRegistry.h"
#include "clang/Analysis/Scalable/Support/ErrorBuilder.h"
#include <vector>

using namespace clang;
using namespace ssaf;

llvm::Expected<std::unique_ptr<SummaryData>>
LUSummaryConsumer::build(const SummaryName &SN) {
  auto LUIt = LU->Data.find(SN);
  if (LUIt == LU->Data.end()) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                "no data for analysis '{0}' in LUSummary",
                                SN.str())
        .build();
  }

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

llvm::Expected<SummaryDataStore>
LUSummaryConsumer::run(llvm::ArrayRef<SummaryName> Names) {
  SummaryDataStore Store;
  for (const auto &SN : Names) {
    auto Result = build(SN);
    if (!Result)
      return Result.takeError();
    Store.Data.emplace(SN, std::move(*Result));
  }
  return Store;
}

SummaryDataStore LUSummaryConsumer::run() && {
  SummaryDataStore Store;
  // Snapshot names first: build() erases entries from LU->Data, so iterating
  // directly over LU->Data while calling build() would invalidate iterators.
  std::vector<SummaryName> Names;
  for (const auto &[SN, _] : LU->Data) {
    Names.push_back(SN);
  }
  for (const auto &SN : Names) {
    auto Result = build(SN);
    if (!Result) {
      llvm::consumeError(Result.takeError());
      continue;
    }
    Store.Data.emplace(SN, std::move(*Result));
  }
  return Store;
}
