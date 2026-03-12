//===- LUSummaryConsumer.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Scalable/SummaryData/LUSummaryConsumer.h"
#include "clang/Analysis/Scalable/SummaryData/SummaryDataBuilderRegistry.h"
#include <cassert>

using namespace clang;
using namespace ssaf;

void LUSummaryConsumer::run(const SummaryName &SN, EntityDataMap &EntityData) {
  auto Builder = SummaryDataBuilderRegistry::instantiate(SN.str());
  if (!Builder) {
    return;
  }

  for (auto &[Id, Summary] : EntityData) {
    Builder->addSummary(Id, std::move(Summary));
  }

  Builder->finalize();

  Data.emplace(SN, std::move(*Builder).getData());
}

void LUSummaryConsumer::run() {
  assert(!WasRun && "run() must be called exactly once");
  WasRun = true;
  for (auto &[SN, EntityData] : LU->Data) {
    run(SN, EntityData);
  }
}
