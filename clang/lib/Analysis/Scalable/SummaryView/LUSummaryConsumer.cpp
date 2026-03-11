//===- LUSummaryConsumer.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Scalable/SummaryView/LUSummaryConsumer.h"
#include "clang/Analysis/Scalable/SummaryView/SummaryViewBuilderRegistry.h"
#include <cassert>
#include <memory>

using namespace clang;
using namespace ssaf;

static std::unique_ptr<SummaryViewBuilderBase>
instantiateBuilder(const SummaryName &SN) {
  for (const auto &Entry : SummaryViewBuilderRegistry::entries()) {
    if (Entry.getName() == SN.str()) {
      return Entry.instantiate();
    }
  }
  return nullptr;
}

void LUSummaryConsumer::run(const SummaryName &SN, EntityDataMap &Data) {
  auto Builder = instantiateBuilder(SN);
  if (!Builder) {
    return;
  }

  assert(Builder->summaryName() == SN &&
         "registry entry name must match SummaryViewBuilder::summaryName()");

  for (auto &[Id, Summary] : Data) {
    Builder->addSummary(Id, std::move(Summary));
  }

  Builder->finalize();

  Views.emplace(SN, std::move(*Builder).getView());
}

void LUSummaryConsumer::run() {
  assert(!WasRun && "run() must be called exactly once");
  WasRun = true;
  for (auto &[SN, Data] : LU->Data) {
    run(SN, Data);
  }
}
