//===- SummaryDataBuilderRegistry.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysisFramework/Core/SummaryData/SummaryDataBuilderRegistry.h"

using namespace clang;
using namespace ssaf;

using RegistryT = llvm::Registry<SummaryDataBuilderBase>;
LLVM_INSTANTIATE_REGISTRY(RegistryT)

namespace {
const RegistryT::entry *findEntry(llvm::StringRef Name) {
  for (const auto &Entry : RegistryT::entries()) {
    if (Entry.getName() == Name) {
      return &Entry;
    }
  }
  return nullptr;
}
} // namespace

bool SummaryDataBuilderRegistry::contains(llvm::StringRef Name) {
  return findEntry(Name) != nullptr;
}

std::unique_ptr<SummaryDataBuilderBase>
SummaryDataBuilderRegistry::instantiate(llvm::StringRef Name) {
  const auto *Entry = findEntry(Name);
  return Entry ? Entry->instantiate() : nullptr;
}
