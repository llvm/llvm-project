//===- SummaryViewBuilderRegistry.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Scalable/SummaryView/SummaryViewBuilderRegistry.h"

using namespace clang;
using namespace ssaf;

using RegistryT = llvm::Registry<SummaryViewBuilderBase>;
LLVM_INSTANTIATE_REGISTRY(RegistryT)

std::unique_ptr<SummaryViewBuilderBase>
SummaryViewBuilderRegistry::instantiate(llvm::StringRef Name) {
  for (const auto &Entry : RegistryT::entries()) {
    if (Entry.getName() == Name) {
      return Entry.instantiate();
    }
  }
  return nullptr;
}

bool SummaryViewBuilderRegistry::isRegistered(llvm::StringRef Name) {
  for (const auto &Entry : RegistryT::entries()) {
    if (Entry.getName() == Name) {
      return true;
    }
  }
  return false;
}
