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

LLVM_INSTANTIATE_REGISTRY(SummaryViewBuilderRegistry)

bool ssaf::isSummaryViewBuilderRegistered(llvm::StringRef Name) {
  for (const auto &Entry : SummaryViewBuilderRegistry::entries()) {
    if (Entry.getName() == Name) {
      return true;
    }
  }
  return false;
}
