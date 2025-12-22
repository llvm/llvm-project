//===- ExtractorRegistry.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Scalable/TUSummary/ExtractorRegistry.h"
#include "clang/Analysis/Scalable/Model/SummaryName.h"
#include <memory>

using namespace clang;
using namespace ssaf;

LLVM_INSTANTIATE_REGISTRY(TUSummaryExtractorRegistry)

bool ssaf::isTUSummaryExtractorRegistered(const SummaryName &Name) {
  for (const auto &Entry : TUSummaryExtractorRegistry::entries()) {
    if (Entry.getName() == Name.str()) {
      return true;
    }
  }
  return false;
}

std::unique_ptr<ASTConsumer>
ssaf::makeTUSummaryExtractor(const SummaryName &Name,
                             TUSummaryBuilder &Builder) {
  for (const auto &Entry : TUSummaryExtractorRegistry::entries()) {
    if (Entry.getName() == Name.str()) {
      return Entry.instantiate(Builder);
    }
  }
  assert(false && "Unknown SummaryExtractor name");
  return nullptr;
}
