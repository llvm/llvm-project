//===- ExtractorRegistry.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Scalable/TUSummary/ExtractorRegistry.h"
#include <memory>

using namespace clang;
using namespace ssaf;

// FIXME: LLVM_INSTANTIATE_REGISTRY can't be used here because it drops extra
// type parameters.
template class CLANG_EXPORT_TEMPLATE
    llvm::Registry<TUSummaryExtractor, TUSummaryBuilder &>;

bool ssaf::isTUSummaryExtractorRegistered(llvm::StringRef SummaryName) {
  for (const auto &Entry : TUSummaryExtractorRegistry::entries())
    if (Entry.getName() == SummaryName)
      return true;
  return false;
}

std::unique_ptr<ASTConsumer>
ssaf::makeTUSummaryExtractor(llvm::StringRef SummaryName,
                             TUSummaryBuilder &Builder) {
  for (const auto &Entry : TUSummaryExtractorRegistry::entries())
    if (Entry.getName() == SummaryName)
      return Entry.instantiate(Builder);
  assert(false && "Unknown SummaryExtractor name");
  return nullptr;
}
