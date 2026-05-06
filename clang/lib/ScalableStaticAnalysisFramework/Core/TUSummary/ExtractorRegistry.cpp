//===- ExtractorRegistry.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysisFramework/Core/TUSummary/ExtractorRegistry.h"
#include "clang/ScalableStaticAnalysisFramework/Core/TUSummary/TUSummaryExtractor.h"
#include <memory>

using namespace clang;
using namespace ssaf;

LLVM_DEFINE_REGISTRY(clang::ssaf::TUSummaryExtractorRegistry)

bool ssaf::isTUSummaryExtractorRegistered(llvm::StringRef SummaryName) {
  for (const auto &Entry : TUSummaryExtractorRegistry::entries())
    if (Entry.getName() == SummaryName)
      return true;
  return false;
}

std::unique_ptr<TUSummaryExtractor>
ssaf::makeTUSummaryExtractor(llvm::StringRef SummaryName,
                             TUSummaryBuilder &Builder) {
  for (const auto &Entry : TUSummaryExtractorRegistry::entries())
    if (Entry.getName() == SummaryName)
      return Entry.instantiate(Builder);
  assert(false && "Unknown SummaryExtractor name");
  return nullptr;
}

void ssaf::printAvailableTUSummaryExtractors(llvm::raw_ostream &OS) {
  OS << "OVERVIEW: Available SSAF summary extractors:\n\n";
  for (const auto &Entry : TUSummaryExtractorRegistry::entries())
    OS << "  " << Entry.getName() << " - " << Entry.getDesc() << "\n";
}
