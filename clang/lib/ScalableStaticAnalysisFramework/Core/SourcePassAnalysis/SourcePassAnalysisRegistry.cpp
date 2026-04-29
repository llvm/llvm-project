//===- SourcePassAnalysisRegistry.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysisFramework/Core/SourcePassAnalysis/SourcePassAnalysisRegistry.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Support/ErrorBuilder.h"
#include "llvm/ADT/STLExtras.h"

using namespace clang;
using namespace ssaf;

// NOLINTNEXTLINE(misc-use-internal-linkage)
volatile int SSAFSourcePassAnalysisRegistryAnchorSource = 0;
LLVM_DEFINE_REGISTRY(SourcePassAnalysisRegistry::RegistryT)

std::vector<AnalysisName> &SourcePassAnalysisRegistry::getAnalysisNames() {
  static std::vector<AnalysisName> Names;
  return Names;
}

bool SourcePassAnalysisRegistry::contains(const AnalysisName &Name) {
  return llvm::is_contained(getAnalysisNames(), Name);
}

const std::vector<AnalysisName> &SourcePassAnalysisRegistry::names() {
  return getAnalysisNames();
}

llvm::Expected<std::unique_ptr<SourcePassAnalysisBase>>
SourcePassAnalysisRegistry::instantiate(const AnalysisName &Name,
                                        std::unique_ptr<WPASuite> WPAResult) {
  for (const auto &Entry : SourcePassAnalysisRegistry::RegistryT::entries()) {
    if (Entry.getName() == Name.str()) {
      auto Analysis = Entry.instantiate(std::move(WPAResult));
      return Analysis;
    }
  }
  return ErrorBuilder::create(std::errc::invalid_argument,
                              "no source-pass analysis registered for '{0}'",
                              Name)
      .build();
}
