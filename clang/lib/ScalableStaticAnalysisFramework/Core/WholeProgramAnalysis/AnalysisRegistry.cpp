//===- AnalysisRegistry.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/AnalysisRegistry.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Support/ErrorBuilder.h"
#include "llvm/ADT/STLExtras.h"

using namespace clang;
using namespace ssaf;

using RegistryT = llvm::Registry<AnalysisBase>;

// NOLINTNEXTLINE(misc-use-internal-linkage)
volatile int SSAFAnalysisRegistryAnchorSource = 0;
LLVM_DEFINE_REGISTRY(RegistryT)

std::vector<AnalysisName> &AnalysisRegistry::getAnalysisNames() {
  static std::vector<AnalysisName> Names;
  return Names;
}

bool AnalysisRegistry::contains(const AnalysisName &Name) {
  return llvm::is_contained(getAnalysisNames(), Name);
}

const std::vector<AnalysisName> &AnalysisRegistry::names() {
  return getAnalysisNames();
}

llvm::Expected<std::unique_ptr<AnalysisBase>>
AnalysisRegistry::instantiate(const AnalysisName &Name) {
  for (const auto &Entry : RegistryT::entries()) {
    if (Entry.getName() == Name.str()) {
      return std::unique_ptr<AnalysisBase>(Entry.instantiate());
    }
  }
  return ErrorBuilder::create(std::errc::invalid_argument,
                              "no analysis registered for '{0}'", Name)
      .build();
}
