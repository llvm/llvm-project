//===- AnalysisRegistry.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysisFramework/Core/Analysis/AnalysisRegistry.h"
#include "llvm/ADT/STLExtras.h"

using namespace clang;
using namespace ssaf;

using RegistryT = llvm::Registry<AnalysisBase>;

LLVM_INSTANTIATE_REGISTRY(RegistryT)

std::vector<AnalysisName> AnalysisRegistry::analysisNames;

bool AnalysisRegistry::contains(llvm::StringRef Name) {
  return llvm::is_contained(analysisNames, AnalysisName(std::string(Name)));
}

const std::vector<AnalysisName> &AnalysisRegistry::names() {
  return analysisNames;
}

std::optional<std::unique_ptr<AnalysisBase>>
AnalysisRegistry::instantiate(llvm::StringRef Name) {
  for (const auto &Entry : RegistryT::entries()) {
    if (Entry.getName() == Name) {
      return std::unique_ptr<AnalysisBase>(Entry.instantiate());
    }
  }
  return std::nullopt;
}
