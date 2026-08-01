//===- TransformationRegistry.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysis/SourceTransformation/TransformationRegistry.h"
#include <memory>

using namespace clang;
using namespace ssaf;

namespace clang::ssaf {
// NOLINTNEXTLINE(misc-use-internal-linkage)
volatile int SSAFSourceTransformationAnchorSource = 0;
} // namespace clang::ssaf

LLVM_DEFINE_REGISTRY_EX(CLANG_ABI_EXPORT, clang::ssaf::TransformationRegistry)

bool ssaf::isTransformationRegistered(llvm::StringRef Name) {
  for (const auto &Entry : TransformationRegistry::entries())
    if (Entry.getName() == Name)
      return true;
  return false;
}

std::unique_ptr<Transformation>
ssaf::makeTransformation(llvm::StringRef Name, const WPASuite &Suite,
                         SourceEditEmitter &Edits,
                         TransformationReportEmitter &Report) {
  for (const auto &Entry : TransformationRegistry::entries())
    if (Entry.getName() == Name)
      return Entry.instantiate(Suite, Edits, Report);
  assert(false && "Unknown Transformation name");
  return nullptr;
}

void ssaf::printAvailableTransformations(llvm::raw_ostream &OS) {
  OS << "OVERVIEW: Available SSAF source transformations:\n\n";
  for (const auto &Entry : TransformationRegistry::entries())
    OS << "  " << Entry.getName() << " - " << Entry.getDesc() << "\n";
}
