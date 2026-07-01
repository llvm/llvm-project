//===- TransformationReportFormatRegistry.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysis/SourceTransformation/TransformationReportFormatRegistry.h"
#include <memory>

using namespace clang;
using namespace ssaf;

LLVM_DEFINE_REGISTRY(clang::ssaf::TransformationReportFormatRegistry)

bool ssaf::isTransformationReportFormatRegistered(llvm::StringRef Extension) {
  for (const auto &Entry : TransformationReportFormatRegistry::entries())
    if (Entry.getName() == Extension)
      return true;
  return false;
}

std::unique_ptr<TransformationReportFormat>
ssaf::makeTransformationReportFormat(llvm::StringRef Extension) {
  for (const auto &Entry : TransformationReportFormatRegistry::entries())
    if (Entry.getName() == Extension)
      return Entry.instantiate();
  return nullptr;
}

void ssaf::printAvailableTransformationReportFormats(llvm::raw_ostream &OS) {
  OS << "OVERVIEW: Available SSAF transformation-report formats:\n\n";
  for (const auto &Entry : TransformationReportFormatRegistry::entries())
    OS << "  " << Entry.getName() << " - " << Entry.getDesc() << "\n";
}
