//===- SummaryViewBuilderRegistry.h
//----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Registry for SummaryViewBuilders and helper functions.
//
// To register a builder, add this to the builder's translation unit:
//
//   static SummaryViewBuilderRegistry::Add<MyViewBuilder>
//       Register("MyAnalysis", "View builder for MyAnalysis");
//
// The registry entry name must match MyView::summaryName().
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_SCALABLE_SUMMARYVIEW_SUMMARYVIEWBUILDERREGISTRY_H
#define LLVM_CLANG_ANALYSIS_SCALABLE_SUMMARYVIEW_SUMMARYVIEWBUILDERREGISTRY_H

#include "clang/Analysis/Scalable/SummaryView/SummaryViewBuilder.h"
#include "clang/Support/Compiler.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Registry.h"

namespace clang::ssaf {

/// Check if a SummaryViewBuilder was registered with a given name.
bool isSummaryViewBuilderRegistered(llvm::StringRef Name);

/// Registry for adding new SummaryViewBuilder implementations.
using SummaryViewBuilderRegistry = llvm::Registry<SummaryViewBuilderBase>;

} // namespace clang::ssaf

namespace llvm {
extern template class CLANG_TEMPLATE_ABI
    Registry<clang::ssaf::SummaryViewBuilderBase>;
} // namespace llvm

#endif // LLVM_CLANG_ANALYSIS_SCALABLE_SUMMARYVIEW_SUMMARYVIEWBUILDERREGISTRY_H
