//===- ExtractorRegistry.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Registry for TUSummaryExtractors, and some helper functions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_SCALABLE_TUSUMMARY_EXTRACTORREGISTRY_H
#define LLVM_CLANG_ANALYSIS_SCALABLE_TUSUMMARY_EXTRACTORREGISTRY_H

#include "clang/Analysis/Scalable/TUSummary/TUSummaryExtractor.h"
#include "clang/Support/Compiler.h"
#include "llvm/Support/Registry.h"

namespace clang::ssaf {
class SummaryName;
class TUSummaryBuilder;

/// Check if a TUSummaryExtractor was registered with a given name.
bool isTUSummaryExtractorRegistered(const SummaryName &Name);

/// Try to instantiate a TUSummaryExtractor with a given name.
std::unique_ptr<ASTConsumer> makeTUSummaryExtractor(const SummaryName &Name,
                                                    TUSummaryBuilder &Builder);

// Registry for adding new TUSummaryExtractor implementations.
using TUSummaryExtractorRegistry =
    llvm::Registry<TUSummaryExtractor, TUSummaryBuilder &>;

} // namespace clang::ssaf

namespace llvm {
extern template class CLANG_TEMPLATE_ABI
    Registry<clang::ssaf::TUSummaryExtractorRegistry>;
} // namespace llvm

#endif // LLVM_CLANG_ANALYSIS_SCALABLE_TUSUMMARY_EXTRACTORREGISTRY_H
