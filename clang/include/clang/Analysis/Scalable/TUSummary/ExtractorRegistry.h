//===- ExtractorRegistry.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Registry for TUSummaryExtractors, and some helper functions.
// To register some custom extractor, insert this code:
//
//   static TUSummaryExtractorRegistry::Add<MyExtractor>
//     X("MyExtractor", "My awesome extractor");
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_SCALABLE_TUSUMMARY_EXTRACTORREGISTRY_H
#define LLVM_CLANG_ANALYSIS_SCALABLE_TUSUMMARY_EXTRACTORREGISTRY_H

#include "clang/Analysis/Scalable/TUSummary/TUSummaryExtractor.h"
#include "clang/Support/Compiler.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Registry.h"

namespace clang::ssaf {

/// Check if a TUSummaryExtractor was registered with a given name.
bool isTUSummaryExtractorRegistered(llvm::StringRef SummaryName);

/// Try to instantiate a TUSummaryExtractor with a given name.
/// This might return null if the construction of the desired TUSummaryExtractor
/// failed.
/// It's a fatal error if there is no extractor registered with the name.
std::unique_ptr<ASTConsumer> makeTUSummaryExtractor(llvm::StringRef SummaryName,
                                                    TUSummaryBuilder &Builder);

// Registry for adding new TUSummaryExtractor implementations.
using TUSummaryExtractorRegistry =
    llvm::Registry<TUSummaryExtractor, TUSummaryBuilder &>;

} // namespace clang::ssaf

namespace llvm {
extern template class CLANG_TEMPLATE_ABI
    Registry<clang::ssaf::TUSummaryExtractor, clang::ssaf::TUSummaryBuilder &>;
} // namespace llvm

#endif // LLVM_CLANG_ANALYSIS_SCALABLE_TUSUMMARY_EXTRACTORREGISTRY_H
