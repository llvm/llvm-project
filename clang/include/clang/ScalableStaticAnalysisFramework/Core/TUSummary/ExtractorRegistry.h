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
//   // NOLINTNEXTLINE(misc-use-internal-linkage)
//   volatile int SSAFMyExtractorAnchorSource = 0;
//   static TUSummaryExtractorRegistry::Add<MyExtractor>
//     X("MyExtractor", "My awesome extractor");
//
// Finally, insert a use of the new anchor symbol into the force-linker header:
// clang/include/clang/ScalableStaticAnalysisFramework/SSAFBuiltinForceLinker.h:
//
//   extern volatile int SSAFMyExtractorAnchorSource;
//   [[maybe_unused]] static int SSAFMyExtractorAnchorDestination =
//       SSAFMyExtractorAnchorSource;
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_TUSUMMARY_EXTRACTORREGISTRY_H
#define LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_TUSUMMARY_EXTRACTORREGISTRY_H

#include "clang/ScalableStaticAnalysisFramework/Core/TUSummary/TUSummaryExtractor.h"
#include "clang/Support/Compiler.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Registry.h"
#include "llvm/Support/raw_ostream.h"

namespace clang::ssaf {

/// Check if a TUSummaryExtractor was registered with a given name.
bool isTUSummaryExtractorRegistered(llvm::StringRef SummaryName);

/// Try to instantiate a TUSummaryExtractor with a given name.
/// This might return null if the construction of the desired TUSummaryExtractor
/// failed.
/// It's a fatal error if there is no extractor registered with the name.
std::unique_ptr<ASTConsumer> makeTUSummaryExtractor(llvm::StringRef SummaryName,
                                                    TUSummaryBuilder &Builder);

/// Print the list of available TUSummaryExtractors.
void printAvailableTUSummaryExtractors(llvm::raw_ostream &OS);

// Registry for adding new TUSummaryExtractor implementations.
using TUSummaryExtractorRegistry =
    llvm::Registry<TUSummaryExtractor, TUSummaryBuilder &>;

} // namespace clang::ssaf

LLVM_DECLARE_REGISTRY(clang::ssaf::TUSummaryExtractorRegistry)

#endif // LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_TUSUMMARY_EXTRACTORREGISTRY_H
