//===- TransformationReportFormatRegistry.h ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Registry for TransformationReportFormats, and some helper functions.
// Formats are keyed by bare file-extension name (no leading `.`,
// case-sensitive).
//
// To register a custom transformation-report format, you will need to add
// some declarations and definitions.
//
// Insert this code to the cpp file:
//
//   namespace clang::ssaf {
//   // NOLINTNEXTLINE(misc-use-internal-linkage)
//   volatile int MyReportFormatAnchorSource = 0;
//   } // namespace clang::ssaf
//   static TransformationReportFormatRegistry::Add<MyReportFormat>
//     RegisterFormat("myext", "My awesome transformation-report format");
//
// Finally, extend the `AnchorSources` list in the force-linker header:
// clang/include/clang/ScalableStaticAnalysis/SSAFBuiltinForceLinker.h:
//
// This anchor is used to force the linker to link the MyReportFormat
// registration.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSIS_SOURCETRANSFORMATION_TRANSFORMATIONREPORTFORMATREGISTRY_H
#define LLVM_CLANG_SCALABLESTATICANALYSIS_SOURCETRANSFORMATION_TRANSFORMATIONREPORTFORMATREGISTRY_H

#include "clang/ScalableStaticAnalysis/SourceTransformation/TransformationReportFormat.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Registry.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

namespace clang::ssaf {

/// Check if a TransformationReportFormat was registered under \p Extension.
bool isTransformationReportFormatRegistered(llvm::StringRef Extension);

/// Try to instantiate a TransformationReportFormat for \p Extension.
/// Returns nullptr when no format is registered under the given extension.
std::unique_ptr<TransformationReportFormat>
makeTransformationReportFormat(llvm::StringRef Extension);

/// Print the list of available TransformationReportFormats.
void printAvailableTransformationReportFormats(llvm::raw_ostream &OS);

using TransformationReportFormatRegistry =
    llvm::Registry<TransformationReportFormat>;

} // namespace clang::ssaf

LLVM_DECLARE_REGISTRY(clang::ssaf::TransformationReportFormatRegistry)

#endif // LLVM_CLANG_SCALABLESTATICANALYSIS_SOURCETRANSFORMATION_TRANSFORMATIONREPORTFORMATREGISTRY_H
