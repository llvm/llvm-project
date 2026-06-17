//===- TUSummaryExtractorOptions.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared, framework-wide options for SSAF TU summary extractors. The same
// instance is passed to every extractor that runs in a single invocation, so
// new cross-cutting toggles can be added here without changing extractor
// constructor signatures again.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_TUSUMMARY_TUSUMMARYEXTRACTOROPTIONS_H
#define LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_TUSUMMARY_TUSUMMARYEXTRACTOROPTIONS_H

namespace clang::ssaf {

/// Cross-cutting options shared by every SSAF TU summary extractor. New
/// options can be added here without changing extractor constructor or
/// registry signatures again.
struct TUSummaryExtractorOptions {
  /// When true (the default), \c findContributors enumerates declarations
  /// in system headers alongside user-source declarations. When false (set
  /// by the clang frontend's `--ssaf-no-extract-from-system-headers` flag,
  /// hardcoded by the clang-ssaf-orchestrator's stage-1 compile task),
  /// \c findContributors skips declarations whose location is in a system
  /// header (per \c clang::SourceManager::isInSystemHeader). The opt-out
  /// avoids USR collisions that can fire the duplicated-contributor
  /// assertion at \c PointerFlowExtractor / \c UnsafeBufferUsageExtractor
  /// when system-header templates produce non-unique USRs, and avoids
  /// wasted analysis work on system code that clang-reforge never
  /// transforms. Default \c true preserves the original SSAF behavior.
  bool ExtractFromSystemHeaders = true;
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_TUSUMMARY_TUSUMMARYEXTRACTOROPTIONS_H
