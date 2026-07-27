//===- SSAFOptions.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_SSAFOPTIONS_H
#define LLVM_CLANG_FRONTEND_SSAFOPTIONS_H

#include "llvm/Support/Compiler.h"
#include <string>
#include <vector>

namespace clang::ssaf {

class SSAFOptions {
public:
  /// List of SSAF extractors to enable.
  /// Controlled by: --ssaf-extract-summaries
  std::vector<std::string> ExtractSummaries;

  /// The TU summary output file with the file extension representing the
  /// serialization format.
  /// Controlled by: --ssaf-tu-summary-file
  std::string TUSummaryFile;

  /// Stable identifier used as the name of the `CompilationUnit`
  /// `BuildNamespace` of every produced TU summary.
  /// Controlled by: --ssaf-compilation-unit-id
  std::string CompilationUnitId;

  /// Name of the SSAF source transformation to run. Exactly one transformation
  /// per invocation; non-empty implies the source-transformation pipeline is
  /// active.
  /// Controlled by: --ssaf-source-transformation
  std::string SourceTransformation;

  /// Path of the WPASuite input consumed by the source transformation. The
  /// extension selects which serialization format reads it.
  /// Controlled by: --ssaf-global-scope-analysis-result
  std::string GlobalScopeAnalysisResult;

  /// Path of the source-edit output file produced by the source
  /// transformation.
  /// Controlled by: --ssaf-src-edit-file
  std::string SrcEditFile;

  /// Path of the transformation-report output file produced by the source
  /// transformation.
  /// Controlled by: --ssaf-transformation-report-file
  std::string TransformationReportFile;

  /// Show the list of available SSAF summary extractors and exit.
  /// Controlled by: --ssaf-list-extractors
  LLVM_PREFERRED_TYPE(bool)
  unsigned ShowExtractors : 1;

  /// Show the list of available SSAF serialization formats and exit.
  /// Controlled by: --ssaf-list-formats
  LLVM_PREFERRED_TYPE(bool)
  unsigned ShowFormats : 1;

  /// Include block-scope (function-local) declarations in extracted SSAF
  /// summaries. Defaults to false to preserve the original behavior.
  /// Controlled by: --ssaf-include-local-entities
  LLVM_PREFERRED_TYPE(bool)
  unsigned IncludeLocalEntities : 1;

  /// Extract from system-header declarations during SSAF contributor
  /// enumeration. Defaults to true to preserve the original behavior.
  /// Controlled by: --ssaf-no-extract-from-system-headers
  LLVM_PREFERRED_TYPE(bool)
  unsigned ExtractFromSystemHeaders : 1;

  SSAFOptions() {
    ShowExtractors = false;
    ShowFormats = false;
    IncludeLocalEntities = false;
    ExtractFromSystemHeaders = true;
  };
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_FRONTEND_SSAFOPTIONS_H
