//===- SARIFTransformationReportFormat.h ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Built-in SARIF 2.1.0 transformation-report writer. Drives clang's existing
// `SarifDocumentWriter`; emits no `fix` keys (source edits live in the
// separate edit file).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_SOURCETRANSFORMATION_SARIFTRANSFORMATIONREPORTFORMAT_H
#define LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_SOURCETRANSFORMATION_SARIFTRANSFORMATIONREPORTFORMAT_H

#include "clang/Basic/Sarif.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <string>
#include <vector>

namespace clang {
class SourceManager;
} // namespace clang

namespace clang::ssaf {

struct ReportResult {
  std::string RuleId;
  clang::SarifResultLevel Level;
  clang::CharSourceRange Range;
  std::string Message;
};

struct ReportDocument {
  std::string TransformationName;
  const clang::SourceManager &SM;
  std::vector<ReportResult> Results;
};

/// Writes \p Doc to \p Path as a SARIF JSON document.
llvm::Error writeSARIFTransformationReport(const ReportDocument &Doc,
                                           llvm::StringRef Path);

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_SOURCETRANSFORMATION_SARIFTRANSFORMATIONREPORTFORMAT_H
