//===- TransformationReport.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSIS_SOURCETRANSFORMATION_TRANSFORMATIONREPORT_H
#define LLVM_CLANG_SCALABLESTATICANALYSIS_SOURCETRANSFORMATION_TRANSFORMATIONREPORT_H

#include "clang/Basic/SourceLocation.h"

#include <string>
#include <vector>

namespace clang {
class SourceManager;

namespace ssaf {

struct ReportResult {
  std::string RuleId;
  std::optional<clang::CharSourceRange> Range;
  std::string Message;
};

struct ReportDocument {
  std::string TransformationName;
  const SourceManager &SM;
  std::vector<ReportResult> Results;
};
} // namespace ssaf
} // namespace clang

#endif // LLVM_CLANG_SCALABLESTATICANALYSIS_SOURCETRANSFORMATION_TRANSFORMATIONREPORT_H
