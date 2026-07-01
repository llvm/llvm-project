//===- TransformationReportFormat.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSIS_SOURCETRANSFORMATION_TRANSFORMATIONREPORTFORMAT_H
#define LLVM_CLANG_SCALABLESTATICANALYSIS_SOURCETRANSFORMATION_TRANSFORMATIONREPORTFORMAT_H

#include "clang/ScalableStaticAnalysis/SourceTransformation/TransformationReport.h"

namespace clang::ssaf {

class TransformationReportFormat {
public:
  virtual ~TransformationReportFormat() = default;

  virtual llvm::Error write(const ReportDocument &Doc,
                            llvm::StringRef Path) = 0;
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSIS_SOURCETRANSFORMATION_TRANSFORMATIONREPORTFORMAT_H
