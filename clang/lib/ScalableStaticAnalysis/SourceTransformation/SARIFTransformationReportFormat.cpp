//===- SARIFTransformationReportFormat.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysis/SourceTransformation/SARIFTransformationReportFormat.h"
#include "clang/Basic/Sarif.h"
#include "clang/Basic/Version.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace ssaf;

llvm::Error ssaf::writeSARIFTransformationReport(const ReportDocument &Doc,
                                                 llvm::StringRef Path) {
  std::error_code EC;
  llvm::raw_fd_ostream OS(Path, EC, llvm::sys::fs::OF_None);
  if (EC)
    return llvm::createStringError(EC, "failed to open '" + Path + "'");

  clang::SarifDocumentWriter Writer(Doc.SM);
  std::string LongToolName =
      "clang ScalableStaticAnalysisFramework source transformation (" +
      Doc.TransformationName + ")";
  Writer.createRun("clang-ssaf", LongToolName, CLANG_VERSION_STRING);

  llvm::StringMap<size_t> RuleIndex;
  for (const ReportResult &R : Doc.Results) {
    if (RuleIndex.contains(R.RuleId))
      continue;
    RuleIndex[R.RuleId] =
        Writer.createRule(clang::SarifRule::create().setRuleId(R.RuleId));
  }

  for (const ReportResult &R : Doc.Results) {
    clang::SarifResult Result = clang::SarifResult::create(RuleIndex[R.RuleId])
                                    .setRuleId(R.RuleId)
                                    .setDiagnosticMessage(R.Message)
                                    .setDiagnosticLevel(R.Level);
    if (R.Range.isValid())
      Result = Result.addLocations({R.Range});
    Writer.appendResult(Result);
  }

  llvm::json::Value Document(Writer.createDocument());
  OS << llvm::formatv("{0:2}", Document) << "\n";
  return llvm::Error::success();
}
