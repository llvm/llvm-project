//===- SARIFTransformationReportFormat.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SARIF writer for source-transformation reports.  Registered as the "sarif"
// report format; see TransformationReportFormatRegistry.h for how formats are
// looked up.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/Sarif.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/Version.h"
#include "clang/ScalableStaticAnalysis/Core/Support/ErrorBuilder.h"
#include "clang/ScalableStaticAnalysis/SourceTransformation/TransformationReportFormat.h"
#include "clang/ScalableStaticAnalysis/SourceTransformation/TransformationReportFormatRegistry.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <system_error>

namespace clang::ssaf {
// NOLINTNEXTLINE(misc-use-internal-linkage)
volatile int SARIFTransformationReportFormatAnchorSource = 0;
} // namespace clang::ssaf

namespace {

namespace ErrorMessages {
constexpr const char *FailedToWriteFile = "failed to write file '{0}': {1}";
constexpr const char *FileExists = "file already exists";
constexpr const char *ParentDirectoryNotFound =
    "parent directory does not exist";
} // namespace ErrorMessages

class SARIFTransformationReportFormat final
    : public clang::ssaf::TransformationReportFormat {
public:
  llvm::Error write(const clang::ssaf::ReportDocument &Doc,
                    llvm::StringRef Path) override;
};

llvm::Error
SARIFTransformationReportFormat::write(const clang::ssaf::ReportDocument &Doc,
                                       llvm::StringRef Path) {
  // Preflight filesystem checks: refuse to clobber an existing file, and fail
  // loudly if the parent directory is missing. Matches the SSAF Serialization
  // convention (see writeJSON in JSONFormat/JSONFormatImpl.cpp).
  if (llvm::sys::fs::exists(Path))
    return clang::ssaf::ErrorBuilder::create(std::errc::file_exists,
                                             ErrorMessages::FailedToWriteFile,
                                             Path, ErrorMessages::FileExists)
        .build();

  llvm::StringRef Dir = llvm::sys::path::parent_path(Path);
  if (!Dir.empty() && !llvm::sys::fs::is_directory(Dir))
    return clang::ssaf::ErrorBuilder::create(
               std::errc::no_such_file_or_directory,
               ErrorMessages::FailedToWriteFile, Path,
               ErrorMessages::ParentDirectoryNotFound)
        .build();

  clang::SarifDocumentWriter Writer(Doc.SM);
  std::string LongToolName =
      "clang ScalableStaticAnalysisFramework source transformation (" +
      Doc.TransformationName + ")";

  Writer.createRun(/*ShortToolName=*/"clang-ssaf",
                   /*LongToolName=*/LongToolName,
                   /*ToolVersion=*/CLANG_VERSION_STRING);

  // A dedup map for ruleIDs:
  llvm::DenseMap<llvm::StringRef, size_t> RuleMap;

  // Translate each ReportResult into a SarifResult. Missing Range -> no
  // `locations` key on the emitted result (run-wide finding).
  for (const clang::ssaf::ReportResult &R : Doc.Results) {
    if (!RuleMap.count(R.RuleId))
      RuleMap[R.RuleId] =
          Writer.createRule(clang::SarifRule::create().setRuleId(R.RuleId));

    clang::SarifResult SR =
        clang::SarifResult::create(RuleMap[R.RuleId])
            .setDiagnosticMessage(R.Message)
            .setDiagnosticLevel(clang::SarifResultLevel::Note);

    // FIXME: why would a transformation be associated with no range?
    if (R.Range)
      SR.addLocations({*R.Range});
    Writer.appendResult(SR);
  }

  llvm::json::Value DocJsonValue(Writer.createDocument());

  std::error_code EC;
  llvm::raw_fd_ostream OS(Path, EC, llvm::sys::fs::OF_Text);

  if (EC)
    return clang::ssaf::ErrorBuilder::create(
               EC, ErrorMessages::FailedToWriteFile, Path, EC.message())
        .build();
  OS << llvm::formatv("{0:2}", DocJsonValue);
  OS.flush();

  if (OS.has_error()) {
    std::error_code StreamEC = OS.error();
    return clang::ssaf::ErrorBuilder::create(StreamEC,
                                             ErrorMessages::FailedToWriteFile,
                                             Path, StreamEC.message())
        .build();
  }
  return llvm::Error::success();
}

} // namespace

static clang::ssaf::TransformationReportFormatRegistry::Add<
    SARIFTransformationReportFormat>
    RegisterSARIFTransformationReportFormat("sarif",
                                            "SARIF transformation report");
