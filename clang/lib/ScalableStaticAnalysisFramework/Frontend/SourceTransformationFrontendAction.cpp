//===- SourceTransformationFrontendAction.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysisFramework/Frontend/SourceTransformationFrontendAction.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/Basic/DiagnosticFrontend.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/MultiplexConsumer.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Serialization/SerializationFormat.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Serialization/SerializationFormatRegistry.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/WPASuite.h"
#include "clang/ScalableStaticAnalysisFramework/SourceTransformation/SARIFTransformationReportFormat.h"
#include "clang/ScalableStaticAnalysisFramework/SourceTransformation/SourceEditEmitter.h"
#include "clang/ScalableStaticAnalysisFramework/SourceTransformation/Transformation.h"
#include "clang/ScalableStaticAnalysisFramework/SourceTransformation/TransformationRegistry.h"
#include "clang/ScalableStaticAnalysisFramework/SourceTransformation/TransformationReportEmitter.h"
#include "clang/ScalableStaticAnalysisFramework/SourceTransformation/YAMLSourceEditFormat.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/IOSandbox.h"
#include "llvm/Support/Path.h"
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace clang;
using namespace ssaf;

namespace {

/// Concrete `SourceEditEmitter` that buffers replacements until flushed.
class AccumulatorSourceEditEmitter final : public SourceEditEmitter {
public:
  void addReplacement(clang::tooling::Replacement R) override {
    Replacements.push_back(std::move(R));
  }

  std::vector<clang::tooling::Replacement> Replacements;
};

/// Concrete `TransformationReportEmitter` that buffers results until flushed.
class AccumulatorReportEmitter final : public TransformationReportEmitter {
public:
  void addResult(StringRef RuleId, clang::SarifResultLevel Level,
                 clang::CharSourceRange Range, StringRef Message) override {
    Results.push_back({RuleId.str(), Level, Range, Message.str()});
  }

  std::vector<ReportResult> Results;
};

/// Per-TU runner: owns the loaded `WPASuite`, the accumulator emitters, and
/// the user-supplied `Transformation`. Inherits from `MultiplexConsumer` so
/// the transformation's `ASTConsumer` virtuals are forwarded for free;
/// serializes both outputs after the AST walk completes.
class SourceTransformationRunner final : public MultiplexConsumer {
public:
  static std::unique_ptr<SourceTransformationRunner>
  create(CompilerInstance &CI, StringRef InFile);

private:
  SourceTransformationRunner(WPASuite Suite, const FrontendOptions &Opts,
                             StringRef InFile);

  void HandleTranslationUnit(ASTContext &Ctx) override;

  WPASuite Suite;
  AccumulatorSourceEditEmitter Edits;
  AccumulatorReportEmitter Report;
  const FrontendOptions &Opts;
  std::string InFile;
};

} // namespace

/// Returns the bare extension of \p Path (no leading dot), or `std::nullopt` if
/// \p Path is empty or has no recognizable extension.
static std::optional<StringRef> bareExtension(StringRef Path) {
  StringRef Ext = llvm::sys::path::extension(Path);
  if (!Ext.consume_front("."))
    return std::nullopt;
  return Ext;
}

/// Returns `true` if any orphan-flag warning was reported. Every missing
/// companion flag fires its own diagnostic in a single pass so the user
/// sees the full list of CLI mistakes at once.
static bool reportOrphanFlagMisuse(DiagnosticsEngine &Diags,
                                   const FrontendOptions &Opts) {
  bool Reported = false;

  if (!Opts.SSAFSourceTransformation.empty()) {
    if (Opts.SSAFGlobalScopeAnalysisResult.empty()) {
      Diags.Report(diag::warn_ssaf_source_transformation_requires_wpa_file);
      Reported = true;
    }
    if (Opts.SSAFSrcEditFile.empty()) {
      Diags.Report(diag::warn_ssaf_source_transformation_requires_edit_file);
      Reported = true;
    }
    if (Opts.SSAFTransformationReportFile.empty()) {
      Diags.Report(diag::warn_ssaf_source_transformation_requires_report_file);
      Reported = true;
    }
    if (Opts.SSAFCompilationUnitId.empty()) {
      Diags.Report(
          diag::warn_ssaf_source_transformation_requires_compilation_unit_id);
      Reported = true;
    }
  } else {
    if (!Opts.SSAFSrcEditFile.empty()) {
      Diags.Report(diag::warn_ssaf_src_edit_file_requires_transformation);
      Reported = true;
    }
    if (!Opts.SSAFTransformationReportFile.empty()) {
      Diags.Report(
          diag::warn_ssaf_transformation_report_file_requires_transformation);
      Reported = true;
    }
  }

  return Reported;
}

std::unique_ptr<SourceTransformationRunner>
SourceTransformationRunner::create(CompilerInstance &CI, StringRef InFile) {
  const FrontendOptions &Opts = CI.getFrontendOpts();
  DiagnosticsEngine &Diags = CI.getDiagnostics();

  if (reportOrphanFlagMisuse(Diags, Opts))
    return nullptr;
  if (Opts.SSAFSourceTransformation.empty())
    return nullptr;

  if (!isTransformationRegistered(Opts.SSAFSourceTransformation)) {
    Diags.Report(diag::warn_ssaf_source_transformation_unknown_name)
        << Opts.SSAFSourceTransformation;
    return nullptr;
  }

  std::optional<StringRef> WPAExt =
      bareExtension(Opts.SSAFGlobalScopeAnalysisResult);
  std::unique_ptr<SerializationFormat> WPAFormat =
      WPAExt && isFormatRegistered(*WPAExt) ? makeFormat(*WPAExt) : nullptr;
  if (!WPAFormat) {
    Diags.Report(diag::warn_ssaf_read_wpa_suite_failed)
        << Opts.SSAFGlobalScopeAnalysisResult << "unknown serialization format";
    return nullptr;
  }
  llvm::sys::sandbox::ScopedSetting Guard = llvm::sys::sandbox::scopedDisable();
  llvm::Expected<WPASuite> SuiteOrErr =
      WPAFormat->readWPASuite(Opts.SSAFGlobalScopeAnalysisResult);
  if (!SuiteOrErr) {
    Diags.Report(diag::warn_ssaf_read_wpa_suite_failed)
        << Opts.SSAFGlobalScopeAnalysisResult
        << llvm::toString(SuiteOrErr.takeError());
    return nullptr;
  }

  return std::unique_ptr<SourceTransformationRunner>{
      new SourceTransformationRunner(std::move(*SuiteOrErr), Opts, InFile)};
}

SourceTransformationRunner::SourceTransformationRunner(
    WPASuite Suite, const FrontendOptions &Opts, StringRef InFile)
    : MultiplexConsumer(std::vector<std::unique_ptr<ASTConsumer>>{}),
      Suite(std::move(Suite)), Opts(Opts), InFile(InFile) {
  // The transformation must be constructed after Suite/Edits/Report start
  // their lifetimes — those references are captured in its base ctor.
  std::vector<std::unique_ptr<ASTConsumer>> Consumers;
  Consumers.push_back(makeTransformation(Opts.SSAFSourceTransformation,
                                         this->Suite, Edits, Report));
  assert(Consumers.front());
  MultiplexConsumer::Consumers = std::move(Consumers);
}

void SourceTransformationRunner::HandleTranslationUnit(ASTContext &Ctx) {
  // First, run the transformation.
  MultiplexConsumer::HandleTranslationUnit(Ctx);

  llvm::sys::sandbox::ScopedSetting Guard = llvm::sys::sandbox::scopedDisable();

  // Then serialize the source edits.
  clang::tooling::TranslationUnitReplacements EditDoc;
  EditDoc.MainSourceFile = InFile;
  EditDoc.Replacements = std::move(Edits.Replacements);
  if (auto Err = writeYAMLSourceEdits(EditDoc, Opts.SSAFSrcEditFile)) {
    Ctx.getDiagnostics().Report(diag::warn_ssaf_write_src_edit_failed)
        << Opts.SSAFSrcEditFile << llvm::toString(std::move(Err));
  }

  // And the transformation report.
  ReportDocument ReportDoc{Opts.SSAFSourceTransformation,
                           Ctx.getSourceManager(), std::move(Report.Results)};
  if (auto Err = writeSARIFTransformationReport(
          ReportDoc, Opts.SSAFTransformationReportFile)) {
    Ctx.getDiagnostics().Report(
        diag::warn_ssaf_write_transformation_report_failed)
        << Opts.SSAFTransformationReportFile << llvm::toString(std::move(Err));
  }
}

SourceTransformationFrontendAction::~SourceTransformationFrontendAction() =
    default;

SourceTransformationFrontendAction::SourceTransformationFrontendAction(
    std::unique_ptr<FrontendAction> WrappedAction)
    : WrapperFrontendAction(std::move(WrappedAction)) {}

std::unique_ptr<ASTConsumer>
SourceTransformationFrontendAction::CreateASTConsumer(CompilerInstance &CI,
                                                      StringRef InFile) {
  auto WrappedConsumer = WrapperFrontendAction::CreateASTConsumer(CI, InFile);
  if (!WrappedConsumer)
    return nullptr;

  if (auto Runner = SourceTransformationRunner::create(CI, InFile)) {
    CI.getCodeGenOpts().ClearASTBeforeBackend = false;
    std::vector<std::unique_ptr<ASTConsumer>> Consumers;
    Consumers.reserve(2);
    Consumers.push_back(std::move(WrappedConsumer));
    Consumers.push_back(std::move(Runner));
    return std::make_unique<MultiplexConsumer>(std::move(Consumers));
  }
  return WrappedConsumer;
}
