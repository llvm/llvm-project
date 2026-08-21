//===- SourceTransformationFrontendAction.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysis/Frontend/SourceTransformationFrontendAction.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/Basic/DiagnosticFrontend.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/MultiplexConsumer.h"
#include "clang/Frontend/SSAFOptions.h"
#include "clang/ScalableStaticAnalysis/Core/Serialization/SerializationFormat.h"
#include "clang/ScalableStaticAnalysis/Core/Serialization/SerializationFormatRegistry.h"
#include "clang/ScalableStaticAnalysis/Core/WholeProgramAnalysis/WPASuite.h"
#include "clang/ScalableStaticAnalysis/SourceTransformation/SARIFTransformationReportFormat.h"
#include "clang/ScalableStaticAnalysis/SourceTransformation/SourceEditEmitter.h"
#include "clang/ScalableStaticAnalysis/SourceTransformation/Transformation.h"
#include "clang/ScalableStaticAnalysis/SourceTransformation/TransformationRegistry.h"
#include "clang/ScalableStaticAnalysis/SourceTransformation/TransformationReportEmitter.h"
#include "clang/ScalableStaticAnalysis/SourceTransformation/YAMLSourceEditFormat.h"
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
  SourceTransformationRunner(WPASuite Suite, const SSAFOptions &Opts,
                             StringRef InFile);

  void HandleTranslationUnit(ASTContext &Ctx) override;

  WPASuite Suite;
  AccumulatorSourceEditEmitter Edits;
  AccumulatorReportEmitter Report;
  const SSAFOptions &Opts;
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

/// Companion options required by `--ssaf-source-transformation=`. Values must
/// match the `%select` branch order in
/// `warn_ssaf_source_transformation_requires`.
enum SourceTransformationCompanion {
  STCompanion_WPAFile,           // --ssaf-global-scope-analysis-result=
  STCompanion_EditFile,          // --ssaf-src-edit-file=
  STCompanion_ReportFile,        // --ssaf-transformation-report-file=
  STCompanion_CompilationUnitId, // --ssaf-compilation-unit-id=
};

/// Options that depend on `--ssaf-source-transformation=` being set. Values
/// must match the `%select` branch order in
/// `warn_ssaf_option_ignored_without_source_transformation`.
enum SourceTransformationDependent {
  STDependent_EditFile,   // --ssaf-src-edit-file=
  STDependent_ReportFile, // --ssaf-transformation-report-file=
};

/// Returns `true` if any orphan-option warning was reported. Every missing
/// companion option fires its own diagnostic in a single pass so the user
/// sees the full list of CLI mistakes at once.
static bool reportOrphanOptionMisuse(DiagnosticsEngine &Diags,
                                     const SSAFOptions &Opts) {
  bool Reported = false;

  if (!Opts.SourceTransformation.empty()) {
    if (Opts.GlobalScopeAnalysisResult.empty()) {
      Diags.Report(diag::warn_ssaf_source_transformation_requires)
          << STCompanion_WPAFile;
      Reported = true;
    }
    if (Opts.SrcEditFile.empty()) {
      Diags.Report(diag::warn_ssaf_source_transformation_requires)
          << STCompanion_EditFile;
      Reported = true;
    }
    if (Opts.TransformationReportFile.empty()) {
      Diags.Report(diag::warn_ssaf_source_transformation_requires)
          << STCompanion_ReportFile;
      Reported = true;
    }
    if (Opts.CompilationUnitId.empty()) {
      Diags.Report(diag::warn_ssaf_source_transformation_requires)
          << STCompanion_CompilationUnitId;
      Reported = true;
    }
  } else {
    if (!Opts.SrcEditFile.empty()) {
      Diags.Report(diag::warn_ssaf_option_ignored_without_source_transformation)
          << STDependent_EditFile;
      Reported = true;
    }
    if (!Opts.TransformationReportFile.empty()) {
      Diags.Report(diag::warn_ssaf_option_ignored_without_source_transformation)
          << STDependent_ReportFile;
      Reported = true;
    }
  }

  return Reported;
}

std::unique_ptr<SourceTransformationRunner>
SourceTransformationRunner::create(CompilerInstance &CI, StringRef InFile) {
  const SSAFOptions &Opts = CI.getSSAFOpts();
  DiagnosticsEngine &Diags = CI.getDiagnostics();

  if (reportOrphanOptionMisuse(Diags, Opts))
    return nullptr;
  if (Opts.SourceTransformation.empty())
    return nullptr;

  if (!isTransformationRegistered(Opts.SourceTransformation)) {
    Diags.Report(diag::warn_ssaf_source_transformation_unknown_name)
        << Opts.SourceTransformation;
    return nullptr;
  }

  std::optional<StringRef> WPAExt =
      bareExtension(Opts.GlobalScopeAnalysisResult);
  std::unique_ptr<SerializationFormat> WPAFormat =
      WPAExt && isFormatRegistered(*WPAExt) ? makeFormat(*WPAExt) : nullptr;
  if (!WPAFormat) {
    Diags.Report(diag::warn_ssaf_read_wpa_suite_failed)
        << Opts.GlobalScopeAnalysisResult << "unknown serialization format";
    return nullptr;
  }
  llvm::sys::sandbox::ScopedSetting Guard = llvm::sys::sandbox::scopedDisable();
  llvm::Expected<WPASuite> SuiteOrErr =
      WPAFormat->readWPASuite(Opts.GlobalScopeAnalysisResult);
  if (!SuiteOrErr) {
    Diags.Report(diag::warn_ssaf_read_wpa_suite_failed)
        << Opts.GlobalScopeAnalysisResult
        << llvm::toString(SuiteOrErr.takeError());
    return nullptr;
  }

  return std::unique_ptr<SourceTransformationRunner>{
      new SourceTransformationRunner(std::move(*SuiteOrErr), Opts, InFile)};
}

SourceTransformationRunner::SourceTransformationRunner(WPASuite Suite,
                                                       const SSAFOptions &Opts,
                                                       StringRef InFile)
    : MultiplexConsumer(std::vector<std::unique_ptr<ASTConsumer>>{}),
      Suite(std::move(Suite)), Opts(Opts), InFile(InFile) {
  // The transformation must be constructed after Suite/Edits/Report start
  // their lifetimes — those references are captured in its base ctor.
  std::vector<std::unique_ptr<ASTConsumer>> Consumers;
  Consumers.push_back(makeTransformation(Opts.SourceTransformation, this->Suite,
                                         Edits, Report));
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
  if (auto Err = writeYAMLSourceEdits(EditDoc, Opts.SrcEditFile)) {
    Ctx.getDiagnostics().Report(diag::warn_ssaf_write_src_edit_failed)
        << Opts.SrcEditFile << llvm::toString(std::move(Err));
  }

  // And the transformation report.
  ReportDocument ReportDoc{Opts.SourceTransformation, Ctx.getSourceManager(),
                           std::move(Report.Results)};
  if (auto Err = writeSARIFTransformationReport(
          ReportDoc, Opts.TransformationReportFile)) {
    Ctx.getDiagnostics().Report(
        diag::warn_ssaf_write_transformation_report_failed)
        << Opts.TransformationReportFile << llvm::toString(std::move(Err));
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
