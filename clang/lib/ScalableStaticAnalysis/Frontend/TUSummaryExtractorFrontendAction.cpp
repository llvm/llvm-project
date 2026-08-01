//===- TUSummaryExtractorFrontendAction.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysis/Frontend/TUSummaryExtractorFrontendAction.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/Basic/DiagnosticFrontend.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Frontend/MultiplexConsumer.h"
#include "clang/Frontend/SSAFOptions.h"
#include "clang/ScalableStaticAnalysis/Core/Serialization/SerializationFormatRegistry.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/ExtractorRegistry.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/TUSummary.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/TUSummaryBuilder.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/TUSummaryExtractor.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/IOSandbox.h"
#include "llvm/Support/Path.h"
#include "llvm/TargetParser/Triple.h"
#include <memory>
#include <string>
#include <vector>

using namespace clang;
using namespace ssaf;

static std::optional<std::pair<llvm::StringRef, llvm::StringRef>>
parseOutputFileFormatAndPathOrReportError(DiagnosticsEngine &Diags,
                                          StringRef SSAFTUSummaryFile) {

  StringRef Ext = llvm::sys::path::extension(SSAFTUSummaryFile);
  StringRef FilePath = SSAFTUSummaryFile.drop_back(Ext.size());

  if (!Ext.consume_front(".") || FilePath.empty()) {
    Diags.Report(diag::warn_ssaf_extract_tu_summary_file_unknown_format)
        << SSAFTUSummaryFile;
    return std::nullopt;
  }

  if (!isFormatRegistered(Ext)) {
    Diags.Report(diag::warn_ssaf_extract_tu_summary_file_unknown_output_format)
        << Ext << SSAFTUSummaryFile;
    return std::nullopt;
  }

  return std::pair{Ext, FilePath};
}

/// Return \c true if reported unrecognized extractors.
static bool
reportUnrecognizedExtractorNames(DiagnosticsEngine &Diags,
                                 ArrayRef<std::string> SSAFExtractSummaries) {
  if (SSAFExtractSummaries.empty()) {
    Diags.Report(diag::warn_ssaf_must_enable_summary_extractors);
    return true;
  }

  std::vector<StringRef> UnrecognizedExtractorNames;
  for (StringRef Name : SSAFExtractSummaries)
    if (!isTUSummaryExtractorRegistered(Name))
      UnrecognizedExtractorNames.push_back(Name);

  if (!UnrecognizedExtractorNames.empty()) {
    Diags.Report(diag::warn_ssaf_extract_summary_unknown_extractor_name)
        << UnrecognizedExtractorNames.size()
        << llvm::join(UnrecognizedExtractorNames, ", ");
    return true;
  }

  return false;
}

static std::vector<std::unique_ptr<ASTConsumer>>
makeTUSummaryExtractors(TUSummaryBuilder &Builder,
                        ArrayRef<std::string> SSAFExtractSummaries) {
  std::vector<std::unique_ptr<ASTConsumer>> Extractors;
  Extractors.reserve(SSAFExtractSummaries.size());
  for (StringRef Name : SSAFExtractSummaries) {
    assert(isTUSummaryExtractorRegistered(Name));
    Extractors.push_back(makeTUSummaryExtractor(Name, Builder));
  }
  return Extractors;
}

namespace {

/// Drives all extractor \c ASTConsumers and serializes the completed
/// \c TUSummary.
///
/// Derives from \c MultiplexConsumer so every \c ASTConsumer virtual method is
/// automatically forwarded to each extractor.
class TUSummaryRunner final : public MultiplexConsumer {
public:
  static std::unique_ptr<TUSummaryRunner> create(CompilerInstance &CI);

private:
  TUSummaryRunner(llvm::Triple TargetTriple,
                  std::unique_ptr<SerializationFormat> Format,
                  const SSAFOptions &Opts);

  void HandleTranslationUnit(ASTContext &Ctx) override;

  TUSummary Summary;

  /// Owned by the \c CompilerInstance.
  const SSAFOptions &Opts;

  TUSummaryBuilder Builder = TUSummaryBuilder(Summary, Opts);
  std::unique_ptr<SerializationFormat> Format;
};
} // namespace

std::unique_ptr<TUSummaryRunner> TUSummaryRunner::create(CompilerInstance &CI) {
  const SSAFOptions &Opts = CI.getSSAFOpts();
  DiagnosticsEngine &Diags = CI.getDiagnostics();

  if (Opts.CompilationUnitId.empty()) {
    Diags.Report(diag::warn_ssaf_tu_summary_requires_compilation_unit_id);
    return nullptr;
  }

  auto MaybePair =
      parseOutputFileFormatAndPathOrReportError(Diags, Opts.TUSummaryFile);
  if (!MaybePair.has_value())
    return nullptr;
  auto [FormatName, OutputPath] = MaybePair.value();

  if (reportUnrecognizedExtractorNames(Diags, Opts.ExtractSummaries))
    return nullptr;

  return std::unique_ptr<TUSummaryRunner>{new TUSummaryRunner{
      CI.getTarget().getTriple(), makeFormat(FormatName), Opts}};
}

TUSummaryRunner::TUSummaryRunner(llvm::Triple TargetTriple,
                                 std::unique_ptr<SerializationFormat> Format,
                                 const SSAFOptions &Opts)
    : MultiplexConsumer(std::vector<std::unique_ptr<ASTConsumer>>{}),
      Summary(std::move(TargetTriple),
              BuildNamespace(BuildNamespaceKind::CompilationUnit,
                             Opts.CompilationUnitId)),
      Opts(Opts), Format(std::move(Format)) {
  assert(this->Format);
  assert(!Opts.CompilationUnitId.empty());

  // Now the Summary and the builders are constructed, we can also construct the
  // extractors.
  auto Extractors = makeTUSummaryExtractors(Builder, Opts.ExtractSummaries);
  assert(!Extractors.empty());

  // We must initialize the Consumers here because our extractors need a
  // Builder that holds a reference to the TUSummary, which would be only
  // initialized after the MultiplexConsumer ctor. This is the only way we can
  // avoid the use of the TUSummary before it starts its lifetime.
  MultiplexConsumer::Consumers = std::move(Extractors);
}

void TUSummaryRunner::HandleTranslationUnit(ASTContext &Ctx) {
  // First, invoke the Summary Extractors.
  MultiplexConsumer::HandleTranslationUnit(Ctx);

  // FIXME(sandboxing): Remove this by adopting `llvm::vfs::OutputBackend`.
  llvm::sys::sandbox::ScopedSetting Guard = llvm::sys::sandbox::scopedDisable();

  // Then serialize the result.
  if (auto Err = Format->writeTUSummary(Summary, Opts.TUSummaryFile)) {
    Ctx.getDiagnostics().Report(diag::warn_ssaf_write_tu_summary_failed)
        << Opts.TUSummaryFile << llvm::toString(std::move(Err));
  }
}

TUSummaryExtractorFrontendAction::~TUSummaryExtractorFrontendAction() = default;

TUSummaryExtractorFrontendAction::TUSummaryExtractorFrontendAction(
    std::unique_ptr<FrontendAction> WrappedAction)
    : WrapperFrontendAction(std::move(WrappedAction)) {}

std::unique_ptr<ASTConsumer>
TUSummaryExtractorFrontendAction::CreateASTConsumer(CompilerInstance &CI,
                                                    StringRef InFile) {
  auto WrappedConsumer = WrapperFrontendAction::CreateASTConsumer(CI, InFile);
  if (!WrappedConsumer)
    return nullptr;

  if (auto Runner = TUSummaryRunner::create(CI)) {
    CI.getCodeGenOpts().ClearASTBeforeBackend = false;
    std::vector<std::unique_ptr<ASTConsumer>> Consumers;
    Consumers.reserve(2);
    Consumers.push_back(std::move(WrappedConsumer));
    Consumers.push_back(std::move(Runner));
    return std::make_unique<MultiplexConsumer>(std::move(Consumers));
  }
  return WrappedConsumer;
}
