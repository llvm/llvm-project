//===- SourcePassAnalysisFrontendAction.cpp
//--------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysisFramework/Frontend/SourcePassAnalysisFrontendAction.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/Basic/DiagnosticFrontend.h"
#include "clang/Frontend/MultiplexConsumer.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Serialization/SerializationFormatRegistry.h"
#include "clang/ScalableStaticAnalysisFramework/Core/SourcePassAnalysis/SourcePassAnalysisRegistry.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/WPASuite.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/IOSandbox.h"
#include "llvm/Support/Path.h"
#include <memory>

using namespace clang;
using namespace ssaf;

SourcePassAnalysisFrontendAction::~SourcePassAnalysisFrontendAction() = default;

SourcePassAnalysisFrontendAction::SourcePassAnalysisFrontendAction(
    std::unique_ptr<FrontendAction> WrappedAction)
    : WrapperFrontendAction(std::move(WrappedAction)) {}

std::unique_ptr<ASTConsumer>
SourcePassAnalysisFrontendAction::CreateASTConsumer(CompilerInstance &CI,
                                                    StringRef InFile) {
  auto WrappedConsumer = WrapperFrontendAction::CreateASTConsumer(CI, InFile);
  if (!WrappedConsumer)
    return nullptr;

  const FrontendOptions &Opts = CI.getFrontendOpts();

  // Parse format from file extension.
  StringRef File = Opts.SSAFLoadWPAResult;
  StringRef Ext = llvm::sys::path::extension(File);

  if (!Ext.consume_front(".") || File.empty()) {
    CI.getDiagnostics().Report(
        diag::warn_ssaf_load_wpa_result_unknown_file_path_format)
        << File;
    return WrappedConsumer;
  }

  if (!isFormatRegistered(Ext)) {
    CI.getDiagnostics().Report(
        diag::warn_ssaf_load_wpa_result_unknown_file_data_format)
        << Ext << File;
    return WrappedConsumer;
  }

  // Load WPA results.
  auto Format = makeFormat(Ext);

  // FIXME(sandboxing): Remove this by adopting `llvm::vfs::OutputBackend`.
  llvm::sys::sandbox::ScopedSetting Guard = llvm::sys::sandbox::scopedDisable();

  CI.getCodeGenOpts().ClearASTBeforeBackend = false;

  // Instantiate each requested source-pass analysis.
  // FIXME: WPASuite is not copyable. For now, each analysis gets a fresh read.
  // Consider shared ownership if multiple analyses need the same suite.
  std::vector<std::unique_ptr<ASTConsumer>> Consumers;
  Consumers.push_back(std::move(WrappedConsumer));

  for (const auto &Name : Opts.SSAFApplySourcePass) {
    auto SuiteOrErr = Format->readWPASuite(File);
    if (!SuiteOrErr) {
      llvm::report_fatal_error(SuiteOrErr.takeError());
      continue;
    }
    auto AnalysisOrErr = SourcePassAnalysisRegistry::instantiate(
        AnalysisName(Name), std::make_unique<WPASuite>(std::move(*SuiteOrErr)));
    if (!AnalysisOrErr) {
      llvm::report_fatal_error(AnalysisOrErr.takeError());
      continue;
    }
    Consumers.push_back(std::move(*AnalysisOrErr));
  }

  return std::make_unique<MultiplexConsumer>(std::move(Consumers));
}
