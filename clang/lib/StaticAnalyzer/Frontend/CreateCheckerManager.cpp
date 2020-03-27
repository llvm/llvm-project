//===- CheckerManager.h - Static Analyzer Checker Manager -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the Static Analyzer Checker Manager.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Frontend/CheckerRegistry.h"
#include <memory>

namespace clang {
namespace ento {

CheckerManager::CheckerManager(
    ASTContext &Context, AnalyzerOptions &AOptions,
    ArrayRef<std::string> plugins,
    ArrayRef<std::function<void(CheckerRegistry &)>> checkerRegistrationFns)
    : Context(&Context), LangOpts(Context.getLangOpts()), AOptions(AOptions),
      Diags(Context.getDiagnostics()),
      Registry(
          std::make_unique<CheckerRegistry>(plugins, Context.getDiagnostics(),
                                            AOptions, checkerRegistrationFns)) {
  Registry->initializeRegistry(*this);
  Registry->initializeManager(*this);
  finishedCheckerRegistration();
}

/// Constructs a CheckerManager without requiring an AST. No checker
/// registration will take place. Only useful for retrieving the
/// CheckerRegistry and print for help flags where the AST is unavalaible.
CheckerManager::CheckerManager(AnalyzerOptions &AOptions,
                               const LangOptions &LangOpts,
                               DiagnosticsEngine &Diags,
                               ArrayRef<std::string> plugins)
    : LangOpts(LangOpts), AOptions(AOptions), Diags(Diags),
      Registry(std::make_unique<CheckerRegistry>(plugins, Diags, AOptions)) {
  Registry->initializeRegistry(*this);
}

CheckerManager::~CheckerManager() {
  for (const auto &CheckerDtor : CheckerDtors)
    CheckerDtor();
}

} // namespace ento
} // namespace clang
