//===- CreateCheckerManager.cpp - Checker Manager constructor ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the constructors and the destructor of the Static Analyzer Checker
// Manager which cannot be placed under 'Core' because they depend on the
// CheckerRegistry.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Frontend/CheckerRegistry.h"
#include <memory>

namespace clang {
namespace ento {

CheckerManager::CheckerManager(
    ASTContext &Context, AnalyzerOptions &AOptions, const Preprocessor &PP,
    ArrayRef<std::string> plugins,
    ArrayRef<std::function<void(CheckerRegistry &)>> checkerRegistrationFns)
    : Context(&Context), LangOpts(Context.getLangOpts()), AOptions(AOptions),
      PP(&PP), Diags(Context.getDiagnostics()),
      RegistryData(std::make_unique<CheckerRegistryData>()) {
  CheckerRegistry Registry(*RegistryData, plugins, Context.getDiagnostics(),
                           AOptions, checkerRegistrationFns);
  Registry.initializeRegistry(*this);
  Registry.initializeManager(*this);
}

CheckerManager::CheckerManager(AnalyzerOptions &AOptions,
                               const LangOptions &LangOpts,
                               DiagnosticsEngine &Diags,
                               ArrayRef<std::string> plugins)
    : LangOpts(LangOpts), AOptions(AOptions), Diags(Diags),
      RegistryData(std::make_unique<CheckerRegistryData>()) {
  CheckerRegistry Registry(*RegistryData, plugins, Diags, AOptions, {});
  Registry.initializeRegistry(*this);
}

// This is declared here to ensure that the destructors of `CheckerBase` and
// `CheckerRegistryData` are available.
CheckerManager::~CheckerManager() = default;

} // namespace ento
} // namespace clang
