//===-- CheckerRegistration.h - Checker Registration Function ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_STATICANALYZER_FRONTEND_CHECKERREGISTRATION_H
#define LLVM_CLANG_STATICANALYZER_FRONTEND_CHECKERREGISTRATION_H

#include "clang/AST/ASTContext.h"
#include "clang/Basic/LLVM.h"
#include <functional>
#include <memory>
#include <string>

namespace clang {
  class AnalyzerOptions;
  class LangOptions;
  class DiagnosticsEngine;

namespace ento {
  class CheckerManager;
  class CheckerRegistry;

  std::unique_ptr<CheckerManager> createCheckerManager(
      ASTContext &context,
      AnalyzerOptions &opts,
      ArrayRef<std::string> plugins,
      ArrayRef<std::function<void(CheckerRegistry &)>> checkerRegistrationFns,
      DiagnosticsEngine &diags);

} // end ento namespace

} // end namespace clang

#endif
