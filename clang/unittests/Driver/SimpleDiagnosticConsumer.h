//===- unittests/Driver/SimpleDiagnosticConsumer.h ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Simple diagnostic consumer to grab up diagnostics for testing.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_UNITTESTS_SIMPLEDIAGNOSTICCONSUMER_H
#define CLANG_UNITTESTS_SIMPLEDIAGNOSTICCONSUMER_H

#include "clang/Basic/Diagnostic.h"
#include "llvm/ADT/SmallString.h"

struct SimpleDiagnosticConsumer : public clang::DiagnosticConsumer {
  void HandleDiagnostic(clang::DiagnosticsEngine::Level DiagLevel,
                        const clang::Diagnostic &Info) override {
    if (DiagLevel == clang::DiagnosticsEngine::Level::Error) {
      Errors.emplace_back();
      Info.FormatDiagnostic(Errors.back());
    } else {
      Msgs.emplace_back();
      Info.FormatDiagnostic(Msgs.back());
    }
  }
  void clear() override {
    Msgs.clear();
    Errors.clear();
    DiagnosticConsumer::clear();
  }
  std::vector<llvm::SmallString<32>> Msgs;
  std::vector<llvm::SmallString<32>> Errors;
};

#endif // CLANG_UNITTESTS_SIMPLEDIAGNOSTICCONSUMER_H
