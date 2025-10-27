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

#include "clang/Driver/Driver.h"
#include "clang/Basic/Diagnostic.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/VirtualFileSystem.h"

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

// Using SimpleDiagnosticConsumer, this function makes a clang Driver, suitable
// for testing situations where it will only ever be used for emitting
// diagnostics, such as being passed to `MultilibSet::select`.
inline clang::driver::Driver diagnostic_test_driver() {
  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> InMemoryFileSystem(
      new llvm::vfs::InMemoryFileSystem);
  auto *DiagConsumer = new SimpleDiagnosticConsumer;
  clang::DiagnosticOptions DiagOpts;
  clang::DiagnosticsEngine Diags(clang::DiagnosticIDs::create(), DiagOpts,
                                 DiagConsumer);
  return clang::driver::Driver("/bin/clang", "", Diags, "", InMemoryFileSystem);
}

#endif // CLANG_UNITTESTS_SIMPLEDIAGNOSTICCONSUMER_H
