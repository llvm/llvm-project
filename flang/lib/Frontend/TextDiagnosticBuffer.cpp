//===- TextDiagnosticBuffer.cpp - Buffer Text Diagnostics -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a concrete diagnostic client, which buffers the diagnostic messages.
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Frontend/TextDiagnosticBuffer.h"
#include "clang/Basic/Diagnostic.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/ErrorHandling.h"

using namespace Fortran::frontend;

/// HandleDiagnostic - Store the errors, warnings, and notes that are
/// reported.
void TextDiagnosticBuffer::HandleDiagnostic(
    clang::DiagnosticsEngine::Level level, const clang::Diagnostic &info) {
  // Default implementation (warnings_/errors count).
  DiagnosticConsumer::HandleDiagnostic(level, info);

  llvm::SmallString<100> buf;
  info.FormatDiagnostic(buf);
  switch (level) {
  default:
    llvm_unreachable("Diagnostic not handled during diagnostic buffering!");
  case clang::DiagnosticsEngine::Note:
    all.emplace_back(level, notes.size());
    notes.emplace_back(info.getLocation(), std::string(buf.str()));
    break;
  case clang::DiagnosticsEngine::Warning:
    all.emplace_back(level, warnings.size());
    warnings.emplace_back(info.getLocation(), std::string(buf.str()));
    break;
  case clang::DiagnosticsEngine::Remark:
    all.emplace_back(level, remarks.size());
    remarks.emplace_back(info.getLocation(), std::string(buf.str()));
    break;
  case clang::DiagnosticsEngine::Error:
  case clang::DiagnosticsEngine::Fatal:
    all.emplace_back(level, errors.size());
    errors.emplace_back(info.getLocation(), std::string(buf.str()));
    break;
  }
}

void TextDiagnosticBuffer::flushDiagnostics(
    clang::DiagnosticsEngine &diags) const {
  for (const auto &i : all) {
    auto diag = diags.Report(diags.getCustomDiagID(i.first, "%0"));
    switch (i.first) {
    default:
      llvm_unreachable("Diagnostic not handled during diagnostic flushing!");
    case clang::DiagnosticsEngine::Note:
      diag << notes[i.second].second;
      break;
    case clang::DiagnosticsEngine::Warning:
      diag << warnings[i.second].second;
      break;
    case clang::DiagnosticsEngine::Remark:
      diag << remarks[i.second].second;
      break;
    case clang::DiagnosticsEngine::Error:
    case clang::DiagnosticsEngine::Fatal:
      diag << errors[i.second].second;
      break;
    }
  }
}
