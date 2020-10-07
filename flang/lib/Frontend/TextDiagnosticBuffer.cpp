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
    all_.emplace_back(level, notes_.size());
    notes_.emplace_back(info.getLocation(), std::string(buf.str()));
    break;
  case clang::DiagnosticsEngine::Warning:
    all_.emplace_back(level, warnings_.size());
    warnings_.emplace_back(info.getLocation(), std::string(buf.str()));
    break;
  case clang::DiagnosticsEngine::Remark:
    all_.emplace_back(level, remarks_.size());
    remarks_.emplace_back(info.getLocation(), std::string(buf.str()));
    break;
  case clang::DiagnosticsEngine::Error:
  case clang::DiagnosticsEngine::Fatal:
    all_.emplace_back(level, errors_.size());
    errors_.emplace_back(info.getLocation(), std::string(buf.str()));
    break;
  }
}

void TextDiagnosticBuffer::FlushDiagnostics(
    clang::DiagnosticsEngine &Diags) const {
  for (const auto &i : all_) {
    auto Diag = Diags.Report(Diags.getCustomDiagID(i.first, "%0"));
    switch (i.first) {
    default:
      llvm_unreachable("Diagnostic not handled during diagnostic flushing!");
    case clang::DiagnosticsEngine::Note:
      Diag << notes_[i.second].second;
      break;
    case clang::DiagnosticsEngine::Warning:
      Diag << warnings_[i.second].second;
      break;
    case clang::DiagnosticsEngine::Remark:
      Diag << remarks_[i.second].second;
      break;
    case clang::DiagnosticsEngine::Error:
    case clang::DiagnosticsEngine::Fatal:
      Diag << errors_[i.second].second;
      break;
    }
  }
}
