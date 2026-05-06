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

static void printWarningOption(llvm::raw_ostream &os,
    clang::DiagnosticsEngine::Level level, const clang::Diagnostic &info) {
  auto &diagIDs = *info.getDiags()->getDiagnosticIDs();

  if (level == clang::DiagnosticsEngine::Warning) {
    llvm::StringRef opt = diagIDs.getWarningOptionForDiag(info.getID());
    if (!opt.empty()) {
      os << " [-W" << opt;
      llvm::StringRef optValue = info.getFlagValue();
      if (!optValue.empty())
        os << "=" << optValue;
      os << ']';
    }
  }
}

/// HandleDiagnostic - Store the errors, warnings, and notes that are
/// reported.
void TextDiagnosticBuffer::HandleDiagnostic(
    clang::DiagnosticsEngine::Level level, const clang::Diagnostic &info) {
  // Default implementation (warnings/errors count).
  DiagnosticConsumer::HandleDiagnostic(level, info);

  llvm::SmallString<100> buf;
  info.FormatDiagnostic(buf);

  // This function dealing with diagnostics emitted directly through the
  // diagnostic engine, e.g. in CompilerInvocation. With -Werror any warning
  // emitted there would become an error, and prevented any part of compilation
  // from happening. In case of OpenMP, this would cause the warning about an
  // incomplete implementation to completely skip the compilation, which is
  // undesirable.
  // Downgrade -Werror'ed warnings back to warnings to avoid this situation.
  const clang::DiagnosticsEngine &diags = *info.getDiags();
  if (level == clang::DiagnosticsEngine::Error) {
    if (!diags.getDiagnosticIDs()->isDefaultMappingAsError(info.getID()))
      level = clang::DiagnosticsEngine::Warning;
  }

  llvm::raw_svector_ostream os(buf);
  printWarningOption(os, level, info);

  switch (level) {
  default:
    llvm_unreachable("Diagnostic not handled during diagnostic buffering!");
  case clang::DiagnosticsEngine::Note:
    all.emplace_back(level, notes.size());
    notes.emplace_back(info.getLocation(), std::string(buf));
    break;
  case clang::DiagnosticsEngine::Warning:
    all.emplace_back(level, warnings.size());
    warnings.emplace_back(info.getLocation(), std::string(buf));
    break;
  case clang::DiagnosticsEngine::Remark:
    all.emplace_back(level, remarks.size());
    remarks.emplace_back(info.getLocation(), std::string(buf));
    break;
  case clang::DiagnosticsEngine::Error:
  case clang::DiagnosticsEngine::Fatal:
    all.emplace_back(level, errors.size());
    errors.emplace_back(info.getLocation(), std::string(buf));
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
