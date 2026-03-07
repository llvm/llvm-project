//===------- SARIFDiagnosticPrinter.cpp - Diagnostic Printer---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This diagnostic client prints out their diagnostic messages in SARIF format.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/SARIFDiagnosticPrinter.h"
#include "clang/Basic/DiagnosticDriver.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/Sarif.h"
#include "clang/Frontend/DiagnosticRenderer.h"
#include "clang/Frontend/SARIFDiagnostic.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {

SARIFDiagnosticPrinter::SARIFDiagnosticPrinter(llvm::StringRef FilePath,
                                               DiagnosticOptions &DiagOpts)
    : FilePath(FilePath), DiagOpts(DiagOpts) {}

std::unique_ptr<SARIFDiagnosticPrinter>
SARIFDiagnosticPrinter::create(ArrayRef<std::pair<StringRef, StringRef>> Config,
                               DiagnosticOptions &DiagOpts,
                               DiagnosticsEngine &Diags) {
  std::optional<std::string> FilePath;
  SarifVersion Version = SarifDocumentWriter::getDefaultVersion();

  for (const auto &Pair : Config) {
    if (Pair.first == "file") {
      FilePath = Pair.second;
    } else if (Pair.first == "version") {
      auto SupportedVersions = SarifDocumentWriter::getSupportedVersions();
      auto FoundVersion =
          std::find_if(SupportedVersions.begin(), SupportedVersions.end(),
                       [=](const SarifVersion &V) {
                         return V.CommandLineVersion == Pair.second;
                       });
      if (FoundVersion != SupportedVersions.end()) {
        Version = *FoundVersion;
      } else {
        SmallString<64> SupportedList;
        bool First = true;
        for (const auto &V : SupportedVersions) {
          if (First) {
            First = false;
          } else {
            SupportedList.append(", ");
          }
          SupportedList.append("'");
          SupportedList.append(V.CommandLineVersion);
          SupportedList.append("'");
        }
        Diags.Report(SourceLocation(), diag::err_invalid_sarif_version)
            << Pair.second << SupportedList;
      }
    } else {
      Diags.Report(SourceLocation(), diag::err_diagnostic_output_unknown_key)
          << "sarif" << Pair.second;
    }
  }

  if (!FilePath) {
    // We should probably have a default here based on the input file name or
    // the output object file name, but I'm not sure how to get that information
    // here.
    Diags.Report(SourceLocation(), diag::err_missing_sarif_file_name);
    return {};
  }

  return std::make_unique<SARIFDiagnosticPrinter>(*FilePath, DiagOpts);
}

void SARIFDiagnosticPrinter::BeginSourceFile(const LangOptions &LO,
                                             const Preprocessor *PP) {
  // Build the SARIFDiagnostic utility.
  if (!hasSarifWriter() && PP) {
    // Use the SourceManager from the preprocessor.
    // REVIEW: Are there cases where we won't have a preprocessor but we will
    // have a SourceManager? If so, we should pass the SourceManager directly to
    // the BeginSourceFile call.
    setSarifWriter(
        std::make_unique<SarifDocumentWriter>(PP->getSourceManager()));
  }
  assert(hasSarifWriter() && "Writer not set!");
  assert(!SARIFDiag && "SARIFDiagnostic already set.");
  SARIFDiag = std::make_unique<SARIFDiagnostic>(LO, DiagOpts, &*Writer);
  // Initialize the SARIF object.
  Writer->createRun("clang", Prefix, getClangFullVersion());
}

void SARIFDiagnosticPrinter::EndSourceFile() {
  assert(SARIFDiag && "SARIFDiagnostic has not been set.");
  Writer->endRun();
  llvm::json::Value Value(Writer->createDocument());
  if (FilePath.empty()) {
    // Write to console.
    llvm::errs() << llvm::formatv("\n{0:2}\n\n", Value);
    llvm::errs().flush();
  } else {
    // Write to file.
    std::error_code EC;
    llvm::raw_fd_ostream OS(FilePath, EC, llvm::sys::fs::OF_TextWithCRLF);
    if (EC) {
      // FIXME: Emit a real diagnostic, similar to how the serialized diagnostic
      // log does via getMetaDiags().
      llvm::errs() << "warning: could not create file: " << EC.message()
                   << '\n';
    } else {
      OS << llvm::formatv("{0:2}\n", Value);
    }
  }

  SARIFDiag.reset();
}

void SARIFDiagnosticPrinter::HandleDiagnostic(DiagnosticsEngine::Level Level,
                                              const Diagnostic &Info) {
  assert(SARIFDiag && "SARIFDiagnostic has not been set.");
  // Default implementation (Warnings/errors count). Keeps track of the
  // number of errors.
  DiagnosticConsumer::HandleDiagnostic(Level, Info);

  // Render the diagnostic message into a temporary buffer eagerly. We'll use
  // this later as we add the diagnostic to the SARIF object.
  SmallString<100> OutStr;
  Info.FormatDiagnostic(OutStr);

  llvm::raw_svector_ostream DiagMessageStream(OutStr);

  // Use a dedicated, simpler path for diagnostics without a valid location.
  // This is important as if the location is missing, we may be emitting
  // diagnostics in a context that lacks language options, a source manager, or
  // other infrastructure necessary when emitting more rich diagnostics.
  if (Info.getLocation().isInvalid()) {
    // FIXME: Enable diagnostics without a source manager
    return;
  }

  // Assert that the rest of our infrastructure is setup properly.
  assert(Info.hasSourceManager() &&
         "Unexpected diagnostic with no source manager");

  SARIFDiag->emitDiagnostic(
      FullSourceLoc(Info.getLocation(), Info.getSourceManager()), Level,
      DiagMessageStream.str(), Info.getRanges(), Info.getFixItHints(), &Info);
}
} // namespace clang
