//===--- LogDiagnosticPrinter.h - Log Diagnostic Client ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_LOGDIAGNOSTICPRINTER_H
#define LLVM_CLANG_FRONTEND_LOGDIAGNOSTICPRINTER_H

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace clang {
class DiagnosticOptions;
class LangOptions;

class LogDiagnosticPrinter : public DiagnosticConsumer {
  struct DiagEntryLocation {
    /// The source file name, if available and if different from DiagEntry's
    /// Filename.
    std::string Filename;

    /// The source file line number, if available.
    unsigned Line;

    /// The source file column number, if available.
    unsigned Column;

    /// The source file offset, if available.
    unsigned Offset;
  };

  struct DiagEntryRange {
    /// The range start.
    LogDiagnosticPrinter::DiagEntryLocation Start;

    /// The range end.
    LogDiagnosticPrinter::DiagEntryLocation End;
  };

  struct DiagEntryFixIt {
    /// The range of existing source file to act upon.
    LogDiagnosticPrinter::DiagEntryRange RemoveRange;

    /// The code to insert at the start of the range,
    ///   after removal of the range; may be empty for pure removal.
    std::string CodeToInsert;
  };

  struct DiagEntry {
    /// The primary message line of the diagnostic.
    std::string Message;

    /// The source file name, if available.
    std::string Filename;

    /// The source file line number, if available.
    unsigned Line;

    /// The source file column number, if available.
    unsigned Column;

    /// The ID of the diagnostic.
    unsigned DiagnosticID;

    /// The Option Flag for the diagnostic
    std::string WarningOption;

    /// The level of the diagnostic.
    DiagnosticsEngine::Level DiagnosticLevel;

    /// The source ranges of the diagnostic.
    SmallVector<LogDiagnosticPrinter::DiagEntryRange, 2> SourceRanges;

    /// The fix-its for the diagnostic.
    SmallVector<LogDiagnosticPrinter::DiagEntryFixIt, 2> FixIts;
  };

  void
  EmitDiagEntryLocation(llvm::raw_ostream &OS, StringRef Indent,
                        const LogDiagnosticPrinter::DiagEntryLocation &Del);

  void EmitDiagEntry(llvm::raw_ostream &OS,
                     const LogDiagnosticPrinter::DiagEntry &DE);

  // Conditional ownership (when StreamOwner is non-null, it's keeping OS
  // alive). We might want to replace this with a wrapper for conditional
  // ownership eventually - it seems to pop up often enough.
  raw_ostream &OS;
  std::unique_ptr<raw_ostream> StreamOwner;
  const LangOptions *LangOpts;
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts;

  SourceLocation LastWarningLoc;
  FullSourceLoc LastLoc;

  SmallVector<DiagEntry, 8> Entries;

  std::string MainFilename;
  std::string DwarfDebugFlags;

public:
  LogDiagnosticPrinter(raw_ostream &OS, DiagnosticOptions *Diags,
                       std::unique_ptr<raw_ostream> StreamOwner);

  void setDwarfDebugFlags(StringRef Value) {
    DwarfDebugFlags = Value;
  }

  void BeginSourceFile(const LangOptions &LO, const Preprocessor *PP) override {
    LangOpts = &LO;
  }

  void EndSourceFile() override;

  void HandleDiagnostic(DiagnosticsEngine::Level DiagLevel,
                        const Diagnostic &Info) override;
};

} // end namespace clang

#endif
