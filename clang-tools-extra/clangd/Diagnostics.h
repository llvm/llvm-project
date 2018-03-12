//===--- Diagnostics.h ------------------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_DIAGNOSTICS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_DIAGNOSTICS_H

#include "Path.h"
#include "Protocol.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LangOptions.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include <cassert>
#include <string>

namespace clang {
namespace clangd {

/// Contains basic information about a diagnostic.
struct DiagBase {
  std::string Message;
  // Intended to be used only in error messages.
  // May be relative, absolute or even artifically constructed.
  std::string File;
  clangd::Range Range;
  DiagnosticsEngine::Level Severity = DiagnosticsEngine::Note;
  // Since File is only descriptive, we store a separate flag to distinguish
  // diags from the main file.
  bool InsideMainFile = false;
};
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const DiagBase &D);

/// Represents a single fix-it that editor can apply to fix the error.
struct Fix {
  /// Message for the fix-it.
  std::string Message;
  /// TextEdits from clang's fix-its. Must be non-empty.
  llvm::SmallVector<TextEdit, 1> Edits;
};
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Fix &F);

/// Represents a note for the diagnostic. Severity of notes can only be 'note'
/// or 'remark'.
struct Note : DiagBase {};

/// A top-level diagnostic that may have Notes and Fixes.
struct Diag : DiagBase {
  /// Elaborate on the problem, usually pointing to a related piece of code.
  std::vector<Note> Notes;
  /// *Alternative* fixes for this diagnostic, one should be chosen.
  std::vector<Fix> Fixes;
};
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Diag &D);

/// Conversion to LSP diagnostics. Formats the error message of each diagnostic
/// to include all its notes. Notes inside main file are also provided as
/// separate diagnostics with their corresponding fixits. Notes outside main
/// file do not have a corresponding LSP diagnostic, but can still be included
/// as part of their main diagnostic's message.
void toLSPDiags(
    const Diag &D,
    llvm::function_ref<void(clangd::Diagnostic, llvm::ArrayRef<Fix>)> OutFn);

/// Convert from clang diagnostic level to LSP severity.
int getSeverity(DiagnosticsEngine::Level L);

/// StoreDiags collects the diagnostics that can later be reported by
/// clangd. It groups all notes for a diagnostic into a single Diag
/// and filters out diagnostics that don't mention the main file (i.e. neither
/// the diag itself nor its notes are in the main file).
class StoreDiags : public DiagnosticConsumer {
public:
  std::vector<Diag> take();

  void BeginSourceFile(const LangOptions &Opts, const Preprocessor *) override;
  void EndSourceFile() override;
  void HandleDiagnostic(DiagnosticsEngine::Level DiagLevel,
                        const clang::Diagnostic &Info) override;

private:
  bool shouldIgnore(DiagnosticsEngine::Level DiagLevel,
                    const clang::Diagnostic &Info);

  void flushLastDiag();

  std::vector<Diag> Output;
  llvm::Optional<LangOptions> LangOpts;
  llvm::Optional<Diag> LastDiag;
  /// Is any diag or note from LastDiag in the main file?
  bool LastDiagMentionsMainFile = false;
};

} // namespace clangd
} // namespace clang

#endif
