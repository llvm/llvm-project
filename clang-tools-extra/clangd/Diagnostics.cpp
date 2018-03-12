//===--- Diagnostics.cpp ----------------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//

#include "Diagnostics.h"
#include "Compiler.h"
#include "Logger.h"
#include "SourceCode.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "llvm/Support/Capacity.h"
#include "llvm/Support/Path.h"
#include <algorithm>

namespace clang {
namespace clangd {

namespace {

bool mentionsMainFile(const Diag &D) {
  if (D.InsideMainFile)
    return true;
  // Fixes are always in the main file.
  if (!D.Fixes.empty())
    return true;
  for (auto &N : D.Notes) {
    if (N.InsideMainFile)
      return true;
  }
  return false;
}

// Checks whether a location is within a half-open range.
// Note that clang also uses closed source ranges, which this can't handle!
bool locationInRange(SourceLocation L, CharSourceRange R,
                     const SourceManager &M) {
  assert(R.isCharRange());
  if (!R.isValid() || M.getFileID(R.getBegin()) != M.getFileID(R.getEnd()) ||
      M.getFileID(R.getBegin()) != M.getFileID(L))
    return false;
  return L != R.getEnd() && M.isPointWithin(L, R.getBegin(), R.getEnd());
}

// Clang diags have a location (shown as ^) and 0 or more ranges (~~~~).
// LSP needs a single range.
Range diagnosticRange(const clang::Diagnostic &D, const LangOptions &L) {
  auto &M = D.getSourceManager();
  auto Loc = M.getFileLoc(D.getLocation());
  // Accept the first range that contains the location.
  for (const auto &CR : D.getRanges()) {
    auto R = Lexer::makeFileCharRange(CR, M, L);
    if (locationInRange(Loc, R, M))
      return halfOpenToRange(M, R);
  }
  // The range may be given as a fixit hint instead.
  for (const auto &F : D.getFixItHints()) {
    auto R = Lexer::makeFileCharRange(F.RemoveRange, M, L);
    if (locationInRange(Loc, R, M))
      return halfOpenToRange(M, R);
  }
  // If no suitable range is found, just use the token at the location.
  auto R = Lexer::makeFileCharRange(CharSourceRange::getTokenRange(Loc), M, L);
  if (!R.isValid()) // Fall back to location only, let the editor deal with it.
    R = CharSourceRange::getCharRange(Loc);
  return halfOpenToRange(M, R);
}

TextEdit toTextEdit(const FixItHint &FixIt, const SourceManager &M,
                    const LangOptions &L) {
  TextEdit Result;
  Result.range =
      halfOpenToRange(M, Lexer::makeFileCharRange(FixIt.RemoveRange, M, L));
  Result.newText = FixIt.CodeToInsert;
  return Result;
}

bool isInsideMainFile(const SourceLocation Loc, const SourceManager &M) {
  return Loc.isValid() && M.isInMainFile(Loc);
}

bool isInsideMainFile(const clang::Diagnostic &D) {
  if (!D.hasSourceManager())
    return false;

  return isInsideMainFile(D.getLocation(), D.getSourceManager());
}

bool isNote(DiagnosticsEngine::Level L) {
  return L == DiagnosticsEngine::Note || L == DiagnosticsEngine::Remark;
}

llvm::StringRef diagLeveltoString(DiagnosticsEngine::Level Lvl) {
  switch (Lvl) {
  case DiagnosticsEngine::Ignored:
    return "ignored";
  case DiagnosticsEngine::Note:
    return "note";
  case DiagnosticsEngine::Remark:
    return "remark";
  case DiagnosticsEngine::Warning:
    return "warning";
  case DiagnosticsEngine::Error:
    return "error";
  case DiagnosticsEngine::Fatal:
    return "fatal error";
  }
  llvm_unreachable("unhandled DiagnosticsEngine::Level");
}

/// Prints a single diagnostic in a clang-like manner, the output includes
/// location, severity and error message. An example of the output message is:
///
///     main.cpp:12:23: error: undeclared identifier
///
/// For main file we only print the basename and for all other files we print
/// the filename on a separate line to provide a slightly more readable output
/// in the editors:
///
///     dir1/dir2/dir3/../../dir4/header.h:12:23
///     error: undeclared identifier
void printDiag(llvm::raw_string_ostream &OS, const DiagBase &D) {
  if (D.InsideMainFile) {
    // Paths to main files are often taken from compile_command.json, where they
    // are typically absolute. To reduce noise we print only basename for them,
    // it should not be confusing and saves space.
    OS << llvm::sys::path::filename(D.File) << ":";
  } else {
    OS << D.File << ":";
  }
  // Note +1 to line and character. clangd::Range is zero-based, but when
  // printing for users we want one-based indexes.
  auto Pos = D.Range.start;
  OS << (Pos.line + 1) << ":" << (Pos.character + 1) << ":";
  // The non-main-file paths are often too long, putting them on a separate
  // line improves readability.
  if (D.InsideMainFile)
    OS << " ";
  else
    OS << "\n";
  OS << diagLeveltoString(D.Severity) << ": " << D.Message;
}

/// Returns a message sent to LSP for the main diagnostic in \p D.
/// The message includes all the notes with their corresponding locations.
/// However, notes with fix-its are excluded as those usually only contain a
/// fix-it message and just add noise if included in the message for diagnostic.
/// Example output:
///
///     no matching function for call to 'foo'
///
///     main.cpp:3:5: note: candidate function not viable: requires 2 arguments
///
///     dir1/dir2/dir3/../../dir4/header.h:12:23
///     note: candidate function not viable: requires 3 arguments
std::string mainMessage(const Diag &D) {
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  OS << D.Message;
  for (auto &Note : D.Notes) {
    OS << "\n\n";
    printDiag(OS, Note);
  }
  OS.flush();
  return Result;
}

/// Returns a message sent to LSP for the note of the main diagnostic.
/// The message includes the main diagnostic to provide the necessary context
/// for the user to understand the note.
std::string noteMessage(const Diag &Main, const DiagBase &Note) {
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  OS << Note.Message;
  OS << "\n\n";
  printDiag(OS, Main);
  OS.flush();
  return Result;
}
} // namespace

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const DiagBase &D) {
  if (!D.InsideMainFile)
    OS << "[in " << D.File << "] ";
  return OS << D.Message;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Fix &F) {
  OS << F.Message << " {";
  const char *Sep = "";
  for (const auto &Edit : F.Edits) {
    OS << Sep << Edit;
    Sep = ", ";
  }
  return OS << "}";
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Diag &D) {
  OS << static_cast<const DiagBase &>(D);
  if (!D.Notes.empty()) {
    OS << ", notes: {";
    const char *Sep = "";
    for (auto &Note : D.Notes) {
      OS << Sep << Note;
      Sep = ", ";
    }
    OS << "}";
  }
  if (!D.Fixes.empty()) {
    OS << ", fixes: {";
    const char *Sep = "";
    for (auto &Fix : D.Fixes) {
      OS << Sep << Fix;
      Sep = ", ";
    }
  }
  return OS;
}

void toLSPDiags(
    const Diag &D,
    llvm::function_ref<void(clangd::Diagnostic, llvm::ArrayRef<Fix>)> OutFn) {
  auto FillBasicFields = [](const DiagBase &D) -> clangd::Diagnostic {
    clangd::Diagnostic Res;
    Res.range = D.Range;
    Res.severity = getSeverity(D.Severity);
    return Res;
  };

  {
    clangd::Diagnostic Main = FillBasicFields(D);
    Main.message = mainMessage(D);
    OutFn(std::move(Main), D.Fixes);
  }

  for (auto &Note : D.Notes) {
    if (!Note.InsideMainFile)
      continue;
    clangd::Diagnostic Res = FillBasicFields(Note);
    Res.message = noteMessage(D, Note);
    OutFn(std::move(Res), llvm::ArrayRef<Fix>());
  }
}

int getSeverity(DiagnosticsEngine::Level L) {
  switch (L) {
  case DiagnosticsEngine::Remark:
    return 4;
  case DiagnosticsEngine::Note:
    return 3;
  case DiagnosticsEngine::Warning:
    return 2;
  case DiagnosticsEngine::Fatal:
  case DiagnosticsEngine::Error:
    return 1;
  case DiagnosticsEngine::Ignored:
    return 0;
  }
  llvm_unreachable("Unknown diagnostic level!");
}

std::vector<Diag> StoreDiags::take() { return std::move(Output); }

void StoreDiags::BeginSourceFile(const LangOptions &Opts,
                                 const Preprocessor *) {
  LangOpts = Opts;
}

void StoreDiags::EndSourceFile() {
  flushLastDiag();
  LangOpts = llvm::None;
}

void StoreDiags::HandleDiagnostic(DiagnosticsEngine::Level DiagLevel,
                                  const clang::Diagnostic &Info) {
  DiagnosticConsumer::HandleDiagnostic(DiagLevel, Info);

  if (!LangOpts || !Info.hasSourceManager()) {
    IgnoreDiagnostics::log(DiagLevel, Info);
    return;
  }

  bool InsideMainFile = isInsideMainFile(Info);

  auto FillDiagBase = [&](DiagBase &D) {
    D.Range = diagnosticRange(Info, *LangOpts);
    llvm::SmallString<64> Message;
    Info.FormatDiagnostic(Message);
    D.Message = Message.str();
    D.InsideMainFile = InsideMainFile;
    D.File = Info.getSourceManager().getFilename(Info.getLocation());
    D.Severity = DiagLevel;
    return D;
  };

  auto AddFix = [&]() -> bool {
    assert(!Info.getFixItHints().empty() &&
           "diagnostic does not have attached fix-its");
    if (!InsideMainFile)
      return false;

    llvm::SmallVector<TextEdit, 1> Edits;
    for (auto &FixIt : Info.getFixItHints()) {
      if (!isInsideMainFile(FixIt.RemoveRange.getBegin(),
                            Info.getSourceManager()))
        return false;
      Edits.push_back(toTextEdit(FixIt, Info.getSourceManager(), *LangOpts));
    }

    llvm::SmallString<64> Message;
    Info.FormatDiagnostic(Message);
    LastDiag->Fixes.push_back(Fix{Message.str(), std::move(Edits)});
    return true;
  };

  if (!isNote(DiagLevel)) {
    // Handle the new main diagnostic.
    flushLastDiag();

    LastDiag = Diag();
    FillDiagBase(*LastDiag);

    if (!Info.getFixItHints().empty())
      AddFix();
  } else {
    // Handle a note to an existing diagnostic.
    if (!LastDiag) {
      assert(false && "Adding a note without main diagnostic");
      IgnoreDiagnostics::log(DiagLevel, Info);
      return;
    }

    if (!Info.getFixItHints().empty()) {
      // A clang note with fix-it is not a separate diagnostic in clangd. We
      // attach it as a Fix to the main diagnostic instead.
      if (!AddFix())
        IgnoreDiagnostics::log(DiagLevel, Info);
    } else {
      // A clang note without fix-its corresponds to clangd::Note.
      Note N;
      FillDiagBase(N);

      LastDiag->Notes.push_back(std::move(N));
    }
  }
}

void StoreDiags::flushLastDiag() {
  if (!LastDiag)
    return;
  if (mentionsMainFile(*LastDiag))
    Output.push_back(std::move(*LastDiag));
  else
    log(Twine("Dropped diagnostic outside main file:") + LastDiag->File + ":" +
        LastDiag->Message);
  LastDiag.reset();
}

} // namespace clangd
} // namespace clang
