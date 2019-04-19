//===--- Diagnostics.cpp -----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Diagnostics.h"
#include "../clang-tidy/ClangTidyDiagnosticConsumer.h"
#include "Compiler.h"
#include "Logger.h"
#include "Protocol.h"
#include "SourceCode.h"
#include "clang/Basic/AllDiagnostics.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "llvm/Support/Capacity.h"
#include "llvm/Support/Path.h"
#include <algorithm>

namespace clang {
namespace clangd {
namespace {

const char *getDiagnosticCode(unsigned ID) {
  switch (ID) {
#define DIAG(ENUM, CLASS, DEFAULT_MAPPING, DESC, GROPU, SFINAE, NOWERROR,      \
             SHOWINSYSHEADER, CATEGORY)                                        \
  case clang::diag::ENUM:                                                      \
    return #ENUM;
#include "clang/Basic/DiagnosticASTKinds.inc"
#include "clang/Basic/DiagnosticAnalysisKinds.inc"
#include "clang/Basic/DiagnosticCommentKinds.inc"
#include "clang/Basic/DiagnosticCommonKinds.inc"
#include "clang/Basic/DiagnosticDriverKinds.inc"
#include "clang/Basic/DiagnosticFrontendKinds.inc"
#include "clang/Basic/DiagnosticLexKinds.inc"
#include "clang/Basic/DiagnosticParseKinds.inc"
#include "clang/Basic/DiagnosticRefactoringKinds.inc"
#include "clang/Basic/DiagnosticSemaKinds.inc"
#include "clang/Basic/DiagnosticSerializationKinds.inc"
#undef DIAG
  default:
    return nullptr;
  }
}

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
  for (const auto &CR : D.getRanges()) {
    auto R = Lexer::makeFileCharRange(CR, M, L);
    if (locationInRange(Loc, R, M))
      return halfOpenToRange(M, R);
  }
  llvm::Optional<Range> FallbackRange;
  // The range may be given as a fixit hint instead.
  for (const auto &F : D.getFixItHints()) {
    auto R = Lexer::makeFileCharRange(F.RemoveRange, M, L);
    if (locationInRange(Loc, R, M))
      return halfOpenToRange(M, R);
    // If there's a fixit that performs insertion, it has zero-width. Therefore
    // it can't contain the location of the diag, but it might be possible that
    // this should be reported as range. For example missing semicolon.
    if (R.getBegin() == R.getEnd() && Loc == R.getBegin())
      FallbackRange = halfOpenToRange(M, R);
  }
  if (FallbackRange)
    return *FallbackRange;
  // If no suitable range is found, just use the token at the location.
  auto R = Lexer::makeFileCharRange(CharSourceRange::getTokenRange(Loc), M, L);
  if (!R.isValid()) // Fall back to location only, let the editor deal with it.
    R = CharSourceRange::getCharRange(Loc);
  return halfOpenToRange(M, R);
}

bool isInsideMainFile(const SourceLocation Loc, const SourceManager &M) {
  return Loc.isValid() && M.isWrittenInMainFile(M.getFileLoc(Loc));
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

/// Capitalizes the first word in the diagnostic's message.
std::string capitalize(std::string Message) {
  if (!Message.empty())
    Message[0] = llvm::toUpper(Message[0]);
  return Message;
}

/// Returns a message sent to LSP for the main diagnostic in \p D.
/// This message may include notes, if they're not emited in some other way.
/// Example output:
///
///     no matching function for call to 'foo'
///
///     main.cpp:3:5: note: candidate function not viable: requires 2 arguments
///
///     dir1/dir2/dir3/../../dir4/header.h:12:23
///     note: candidate function not viable: requires 3 arguments
std::string mainMessage(const Diag &D, const ClangdDiagnosticOptions &Opts) {
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  OS << D.Message;
  if (Opts.DisplayFixesCount && !D.Fixes.empty())
    OS << " (" << (D.Fixes.size() > 1 ? "fixes" : "fix") << " available)";
  // If notes aren't emitted as structured info, add them to the message.
  if (!Opts.EmitRelatedLocations)
    for (auto &Note : D.Notes) {
      OS << "\n\n";
      printDiag(OS, Note);
    }
  OS.flush();
  return capitalize(std::move(Result));
}

/// Returns a message sent to LSP for the note of the main diagnostic.
std::string noteMessage(const Diag &Main, const DiagBase &Note,
                        const ClangdDiagnosticOptions &Opts) {
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  OS << Note.Message;
  // If the client doesn't support structured links between the note and the
  // original diagnostic, then emit the main diagnostic to give context.
  if (!Opts.EmitRelatedLocations) {
    OS << "\n\n";
    printDiag(OS, Main);
  }
  OS.flush();
  return capitalize(std::move(Result));
}
} // namespace

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const DiagBase &D) {
  OS << "[";
  if (!D.InsideMainFile)
    OS << D.File << ":";
  OS << D.Range.start << "-" << D.Range.end << "] ";

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

CodeAction toCodeAction(const Fix &F, const URIForFile &File) {
  CodeAction Action;
  Action.title = F.Message;
  Action.kind = CodeAction::QUICKFIX_KIND;
  Action.edit.emplace();
  Action.edit->changes.emplace();
  (*Action.edit->changes)[File.uri()] = {F.Edits.begin(), F.Edits.end()};
  return Action;
}

void toLSPDiags(
    const Diag &D, const URIForFile &File, const ClangdDiagnosticOptions &Opts,
    llvm::function_ref<void(clangd::Diagnostic, llvm::ArrayRef<Fix>)> OutFn) {
  auto FillBasicFields = [](const DiagBase &D) -> clangd::Diagnostic {
    clangd::Diagnostic Res;
    Res.range = D.Range;
    Res.severity = getSeverity(D.Severity);
    return Res;
  };

  clangd::Diagnostic Main = FillBasicFields(D);
  Main.code = D.Name;
  switch (D.Source) {
  case Diag::Clang:
    Main.source = "clang";
    break;
  case Diag::ClangTidy:
    Main.source = "clang-tidy";
    break;
  case Diag::Unknown:
    break;
  }
  if (Opts.EmbedFixesInDiagnostics) {
    Main.codeActions.emplace();
    for (const auto &Fix : D.Fixes)
      Main.codeActions->push_back(toCodeAction(Fix, File));
  }
  if (Opts.SendDiagnosticCategory && !D.Category.empty())
    Main.category = D.Category;

  Main.message = mainMessage(D, Opts);
  if (Opts.EmitRelatedLocations) {
    Main.relatedInformation.emplace();
    for (auto &Note : D.Notes) {
      if (!Note.AbsFile) {
        vlog("Dropping note from unknown file: {0}", Note);
        continue;
      }
      DiagnosticRelatedInformation RelInfo;
      RelInfo.location.range = Note.Range;
      RelInfo.location.uri =
          URIForFile::canonicalize(*Note.AbsFile, File.file());
      RelInfo.message = noteMessage(D, Note, Opts);
      Main.relatedInformation->push_back(std::move(RelInfo));
    }
  }
  OutFn(std::move(Main), D.Fixes);

  // If we didn't emit the notes as relatedLocations, emit separate diagnostics
  // so the user can find the locations easily.
  if (!Opts.EmitRelatedLocations)
    for (auto &Note : D.Notes) {
      if (!Note.InsideMainFile)
        continue;
      clangd::Diagnostic Res = FillBasicFields(Note);
      Res.message = noteMessage(D, Note, Opts);
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

std::vector<Diag> StoreDiags::take(const clang::tidy::ClangTidyContext *Tidy) {
  // Fill in name/source now that we have all the context needed to map them.
  for (auto &Diag : Output) {
    if (const char *ClangDiag = getDiagnosticCode(Diag.ID)) {
      // Warnings controlled by -Wfoo are better recognized by that name.
      StringRef Warning = DiagnosticIDs::getWarningOptionForDiag(Diag.ID);
      if (!Warning.empty()) {
        Diag.Name = ("-W" + Warning).str();
      } else {
        StringRef Name(ClangDiag);
        // Almost always an error, with a name like err_enum_class_reference.
        // Drop the err_ prefix for brevity.
        Name.consume_front("err_");
        Diag.Name = Name;
      }
      Diag.Source = Diag::Clang;
      continue;
    }
    if (Tidy != nullptr) {
      std::string TidyDiag = Tidy->getCheckName(Diag.ID);
      if (!TidyDiag.empty()) {
        Diag.Name = std::move(TidyDiag);
        Diag.Source = Diag::ClangTidy;
        // clang-tidy bakes the name into diagnostic messages. Strip it out.
        // It would be much nicer to make clang-tidy not do this.
        auto CleanMessage = [&](std::string &Msg) {
          StringRef Rest(Msg);
          if (Rest.consume_back("]") && Rest.consume_back(Diag.Name) &&
              Rest.consume_back(" ["))
            Msg.resize(Rest.size());
        };
        CleanMessage(Diag.Message);
        for (auto& Note : Diag.Notes)
          CleanMessage(Note.Message);
        continue;
      }
    }
  }
  return std::move(Output);
}

void StoreDiags::BeginSourceFile(const LangOptions &Opts,
                                 const Preprocessor *) {
  LangOpts = Opts;
}

void StoreDiags::EndSourceFile() {
  flushLastDiag();
  LangOpts = None;
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
    auto &SM = Info.getSourceManager();
    D.AbsFile = getCanonicalPath(
        SM.getFileEntryForID(SM.getFileID(Info.getLocation())), SM);
    D.Severity = DiagLevel;
    D.Category = DiagnosticIDs::getCategoryNameFromID(
                     DiagnosticIDs::getCategoryNumberForDiag(Info.getID()))
                     .str();
    return D;
  };

  auto AddFix = [&](bool SyntheticMessage) -> bool {
    assert(!Info.getFixItHints().empty() &&
           "diagnostic does not have attached fix-its");
    if (!InsideMainFile)
      return false;

    llvm::SmallVector<TextEdit, 1> Edits;
    for (auto &FixIt : Info.getFixItHints()) {
      // Follow clang's behavior, don't apply FixIt to the code in macros,
      // we are less certain it is the right fix.
      if (FixIt.RemoveRange.getBegin().isMacroID() ||
          FixIt.RemoveRange.getEnd().isMacroID())
        return false;
      if (!isInsideMainFile(FixIt.RemoveRange.getBegin(),
                            Info.getSourceManager()))
        return false;
      Edits.push_back(toTextEdit(FixIt, Info.getSourceManager(), *LangOpts));
    }

    llvm::SmallString<64> Message;
    // If requested and possible, create a message like "change 'foo' to 'bar'".
    if (SyntheticMessage && Info.getNumFixItHints() == 1) {
      const auto &FixIt = Info.getFixItHint(0);
      bool Invalid = false;
      llvm::StringRef Remove = Lexer::getSourceText(
          FixIt.RemoveRange, Info.getSourceManager(), *LangOpts, &Invalid);
      llvm::StringRef Insert = FixIt.CodeToInsert;
      if (!Invalid) {
        llvm::raw_svector_ostream M(Message);
        if (!Remove.empty() && !Insert.empty())
          M << "change '" << Remove << "' to '" << Insert << "'";
        else if (!Remove.empty())
          M << "remove '" << Remove << "'";
        else if (!Insert.empty())
          M << "insert '" << Insert << "'";
        // Don't allow source code to inject newlines into diagnostics.
        std::replace(Message.begin(), Message.end(), '\n', ' ');
      }
    }
    if (Message.empty()) // either !SytheticMessage, or we failed to make one.
      Info.FormatDiagnostic(Message);
    LastDiag->Fixes.push_back(Fix{Message.str(), std::move(Edits)});
    return true;
  };

  if (!isNote(DiagLevel)) {
    // Handle the new main diagnostic.
    flushLastDiag();

    LastDiag = Diag();
    LastDiag->ID = Info.getID();
    FillDiagBase(*LastDiag);

    if (!Info.getFixItHints().empty())
      AddFix(true /* try to invent a message instead of repeating the diag */);
    if (Fixer) {
      auto ExtraFixes = Fixer(DiagLevel, Info);
      LastDiag->Fixes.insert(LastDiag->Fixes.end(), ExtraFixes.begin(),
                             ExtraFixes.end());
    }
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
      if (!AddFix(false /* use the note as the message */))
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
    vlog("Dropped diagnostic outside main file: {0}: {1}", LastDiag->File,
         LastDiag->Message);
  LastDiag.reset();
}

} // namespace clangd
} // namespace clang
