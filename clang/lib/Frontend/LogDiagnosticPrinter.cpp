//===--- LogDiagnosticPrinter.cpp - Log Diagnostic Printer ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/LogDiagnosticPrinter.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/PlistSupport.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;
using namespace markup;

LogDiagnosticPrinter::LogDiagnosticPrinter(
    raw_ostream &os, DiagnosticOptions *diags,
    std::unique_ptr<raw_ostream> StreamOwner)
    : OS(os), StreamOwner(std::move(StreamOwner)), LangOpts(nullptr),
      DiagOpts(diags) {}

static StringRef getLevelName(DiagnosticsEngine::Level Level) {
  switch (Level) {
  case DiagnosticsEngine::Ignored: return "ignored";
  case DiagnosticsEngine::Remark:  return "remark";
  case DiagnosticsEngine::Note:    return "note";
  case DiagnosticsEngine::Warning: return "warning";
  case DiagnosticsEngine::Error:   return "error";
  case DiagnosticsEngine::Fatal:   return "fatal error";
  }
  llvm_unreachable("Invalid DiagnosticsEngine level!");
}

void LogDiagnosticPrinter::EmitDiagEntryLocation(
    llvm::raw_ostream &OS, StringRef Indent,
    const LogDiagnosticPrinter::DiagEntryLocation &Del) {
  OS << Indent << "<dict>\n";
  if (!Del.Filename.empty()) {
    OS << Indent << "  <key>filename</key>\n";
    OS << Indent << "  ";
    EmitString(OS, Del.Filename) << '\n';
  }
  OS << Indent << "  <key>line</key>\n";
  OS << Indent << "  ";
  EmitInteger(OS, Del.Line) << '\n';
  OS << Indent << "  <key>column</key>\n";
  OS << Indent << "  ";
  EmitInteger(OS, Del.Column) << '\n';
  OS << Indent << "  <key>offset</key>\n";
  OS << Indent << "  ";
  EmitInteger(OS, Del.Offset) << '\n';
  OS << Indent << "</dict>\n";
}

void
LogDiagnosticPrinter::EmitDiagEntry(llvm::raw_ostream &OS,
                                    const LogDiagnosticPrinter::DiagEntry &DE) {
  OS << "    <dict>\n";
  OS << "      <key>level</key>\n"
     << "      ";
  EmitString(OS, getLevelName(DE.DiagnosticLevel)) << '\n';
  if (!DE.Filename.empty()) {
    OS << "      <key>filename</key>\n"
       << "      ";
    EmitString(OS, DE.Filename) << '\n';
  }
  if (DE.Line != 0) {
    OS << "      <key>line</key>\n"
       << "      ";
    EmitInteger(OS, DE.Line) << '\n';
  }
  if (DE.Column != 0) {
    OS << "      <key>column</key>\n"
       << "      ";
    EmitInteger(OS, DE.Column) << '\n';
  }
  if (!DE.Message.empty()) {
    OS << "      <key>message</key>\n"
       << "      ";
    EmitString(OS, DE.Message) << '\n';
  }
  OS << "      <key>ID</key>\n"
     << "      ";
  EmitInteger(OS, DE.DiagnosticID) << '\n';
  if (!DE.WarningOption.empty()) {
    OS << "      <key>WarningOption</key>\n"
       << "      ";
    EmitString(OS, DE.WarningOption) << '\n';
  }
  if (!DE.SourceRanges.empty()) {
    OS << "      <key>source-ranges</key>\n"
       << "      <array>\n";
    for (auto R = DE.SourceRanges.begin(), E = DE.SourceRanges.end(); R != E;
         ++R) {
      OS << "        <dict>\n";
      OS << "          <key>start-at</key>\n";
      EmitDiagEntryLocation(OS, "          ", R->Start);
      OS << "          <key>end-before</key>\n";
      EmitDiagEntryLocation(OS, "          ", R->End);
      OS << "        </dict>\n";
    }
    OS << "      </array>\n";
  }
  if (!DE.FixIts.empty()) {
    OS << "      <key>fixits</key>\n"
       << "      <array>\n";
    for (auto F = DE.FixIts.begin(), E = DE.FixIts.end(); F != E; ++F) {
      OS << "        <dict>\n";
      OS << "          <key>start-at</key>\n";
      EmitDiagEntryLocation(OS, "          ", F->RemoveRange.Start);
      OS << "          <key>end-before</key>\n";
      EmitDiagEntryLocation(OS, "          ", F->RemoveRange.End);
      // Always issue a replacement key/value, even if CodeToInsert is empty.
      OS << "          <key>replacement</key>\n"
         << "          ";
      if (F->CodeToInsert.empty()) {
        OS << "<string/>\n";
      } else {
        EmitString(OS, F->CodeToInsert) << '\n';
      }
      OS << "        </dict>\n";
    }
    OS << "      </array>\n";
  }
  OS << "    </dict>\n";
}

void LogDiagnosticPrinter::EndSourceFile() {
  // We emit all the diagnostics in EndSourceFile. However, we don't emit any
  // entry if no diagnostics were present.
  //
  // Note that DiagnosticConsumer has no "end-of-compilation" callback, so we
  // will miss any diagnostics which are emitted after and outside the
  // translation unit processing.
  if (Entries.empty())
    return;

  // Write to a temporary string to ensure atomic write of diagnostic object.
  SmallString<512> Msg;
  llvm::raw_svector_ostream OS(Msg);

  OS << "<dict>\n";
  if (!MainFilename.empty()) {
    OS << "  <key>main-file</key>\n"
       << "  ";
    EmitString(OS, MainFilename) << '\n';
  }
  if (!DwarfDebugFlags.empty()) {
    OS << "  <key>dwarf-debug-flags</key>\n"
       << "  ";
    EmitString(OS, DwarfDebugFlags) << '\n';
  }
  OS << "  <key>diagnostics</key>\n";
  OS << "  <array>\n";
  for (auto &DE : Entries)
    EmitDiagEntry(OS, DE);
  OS << "  </array>\n";
  OS << "</dict>\n";

  this->OS << OS.str();
}

void LogDiagnosticPrinter::HandleDiagnostic(DiagnosticsEngine::Level Level,
                                            const Diagnostic &Info) {
  // Default implementation (Warnings/errors count).
  DiagnosticConsumer::HandleDiagnostic(Level, Info);

  // Initialize the main file name, if we haven't already fetched it.
  if (MainFilename.empty() && Info.hasSourceManager()) {
    const SourceManager &SM = Info.getSourceManager();
    FileID FID = SM.getMainFileID();
    if (FID.isValid()) {
      const FileEntry *FE = SM.getFileEntryForID(FID);
      if (FE && FE->isValid())
        MainFilename = FE->getName();
    }
  }

  // Create the diag entry.
  DiagEntry DE;
  DE.DiagnosticID = Info.getID();
  DE.DiagnosticLevel = Level;

  DE.WarningOption = DiagnosticIDs::getWarningOptionForDiag(DE.DiagnosticID);

  // Format the message.
  SmallString<100> MessageStr;
  Info.FormatDiagnostic(MessageStr);
  DE.Message = MessageStr.str();

  // Set the location information.
  DE.Filename = "";
  DE.Line = DE.Column = 0;
  if (Info.getLocation().isValid() && Info.hasSourceManager()) {
    const SourceManager &SM = Info.getSourceManager();
    PresumedLoc PLoc = SM.getPresumedLoc(Info.getLocation());

    if (PLoc.isInvalid()) {
      // At least print the file name if available:
      FileID FID = SM.getFileID(Info.getLocation());
      if (FID.isValid()) {
        const FileEntry *FE = SM.getFileEntryForID(FID);
        if (FE && FE->isValid())
          DE.Filename = FE->getName();
      }
    } else {
      DE.Filename = PLoc.getFilename();
      DE.Line = PLoc.getLine();
      DE.Column = PLoc.getColumn();
    }
  }

  auto InitDer = [](DiagEntryRange &Der, const CharSourceRange &R,
                    const DiagEntry &DE, const Diagnostic &Info,
                    const LangOptions *LangOpts) -> bool {
    if (!R.isValid() || !Info.hasSourceManager())
      return false;

    SourceManager &SM = Info.getSourceManager();
    FullSourceLoc StartLoc = FullSourceLoc(R.getBegin(), SM);
    FullSourceLoc EndLoc = FullSourceLoc(R.getEnd(), SM);
    if (StartLoc.isInvalid() || EndLoc.isInvalid())
      return false;

    PresumedLoc StartPLoc =
        StartLoc.hasManager() ? StartLoc.getPresumedLoc() : PresumedLoc();
    StringRef StartFilename = StartPLoc.getFilename();
    if (!DE.Filename.empty() && DE.Filename == StartFilename)
      StartFilename = "";

    PresumedLoc EndPLoc =
        EndLoc.hasManager() ? EndLoc.getPresumedLoc() : PresumedLoc();
    StringRef EndFilename = EndPLoc.getFilename();
    if (!DE.Filename.empty() && DE.Filename == EndFilename)
      EndFilename = "";

    unsigned TokSize = 0;
    if (R.isTokenRange())
      TokSize = Lexer::MeasureTokenLength(R.getEnd(), SM, *LangOpts);
    Der = {{StartFilename, StartPLoc.getLine(), StartPLoc.getColumn(),
            StartLoc.getFileOffset()},
           {EndFilename, EndPLoc.getLine(), EndPLoc.getColumn() + TokSize,
            EndLoc.getFileOffset() + TokSize}};
    return true;
  };

  ArrayRef<CharSourceRange> ranges = Info.getRanges();
  for (ArrayRef<CharSourceRange>::iterator R = ranges.begin(),
       E = ranges.end(); R != E; ++R) {
    DiagEntryRange Der;
    bool Success = InitDer(Der, *R, DE, Info, LangOpts);
    if (Success)
      DE.SourceRanges.push_back(Der);
  }

  ArrayRef<FixItHint> fixits = Info.getFixItHints();
  for (ArrayRef<FixItHint>::iterator F = fixits.begin(), E = fixits.end();
       F != E; ++F) {
    // We follow FixItRewriter's example in not (yet) handling
    // fix-its in macros.
    if (F->RemoveRange.getBegin().isMacroID() ||
        F->RemoveRange.getEnd().isMacroID()) {
      // If any bad FixItHint, skip all of them; the rest
      // might not make sense independent of the skipped ones.
      DE.FixIts.clear();
      break;
    }
    if (F->isNull())
      continue;

    DiagEntryFixIt FI;
    bool Success = InitDer(FI.RemoveRange, F->RemoveRange, DE, Info, LangOpts);
    if (Success) {
      FI.CodeToInsert = F->CodeToInsert;
      DE.FixIts.push_back(FI);
    }
  }

  // Record the diagnostic entry.
  Entries.push_back(DE);
}
