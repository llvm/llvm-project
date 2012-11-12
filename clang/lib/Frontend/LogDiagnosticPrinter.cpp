//===--- LogDiagnosticPrinter.cpp - Log Diagnostic Printer ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/LogDiagnosticPrinter.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ErrorHandling.h"
using namespace clang;

LogDiagnosticPrinter::LogDiagnosticPrinter(raw_ostream &os,
                                           DiagnosticOptions *diags,
                                           bool _OwnsOutputStream)
  : OS(os), LangOpts(0), DiagOpts(diags),
    OwnsOutputStream(_OwnsOutputStream) {
}

LogDiagnosticPrinter::~LogDiagnosticPrinter() {
  if (OwnsOutputStream)
    delete &OS;
}

static StringRef getLevelName(DiagnosticsEngine::Level Level) {
  switch (Level) {
  case DiagnosticsEngine::Ignored: return "ignored";
  case DiagnosticsEngine::Note:    return "note";
  case DiagnosticsEngine::Warning: return "warning";
  case DiagnosticsEngine::Error:   return "error";
  case DiagnosticsEngine::Fatal:   return "fatal error";
  }
  llvm_unreachable("Invalid DiagnosticsEngine level!");
}

// Escape XML characters inside the raw string.
static void emitString(llvm::raw_svector_ostream &OS, const StringRef Raw) {
  for (StringRef::iterator I = Raw.begin(), E = Raw.end(); I != E; ++I) {
    char c = *I;
    switch (c) {
    default:   OS << c; break;
    case '&':  OS << "&amp;"; break;
    case '<':  OS << "&lt;"; break;
    case '>':  OS << "&gt;"; break;
    case '\'': OS << "&apos;"; break;
    case '\"': OS << "&quot;"; break;
    }
  }
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
       << "  <string>";
    emitString(OS, MainFilename);
    OS << "</string>\n";
  }
  if (!DwarfDebugFlags.empty()) {
    OS << "  <key>dwarf-debug-flags</key>\n"
       << "  <string>";
    emitString(OS, DwarfDebugFlags);
    OS << "</string>\n";
  }
  OS << "  <key>diagnostics</key>\n";
  OS << "  <array>\n";
  for (unsigned i = 0, e = Entries.size(); i != e; ++i) {
    DiagEntry &DE = Entries[i];

    OS << "    <dict>\n";
    OS << "      <key>level</key>\n"
       << "      <string>";
    emitString(OS, getLevelName(DE.DiagnosticLevel));
    OS << "</string>\n";
    if (!DE.Filename.empty()) {
      OS << "      <key>filename</key>\n"
         << "      <string>";
      emitString(OS, DE.Filename);
      OS << "</string>\n";
    }
    if (DE.Line != 0) {
      OS << "      <key>line</key>\n"
         << "      <integer>" << DE.Line << "</integer>\n";
    }
    if (DE.Column != 0) {
      OS << "      <key>column</key>\n"
         << "      <integer>" << DE.Column << "</integer>\n";
    }
    if (!DE.Message.empty()) {
      OS << "      <key>message</key>\n"
         << "      <string>";
      emitString(OS, DE.Message);
      OS << "</string>\n";
    }
    OS << "    </dict>\n";
  }
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
    if (!FID.isInvalid()) {
      const FileEntry *FE = SM.getFileEntryForID(FID);
      if (FE && FE->getName())
        MainFilename = FE->getName();
    }
  }

  // Create the diag entry.
  DiagEntry DE;
  DE.DiagnosticID = Info.getID();
  DE.DiagnosticLevel = Level;

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
      if (!FID.isInvalid()) {
        const FileEntry *FE = SM.getFileEntryForID(FID);
        if (FE && FE->getName())
          DE.Filename = FE->getName();
      }
    } else {
      DE.Filename = PLoc.getFilename();
      DE.Line = PLoc.getLine();
      DE.Column = PLoc.getColumn();
    }
  }

  // Record the diagnostic entry.
  Entries.push_back(DE);
}

DiagnosticConsumer *
LogDiagnosticPrinter::clone(DiagnosticsEngine &Diags) const {
  return new LogDiagnosticPrinter(OS, &*DiagOpts, /*OwnsOutputStream=*/false);
}

