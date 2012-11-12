//===--- PlistReporter.cpp - ARC Migrate Tool Plist Reporter ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Internals.h"
#include "clang/Lex/Lexer.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/FileManager.h"
using namespace clang;
using namespace arcmt;

// FIXME: This duplicates significant functionality from PlistDiagnostics.cpp,
// it would be jolly good if there was a reusable PlistWriter or something.

typedef llvm::DenseMap<FileID, unsigned> FIDMap;

static void AddFID(FIDMap &FIDs, SmallVectorImpl<FileID> &V,
                   const SourceManager &SM, SourceLocation L) {

  FileID FID = SM.getFileID(SM.getExpansionLoc(L));
  FIDMap::iterator I = FIDs.find(FID);
  if (I != FIDs.end()) return;
  FIDs[FID] = V.size();
  V.push_back(FID);
}

static unsigned GetFID(const FIDMap& FIDs, const SourceManager &SM,
                       SourceLocation L) {
  FileID FID = SM.getFileID(SM.getExpansionLoc(L));
  FIDMap::const_iterator I = FIDs.find(FID);
  assert(I != FIDs.end());
  return I->second;
}

static raw_ostream& Indent(raw_ostream& o, const unsigned indent) {
  for (unsigned i = 0; i < indent; ++i) o << ' ';
  return o;
}

static void EmitLocation(raw_ostream& o, const SourceManager &SM,
                         const LangOptions &LangOpts,
                         SourceLocation L, const FIDMap &FM,
                         unsigned indent, bool extend = false) {

  FullSourceLoc Loc(SM.getExpansionLoc(L), const_cast<SourceManager&>(SM));

  // Add in the length of the token, so that we cover multi-char tokens.
  unsigned offset =
    extend ? Lexer::MeasureTokenLength(Loc, SM, LangOpts) - 1 : 0;

  Indent(o, indent) << "<dict>\n";
  Indent(o, indent) << " <key>line</key><integer>"
                    << Loc.getExpansionLineNumber() << "</integer>\n";
  Indent(o, indent) << " <key>col</key><integer>"
                    << Loc.getExpansionColumnNumber() + offset << "</integer>\n";
  Indent(o, indent) << " <key>file</key><integer>"
                    << GetFID(FM, SM, Loc) << "</integer>\n";
  Indent(o, indent) << "</dict>\n";
}

static void EmitRange(raw_ostream& o, const SourceManager &SM,
                      const LangOptions &LangOpts,
                      CharSourceRange R, const FIDMap &FM,
                      unsigned indent) {
  Indent(o, indent) << "<array>\n";
  EmitLocation(o, SM, LangOpts, R.getBegin(), FM, indent+1);
  EmitLocation(o, SM, LangOpts, R.getEnd(), FM, indent+1, R.isTokenRange());
  Indent(o, indent) << "</array>\n";
}

static raw_ostream& EmitString(raw_ostream& o,
                                     StringRef s) {
  o << "<string>";
  for (StringRef::const_iterator I=s.begin(), E=s.end(); I!=E; ++I) {
    char c = *I;
    switch (c) {
    default:   o << c; break;
    case '&':  o << "&amp;"; break;
    case '<':  o << "&lt;"; break;
    case '>':  o << "&gt;"; break;
    case '\'': o << "&apos;"; break;
    case '\"': o << "&quot;"; break;
    }
  }
  o << "</string>";
  return o;
}

void arcmt::writeARCDiagsToPlist(const std::string &outPath,
                                 ArrayRef<StoredDiagnostic> diags,
                                 SourceManager &SM,
                                 const LangOptions &LangOpts) {
  DiagnosticIDs DiagIDs;

  // Build up a set of FIDs that we use by scanning the locations and
  // ranges of the diagnostics.
  FIDMap FM;
  SmallVector<FileID, 10> Fids;

  for (ArrayRef<StoredDiagnostic>::iterator
         I = diags.begin(), E = diags.end(); I != E; ++I) {
    const StoredDiagnostic &D = *I;

    AddFID(FM, Fids, SM, D.getLocation());

    for (StoredDiagnostic::range_iterator
           RI = D.range_begin(), RE = D.range_end(); RI != RE; ++RI) {
      AddFID(FM, Fids, SM, RI->getBegin());
      AddFID(FM, Fids, SM, RI->getEnd());
    }
  }

  std::string errMsg;
  llvm::raw_fd_ostream o(outPath.c_str(), errMsg);
  if (!errMsg.empty()) {
    llvm::errs() << "error: could not create file: " << outPath << '\n';
    return;
  }

  // Write the plist header.
  o << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
  "<!DOCTYPE plist PUBLIC \"-//Apple Computer//DTD PLIST 1.0//EN\" "
  "\"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n"
  "<plist version=\"1.0\">\n";

  // Write the root object: a <dict> containing...
  //  - "files", an <array> mapping from FIDs to file names
  //  - "diagnostics", an <array> containing the diagnostics
  o << "<dict>\n"
       " <key>files</key>\n"
       " <array>\n";

  for (SmallVectorImpl<FileID>::iterator I=Fids.begin(), E=Fids.end();
       I!=E; ++I) {
    o << "  ";
    EmitString(o, SM.getFileEntryForID(*I)->getName()) << '\n';
  }

  o << " </array>\n"
       " <key>diagnostics</key>\n"
       " <array>\n";

  for (ArrayRef<StoredDiagnostic>::iterator
         DI = diags.begin(), DE = diags.end(); DI != DE; ++DI) {
    
    const StoredDiagnostic &D = *DI;

    if (D.getLevel() == DiagnosticsEngine::Ignored)
      continue;

    o << "  <dict>\n";

    // Output the diagnostic.
    o << "   <key>description</key>";
    EmitString(o, D.getMessage()) << '\n';
    o << "   <key>category</key>";
    EmitString(o, DiagIDs.getCategoryNameFromID(
                          DiagIDs.getCategoryNumberForDiag(D.getID()))) << '\n';
    o << "   <key>type</key>";
    if (D.getLevel() >= DiagnosticsEngine::Error)
      EmitString(o, "error") << '\n';
    else if (D.getLevel() == DiagnosticsEngine::Warning)
      EmitString(o, "warning") << '\n';
    else
      EmitString(o, "note") << '\n';

    // Output the location of the bug.
    o << "  <key>location</key>\n";
    EmitLocation(o, SM, LangOpts, D.getLocation(), FM, 2);

    // Output the ranges (if any).
    StoredDiagnostic::range_iterator RI = D.range_begin(), RE = D.range_end();

    if (RI != RE) {
      o << "   <key>ranges</key>\n";
      o << "   <array>\n";
      for (; RI != RE; ++RI)
        EmitRange(o, SM, LangOpts, *RI, FM, 4);
      o << "   </array>\n";
    }

    // Close up the entry.
    o << "  </dict>\n";
  }

  o << " </array>\n";

  // Finish.
  o << "</dict>\n</plist>";
}
