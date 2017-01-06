//===--- PlistDiagnostics.cpp - Plist Diagnostics for Paths -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the PlistDiagnostics object.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/FileManager.h"
#include "clang/Basic/PlistSupport.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/Version.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/StaticAnalyzer/Core/BugReporter/PathDiagnostic.h"
#include "clang/StaticAnalyzer/Core/IssueHash.h"
#include "clang/StaticAnalyzer/Core/PathDiagnosticConsumers.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
using namespace clang;
using namespace ento;
using namespace markup;

namespace {
  class PlistDiagnostics : public PathDiagnosticConsumer {
    const std::string OutputFile;
    const LangOptions &LangOpts;
    const bool SupportsCrossFileDiagnostics;
  public:
    PlistDiagnostics(AnalyzerOptions &AnalyzerOpts,
                     const std::string& prefix,
                     const LangOptions &LangOpts,
                     bool supportsMultipleFiles);

    ~PlistDiagnostics() override {}

    void FlushDiagnosticsImpl(std::vector<const PathDiagnostic *> &Diags,
                              FilesMade *filesMade) override;

    StringRef getName() const override {
      return "PlistDiagnostics";
    }

    PathGenerationScheme getGenerationScheme() const override {
      return Extensive;
    }
    bool supportsLogicalOpControlFlow() const override { return true; }
    bool supportsCrossFileDiagnostics() const override {
      return SupportsCrossFileDiagnostics;
    }
  };
} // end anonymous namespace

PlistDiagnostics::PlistDiagnostics(AnalyzerOptions &AnalyzerOpts,
                                   const std::string& output,
                                   const LangOptions &LO,
                                   bool supportsMultipleFiles)
  : OutputFile(output),
    LangOpts(LO),
    SupportsCrossFileDiagnostics(supportsMultipleFiles) {}

void ento::createPlistDiagnosticConsumer(AnalyzerOptions &AnalyzerOpts,
                                         PathDiagnosticConsumers &C,
                                         const std::string& s,
                                         const Preprocessor &PP) {
  C.push_back(new PlistDiagnostics(AnalyzerOpts, s,
                                   PP.getLangOpts(), false));
}

void ento::createPlistMultiFileDiagnosticConsumer(AnalyzerOptions &AnalyzerOpts,
                                                  PathDiagnosticConsumers &C,
                                                  const std::string &s,
                                                  const Preprocessor &PP) {
  C.push_back(new PlistDiagnostics(AnalyzerOpts, s,
                                   PP.getLangOpts(), true));
}

static void ReportControlFlow(raw_ostream &o,
                              const PathDiagnosticControlFlowPiece& P,
                              const FIDMap& FM,
                              const SourceManager &SM,
                              const LangOptions &LangOpts,
                              unsigned indent) {

  Indent(o, indent) << "<dict>\n";
  ++indent;

  Indent(o, indent) << "<key>kind</key><string>control</string>\n";

  // Emit edges.
  Indent(o, indent) << "<key>edges</key>\n";
  ++indent;
  Indent(o, indent) << "<array>\n";
  ++indent;
  for (PathDiagnosticControlFlowPiece::const_iterator I=P.begin(), E=P.end();
       I!=E; ++I) {
    Indent(o, indent) << "<dict>\n";
    ++indent;

    // Make the ranges of the start and end point self-consistent with adjacent edges
    // by forcing to use only the beginning of the range.  This simplifies the layout
    // logic for clients.
    Indent(o, indent) << "<key>start</key>\n";
    SourceRange StartEdge(
        SM.getExpansionLoc(I->getStart().asRange().getBegin()));
    EmitRange(o, SM, Lexer::getAsCharRange(StartEdge, SM, LangOpts), FM,
              indent + 1);

    Indent(o, indent) << "<key>end</key>\n";
    SourceRange EndEdge(SM.getExpansionLoc(I->getEnd().asRange().getBegin()));
    EmitRange(o, SM, Lexer::getAsCharRange(EndEdge, SM, LangOpts), FM,
              indent + 1);

    --indent;
    Indent(o, indent) << "</dict>\n";
  }
  --indent;
  Indent(o, indent) << "</array>\n";
  --indent;

  // Output any helper text.
  const auto &s = P.getString();
  if (!s.empty()) {
    Indent(o, indent) << "<key>alternate</key>";
    EmitString(o, s) << '\n';
  }

  --indent;
  Indent(o, indent) << "</dict>\n";
}

static void ReportEvent(raw_ostream &o, const PathDiagnosticPiece& P,
                        const FIDMap& FM,
                        const SourceManager &SM,
                        const LangOptions &LangOpts,
                        unsigned indent,
                        unsigned depth,
                        bool isKeyEvent = false) {

  Indent(o, indent) << "<dict>\n";
  ++indent;

  Indent(o, indent) << "<key>kind</key><string>event</string>\n";

  if (isKeyEvent) {
    Indent(o, indent) << "<key>key_event</key><true/>\n";
  }

  // Output the location.
  FullSourceLoc L = P.getLocation().asLocation();

  Indent(o, indent) << "<key>location</key>\n";
  EmitLocation(o, SM, L, FM, indent);

  // Output the ranges (if any).
  ArrayRef<SourceRange> Ranges = P.getRanges();

  if (!Ranges.empty()) {
    Indent(o, indent) << "<key>ranges</key>\n";
    Indent(o, indent) << "<array>\n";
    ++indent;
    for (auto &R : Ranges)
      EmitRange(o, SM,
                Lexer::getAsCharRange(SM.getExpansionRange(R), SM, LangOpts),
                FM, indent + 1);
    --indent;
    Indent(o, indent) << "</array>\n";
  }

  // Output the call depth.
  Indent(o, indent) << "<key>depth</key>";
  EmitInteger(o, depth) << '\n';

  // Output the text.
  assert(!P.getString().empty());
  Indent(o, indent) << "<key>extended_message</key>\n";
  Indent(o, indent);
  EmitString(o, P.getString()) << '\n';

  // Output the short text.
  // FIXME: Really use a short string.
  Indent(o, indent) << "<key>message</key>\n";
  Indent(o, indent);
  EmitString(o, P.getString()) << '\n';

  // Finish up.
  --indent;
  Indent(o, indent); o << "</dict>\n";
}

static void ReportPiece(raw_ostream &o,
                        const PathDiagnosticPiece &P,
                        const FIDMap& FM, const SourceManager &SM,
                        const LangOptions &LangOpts,
                        unsigned indent,
                        unsigned depth,
                        bool includeControlFlow,
                        bool isKeyEvent = false);

static void ReportCall(raw_ostream &o,
                       const PathDiagnosticCallPiece &P,
                       const FIDMap& FM, const SourceManager &SM,
                       const LangOptions &LangOpts,
                       unsigned indent,
                       unsigned depth) {

  if (auto callEnter = P.getCallEnterEvent())
    ReportPiece(o, *callEnter, FM, SM, LangOpts, indent, depth, true,
                P.isLastInMainSourceFile());


  ++depth;

  if (auto callEnterWithinCaller = P.getCallEnterWithinCallerEvent())
    ReportPiece(o, *callEnterWithinCaller, FM, SM, LangOpts,
                indent, depth, true);

  for (PathPieces::const_iterator I = P.path.begin(), E = P.path.end();I!=E;++I)
    ReportPiece(o, **I, FM, SM, LangOpts, indent, depth, true);

  --depth;

  if (auto callExit = P.getCallExitEvent())
    ReportPiece(o, *callExit, FM, SM, LangOpts, indent, depth, true);
}

static void ReportMacro(raw_ostream &o,
                        const PathDiagnosticMacroPiece& P,
                        const FIDMap& FM, const SourceManager &SM,
                        const LangOptions &LangOpts,
                        unsigned indent,
                        unsigned depth) {

  for (PathPieces::const_iterator I = P.subPieces.begin(), E=P.subPieces.end();
       I!=E; ++I) {
    ReportPiece(o, **I, FM, SM, LangOpts, indent, depth, false);
  }
}

static void ReportDiag(raw_ostream &o, const PathDiagnosticPiece& P,
                       const FIDMap& FM, const SourceManager &SM,
                       const LangOptions &LangOpts) {
  ReportPiece(o, P, FM, SM, LangOpts, 4, 0, true);
}

static void ReportPiece(raw_ostream &o,
                        const PathDiagnosticPiece &P,
                        const FIDMap& FM, const SourceManager &SM,
                        const LangOptions &LangOpts,
                        unsigned indent,
                        unsigned depth,
                        bool includeControlFlow,
                        bool isKeyEvent) {
  switch (P.getKind()) {
    case PathDiagnosticPiece::ControlFlow:
      if (includeControlFlow)
        ReportControlFlow(o, cast<PathDiagnosticControlFlowPiece>(P), FM, SM,
                          LangOpts, indent);
      break;
    case PathDiagnosticPiece::Call:
      ReportCall(o, cast<PathDiagnosticCallPiece>(P), FM, SM, LangOpts,
                 indent, depth);
      break;
    case PathDiagnosticPiece::Event:
      ReportEvent(o, cast<PathDiagnosticSpotPiece>(P), FM, SM, LangOpts,
                  indent, depth, isKeyEvent);
      break;
    case PathDiagnosticPiece::Macro:
      ReportMacro(o, cast<PathDiagnosticMacroPiece>(P), FM, SM, LangOpts,
                  indent, depth);
      break;
    case PathDiagnosticPiece::Note:
      // FIXME: Extend the plist format to support those.
      break;
  }
}

void PlistDiagnostics::FlushDiagnosticsImpl(
                                    std::vector<const PathDiagnostic *> &Diags,
                                    FilesMade *filesMade) {
  // Build up a set of FIDs that we use by scanning the locations and
  // ranges of the diagnostics.
  FIDMap FM;
  SmallVector<FileID, 10> Fids;
  const SourceManager* SM = nullptr;

  if (!Diags.empty())
    SM = &Diags.front()->path.front()->getLocation().getManager();

  auto AddPieceFID = [&FM, &Fids, SM](const PathDiagnosticPiece &Piece) {
    AddFID(FM, Fids, *SM, Piece.getLocation().asLocation());
    ArrayRef<SourceRange> Ranges = Piece.getRanges();
    for (const SourceRange &Range : Ranges) {
      AddFID(FM, Fids, *SM, Range.getBegin());
      AddFID(FM, Fids, *SM, Range.getEnd());
    }
  };

  for (const PathDiagnostic *D : Diags) {

    SmallVector<const PathPieces *, 5> WorkList;
    WorkList.push_back(&D->path);

    while (!WorkList.empty()) {
      const PathPieces &Path = *WorkList.pop_back_val();

      for (const auto &Iter : Path) {
        const PathDiagnosticPiece &Piece = *Iter;
        AddPieceFID(Piece);

        if (const PathDiagnosticCallPiece *Call =
                dyn_cast<PathDiagnosticCallPiece>(&Piece)) {
          if (auto CallEnterWithin = Call->getCallEnterWithinCallerEvent())
            AddPieceFID(*CallEnterWithin);

          if (auto CallEnterEvent = Call->getCallEnterEvent())
            AddPieceFID(*CallEnterEvent);

          WorkList.push_back(&Call->path);
        } else if (const PathDiagnosticMacroPiece *Macro =
                       dyn_cast<PathDiagnosticMacroPiece>(&Piece)) {
          WorkList.push_back(&Macro->subPieces);
        }
      }
    }
  }

  // Open the file.
  std::error_code EC;
  llvm::raw_fd_ostream o(OutputFile, EC, llvm::sys::fs::F_Text);
  if (EC) {
    llvm::errs() << "warning: could not create file: " << EC.message() << '\n';
    return;
  }

  EmitPlistHeader(o);

  // Write the root object: a <dict> containing...
  //  - "clang_version", the string representation of clang version
  //  - "files", an <array> mapping from FIDs to file names
  //  - "diagnostics", an <array> containing the path diagnostics
  o << "<dict>\n" <<
       " <key>clang_version</key>\n";
  EmitString(o, getClangFullVersion()) << '\n';
  o << " <key>files</key>\n"
       " <array>\n";

  for (FileID FID : Fids)
    EmitString(o << "  ", SM->getFileEntryForID(FID)->getName()) << '\n';

  o << " </array>\n"
       " <key>diagnostics</key>\n"
       " <array>\n";

  for (std::vector<const PathDiagnostic*>::iterator DI=Diags.begin(),
       DE = Diags.end(); DI!=DE; ++DI) {

    o << "  <dict>\n"
         "   <key>path</key>\n";

    const PathDiagnostic *D = *DI;

    o << "   <array>\n";

    for (PathPieces::const_iterator I = D->path.begin(), E = D->path.end();
         I != E; ++I)
      ReportDiag(o, **I, FM, *SM, LangOpts);

    o << "   </array>\n";

    // Output the bug type and bug category.
    o << "   <key>description</key>";
    EmitString(o, D->getShortDescription()) << '\n';
    o << "   <key>category</key>";
    EmitString(o, D->getCategory()) << '\n';
    o << "   <key>type</key>";
    EmitString(o, D->getBugType()) << '\n';
    o << "   <key>check_name</key>";
    EmitString(o, D->getCheckName()) << '\n';

    o << "   <!-- This hash is experimental and going to change! -->\n";
    o << "   <key>issue_hash_content_of_line_in_context</key>";
    PathDiagnosticLocation UPDLoc = D->getUniqueingLoc();
    FullSourceLoc L(SM->getExpansionLoc(UPDLoc.isValid()
                                            ? UPDLoc.asLocation()
                                            : D->getLocation().asLocation()),
                    *SM);
    const Decl *DeclWithIssue = D->getDeclWithIssue();
    EmitString(o, GetIssueHash(*SM, L, D->getCheckName(), D->getBugType(),
                               DeclWithIssue, LangOpts))
        << '\n';

    // Output information about the semantic context where
    // the issue occurred.
    if (const Decl *DeclWithIssue = D->getDeclWithIssue()) {
      // FIXME: handle blocks, which have no name.
      if (const NamedDecl *ND = dyn_cast<NamedDecl>(DeclWithIssue)) {
        StringRef declKind;
        switch (ND->getKind()) {
          case Decl::CXXRecord:
            declKind = "C++ class";
            break;
          case Decl::CXXMethod:
            declKind = "C++ method";
            break;
          case Decl::ObjCMethod:
            declKind = "Objective-C method";
            break;
          case Decl::Function:
            declKind = "function";
            break;
          default:
            break;
        }
        if (!declKind.empty()) {
          const std::string &declName = ND->getDeclName().getAsString();
          o << "  <key>issue_context_kind</key>";
          EmitString(o, declKind) << '\n';
          o << "  <key>issue_context</key>";
          EmitString(o, declName) << '\n';
        }

        // Output the bug hash for issue unique-ing. Currently, it's just an
        // offset from the beginning of the function.
        if (const Stmt *Body = DeclWithIssue->getBody()) {

          // If the bug uniqueing location exists, use it for the hash.
          // For example, this ensures that two leaks reported on the same line
          // will have different issue_hashes and that the hash will identify
          // the leak location even after code is added between the allocation
          // site and the end of scope (leak report location).
          if (UPDLoc.isValid()) {
            FullSourceLoc UFunL(SM->getExpansionLoc(
              D->getUniqueingDecl()->getBody()->getLocStart()), *SM);
            o << "  <key>issue_hash_function_offset</key><string>"
              << L.getExpansionLineNumber() - UFunL.getExpansionLineNumber()
              << "</string>\n";

          // Otherwise, use the location on which the bug is reported.
          } else {
            FullSourceLoc FunL(SM->getExpansionLoc(Body->getLocStart()), *SM);
            o << "  <key>issue_hash_function_offset</key><string>"
              << L.getExpansionLineNumber() - FunL.getExpansionLineNumber()
              << "</string>\n";
          }

        }
      }
    }

    // Output the location of the bug.
    o << "  <key>location</key>\n";
    EmitLocation(o, *SM, D->getLocation().asLocation(), FM, 2);

    // Output the diagnostic to the sub-diagnostic client, if any.
    if (!filesMade->empty()) {
      StringRef lastName;
      PDFileEntry::ConsumerFiles *files = filesMade->getFiles(*D);
      if (files) {
        for (PDFileEntry::ConsumerFiles::const_iterator CI = files->begin(),
                CE = files->end(); CI != CE; ++CI) {
          StringRef newName = CI->first;
          if (newName != lastName) {
            if (!lastName.empty()) {
              o << "  </array>\n";
            }
            lastName = newName;
            o <<  "  <key>" << lastName << "_files</key>\n";
            o << "  <array>\n";
          }
          o << "   <string>" << CI->second << "</string>\n";
        }
        o << "  </array>\n";
      }
    }

    // Close up the entry.
    o << "  </dict>\n";
  }

  o << " </array>\n";

  // Finish.
  o << "</dict>\n</plist>";
}
