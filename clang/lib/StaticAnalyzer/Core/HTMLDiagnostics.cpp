//===--- HTMLDiagnostics.cpp - HTML Diagnostics for Paths ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the HTMLDiagnostics object.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Rewrite/Core/HTMLRewrite.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/StaticAnalyzer/Core/BugReporter/PathDiagnostic.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/IssueHash.h"
#include "clang/StaticAnalyzer/Core/PathDiagnosticConsumers.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <sstream>

using namespace clang;
using namespace ento;

//===----------------------------------------------------------------------===//
// Boilerplate.
//===----------------------------------------------------------------------===//

namespace {

class HTMLDiagnostics : public PathDiagnosticConsumer {
  std::string Directory;
  bool createdDir, noDir;
  const Preprocessor &PP;
  AnalyzerOptions &AnalyzerOpts;
public:
  HTMLDiagnostics(AnalyzerOptions &AnalyzerOpts, const std::string& prefix, const Preprocessor &pp);

  ~HTMLDiagnostics() override { FlushDiagnostics(nullptr); }

  void FlushDiagnosticsImpl(std::vector<const PathDiagnostic *> &Diags,
                            FilesMade *filesMade) override;

  StringRef getName() const override {
    return "HTMLDiagnostics";
  }

  unsigned ProcessMacroPiece(raw_ostream &os,
                             const PathDiagnosticMacroPiece& P,
                             unsigned num);

  void HandlePiece(Rewriter& R, FileID BugFileID,
                   const PathDiagnosticPiece& P, unsigned num, unsigned max);

  void HighlightRange(Rewriter& R, FileID BugFileID, SourceRange Range,
                      const char *HighlightStart = "<span class=\"mrange\">",
                      const char *HighlightEnd = "</span>");

  void ReportDiag(const PathDiagnostic& D,
                  FilesMade *filesMade);
};

} // end anonymous namespace

HTMLDiagnostics::HTMLDiagnostics(AnalyzerOptions &AnalyzerOpts,
                                 const std::string& prefix,
                                 const Preprocessor &pp)
    : Directory(prefix), createdDir(false), noDir(false), PP(pp), AnalyzerOpts(AnalyzerOpts) {
}

void ento::createHTMLDiagnosticConsumer(AnalyzerOptions &AnalyzerOpts,
                                        PathDiagnosticConsumers &C,
                                        const std::string& prefix,
                                        const Preprocessor &PP) {
  C.push_back(new HTMLDiagnostics(AnalyzerOpts, prefix, PP));
}

//===----------------------------------------------------------------------===//
// Report processing.
//===----------------------------------------------------------------------===//

void HTMLDiagnostics::FlushDiagnosticsImpl(
  std::vector<const PathDiagnostic *> &Diags,
  FilesMade *filesMade) {
  for (std::vector<const PathDiagnostic *>::iterator it = Diags.begin(),
       et = Diags.end(); it != et; ++it) {
    ReportDiag(**it, filesMade);
  }
}

void HTMLDiagnostics::ReportDiag(const PathDiagnostic& D,
                                 FilesMade *filesMade) {

  // Create the HTML directory if it is missing.
  if (!createdDir) {
    createdDir = true;
    if (std::error_code ec = llvm::sys::fs::create_directories(Directory)) {
      llvm::errs() << "warning: could not create directory '"
                   << Directory << "': " << ec.message() << '\n';

      noDir = true;

      return;
    }
  }

  if (noDir)
    return;

  // First flatten out the entire path to make it easier to use.
  PathPieces path = D.path.flatten(/*ShouldFlattenMacros=*/false);

  // The path as already been prechecked that all parts of the path are
  // from the same file and that it is non-empty.
  const SourceManager &SMgr = path.front()->getLocation().getManager();
  assert(!path.empty());
  FileID FID =
    path.front()->getLocation().asLocation().getExpansionLoc().getFileID();
  assert(FID.isValid());

  // Create a new rewriter to generate HTML.
  Rewriter R(const_cast<SourceManager&>(SMgr), PP.getLangOpts());

  // Get the function/method name
  SmallString<128> declName("unknown");
  int offsetDecl = 0;
  if (const Decl *DeclWithIssue = D.getDeclWithIssue()) {
      if (const NamedDecl *ND = dyn_cast<NamedDecl>(DeclWithIssue)) {
          declName = ND->getDeclName().getAsString();
      }

      if (const Stmt *Body = DeclWithIssue->getBody()) {
          // Retrieve the relative position of the declaration which will be used
          // for the file name
          FullSourceLoc L(
              SMgr.getExpansionLoc(path.back()->getLocation().asLocation()),
              SMgr);
          FullSourceLoc FunL(SMgr.getExpansionLoc(Body->getLocStart()), SMgr);
          offsetDecl = L.getExpansionLineNumber() - FunL.getExpansionLineNumber();
      }
  }

  // Process the path.
  unsigned n = path.size();
  unsigned max = n;

  for (PathPieces::const_reverse_iterator I = path.rbegin(),
       E = path.rend();
        I != E; ++I, --n)
    HandlePiece(R, FID, **I, n, max);

  // Add line numbers, header, footer, etc.

  // unsigned FID = R.getSourceMgr().getMainFileID();
  html::EscapeText(R, FID);
  html::AddLineNumbers(R, FID);

  // If we have a preprocessor, relex the file and syntax highlight.
  // We might not have a preprocessor if we come from a deserialized AST file,
  // for example.

  html::SyntaxHighlight(R, FID, PP);
  html::HighlightMacros(R, FID, PP);

  // Get the full directory name of the analyzed file.

  const FileEntry* Entry = SMgr.getFileEntryForID(FID);

  // This is a cludge; basically we want to append either the full
  // working directory if we have no directory information.  This is
  // a work in progress.

  llvm::SmallString<0> DirName;

  if (llvm::sys::path::is_relative(Entry->getName())) {
    llvm::sys::fs::current_path(DirName);
    DirName += '/';
  }

  int LineNumber = path.back()->getLocation().asLocation().getExpansionLineNumber();
  int ColumnNumber = path.back()->getLocation().asLocation().getExpansionColumnNumber();

  // Add the name of the file as an <h1> tag.

  {
    std::string s;
    llvm::raw_string_ostream os(s);

    os << "<!-- REPORTHEADER -->\n"
      << "<h3>Bug Summary</h3>\n<table class=\"simpletable\">\n"
          "<tr><td class=\"rowname\">File:</td><td>"
      << html::EscapeText(DirName)
      << html::EscapeText(Entry->getName())
      << "</td></tr>\n<tr><td class=\"rowname\">Location:</td><td>"
         "<a href=\"#EndPath\">line "
      << LineNumber
      << ", column "
      << ColumnNumber
      << "</a></td></tr>\n"
         "<tr><td class=\"rowname\">Description:</td><td>"
      << D.getVerboseDescription() << "</td></tr>\n";

    // Output any other meta data.

    for (PathDiagnostic::meta_iterator I=D.meta_begin(), E=D.meta_end();
         I!=E; ++I) {
      os << "<tr><td></td><td>" << html::EscapeText(*I) << "</td></tr>\n";
    }

    os << "</table>\n<!-- REPORTSUMMARYEXTRA -->\n"
          "<h3>Annotated Source Code</h3>\n";

    R.InsertTextBefore(SMgr.getLocForStartOfFile(FID), os.str());
  }

  // Embed meta-data tags.
  {
    std::string s;
    llvm::raw_string_ostream os(s);

    StringRef BugDesc = D.getVerboseDescription();
    if (!BugDesc.empty())
      os << "\n<!-- BUGDESC " << BugDesc << " -->\n";

    StringRef BugType = D.getBugType();
    if (!BugType.empty())
      os << "\n<!-- BUGTYPE " << BugType << " -->\n";

    PathDiagnosticLocation UPDLoc = D.getUniqueingLoc();
    FullSourceLoc L(SMgr.getExpansionLoc(UPDLoc.isValid()
                                             ? UPDLoc.asLocation()
                                             : D.getLocation().asLocation()),
                    SMgr);
    const Decl *DeclWithIssue = D.getDeclWithIssue();

    StringRef BugCategory = D.getCategory();
    if (!BugCategory.empty())
      os << "\n<!-- BUGCATEGORY " << BugCategory << " -->\n";

    os << "\n<!-- BUGFILE " << DirName << Entry->getName() << " -->\n";

    os << "\n<!-- FILENAME " << llvm::sys::path::filename(Entry->getName()) << " -->\n";

    os  << "\n<!-- FUNCTIONNAME " <<  declName << " -->\n";

    os << "\n<!-- ISSUEHASHCONTENTOFLINEINCONTEXT "
       << GetIssueHash(SMgr, L, D.getCheckName(), D.getBugType(), DeclWithIssue,
                       PP.getLangOpts()) << " -->\n";

    os << "\n<!-- BUGLINE "
       << LineNumber
       << " -->\n";

    os << "\n<!-- BUGCOLUMN "
      << ColumnNumber
      << " -->\n";

    os << "\n<!-- BUGPATHLENGTH " << path.size() << " -->\n";

    // Mark the end of the tags.
    os << "\n<!-- BUGMETAEND -->\n";

    // Insert the text.
    R.InsertTextBefore(SMgr.getLocForStartOfFile(FID), os.str());
  }

  // Add CSS, header, and footer.

  html::AddHeaderFooterInternalBuiltinCSS(R, FID, Entry->getName());

  // Get the rewrite buffer.
  const RewriteBuffer *Buf = R.getRewriteBufferFor(FID);

  if (!Buf) {
    llvm::errs() << "warning: no diagnostics generated for main file.\n";
    return;
  }

  // Create a path for the target HTML file.
  int FD;
  SmallString<128> Model, ResultPath;

  if (!AnalyzerOpts.shouldWriteStableReportFilename()) {
      llvm::sys::path::append(Model, Directory, "report-%%%%%%.html");
      if (std::error_code EC =
          llvm::sys::fs::make_absolute(Model)) {
          llvm::errs() << "warning: could not make '" << Model
                       << "' absolute: " << EC.message() << '\n';
        return;
      }
      if (std::error_code EC =
          llvm::sys::fs::createUniqueFile(Model, FD, ResultPath)) {
          llvm::errs() << "warning: could not create file in '" << Directory
                       << "': " << EC.message() << '\n';
          return;
      }

  } else {
      int i = 1;
      std::error_code EC;
      do {
          // Find a filename which is not already used
          std::stringstream filename;
          Model = "";
          filename << "report-"
                   << llvm::sys::path::filename(Entry->getName()).str()
                   << "-" << declName.c_str()
                   << "-" << offsetDecl
                   << "-" << i << ".html";
          llvm::sys::path::append(Model, Directory,
                                  filename.str());
          EC = llvm::sys::fs::openFileForWrite(Model,
                                               FD,
                                               llvm::sys::fs::F_RW |
                                               llvm::sys::fs::F_Excl);
          if (EC && EC != llvm::errc::file_exists) {
              llvm::errs() << "warning: could not create file '" << Model
                           << "': " << EC.message() << '\n';
              return;
          }
          i++;
      } while (EC);
  }

  llvm::raw_fd_ostream os(FD, true);

  if (filesMade)
    filesMade->addDiagnostic(D, getName(),
                             llvm::sys::path::filename(ResultPath));

  // Emit the HTML to disk.
  for (RewriteBuffer::iterator I = Buf->begin(), E = Buf->end(); I!=E; ++I)
      os << *I;
}

void HTMLDiagnostics::HandlePiece(Rewriter& R, FileID BugFileID,
                                  const PathDiagnosticPiece& P,
                                  unsigned num, unsigned max) {

  // For now, just draw a box above the line in question, and emit the
  // warning.
  FullSourceLoc Pos = P.getLocation().asLocation();

  if (!Pos.isValid())
    return;

  SourceManager &SM = R.getSourceMgr();
  assert(&Pos.getManager() == &SM && "SourceManagers are different!");
  std::pair<FileID, unsigned> LPosInfo = SM.getDecomposedExpansionLoc(Pos);

  if (LPosInfo.first != BugFileID)
    return;

  const llvm::MemoryBuffer *Buf = SM.getBuffer(LPosInfo.first);
  const char* FileStart = Buf->getBufferStart();

  // Compute the column number.  Rewind from the current position to the start
  // of the line.
  unsigned ColNo = SM.getColumnNumber(LPosInfo.first, LPosInfo.second);
  const char *TokInstantiationPtr =Pos.getExpansionLoc().getCharacterData();
  const char *LineStart = TokInstantiationPtr-ColNo;

  // Compute LineEnd.
  const char *LineEnd = TokInstantiationPtr;
  const char* FileEnd = Buf->getBufferEnd();
  while (*LineEnd != '\n' && LineEnd != FileEnd)
    ++LineEnd;

  // Compute the margin offset by counting tabs and non-tabs.
  unsigned PosNo = 0;
  for (const char* c = LineStart; c != TokInstantiationPtr; ++c)
    PosNo += *c == '\t' ? 8 : 1;

  // Create the html for the message.

  const char *Kind = nullptr;
  switch (P.getKind()) {
  case PathDiagnosticPiece::Call:
      llvm_unreachable("Calls should already be handled");
  case PathDiagnosticPiece::Event:  Kind = "Event"; break;
  case PathDiagnosticPiece::ControlFlow: Kind = "Control"; break;
    // Setting Kind to "Control" is intentional.
  case PathDiagnosticPiece::Macro: Kind = "Control"; break;
  }

  std::string sbuf;
  llvm::raw_string_ostream os(sbuf);

  os << "\n<tr><td class=\"num\"></td><td class=\"line\"><div id=\"";

  if (num == max)
    os << "EndPath";
  else
    os << "Path" << num;

  os << "\" class=\"msg";
  if (Kind)
    os << " msg" << Kind;
  os << "\" style=\"margin-left:" << PosNo << "ex";

  // Output a maximum size.
  if (!isa<PathDiagnosticMacroPiece>(P)) {
    // Get the string and determining its maximum substring.
    const std::string& Msg = P.getString();
    unsigned max_token = 0;
    unsigned cnt = 0;
    unsigned len = Msg.size();

    for (std::string::const_iterator I=Msg.begin(), E=Msg.end(); I!=E; ++I)
      switch (*I) {
      default:
        ++cnt;
        continue;
      case ' ':
      case '\t':
      case '\n':
        if (cnt > max_token) max_token = cnt;
        cnt = 0;
      }

    if (cnt > max_token)
      max_token = cnt;

    // Determine the approximate size of the message bubble in em.
    unsigned em;
    const unsigned max_line = 120;

    if (max_token >= max_line)
      em = max_token / 2;
    else {
      unsigned characters = max_line;
      unsigned lines = len / max_line;

      if (lines > 0) {
        for (; characters > max_token; --characters)
          if (len / characters > lines) {
            ++characters;
            break;
          }
      }

      em = characters / 2;
    }

    if (em < max_line/2)
      os << "; max-width:" << em << "em";
  }
  else
    os << "; max-width:100em";

  os << "\">";

  if (max > 1) {
    os << "<table class=\"msgT\"><tr><td valign=\"top\">";
    os << "<div class=\"PathIndex";
    if (Kind) os << " PathIndex" << Kind;
    os << "\">" << num << "</div>";

    if (num > 1) {
      os << "</td><td><div class=\"PathNav\"><a href=\"#Path"
         << (num - 1)
         << "\" title=\"Previous event ("
         << (num - 1)
         << ")\">&#x2190;</a></div></td>";
    }

    os << "</td><td>";
  }

  if (const PathDiagnosticMacroPiece *MP =
        dyn_cast<PathDiagnosticMacroPiece>(&P)) {

    os << "Within the expansion of the macro '";

    // Get the name of the macro by relexing it.
    {
      FullSourceLoc L = MP->getLocation().asLocation().getExpansionLoc();
      assert(L.isFileID());
      StringRef BufferInfo = L.getBufferData();
      std::pair<FileID, unsigned> LocInfo = L.getDecomposedLoc();
      const char* MacroName = LocInfo.second + BufferInfo.data();
      Lexer rawLexer(SM.getLocForStartOfFile(LocInfo.first), PP.getLangOpts(),
                     BufferInfo.begin(), MacroName, BufferInfo.end());

      Token TheTok;
      rawLexer.LexFromRawLexer(TheTok);
      for (unsigned i = 0, n = TheTok.getLength(); i < n; ++i)
        os << MacroName[i];
    }

    os << "':\n";

    if (max > 1) {
      os << "</td>";
      if (num < max) {
        os << "<td><div class=\"PathNav\"><a href=\"#";
        if (num == max - 1)
          os << "EndPath";
        else
          os << "Path" << (num + 1);
        os << "\" title=\"Next event ("
        << (num + 1)
        << ")\">&#x2192;</a></div></td>";
      }

      os << "</tr></table>";
    }

    // Within a macro piece.  Write out each event.
    ProcessMacroPiece(os, *MP, 0);
  }
  else {
    os << html::EscapeText(P.getString());

    if (max > 1) {
      os << "</td>";
      if (num < max) {
        os << "<td><div class=\"PathNav\"><a href=\"#";
        if (num == max - 1)
          os << "EndPath";
        else
          os << "Path" << (num + 1);
        os << "\" title=\"Next event ("
           << (num + 1)
           << ")\">&#x2192;</a></div></td>";
      }

      os << "</tr></table>";
    }
  }

  os << "</div></td></tr>";

  // Insert the new html.
  unsigned DisplayPos = LineEnd - FileStart;
  SourceLocation Loc =
    SM.getLocForStartOfFile(LPosInfo.first).getLocWithOffset(DisplayPos);

  R.InsertTextBefore(Loc, os.str());

  // Now highlight the ranges.
  ArrayRef<SourceRange> Ranges = P.getRanges();
  for (ArrayRef<SourceRange>::iterator I = Ranges.begin(),
                                       E = Ranges.end(); I != E; ++I) {
    HighlightRange(R, LPosInfo.first, *I);
  }
}

static void EmitAlphaCounter(raw_ostream &os, unsigned n) {
  unsigned x = n % ('z' - 'a');
  n /= 'z' - 'a';

  if (n > 0)
    EmitAlphaCounter(os, n);

  os << char('a' + x);
}

unsigned HTMLDiagnostics::ProcessMacroPiece(raw_ostream &os,
                                            const PathDiagnosticMacroPiece& P,
                                            unsigned num) {

  for (PathPieces::const_iterator I = P.subPieces.begin(), E=P.subPieces.end();
        I!=E; ++I) {

    if (const PathDiagnosticMacroPiece *MP =
          dyn_cast<PathDiagnosticMacroPiece>(*I)) {
      num = ProcessMacroPiece(os, *MP, num);
      continue;
    }

    if (PathDiagnosticEventPiece *EP = dyn_cast<PathDiagnosticEventPiece>(*I)) {
      os << "<div class=\"msg msgEvent\" style=\"width:94%; "
            "margin-left:5px\">"
            "<table class=\"msgT\"><tr>"
            "<td valign=\"top\"><div class=\"PathIndex PathIndexEvent\">";
      EmitAlphaCounter(os, num++);
      os << "</div></td><td valign=\"top\">"
         << html::EscapeText(EP->getString())
         << "</td></tr></table></div>\n";
    }
  }

  return num;
}

void HTMLDiagnostics::HighlightRange(Rewriter& R, FileID BugFileID,
                                     SourceRange Range,
                                     const char *HighlightStart,
                                     const char *HighlightEnd) {
  SourceManager &SM = R.getSourceMgr();
  const LangOptions &LangOpts = R.getLangOpts();

  SourceLocation InstantiationStart = SM.getExpansionLoc(Range.getBegin());
  unsigned StartLineNo = SM.getExpansionLineNumber(InstantiationStart);

  SourceLocation InstantiationEnd = SM.getExpansionLoc(Range.getEnd());
  unsigned EndLineNo = SM.getExpansionLineNumber(InstantiationEnd);

  if (EndLineNo < StartLineNo)
    return;

  if (SM.getFileID(InstantiationStart) != BugFileID ||
      SM.getFileID(InstantiationEnd) != BugFileID)
    return;

  // Compute the column number of the end.
  unsigned EndColNo = SM.getExpansionColumnNumber(InstantiationEnd);
  unsigned OldEndColNo = EndColNo;

  if (EndColNo) {
    // Add in the length of the token, so that we cover multi-char tokens.
    EndColNo += Lexer::MeasureTokenLength(Range.getEnd(), SM, LangOpts)-1;
  }

  // Highlight the range.  Make the span tag the outermost tag for the
  // selected range.

  SourceLocation E =
    InstantiationEnd.getLocWithOffset(EndColNo - OldEndColNo);

  html::HighlightRange(R, InstantiationStart, E, HighlightStart, HighlightEnd);
}
