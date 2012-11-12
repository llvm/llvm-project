//===--- InclusionRewriter.cpp - Rewrite includes into their expansions ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This code rewrites include invocations into their expansions.  This gives you
// a file with all included files merged into it.
//
//===----------------------------------------------------------------------===//

#include "clang/Rewrite/Frontend/Rewriters.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/PreprocessorOutputOptions.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace llvm;

namespace {

class InclusionRewriter : public PPCallbacks {
  /// Information about which #includes were actually performed,
  /// created by preprocessor callbacks.
  struct FileChange {
    SourceLocation From;
    FileID Id;
    SrcMgr::CharacteristicKind FileType;
    FileChange(SourceLocation From) : From(From) {
    }
  };
  Preprocessor &PP; ///< Used to find inclusion directives.
  SourceManager &SM; ///< Used to read and manage source files.
  raw_ostream &OS; ///< The destination stream for rewritten contents.
  bool ShowLineMarkers; ///< Show #line markers.
  bool UseLineDirective; ///< Use of line directives or line markers.
  typedef std::map<unsigned, FileChange> FileChangeMap;
  FileChangeMap FileChanges; /// Tracks which files were included where.
  /// Used transitively for building up the FileChanges mapping over the
  /// various \c PPCallbacks callbacks.
  FileChangeMap::iterator LastInsertedFileChange;
public:
  InclusionRewriter(Preprocessor &PP, raw_ostream &OS, bool ShowLineMarkers);
  bool Process(FileID FileId, SrcMgr::CharacteristicKind FileType);
private:
  virtual void FileChanged(SourceLocation Loc, FileChangeReason Reason,
                           SrcMgr::CharacteristicKind FileType,
                           FileID PrevFID);
  virtual void FileSkipped(const FileEntry &ParentFile,
                           const Token &FilenameTok,
                           SrcMgr::CharacteristicKind FileType);
  virtual void InclusionDirective(SourceLocation HashLoc,
                                  const Token &IncludeTok,
                                  StringRef FileName,
                                  bool IsAngled,
                                  CharSourceRange FilenameRange,
                                  const FileEntry *File,
                                  StringRef SearchPath,
                                  StringRef RelativePath,
                                  const Module *Imported);
  void WriteLineInfo(const char *Filename, int Line,
                     SrcMgr::CharacteristicKind FileType,
                     StringRef EOL, StringRef Extra = StringRef());
  void OutputContentUpTo(const MemoryBuffer &FromFile,
                         unsigned &WriteFrom, unsigned WriteTo,
                         StringRef EOL, int &lines,
                         bool EnsureNewline = false);
  void CommentOutDirective(Lexer &DirectivesLex, const Token &StartToken,
                           const MemoryBuffer &FromFile, StringRef EOL,
                           unsigned &NextToWrite, int &Lines);
  const FileChange *FindFileChangeLocation(SourceLocation Loc) const;
  StringRef NextIdentifierName(Lexer &RawLex, Token &RawToken);
};

}  // end anonymous namespace

/// Initializes an InclusionRewriter with a \p PP source and \p OS destination.
InclusionRewriter::InclusionRewriter(Preprocessor &PP, raw_ostream &OS,
                                     bool ShowLineMarkers)
    : PP(PP), SM(PP.getSourceManager()), OS(OS),
    ShowLineMarkers(ShowLineMarkers),
    LastInsertedFileChange(FileChanges.end()) {
  // If we're in microsoft mode, use normal #line instead of line markers.
  UseLineDirective = PP.getLangOpts().MicrosoftExt;
}

/// Write appropriate line information as either #line directives or GNU line
/// markers depending on what mode we're in, including the \p Filename and
/// \p Line we are located at, using the specified \p EOL line separator, and
/// any \p Extra context specifiers in GNU line directives.
void InclusionRewriter::WriteLineInfo(const char *Filename, int Line,
                                      SrcMgr::CharacteristicKind FileType,
                                      StringRef EOL, StringRef Extra) {
  if (!ShowLineMarkers)
    return;
  if (UseLineDirective) {
    OS << "#line" << ' ' << Line << ' ' << '"' << Filename << '"';
  } else {
    // Use GNU linemarkers as described here:
    // http://gcc.gnu.org/onlinedocs/cpp/Preprocessor-Output.html
    OS << '#' << ' ' << Line << ' ' << '"' << Filename << '"';
    if (!Extra.empty())
      OS << Extra;
    if (FileType == SrcMgr::C_System)
      // "`3' This indicates that the following text comes from a system header
      // file, so certain warnings should be suppressed."
      OS << " 3";
    else if (FileType == SrcMgr::C_ExternCSystem)
      // as above for `3', plus "`4' This indicates that the following text
      // should be treated as being wrapped in an implicit extern "C" block."
      OS << " 3 4";
  }
  OS << EOL;
}

/// FileChanged - Whenever the preprocessor enters or exits a #include file
/// it invokes this handler.
void InclusionRewriter::FileChanged(SourceLocation Loc,
                                    FileChangeReason Reason,
                                    SrcMgr::CharacteristicKind NewFileType,
                                    FileID) {
  if (Reason != EnterFile)
    return;
  if (LastInsertedFileChange == FileChanges.end())
    // we didn't reach this file (eg: the main file) via an inclusion directive
    return;
  LastInsertedFileChange->second.Id = FullSourceLoc(Loc, SM).getFileID();
  LastInsertedFileChange->second.FileType = NewFileType;
  LastInsertedFileChange = FileChanges.end();
}

/// Called whenever an inclusion is skipped due to canonical header protection
/// macros.
void InclusionRewriter::FileSkipped(const FileEntry &/*ParentFile*/,
                                    const Token &/*FilenameTok*/,
                                    SrcMgr::CharacteristicKind /*FileType*/) {
  assert(LastInsertedFileChange != FileChanges.end() && "A file, that wasn't "
    "found via an inclusion directive, was skipped");
  FileChanges.erase(LastInsertedFileChange);
  LastInsertedFileChange = FileChanges.end();
}

/// This should be called whenever the preprocessor encounters include
/// directives. It does not say whether the file has been included, but it
/// provides more information about the directive (hash location instead
/// of location inside the included file). It is assumed that the matching
/// FileChanged() or FileSkipped() is called after this.
void InclusionRewriter::InclusionDirective(SourceLocation HashLoc,
                                           const Token &/*IncludeTok*/,
                                           StringRef /*FileName*/,
                                           bool /*IsAngled*/,
                                           CharSourceRange /*FilenameRange*/,
                                           const FileEntry * /*File*/,
                                           StringRef /*SearchPath*/,
                                           StringRef /*RelativePath*/,
                                           const Module * /*Imported*/) {
  assert(LastInsertedFileChange == FileChanges.end() && "Another inclusion "
    "directive was found before the previous one was processed");
  std::pair<FileChangeMap::iterator, bool> p = FileChanges.insert(
    std::make_pair(HashLoc.getRawEncoding(), FileChange(HashLoc)));
  assert(p.second && "Unexpected revisitation of the same include directive");
  LastInsertedFileChange = p.first;
}

/// Simple lookup for a SourceLocation (specifically one denoting the hash in
/// an inclusion directive) in the map of inclusion information, FileChanges.
const InclusionRewriter::FileChange *
InclusionRewriter::FindFileChangeLocation(SourceLocation Loc) const {
  FileChangeMap::const_iterator I = FileChanges.find(Loc.getRawEncoding());
  if (I != FileChanges.end())
    return &I->second;
  return NULL;
}

/// Detect the likely line ending style of \p FromFile by examining the first
/// newline found within it.
static StringRef DetectEOL(const MemoryBuffer &FromFile) {
  // detect what line endings the file uses, so that added content does not mix
  // the style
  const char *Pos = strchr(FromFile.getBufferStart(), '\n');
  if (Pos == NULL)
    return "\n";
  if (Pos + 1 < FromFile.getBufferEnd() && Pos[1] == '\r')
    return "\n\r";
  if (Pos - 1 >= FromFile.getBufferStart() && Pos[-1] == '\r')
    return "\r\n";
  return "\n";
}

/// Writes out bytes from \p FromFile, starting at \p NextToWrite and ending at
/// \p WriteTo - 1.
void InclusionRewriter::OutputContentUpTo(const MemoryBuffer &FromFile,
                                          unsigned &WriteFrom, unsigned WriteTo,
                                          StringRef EOL, int &Line,
                                          bool EnsureNewline) {
  if (WriteTo <= WriteFrom)
    return;
  OS.write(FromFile.getBufferStart() + WriteFrom, WriteTo - WriteFrom);
  // count lines manually, it's faster than getPresumedLoc()
  Line += std::count(FromFile.getBufferStart() + WriteFrom,
                     FromFile.getBufferStart() + WriteTo, '\n');
  if (EnsureNewline) {
    char LastChar = FromFile.getBufferStart()[WriteTo - 1];
    if (LastChar != '\n' && LastChar != '\r')
      OS << EOL;
  }
  WriteFrom = WriteTo;
}

/// Print characters from \p FromFile starting at \p NextToWrite up until the
/// inclusion directive at \p StartToken, then print out the inclusion
/// inclusion directive disabled by a #if directive, updating \p NextToWrite
/// and \p Line to track the number of source lines visited and the progress
/// through the \p FromFile buffer.
void InclusionRewriter::CommentOutDirective(Lexer &DirectiveLex,
                                            const Token &StartToken,
                                            const MemoryBuffer &FromFile,
                                            StringRef EOL,
                                            unsigned &NextToWrite, int &Line) {
  OutputContentUpTo(FromFile, NextToWrite,
    SM.getFileOffset(StartToken.getLocation()), EOL, Line);
  Token DirectiveToken;
  do {
    DirectiveLex.LexFromRawLexer(DirectiveToken);
  } while (!DirectiveToken.is(tok::eod) && DirectiveToken.isNot(tok::eof));
  OS << "#if 0 /* expanded by -frewrite-includes */" << EOL;
  OutputContentUpTo(FromFile, NextToWrite,
    SM.getFileOffset(DirectiveToken.getLocation()) + DirectiveToken.getLength(),
    EOL, Line);
  OS << "#endif /* expanded by -frewrite-includes */" << EOL;
}

/// Find the next identifier in the pragma directive specified by \p RawToken.
StringRef InclusionRewriter::NextIdentifierName(Lexer &RawLex,
                                                Token &RawToken) {
  RawLex.LexFromRawLexer(RawToken);
  if (RawToken.is(tok::raw_identifier))
    PP.LookUpIdentifierInfo(RawToken);
  if (RawToken.is(tok::identifier))
    return RawToken.getIdentifierInfo()->getName();
  return StringRef();
}

/// Use a raw lexer to analyze \p FileId, inccrementally copying parts of it
/// and including content of included files recursively.
bool InclusionRewriter::Process(FileID FileId,
                                SrcMgr::CharacteristicKind FileType)
{
  bool Invalid;
  const MemoryBuffer &FromFile = *SM.getBuffer(FileId, &Invalid);
  if (Invalid) // invalid inclusion
    return true;
  const char *FileName = FromFile.getBufferIdentifier();
  Lexer RawLex(FileId, &FromFile, PP.getSourceManager(), PP.getLangOpts());
  RawLex.SetCommentRetentionState(false);

  StringRef EOL = DetectEOL(FromFile);

  // Per the GNU docs: "1" indicates the start of a new file.
  WriteLineInfo(FileName, 1, FileType, EOL, " 1");

  if (SM.getFileIDSize(FileId) == 0)
    return true;

  // The next byte to be copied from the source file
  unsigned NextToWrite = 0;
  int Line = 1; // The current input file line number.

  Token RawToken;
  RawLex.LexFromRawLexer(RawToken);

  // TODO: Consider adding a switch that strips possibly unimportant content,
  // such as comments, to reduce the size of repro files.
  while (RawToken.isNot(tok::eof)) {
    if (RawToken.is(tok::hash) && RawToken.isAtStartOfLine()) {
      RawLex.setParsingPreprocessorDirective(true);
      Token HashToken = RawToken;
      RawLex.LexFromRawLexer(RawToken);
      if (RawToken.is(tok::raw_identifier))
        PP.LookUpIdentifierInfo(RawToken);
      if (RawToken.is(tok::identifier)) {
        switch (RawToken.getIdentifierInfo()->getPPKeywordID()) {
          case tok::pp_include:
          case tok::pp_include_next:
          case tok::pp_import: {
            CommentOutDirective(RawLex, HashToken, FromFile, EOL, NextToWrite,
              Line);
            if (const FileChange *Change = FindFileChangeLocation(
                HashToken.getLocation())) {
              // now include and recursively process the file
              if (Process(Change->Id, Change->FileType))
                // and set lineinfo back to this file, if the nested one was
                // actually included
                // `2' indicates returning to a file (after having included
                // another file.
                WriteLineInfo(FileName, Line, FileType, EOL, " 2");
            } else
              // fix up lineinfo (since commented out directive changed line
              // numbers) for inclusions that were skipped due to header guards
              WriteLineInfo(FileName, Line, FileType, EOL);
            break;
          }
          case tok::pp_pragma: {
            StringRef Identifier = NextIdentifierName(RawLex, RawToken);
            if (Identifier == "clang" || Identifier == "GCC") {
              if (NextIdentifierName(RawLex, RawToken) == "system_header") {
                // keep the directive in, commented out
                CommentOutDirective(RawLex, HashToken, FromFile, EOL,
                  NextToWrite, Line);
                // update our own type
                FileType = SM.getFileCharacteristic(RawToken.getLocation());
                WriteLineInfo(FileName, Line, FileType, EOL);
              }
            } else if (Identifier == "once") {
              // keep the directive in, commented out
              CommentOutDirective(RawLex, HashToken, FromFile, EOL,
                NextToWrite, Line);
              WriteLineInfo(FileName, Line, FileType, EOL);
            }
            break;
          }
          default:
            break;
        }
      }
      RawLex.setParsingPreprocessorDirective(false);
    }
    RawLex.LexFromRawLexer(RawToken);
  }
  OutputContentUpTo(FromFile, NextToWrite,
    SM.getFileOffset(SM.getLocForEndOfFile(FileId)) + 1, EOL, Line,
    /*EnsureNewline*/true);
  return true;
}

/// InclusionRewriterInInput - Implement -frewrite-includes mode.
void clang::RewriteIncludesInInput(Preprocessor &PP, raw_ostream *OS,
                                   const PreprocessorOutputOptions &Opts) {
  SourceManager &SM = PP.getSourceManager();
  InclusionRewriter *Rewrite = new InclusionRewriter(PP, *OS,
                                                     Opts.ShowLineMarkers);
  PP.addPPCallbacks(Rewrite);

  // First let the preprocessor process the entire file and call callbacks.
  // Callbacks will record which #include's were actually performed.
  PP.EnterMainSourceFile();
  Token Tok;
  // Only preprocessor directives matter here, so disable macro expansion
  // everywhere else as an optimization.
  // TODO: It would be even faster if the preprocessor could be switched
  // to a mode where it would parse only preprocessor directives and comments,
  // nothing else matters for parsing or processing.
  PP.SetMacroExpansionOnlyInDirectives();
  do {
    PP.Lex(Tok);
  } while (Tok.isNot(tok::eof));
  Rewrite->Process(SM.getMainFileID(), SrcMgr::C_User);
  OS->flush();
}
