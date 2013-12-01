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
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/PreprocessorOutputOptions.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/Pragma.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace llvm;

namespace {

class InclusionRewriter : public PPCallbacks {
  /// Information about which #includes were actually performed,
  /// created by preprocessor callbacks.
  struct FileChange {
    const Module *Mod;
    SourceLocation From;
    FileID Id;
    SrcMgr::CharacteristicKind FileType;
    FileChange(SourceLocation From, const Module *Mod) : Mod(Mod), From(From) {
    }
  };
  Preprocessor &PP; ///< Used to find inclusion directives.
  SourceManager &SM; ///< Used to read and manage source files.
  raw_ostream &OS; ///< The destination stream for rewritten contents.
  const llvm::MemoryBuffer *PredefinesBuffer; ///< The preprocessor predefines.
  bool ShowLineMarkers; ///< Show #line markers.
  bool UseLineDirective; ///< Use of line directives or line markers.
  typedef std::map<unsigned, FileChange> FileChangeMap;
  FileChangeMap FileChanges; ///< Tracks which files were included where.
  /// Used transitively for building up the FileChanges mapping over the
  /// various \c PPCallbacks callbacks.
  FileChangeMap::iterator LastInsertedFileChange;
public:
  InclusionRewriter(Preprocessor &PP, raw_ostream &OS, bool ShowLineMarkers);
  bool Process(FileID FileId, SrcMgr::CharacteristicKind FileType);
  void setPredefinesBuffer(const llvm::MemoryBuffer *Buf) {
    PredefinesBuffer = Buf;
  }
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
  void WriteImplicitModuleImport(const Module *Mod, StringRef EOL);
  void OutputContentUpTo(const MemoryBuffer &FromFile,
                         unsigned &WriteFrom, unsigned WriteTo,
                         StringRef EOL, int &lines,
                         bool EnsureNewline = false);
  void CommentOutDirective(Lexer &DirectivesLex, const Token &StartToken,
                           const MemoryBuffer &FromFile, StringRef EOL,
                           unsigned &NextToWrite, int &Lines);
  bool HandleHasInclude(FileID FileId, Lexer &RawLex,
                        const DirectoryLookup *Lookup, Token &Tok,
                        bool &FileExists);
  const FileChange *FindFileChangeLocation(SourceLocation Loc) const;
  StringRef NextIdentifierName(Lexer &RawLex, Token &RawToken);
};

}  // end anonymous namespace

/// Initializes an InclusionRewriter with a \p PP source and \p OS destination.
InclusionRewriter::InclusionRewriter(Preprocessor &PP, raw_ostream &OS,
                                     bool ShowLineMarkers)
    : PP(PP), SM(PP.getSourceManager()), OS(OS), PredefinesBuffer(0),
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
    OS << "#line" << ' ' << Line << ' ' << '"';
    OS.write_escaped(Filename);
    OS << '"';
  } else {
    // Use GNU linemarkers as described here:
    // http://gcc.gnu.org/onlinedocs/cpp/Preprocessor-Output.html
    OS << '#' << ' ' << Line << ' ' << '"';
    OS.write_escaped(Filename);
    OS << '"';
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

void InclusionRewriter::WriteImplicitModuleImport(const Module *Mod,
                                                  StringRef EOL) {
  OS << "@import " << Mod->getFullModuleName() << ";"
     << " /* clang -frewrite-includes: implicit import */" << EOL;
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
                                           const Module *Imported) {
  assert(LastInsertedFileChange == FileChanges.end() && "Another inclusion "
    "directive was found before the previous one was processed");
  std::pair<FileChangeMap::iterator, bool> p = FileChanges.insert(
    std::make_pair(HashLoc.getRawEncoding(), FileChange(HashLoc, Imported)));
  assert(p.second && "Unexpected revisitation of the same include directive");
  if (!Imported)
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
  if (&FromFile == PredefinesBuffer) {
    // Ignore the #defines of the predefines buffer.
    WriteFrom = WriteTo;
    return;
  }
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

// Expand __has_include and __has_include_next if possible. If there's no
// definitive answer return false.
bool InclusionRewriter::HandleHasInclude(
    FileID FileId, Lexer &RawLex, const DirectoryLookup *Lookup, Token &Tok,
    bool &FileExists) {
  // Lex the opening paren.
  RawLex.LexFromRawLexer(Tok);
  if (Tok.isNot(tok::l_paren))
    return false;

  RawLex.LexFromRawLexer(Tok);

  SmallString<128> FilenameBuffer;
  StringRef Filename;
  // Since the raw lexer doesn't give us angle_literals we have to parse them
  // ourselves.
  // FIXME: What to do if the file name is a macro?
  if (Tok.is(tok::less)) {
    RawLex.LexFromRawLexer(Tok);

    FilenameBuffer += '<';
    do {
      if (Tok.is(tok::eod)) // Sanity check.
        return false;

      if (Tok.is(tok::raw_identifier))
        PP.LookUpIdentifierInfo(Tok);

      // Get the string piece.
      SmallVector<char, 128> TmpBuffer;
      bool Invalid = false;
      StringRef TmpName = PP.getSpelling(Tok, TmpBuffer, &Invalid);
      if (Invalid)
        return false;

      FilenameBuffer += TmpName;

      RawLex.LexFromRawLexer(Tok);
    } while (Tok.isNot(tok::greater));

    FilenameBuffer += '>';
    Filename = FilenameBuffer;
  } else {
    if (Tok.isNot(tok::string_literal))
      return false;

    bool Invalid = false;
    Filename = PP.getSpelling(Tok, FilenameBuffer, &Invalid);
    if (Invalid)
      return false;
  }

  // Lex the closing paren.
  RawLex.LexFromRawLexer(Tok);
  if (Tok.isNot(tok::r_paren))
    return false;

  // Now ask HeaderInfo if it knows about the header.
  // FIXME: Subframeworks aren't handled here. Do we care?
  bool isAngled = PP.GetIncludeFilenameSpelling(Tok.getLocation(), Filename);
  const DirectoryLookup *CurDir;
  const FileEntry *File = PP.getHeaderSearchInfo().LookupFile(
      Filename, isAngled, 0, CurDir,
      PP.getSourceManager().getFileEntryForID(FileId), 0, 0, 0, false);

  FileExists = File != 0;
  return true;
}

/// Use a raw lexer to analyze \p FileId, incrementally copying parts of it
/// and including content of included files recursively.
bool InclusionRewriter::Process(FileID FileId,
                                SrcMgr::CharacteristicKind FileType)
{
  bool Invalid;
  const MemoryBuffer &FromFile = *SM.getBuffer(FileId, &Invalid);
  if (Invalid) // invalid inclusion
    return false;
  const char *FileName = FromFile.getBufferIdentifier();
  Lexer RawLex(FileId, &FromFile, PP.getSourceManager(), PP.getLangOpts());
  RawLex.SetCommentRetentionState(false);

  StringRef EOL = DetectEOL(FromFile);

  // Per the GNU docs: "1" indicates the start of a new file.
  WriteLineInfo(FileName, 1, FileType, EOL, " 1");

  if (SM.getFileIDSize(FileId) == 0)
    return false;

  // The next byte to be copied from the source file
  unsigned NextToWrite = 0;
  int Line = 1; // The current input file line number.

  // Ignore UTF-8 BOM, otherwise it'd end up somewhere else than the start
  // of the resulting file.
  if (FromFile.getBuffer().startswith("\xEF\xBB\xBF"))
    NextToWrite = 3;

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
      if (RawToken.getIdentifierInfo() != NULL) {
        switch (RawToken.getIdentifierInfo()->getPPKeywordID()) {
          case tok::pp_include:
          case tok::pp_include_next:
          case tok::pp_import: {
            CommentOutDirective(RawLex, HashToken, FromFile, EOL, NextToWrite,
              Line);
            StringRef LineInfoExtra;
            if (const FileChange *Change = FindFileChangeLocation(
                HashToken.getLocation())) {
              if (Change->Mod) {
                WriteImplicitModuleImport(Change->Mod, EOL);

              // else now include and recursively process the file
              } else if (Process(Change->Id, Change->FileType)) {
                // and set lineinfo back to this file, if the nested one was
                // actually included
                // `2' indicates returning to a file (after having included
                // another file.
                LineInfoExtra = " 2";
              }
            }
            // fix up lineinfo (since commented out directive changed line
            // numbers) for inclusions that were skipped due to header guards
            WriteLineInfo(FileName, Line, FileType, EOL, LineInfoExtra);
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
          case tok::pp_if:
          case tok::pp_elif: {
            bool elif = (RawToken.getIdentifierInfo()->getPPKeywordID() ==
                         tok::pp_elif);
            // Rewrite special builtin macros to avoid pulling in host details.
            do {
              // Walk over the directive.
              RawLex.LexFromRawLexer(RawToken);
              if (RawToken.is(tok::raw_identifier))
                PP.LookUpIdentifierInfo(RawToken);

              if (RawToken.is(tok::identifier)) {
                bool HasFile;
                SourceLocation Loc = RawToken.getLocation();

                // Rewrite __has_include(x)
                if (RawToken.getIdentifierInfo()->isStr("__has_include")) {
                  if (!HandleHasInclude(FileId, RawLex, 0, RawToken, HasFile))
                    continue;
                  // Rewrite __has_include_next(x)
                } else if (RawToken.getIdentifierInfo()->isStr(
                               "__has_include_next")) {
                  const DirectoryLookup *Lookup = PP.GetCurDirLookup();
                  if (Lookup)
                    ++Lookup;

                  if (!HandleHasInclude(FileId, RawLex, Lookup, RawToken,
                                        HasFile))
                    continue;
                } else {
                  continue;
                }
                // Replace the macro with (0) or (1), followed by the commented
                // out macro for reference.
                OutputContentUpTo(FromFile, NextToWrite, SM.getFileOffset(Loc),
                                  EOL, Line);
                OS << '(' << (int) HasFile << ")/*";
                OutputContentUpTo(FromFile, NextToWrite,
                                  SM.getFileOffset(RawToken.getLocation()) +
                                  RawToken.getLength(),
                                  EOL, Line);
                OS << "*/";
              }
            } while (RawToken.isNot(tok::eod));
            if (elif) {
              OutputContentUpTo(FromFile, NextToWrite,
                                SM.getFileOffset(RawToken.getLocation()) +
                                    RawToken.getLength(),
                                EOL, Line, /*EnsureNewLine*/ true);
              WriteLineInfo(FileName, Line, FileType, EOL);
            }
            break;
          }
          case tok::pp_endif:
          case tok::pp_else: {
            // We surround every #include by #if 0 to comment it out, but that
            // changes line numbers. These are fixed up right after that, but
            // the whole #include could be inside a preprocessor conditional
            // that is not processed. So it is necessary to fix the line
            // numbers one the next line after each #else/#endif as well.
            RawLex.SetKeepWhitespaceMode(true);
            do {
              RawLex.LexFromRawLexer(RawToken);
            } while (RawToken.isNot(tok::eod) && RawToken.isNot(tok::eof));
            OutputContentUpTo(
                FromFile, NextToWrite,
                SM.getFileOffset(RawToken.getLocation()) + RawToken.getLength(),
                EOL, Line, /*EnsureNewLine*/ true);
            WriteLineInfo(FileName, Line, FileType, EOL);
            RawLex.SetKeepWhitespaceMode(false);
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
    SM.getFileOffset(SM.getLocForEndOfFile(FileId)), EOL, Line,
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
  // Ignore all pragmas, otherwise there will be warnings about unknown pragmas
  // (because there's nothing to handle them).
  PP.AddPragmaHandler(new EmptyPragmaHandler());
  // Ignore also all pragma in all namespaces created
  // in Preprocessor::RegisterBuiltinPragmas().
  PP.AddPragmaHandler("GCC", new EmptyPragmaHandler());
  PP.AddPragmaHandler("clang", new EmptyPragmaHandler());

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
  Rewrite->setPredefinesBuffer(SM.getBuffer(PP.getPredefinesFileID()));
  Rewrite->Process(PP.getPredefinesFileID(), SrcMgr::C_User);
  Rewrite->Process(SM.getMainFileID(), SrcMgr::C_User);
  OS->flush();
}
