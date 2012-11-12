//===--- PrintPreprocessedOutput.cpp - Implement the -E mode --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This code simply runs the preprocessor on the input file and prints out the
// result.  This is the traditional behavior of the -E option.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/Utils.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/PreprocessorOutputOptions.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Pragma.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/TokenConcatenation.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ErrorHandling.h"
#include <cctype>
#include <cstdio>
using namespace clang;

/// PrintMacroDefinition - Print a macro definition in a form that will be
/// properly accepted back as a definition.
static void PrintMacroDefinition(const IdentifierInfo &II, const MacroInfo &MI,
                                 Preprocessor &PP, raw_ostream &OS) {
  OS << "#define " << II.getName();

  if (MI.isFunctionLike()) {
    OS << '(';
    if (!MI.arg_empty()) {
      MacroInfo::arg_iterator AI = MI.arg_begin(), E = MI.arg_end();
      for (; AI+1 != E; ++AI) {
        OS << (*AI)->getName();
        OS << ',';
      }

      // Last argument.
      if ((*AI)->getName() == "__VA_ARGS__")
        OS << "...";
      else
        OS << (*AI)->getName();
    }

    if (MI.isGNUVarargs())
      OS << "...";  // #define foo(x...)

    OS << ')';
  }

  // GCC always emits a space, even if the macro body is empty.  However, do not
  // want to emit two spaces if the first token has a leading space.
  if (MI.tokens_empty() || !MI.tokens_begin()->hasLeadingSpace())
    OS << ' ';

  SmallString<128> SpellingBuffer;
  for (MacroInfo::tokens_iterator I = MI.tokens_begin(), E = MI.tokens_end();
       I != E; ++I) {
    if (I->hasLeadingSpace())
      OS << ' ';

    OS << PP.getSpelling(*I, SpellingBuffer);
  }
}

//===----------------------------------------------------------------------===//
// Preprocessed token printer
//===----------------------------------------------------------------------===//

namespace {
class PrintPPOutputPPCallbacks : public PPCallbacks {
  Preprocessor &PP;
  SourceManager &SM;
  TokenConcatenation ConcatInfo;
public:
  raw_ostream &OS;
private:
  unsigned CurLine;

  bool EmittedTokensOnThisLine;
  bool EmittedDirectiveOnThisLine;
  SrcMgr::CharacteristicKind FileType;
  SmallString<512> CurFilename;
  bool Initialized;
  bool DisableLineMarkers;
  bool DumpDefines;
  bool UseLineDirective;
public:
  PrintPPOutputPPCallbacks(Preprocessor &pp, raw_ostream &os,
                           bool lineMarkers, bool defines)
     : PP(pp), SM(PP.getSourceManager()),
       ConcatInfo(PP), OS(os), DisableLineMarkers(lineMarkers),
       DumpDefines(defines) {
    CurLine = 0;
    CurFilename += "<uninit>";
    EmittedTokensOnThisLine = false;
    EmittedDirectiveOnThisLine = false;
    FileType = SrcMgr::C_User;
    Initialized = false;

    // If we're in microsoft mode, use normal #line instead of line markers.
    UseLineDirective = PP.getLangOpts().MicrosoftExt;
  }

  void setEmittedTokensOnThisLine() { EmittedTokensOnThisLine = true; }
  bool hasEmittedTokensOnThisLine() const { return EmittedTokensOnThisLine; }

  void setEmittedDirectiveOnThisLine() { EmittedDirectiveOnThisLine = true; }
  bool hasEmittedDirectiveOnThisLine() const {
    return EmittedDirectiveOnThisLine;
  }

  bool startNewLineIfNeeded(bool ShouldUpdateCurrentLine = true);
  
  virtual void FileChanged(SourceLocation Loc, FileChangeReason Reason,
                           SrcMgr::CharacteristicKind FileType,
                           FileID PrevFID);
  virtual void Ident(SourceLocation Loc, const std::string &str);
  virtual void PragmaComment(SourceLocation Loc, const IdentifierInfo *Kind,
                             const std::string &Str);
  virtual void PragmaMessage(SourceLocation Loc, StringRef Str);
  virtual void PragmaDiagnosticPush(SourceLocation Loc,
                                    StringRef Namespace);
  virtual void PragmaDiagnosticPop(SourceLocation Loc,
                                   StringRef Namespace);
  virtual void PragmaDiagnostic(SourceLocation Loc, StringRef Namespace,
                                diag::Mapping Map, StringRef Str);

  bool HandleFirstTokOnLine(Token &Tok);
  bool MoveToLine(SourceLocation Loc) {
    PresumedLoc PLoc = SM.getPresumedLoc(Loc);
    if (PLoc.isInvalid())
      return false;
    return MoveToLine(PLoc.getLine());
  }
  bool MoveToLine(unsigned LineNo);

  bool AvoidConcat(const Token &PrevPrevTok, const Token &PrevTok, 
                   const Token &Tok) {
    return ConcatInfo.AvoidConcat(PrevPrevTok, PrevTok, Tok);
  }
  void WriteLineInfo(unsigned LineNo, const char *Extra=0, unsigned ExtraLen=0);
  bool LineMarkersAreDisabled() const { return DisableLineMarkers; }
  void HandleNewlinesInToken(const char *TokStr, unsigned Len);

  /// MacroDefined - This hook is called whenever a macro definition is seen.
  void MacroDefined(const Token &MacroNameTok, const MacroInfo *MI);

  /// MacroUndefined - This hook is called whenever a macro #undef is seen.
  void MacroUndefined(const Token &MacroNameTok, const MacroInfo *MI);
};
}  // end anonymous namespace

void PrintPPOutputPPCallbacks::WriteLineInfo(unsigned LineNo,
                                             const char *Extra,
                                             unsigned ExtraLen) {
  startNewLineIfNeeded(/*ShouldUpdateCurrentLine=*/false);

  // Emit #line directives or GNU line markers depending on what mode we're in.
  if (UseLineDirective) {
    OS << "#line" << ' ' << LineNo << ' ' << '"';
    OS.write(CurFilename.data(), CurFilename.size());
    OS << '"';
  } else {
    OS << '#' << ' ' << LineNo << ' ' << '"';
    OS.write(CurFilename.data(), CurFilename.size());
    OS << '"';

    if (ExtraLen)
      OS.write(Extra, ExtraLen);

    if (FileType == SrcMgr::C_System)
      OS.write(" 3", 2);
    else if (FileType == SrcMgr::C_ExternCSystem)
      OS.write(" 3 4", 4);
  }
  OS << '\n';
}

/// MoveToLine - Move the output to the source line specified by the location
/// object.  We can do this by emitting some number of \n's, or be emitting a
/// #line directive.  This returns false if already at the specified line, true
/// if some newlines were emitted.
bool PrintPPOutputPPCallbacks::MoveToLine(unsigned LineNo) {
  // If this line is "close enough" to the original line, just print newlines,
  // otherwise print a #line directive.
  if (LineNo-CurLine <= 8) {
    if (LineNo-CurLine == 1)
      OS << '\n';
    else if (LineNo == CurLine)
      return false;    // Spelling line moved, but expansion line didn't.
    else {
      const char *NewLines = "\n\n\n\n\n\n\n\n";
      OS.write(NewLines, LineNo-CurLine);
    }
  } else if (!DisableLineMarkers) {
    // Emit a #line or line marker.
    WriteLineInfo(LineNo, 0, 0);
  } else {
    // Okay, we're in -P mode, which turns off line markers.  However, we still
    // need to emit a newline between tokens on different lines.
    startNewLineIfNeeded(/*ShouldUpdateCurrentLine=*/false);
  }

  CurLine = LineNo;
  return true;
}

bool
PrintPPOutputPPCallbacks::startNewLineIfNeeded(bool ShouldUpdateCurrentLine) {
  if (EmittedTokensOnThisLine || EmittedDirectiveOnThisLine) {
    OS << '\n';
    EmittedTokensOnThisLine = false;
    EmittedDirectiveOnThisLine = false;
    if (ShouldUpdateCurrentLine)
      ++CurLine;
    return true;
  }
  
  return false;
}

/// FileChanged - Whenever the preprocessor enters or exits a #include file
/// it invokes this handler.  Update our conception of the current source
/// position.
void PrintPPOutputPPCallbacks::FileChanged(SourceLocation Loc,
                                           FileChangeReason Reason,
                                       SrcMgr::CharacteristicKind NewFileType,
                                       FileID PrevFID) {
  // Unless we are exiting a #include, make sure to skip ahead to the line the
  // #include directive was at.
  SourceManager &SourceMgr = SM;
  
  PresumedLoc UserLoc = SourceMgr.getPresumedLoc(Loc);
  if (UserLoc.isInvalid())
    return;
  
  unsigned NewLine = UserLoc.getLine();

  if (Reason == PPCallbacks::EnterFile) {
    SourceLocation IncludeLoc = UserLoc.getIncludeLoc();
    if (IncludeLoc.isValid())
      MoveToLine(IncludeLoc);
  } else if (Reason == PPCallbacks::SystemHeaderPragma) {
    MoveToLine(NewLine);

    // TODO GCC emits the # directive for this directive on the line AFTER the
    // directive and emits a bunch of spaces that aren't needed.  Emulate this
    // strange behavior.
  }
  
  CurLine = NewLine;

  CurFilename.clear();
  CurFilename += UserLoc.getFilename();
  Lexer::Stringify(CurFilename);
  FileType = NewFileType;

  if (DisableLineMarkers) return;
  
  if (!Initialized) {
    WriteLineInfo(CurLine);
    Initialized = true;
  }

  switch (Reason) {
  case PPCallbacks::EnterFile:
    WriteLineInfo(CurLine, " 1", 2);
    break;
  case PPCallbacks::ExitFile:
    WriteLineInfo(CurLine, " 2", 2);
    break;
  case PPCallbacks::SystemHeaderPragma:
  case PPCallbacks::RenameFile:
    WriteLineInfo(CurLine);
    break;
  }
}

/// Ident - Handle #ident directives when read by the preprocessor.
///
void PrintPPOutputPPCallbacks::Ident(SourceLocation Loc, const std::string &S) {
  MoveToLine(Loc);

  OS.write("#ident ", strlen("#ident "));
  OS.write(&S[0], S.size());
  EmittedTokensOnThisLine = true;
}

/// MacroDefined - This hook is called whenever a macro definition is seen.
void PrintPPOutputPPCallbacks::MacroDefined(const Token &MacroNameTok,
                                            const MacroInfo *MI) {
  // Only print out macro definitions in -dD mode.
  if (!DumpDefines ||
      // Ignore __FILE__ etc.
      MI->isBuiltinMacro()) return;

  MoveToLine(MI->getDefinitionLoc());
  PrintMacroDefinition(*MacroNameTok.getIdentifierInfo(), *MI, PP, OS);
  setEmittedDirectiveOnThisLine();
}

void PrintPPOutputPPCallbacks::MacroUndefined(const Token &MacroNameTok,
                                              const MacroInfo *MI) {
  // Only print out macro definitions in -dD mode.
  if (!DumpDefines) return;

  MoveToLine(MacroNameTok.getLocation());
  OS << "#undef " << MacroNameTok.getIdentifierInfo()->getName();
  setEmittedDirectiveOnThisLine();
}

void PrintPPOutputPPCallbacks::PragmaComment(SourceLocation Loc,
                                             const IdentifierInfo *Kind,
                                             const std::string &Str) {
  startNewLineIfNeeded();
  MoveToLine(Loc);
  OS << "#pragma comment(" << Kind->getName();

  if (!Str.empty()) {
    OS << ", \"";

    for (unsigned i = 0, e = Str.size(); i != e; ++i) {
      unsigned char Char = Str[i];
      if (isprint(Char) && Char != '\\' && Char != '"')
        OS << (char)Char;
      else  // Output anything hard as an octal escape.
        OS << '\\'
           << (char)('0'+ ((Char >> 6) & 7))
           << (char)('0'+ ((Char >> 3) & 7))
           << (char)('0'+ ((Char >> 0) & 7));
    }
    OS << '"';
  }

  OS << ')';
  setEmittedDirectiveOnThisLine();
}

void PrintPPOutputPPCallbacks::PragmaMessage(SourceLocation Loc,
                                             StringRef Str) {
  startNewLineIfNeeded();
  MoveToLine(Loc);
  OS << "#pragma message(";

  OS << '"';

  for (unsigned i = 0, e = Str.size(); i != e; ++i) {
    unsigned char Char = Str[i];
    if (isprint(Char) && Char != '\\' && Char != '"')
      OS << (char)Char;
    else  // Output anything hard as an octal escape.
      OS << '\\'
         << (char)('0'+ ((Char >> 6) & 7))
         << (char)('0'+ ((Char >> 3) & 7))
         << (char)('0'+ ((Char >> 0) & 7));
  }
  OS << '"';

  OS << ')';
  setEmittedDirectiveOnThisLine();
}

void PrintPPOutputPPCallbacks::
PragmaDiagnosticPush(SourceLocation Loc, StringRef Namespace) {
  startNewLineIfNeeded();
  MoveToLine(Loc);
  OS << "#pragma " << Namespace << " diagnostic push";
  setEmittedDirectiveOnThisLine();
}

void PrintPPOutputPPCallbacks::
PragmaDiagnosticPop(SourceLocation Loc, StringRef Namespace) {
  startNewLineIfNeeded();
  MoveToLine(Loc);
  OS << "#pragma " << Namespace << " diagnostic pop";
  setEmittedDirectiveOnThisLine();
}

void PrintPPOutputPPCallbacks::
PragmaDiagnostic(SourceLocation Loc, StringRef Namespace,
                 diag::Mapping Map, StringRef Str) {
  startNewLineIfNeeded();
  MoveToLine(Loc);
  OS << "#pragma " << Namespace << " diagnostic ";
  switch (Map) {
  case diag::MAP_WARNING:
    OS << "warning";
    break;
  case diag::MAP_ERROR:
    OS << "error";
    break;
  case diag::MAP_IGNORE:
    OS << "ignored";
    break;
  case diag::MAP_FATAL:
    OS << "fatal";
    break;
  }
  OS << " \"" << Str << '"';
  setEmittedDirectiveOnThisLine();
}

/// HandleFirstTokOnLine - When emitting a preprocessed file in -E mode, this
/// is called for the first token on each new line.  If this really is the start
/// of a new logical line, handle it and return true, otherwise return false.
/// This may not be the start of a logical line because the "start of line"
/// marker is set for spelling lines, not expansion ones.
bool PrintPPOutputPPCallbacks::HandleFirstTokOnLine(Token &Tok) {
  // Figure out what line we went to and insert the appropriate number of
  // newline characters.
  if (!MoveToLine(Tok.getLocation()))
    return false;

  // Print out space characters so that the first token on a line is
  // indented for easy reading.
  unsigned ColNo = SM.getExpansionColumnNumber(Tok.getLocation());

  // This hack prevents stuff like:
  // #define HASH #
  // HASH define foo bar
  // From having the # character end up at column 1, which makes it so it
  // is not handled as a #define next time through the preprocessor if in
  // -fpreprocessed mode.
  if (ColNo <= 1 && Tok.is(tok::hash))
    OS << ' ';

  // Otherwise, indent the appropriate number of spaces.
  for (; ColNo > 1; --ColNo)
    OS << ' ';

  return true;
}

void PrintPPOutputPPCallbacks::HandleNewlinesInToken(const char *TokStr,
                                                     unsigned Len) {
  unsigned NumNewlines = 0;
  for (; Len; --Len, ++TokStr) {
    if (*TokStr != '\n' &&
        *TokStr != '\r')
      continue;

    ++NumNewlines;

    // If we have \n\r or \r\n, skip both and count as one line.
    if (Len != 1 &&
        (TokStr[1] == '\n' || TokStr[1] == '\r') &&
        TokStr[0] != TokStr[1])
      ++TokStr, --Len;
  }

  if (NumNewlines == 0) return;

  CurLine += NumNewlines;
}


namespace {
struct UnknownPragmaHandler : public PragmaHandler {
  const char *Prefix;
  PrintPPOutputPPCallbacks *Callbacks;

  UnknownPragmaHandler(const char *prefix, PrintPPOutputPPCallbacks *callbacks)
    : Prefix(prefix), Callbacks(callbacks) {}
  virtual void HandlePragma(Preprocessor &PP, PragmaIntroducerKind Introducer,
                            Token &PragmaTok) {
    // Figure out what line we went to and insert the appropriate number of
    // newline characters.
    Callbacks->startNewLineIfNeeded();
    Callbacks->MoveToLine(PragmaTok.getLocation());
    Callbacks->OS.write(Prefix, strlen(Prefix));
    // Read and print all of the pragma tokens.
    while (PragmaTok.isNot(tok::eod)) {
      if (PragmaTok.hasLeadingSpace())
        Callbacks->OS << ' ';
      std::string TokSpell = PP.getSpelling(PragmaTok);
      Callbacks->OS.write(&TokSpell[0], TokSpell.size());
      PP.LexUnexpandedToken(PragmaTok);
    }
    Callbacks->setEmittedDirectiveOnThisLine();
  }
};
} // end anonymous namespace


static void PrintPreprocessedTokens(Preprocessor &PP, Token &Tok,
                                    PrintPPOutputPPCallbacks *Callbacks,
                                    raw_ostream &OS) {
  char Buffer[256];
  Token PrevPrevTok, PrevTok;
  PrevPrevTok.startToken();
  PrevTok.startToken();
  while (1) {
    if (Callbacks->hasEmittedDirectiveOnThisLine()) {
      Callbacks->startNewLineIfNeeded();
      Callbacks->MoveToLine(Tok.getLocation());
    }

    // If this token is at the start of a line, emit newlines if needed.
    if (Tok.isAtStartOfLine() && Callbacks->HandleFirstTokOnLine(Tok)) {
      // done.
    } else if (Tok.hasLeadingSpace() ||
               // If we haven't emitted a token on this line yet, PrevTok isn't
               // useful to look at and no concatenation could happen anyway.
               (Callbacks->hasEmittedTokensOnThisLine() &&
                // Don't print "-" next to "-", it would form "--".
                Callbacks->AvoidConcat(PrevPrevTok, PrevTok, Tok))) {
      OS << ' ';
    }

    if (IdentifierInfo *II = Tok.getIdentifierInfo()) {
      OS << II->getName();
    } else if (Tok.isLiteral() && !Tok.needsCleaning() &&
               Tok.getLiteralData()) {
      OS.write(Tok.getLiteralData(), Tok.getLength());
    } else if (Tok.getLength() < 256) {
      const char *TokPtr = Buffer;
      unsigned Len = PP.getSpelling(Tok, TokPtr);
      OS.write(TokPtr, Len);

      // Tokens that can contain embedded newlines need to adjust our current
      // line number.
      if (Tok.getKind() == tok::comment)
        Callbacks->HandleNewlinesInToken(TokPtr, Len);
    } else {
      std::string S = PP.getSpelling(Tok);
      OS.write(&S[0], S.size());

      // Tokens that can contain embedded newlines need to adjust our current
      // line number.
      if (Tok.getKind() == tok::comment)
        Callbacks->HandleNewlinesInToken(&S[0], S.size());
    }
    Callbacks->setEmittedTokensOnThisLine();

    if (Tok.is(tok::eof)) break;

    PrevPrevTok = PrevTok;
    PrevTok = Tok;
    PP.Lex(Tok);
  }
}

typedef std::pair<IdentifierInfo*, MacroInfo*> id_macro_pair;
static int MacroIDCompare(const void* a, const void* b) {
  const id_macro_pair *LHS = static_cast<const id_macro_pair*>(a);
  const id_macro_pair *RHS = static_cast<const id_macro_pair*>(b);
  return LHS->first->getName().compare(RHS->first->getName());
}

static void DoPrintMacros(Preprocessor &PP, raw_ostream *OS) {
  // Ignore unknown pragmas.
  PP.AddPragmaHandler(new EmptyPragmaHandler());

  // -dM mode just scans and ignores all tokens in the files, then dumps out
  // the macro table at the end.
  PP.EnterMainSourceFile();

  Token Tok;
  do PP.Lex(Tok);
  while (Tok.isNot(tok::eof));

  SmallVector<id_macro_pair, 128> MacrosByID;
  for (Preprocessor::macro_iterator I = PP.macro_begin(), E = PP.macro_end();
       I != E; ++I) {
    if (I->first->hasMacroDefinition())
      MacrosByID.push_back(id_macro_pair(I->first, I->second));
  }
  llvm::array_pod_sort(MacrosByID.begin(), MacrosByID.end(), MacroIDCompare);

  for (unsigned i = 0, e = MacrosByID.size(); i != e; ++i) {
    MacroInfo &MI = *MacrosByID[i].second;
    // Ignore computed macros like __LINE__ and friends.
    if (MI.isBuiltinMacro()) continue;

    PrintMacroDefinition(*MacrosByID[i].first, MI, PP, *OS);
    *OS << '\n';
  }
}

/// DoPrintPreprocessedInput - This implements -E mode.
///
void clang::DoPrintPreprocessedInput(Preprocessor &PP, raw_ostream *OS,
                                     const PreprocessorOutputOptions &Opts) {
  // Show macros with no output is handled specially.
  if (!Opts.ShowCPP) {
    assert(Opts.ShowMacros && "Not yet implemented!");
    DoPrintMacros(PP, OS);
    return;
  }

  // Inform the preprocessor whether we want it to retain comments or not, due
  // to -C or -CC.
  PP.SetCommentRetentionState(Opts.ShowComments, Opts.ShowMacroComments);

  PrintPPOutputPPCallbacks *Callbacks =
      new PrintPPOutputPPCallbacks(PP, *OS, !Opts.ShowLineMarkers,
                                   Opts.ShowMacros);
  PP.AddPragmaHandler(new UnknownPragmaHandler("#pragma", Callbacks));
  PP.AddPragmaHandler("GCC", new UnknownPragmaHandler("#pragma GCC",Callbacks));
  PP.AddPragmaHandler("clang",
                      new UnknownPragmaHandler("#pragma clang", Callbacks));

  PP.addPPCallbacks(Callbacks);

  // After we have configured the preprocessor, enter the main file.
  PP.EnterMainSourceFile();

  // Consume all of the tokens that come from the predefines buffer.  Those
  // should not be emitted into the output and are guaranteed to be at the
  // start.
  const SourceManager &SourceMgr = PP.getSourceManager();
  Token Tok;
  do {
    PP.Lex(Tok);
    if (Tok.is(tok::eof) || !Tok.getLocation().isFileID())
      break;

    PresumedLoc PLoc = SourceMgr.getPresumedLoc(Tok.getLocation());
    if (PLoc.isInvalid())
      break;

    if (strcmp(PLoc.getFilename(), "<built-in>"))
      break;
  } while (true);

  // Read all the preprocessed tokens, printing them out to the stream.
  PrintPreprocessedTokens(PP, Tok, Callbacks, *OS);
  *OS << '\n';
}
