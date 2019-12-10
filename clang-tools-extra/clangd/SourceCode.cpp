//===--- SourceCode.h - Manipulating source code as strings -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "SourceCode.h"

#include "Context.h"
#include "FuzzyMatch.h"
#include "Logger.h"
#include "Protocol.h"
#include "refactor/Tweak.h"
#include "clang/AST/ASTContext.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Driver/Types.h"
#include "clang/Format/Format.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/Token.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA1.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/xxhash.h"
#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

namespace clang {
namespace clangd {

// Here be dragons. LSP positions use columns measured in *UTF-16 code units*!
// Clangd uses UTF-8 and byte-offsets internally, so conversion is nontrivial.

// Iterates over unicode codepoints in the (UTF-8) string. For each,
// invokes CB(UTF-8 length, UTF-16 length), and breaks if it returns true.
// Returns true if CB returned true, false if we hit the end of string.
template <typename Callback>
static bool iterateCodepoints(llvm::StringRef U8, const Callback &CB) {
  // A codepoint takes two UTF-16 code unit if it's astral (outside BMP).
  // Astral codepoints are encoded as 4 bytes in UTF-8, starting with 11110xxx.
  for (size_t I = 0; I < U8.size();) {
    unsigned char C = static_cast<unsigned char>(U8[I]);
    if (LLVM_LIKELY(!(C & 0x80))) { // ASCII character.
      if (CB(1, 1))
        return true;
      ++I;
      continue;
    }
    // This convenient property of UTF-8 holds for all non-ASCII characters.
    size_t UTF8Length = llvm::countLeadingOnes(C);
    // 0xxx is ASCII, handled above. 10xxx is a trailing byte, invalid here.
    // 11111xxx is not valid UTF-8 at all. Assert because it's probably our bug.
    assert((UTF8Length >= 2 && UTF8Length <= 4) &&
           "Invalid UTF-8, or transcoding bug?");
    I += UTF8Length; // Skip over all trailing bytes.
    // A codepoint takes two UTF-16 code unit if it's astral (outside BMP).
    // Astral codepoints are encoded as 4 bytes in UTF-8 (11110xxx ...)
    if (CB(UTF8Length, UTF8Length == 4 ? 2 : 1))
      return true;
  }
  return false;
}

// Returns the byte offset into the string that is an offset of \p Units in
// the specified encoding.
// Conceptually, this converts to the encoding, truncates to CodeUnits,
// converts back to UTF-8, and returns the length in bytes.
static size_t measureUnits(llvm::StringRef U8, int Units, OffsetEncoding Enc,
                           bool &Valid) {
  Valid = Units >= 0;
  if (Units <= 0)
    return 0;
  size_t Result = 0;
  switch (Enc) {
  case OffsetEncoding::UTF8:
    Result = Units;
    break;
  case OffsetEncoding::UTF16:
    Valid = iterateCodepoints(U8, [&](int U8Len, int U16Len) {
      Result += U8Len;
      Units -= U16Len;
      return Units <= 0;
    });
    if (Units < 0) // Offset in the middle of a surrogate pair.
      Valid = false;
    break;
  case OffsetEncoding::UTF32:
    Valid = iterateCodepoints(U8, [&](int U8Len, int U16Len) {
      Result += U8Len;
      Units--;
      return Units <= 0;
    });
    break;
  case OffsetEncoding::UnsupportedEncoding:
    llvm_unreachable("unsupported encoding");
  }
  // Don't return an out-of-range index if we overran.
  if (Result > U8.size()) {
    Valid = false;
    return U8.size();
  }
  return Result;
}

Key<OffsetEncoding> kCurrentOffsetEncoding;
static OffsetEncoding lspEncoding() {
  auto *Enc = Context::current().get(kCurrentOffsetEncoding);
  return Enc ? *Enc : OffsetEncoding::UTF16;
}

// Like most strings in clangd, the input is UTF-8 encoded.
size_t lspLength(llvm::StringRef Code) {
  size_t Count = 0;
  switch (lspEncoding()) {
  case OffsetEncoding::UTF8:
    Count = Code.size();
    break;
  case OffsetEncoding::UTF16:
    iterateCodepoints(Code, [&](int U8Len, int U16Len) {
      Count += U16Len;
      return false;
    });
    break;
  case OffsetEncoding::UTF32:
    iterateCodepoints(Code, [&](int U8Len, int U16Len) {
      ++Count;
      return false;
    });
    break;
  case OffsetEncoding::UnsupportedEncoding:
    llvm_unreachable("unsupported encoding");
  }
  return Count;
}

llvm::Expected<size_t> positionToOffset(llvm::StringRef Code, Position P,
                                        bool AllowColumnsBeyondLineLength) {
  if (P.line < 0)
    return llvm::make_error<llvm::StringError>(
        llvm::formatv("Line value can't be negative ({0})", P.line),
        llvm::errc::invalid_argument);
  if (P.character < 0)
    return llvm::make_error<llvm::StringError>(
        llvm::formatv("Character value can't be negative ({0})", P.character),
        llvm::errc::invalid_argument);
  size_t StartOfLine = 0;
  for (int I = 0; I != P.line; ++I) {
    size_t NextNL = Code.find('\n', StartOfLine);
    if (NextNL == llvm::StringRef::npos)
      return llvm::make_error<llvm::StringError>(
          llvm::formatv("Line value is out of range ({0})", P.line),
          llvm::errc::invalid_argument);
    StartOfLine = NextNL + 1;
  }
  StringRef Line =
      Code.substr(StartOfLine).take_until([](char C) { return C == '\n'; });

  // P.character may be in UTF-16, transcode if necessary.
  bool Valid;
  size_t ByteInLine = measureUnits(Line, P.character, lspEncoding(), Valid);
  if (!Valid && !AllowColumnsBeyondLineLength)
    return llvm::make_error<llvm::StringError>(
        llvm::formatv("{0} offset {1} is invalid for line {2}", lspEncoding(),
                      P.character, P.line),
        llvm::errc::invalid_argument);
  return StartOfLine + ByteInLine;
}

Position offsetToPosition(llvm::StringRef Code, size_t Offset) {
  Offset = std::min(Code.size(), Offset);
  llvm::StringRef Before = Code.substr(0, Offset);
  int Lines = Before.count('\n');
  size_t PrevNL = Before.rfind('\n');
  size_t StartOfLine = (PrevNL == llvm::StringRef::npos) ? 0 : (PrevNL + 1);
  Position Pos;
  Pos.line = Lines;
  Pos.character = lspLength(Before.substr(StartOfLine));
  return Pos;
}

Position sourceLocToPosition(const SourceManager &SM, SourceLocation Loc) {
  // We use the SourceManager's line tables, but its column number is in bytes.
  FileID FID;
  unsigned Offset;
  std::tie(FID, Offset) = SM.getDecomposedSpellingLoc(Loc);
  Position P;
  P.line = static_cast<int>(SM.getLineNumber(FID, Offset)) - 1;
  bool Invalid = false;
  llvm::StringRef Code = SM.getBufferData(FID, &Invalid);
  if (!Invalid) {
    auto ColumnInBytes = SM.getColumnNumber(FID, Offset) - 1;
    auto LineSoFar = Code.substr(Offset - ColumnInBytes, ColumnInBytes);
    P.character = lspLength(LineSoFar);
  }
  return P;
}

bool isSpelledInSource(SourceLocation Loc, const SourceManager &SM) {
  if (Loc.isMacroID()) {
    std::string PrintLoc = SM.getSpellingLoc(Loc).printToString(SM);
    if (llvm::StringRef(PrintLoc).startswith("<scratch") ||
        llvm::StringRef(PrintLoc).startswith("<command line>"))
      return false;
  }
  return true;
}

llvm::Optional<Range> getTokenRange(const SourceManager &SM,
                                    const LangOptions &LangOpts,
                                    SourceLocation TokLoc) {
  if (!TokLoc.isValid())
    return llvm::None;
  SourceLocation End = Lexer::getLocForEndOfToken(TokLoc, 0, SM, LangOpts);
  if (!End.isValid())
    return llvm::None;
  return halfOpenToRange(SM, CharSourceRange::getCharRange(TokLoc, End));
}

namespace {

enum TokenFlavor { Identifier, Operator, Whitespace, Other };

bool isOverloadedOperator(const Token &Tok) {
  switch (Tok.getKind()) {
#define OVERLOADED_OPERATOR(Name, Spelling, Token, Unary, Binary, MemOnly)     \
  case tok::Token:
#define OVERLOADED_OPERATOR_MULTI(Name, Spelling, Unary, Binary, MemOnly)
#include "clang/Basic/OperatorKinds.def"
    return true;

  default:
    break;
  }
  return false;
}

TokenFlavor getTokenFlavor(SourceLocation Loc, const SourceManager &SM,
                           const LangOptions &LangOpts) {
  Token Tok;
  Tok.setKind(tok::NUM_TOKENS);
  if (Lexer::getRawToken(Loc, Tok, SM, LangOpts,
                         /*IgnoreWhiteSpace*/ false))
    return Other;

  // getRawToken will return false without setting Tok when the token is
  // whitespace, so if the flag is not set, we are sure this is a whitespace.
  if (Tok.is(tok::TokenKind::NUM_TOKENS))
    return Whitespace;
  if (Tok.is(tok::TokenKind::raw_identifier))
    return Identifier;
  if (isOverloadedOperator(Tok))
    return Operator;
  return Other;
}

} // namespace

SourceLocation getBeginningOfIdentifier(const Position &Pos,
                                        const SourceManager &SM,
                                        const LangOptions &LangOpts) {
  FileID FID = SM.getMainFileID();
  auto Offset = positionToOffset(SM.getBufferData(FID), Pos);
  if (!Offset) {
    log("getBeginningOfIdentifier: {0}", Offset.takeError());
    return SourceLocation();
  }

  // GetBeginningOfToken(InputLoc) is almost what we want, but does the wrong
  // thing if the cursor is at the end of the token (identifier or operator).
  // The cases are:
  //   1) at the beginning of the token
  //   2) at the middle of the token
  //   3) at the end of the token
  //   4) anywhere outside the identifier or operator
  // To distinguish all cases, we lex both at the
  // GetBeginningOfToken(InputLoc-1) and GetBeginningOfToken(InputLoc), for
  // cases 1 and 4, we just return the original location.
  SourceLocation InputLoc = SM.getComposedLoc(FID, *Offset);
  if (*Offset == 0) // Case 1 or 4.
    return InputLoc;
  SourceLocation Before = SM.getComposedLoc(FID, *Offset - 1);
  SourceLocation BeforeTokBeginning =
      Lexer::GetBeginningOfToken(Before, SM, LangOpts);
  TokenFlavor BeforeKind = getTokenFlavor(BeforeTokBeginning, SM, LangOpts);

  SourceLocation CurrentTokBeginning =
      Lexer::GetBeginningOfToken(InputLoc, SM, LangOpts);
  TokenFlavor CurrentKind = getTokenFlavor(CurrentTokBeginning, SM, LangOpts);

  // At the middle of the token.
  if (BeforeTokBeginning == CurrentTokBeginning) {
    // For interesting token, we return the beginning of the token.
    if (CurrentKind == Identifier || CurrentKind == Operator)
      return CurrentTokBeginning;
    // otherwise, we return the original loc.
    return InputLoc;
  }

  // Whitespace is not interesting.
  if (BeforeKind == Whitespace)
    return CurrentTokBeginning;
  if (CurrentKind == Whitespace)
    return BeforeTokBeginning;

  // The cursor is at the token boundary, e.g. "Before^Current", we prefer
  // identifiers to other tokens.
  if (CurrentKind == Identifier)
    return CurrentTokBeginning;
  if (BeforeKind == Identifier)
    return BeforeTokBeginning;
  // Then prefer overloaded operators to other tokens.
  if (CurrentKind == Operator)
    return CurrentTokBeginning;
  if (BeforeKind == Operator)
    return BeforeTokBeginning;

  // Non-interesting case, we just return the original location.
  return InputLoc;
}

bool isValidFileRange(const SourceManager &Mgr, SourceRange R) {
  if (!R.getBegin().isValid() || !R.getEnd().isValid())
    return false;

  FileID BeginFID;
  size_t BeginOffset = 0;
  std::tie(BeginFID, BeginOffset) = Mgr.getDecomposedLoc(R.getBegin());

  FileID EndFID;
  size_t EndOffset = 0;
  std::tie(EndFID, EndOffset) = Mgr.getDecomposedLoc(R.getEnd());

  return BeginFID.isValid() && BeginFID == EndFID && BeginOffset <= EndOffset;
}

bool halfOpenRangeContains(const SourceManager &Mgr, SourceRange R,
                           SourceLocation L) {
  assert(isValidFileRange(Mgr, R));

  FileID BeginFID;
  size_t BeginOffset = 0;
  std::tie(BeginFID, BeginOffset) = Mgr.getDecomposedLoc(R.getBegin());
  size_t EndOffset = Mgr.getFileOffset(R.getEnd());

  FileID LFid;
  size_t LOffset;
  std::tie(LFid, LOffset) = Mgr.getDecomposedLoc(L);
  return BeginFID == LFid && BeginOffset <= LOffset && LOffset < EndOffset;
}

bool halfOpenRangeTouches(const SourceManager &Mgr, SourceRange R,
                          SourceLocation L) {
  return L == R.getEnd() || halfOpenRangeContains(Mgr, R, L);
}

SourceLocation includeHashLoc(FileID IncludedFile, const SourceManager &SM) {
  assert(SM.getLocForEndOfFile(IncludedFile).isFileID());
  FileID IncludingFile;
  unsigned Offset;
  std::tie(IncludingFile, Offset) =
      SM.getDecomposedExpansionLoc(SM.getIncludeLoc(IncludedFile));
  bool Invalid = false;
  llvm::StringRef Buf = SM.getBufferData(IncludingFile, &Invalid);
  if (Invalid)
    return SourceLocation();
  // Now buf is "...\n#include <foo>\n..."
  // and Offset points here:   ^
  // Rewind to the preceding # on the line.
  assert(Offset < Buf.size());
  for (;; --Offset) {
    if (Buf[Offset] == '#')
      return SM.getComposedLoc(IncludingFile, Offset);
    if (Buf[Offset] == '\n' || Offset == 0) // no hash, what's going on?
      return SourceLocation();
  }
}

static unsigned getTokenLengthAtLoc(SourceLocation Loc, const SourceManager &SM,
                                    const LangOptions &LangOpts) {
  Token TheTok;
  if (Lexer::getRawToken(Loc, TheTok, SM, LangOpts))
    return 0;
  // FIXME: Here we check whether the token at the location is a greatergreater
  // (>>) token and consider it as a single greater (>). This is to get it
  // working for templates but it isn't correct for the right shift operator. We
  // can avoid this by using half open char ranges in getFileRange() but getting
  // token ending is not well supported in macroIDs.
  if (TheTok.is(tok::greatergreater))
    return 1;
  return TheTok.getLength();
}

// Returns location of the last character of the token at a given loc
static SourceLocation getLocForTokenEnd(SourceLocation BeginLoc,
                                        const SourceManager &SM,
                                        const LangOptions &LangOpts) {
  unsigned Len = getTokenLengthAtLoc(BeginLoc, SM, LangOpts);
  return BeginLoc.getLocWithOffset(Len ? Len - 1 : 0);
}

// Returns location of the starting of the token at a given EndLoc
static SourceLocation getLocForTokenBegin(SourceLocation EndLoc,
                                          const SourceManager &SM,
                                          const LangOptions &LangOpts) {
  return EndLoc.getLocWithOffset(
      -(signed)getTokenLengthAtLoc(EndLoc, SM, LangOpts));
}

// Converts a char source range to a token range.
static SourceRange toTokenRange(CharSourceRange Range, const SourceManager &SM,
                                const LangOptions &LangOpts) {
  if (!Range.isTokenRange())
    Range.setEnd(getLocForTokenBegin(Range.getEnd(), SM, LangOpts));
  return Range.getAsRange();
}
// Returns the union of two token ranges.
// To find the maximum of the Ends of the ranges, we compare the location of the
// last character of the token.
static SourceRange unionTokenRange(SourceRange R1, SourceRange R2,
                                   const SourceManager &SM,
                                   const LangOptions &LangOpts) {
  SourceLocation Begin =
      SM.isBeforeInTranslationUnit(R1.getBegin(), R2.getBegin())
          ? R1.getBegin()
          : R2.getBegin();
  SourceLocation End =
      SM.isBeforeInTranslationUnit(getLocForTokenEnd(R1.getEnd(), SM, LangOpts),
                                   getLocForTokenEnd(R2.getEnd(), SM, LangOpts))
          ? R2.getEnd()
          : R1.getEnd();
  return SourceRange(Begin, End);
}

// Given a range whose endpoints may be in different expansions or files,
// tries to find a range within a common file by following up the expansion and
// include location in each.
static SourceRange rangeInCommonFile(SourceRange R, const SourceManager &SM,
                                     const LangOptions &LangOpts) {
  // Fast path for most common cases.
  if (SM.isWrittenInSameFile(R.getBegin(), R.getEnd()))
    return R;
  // Record the stack of expansion locations for the beginning, keyed by FileID.
  llvm::DenseMap<FileID, SourceLocation> BeginExpansions;
  for (SourceLocation Begin = R.getBegin(); Begin.isValid();
       Begin = Begin.isFileID()
                   ? includeHashLoc(SM.getFileID(Begin), SM)
                   : SM.getImmediateExpansionRange(Begin).getBegin()) {
    BeginExpansions[SM.getFileID(Begin)] = Begin;
  }
  // Move up the stack of expansion locations for the end until we find the
  // location in BeginExpansions with that has the same file id.
  for (SourceLocation End = R.getEnd(); End.isValid();
       End = End.isFileID() ? includeHashLoc(SM.getFileID(End), SM)
                            : toTokenRange(SM.getImmediateExpansionRange(End),
                                           SM, LangOpts)
                                  .getEnd()) {
    auto It = BeginExpansions.find(SM.getFileID(End));
    if (It != BeginExpansions.end()) {
      if (SM.getFileOffset(It->second) > SM.getFileOffset(End))
        return SourceLocation();
      return {It->second, End};
    }
  }
  return SourceRange();
}

// Find an expansion range (not necessarily immediate) the ends of which are in
// the same file id.
static SourceRange
getExpansionTokenRangeInSameFile(SourceLocation Loc, const SourceManager &SM,
                                 const LangOptions &LangOpts) {
  return rangeInCommonFile(
      toTokenRange(SM.getImmediateExpansionRange(Loc), SM, LangOpts), SM,
      LangOpts);
}

// Returns the file range for a given Location as a Token Range
// This is quite similar to getFileLoc in SourceManager as both use
// getImmediateExpansionRange and getImmediateSpellingLoc (for macro IDs).
// However:
// - We want to maintain the full range information as we move from one file to
//   the next. getFileLoc only uses the BeginLoc of getImmediateExpansionRange.
// - We want to split '>>' tokens as the lexer parses the '>>' in nested
//   template instantiations as a '>>' instead of two '>'s.
// There is also getExpansionRange but it simply calls
// getImmediateExpansionRange on the begin and ends separately which is wrong.
static SourceRange getTokenFileRange(SourceLocation Loc,
                                     const SourceManager &SM,
                                     const LangOptions &LangOpts) {
  SourceRange FileRange = Loc;
  while (!FileRange.getBegin().isFileID()) {
    if (SM.isMacroArgExpansion(FileRange.getBegin())) {
      FileRange = unionTokenRange(
          SM.getImmediateSpellingLoc(FileRange.getBegin()),
          SM.getImmediateSpellingLoc(FileRange.getEnd()), SM, LangOpts);
      assert(SM.isWrittenInSameFile(FileRange.getBegin(), FileRange.getEnd()));
    } else {
      SourceRange ExpansionRangeForBegin =
          getExpansionTokenRangeInSameFile(FileRange.getBegin(), SM, LangOpts);
      SourceRange ExpansionRangeForEnd =
          getExpansionTokenRangeInSameFile(FileRange.getEnd(), SM, LangOpts);
      if (ExpansionRangeForBegin.isInvalid() ||
          ExpansionRangeForEnd.isInvalid())
        return SourceRange();
      assert(SM.isWrittenInSameFile(ExpansionRangeForBegin.getBegin(),
                                    ExpansionRangeForEnd.getBegin()) &&
             "Both Expansion ranges should be in same file.");
      FileRange = unionTokenRange(ExpansionRangeForBegin, ExpansionRangeForEnd,
                                  SM, LangOpts);
    }
  }
  return FileRange;
}

bool isInsideMainFile(SourceLocation Loc, const SourceManager &SM) {
  return Loc.isValid() && SM.isWrittenInMainFile(SM.getExpansionLoc(Loc));
}

llvm::Optional<SourceRange> toHalfOpenFileRange(const SourceManager &SM,
                                                const LangOptions &LangOpts,
                                                SourceRange R) {
  SourceRange R1 = getTokenFileRange(R.getBegin(), SM, LangOpts);
  if (!isValidFileRange(SM, R1))
    return llvm::None;

  SourceRange R2 = getTokenFileRange(R.getEnd(), SM, LangOpts);
  if (!isValidFileRange(SM, R2))
    return llvm::None;

  SourceRange Result =
      rangeInCommonFile(unionTokenRange(R1, R2, SM, LangOpts), SM, LangOpts);
  unsigned TokLen = getTokenLengthAtLoc(Result.getEnd(), SM, LangOpts);
  // Convert from closed token range to half-open (char) range
  Result.setEnd(Result.getEnd().getLocWithOffset(TokLen));
  if (!isValidFileRange(SM, Result))
    return llvm::None;

  return Result;
}

llvm::StringRef toSourceCode(const SourceManager &SM, SourceRange R) {
  assert(isValidFileRange(SM, R));
  bool Invalid = false;
  auto *Buf = SM.getBuffer(SM.getFileID(R.getBegin()), &Invalid);
  assert(!Invalid);

  size_t BeginOffset = SM.getFileOffset(R.getBegin());
  size_t EndOffset = SM.getFileOffset(R.getEnd());
  return Buf->getBuffer().substr(BeginOffset, EndOffset - BeginOffset);
}

llvm::Expected<SourceLocation> sourceLocationInMainFile(const SourceManager &SM,
                                                        Position P) {
  llvm::StringRef Code = SM.getBuffer(SM.getMainFileID())->getBuffer();
  auto Offset =
      positionToOffset(Code, P, /*AllowColumnBeyondLineLength=*/false);
  if (!Offset)
    return Offset.takeError();
  return SM.getLocForStartOfFile(SM.getMainFileID()).getLocWithOffset(*Offset);
}

Range halfOpenToRange(const SourceManager &SM, CharSourceRange R) {
  // Clang is 1-based, LSP uses 0-based indexes.
  Position Begin = sourceLocToPosition(SM, R.getBegin());
  Position End = sourceLocToPosition(SM, R.getEnd());

  return {Begin, End};
}

std::pair<size_t, size_t> offsetToClangLineColumn(llvm::StringRef Code,
                                                  size_t Offset) {
  Offset = std::min(Code.size(), Offset);
  llvm::StringRef Before = Code.substr(0, Offset);
  int Lines = Before.count('\n');
  size_t PrevNL = Before.rfind('\n');
  size_t StartOfLine = (PrevNL == llvm::StringRef::npos) ? 0 : (PrevNL + 1);
  return {Lines + 1, Offset - StartOfLine + 1};
}

std::pair<StringRef, StringRef> splitQualifiedName(StringRef QName) {
  size_t Pos = QName.rfind("::");
  if (Pos == llvm::StringRef::npos)
    return {llvm::StringRef(), QName};
  return {QName.substr(0, Pos + 2), QName.substr(Pos + 2)};
}

TextEdit replacementToEdit(llvm::StringRef Code,
                           const tooling::Replacement &R) {
  Range ReplacementRange = {
      offsetToPosition(Code, R.getOffset()),
      offsetToPosition(Code, R.getOffset() + R.getLength())};
  return {ReplacementRange, R.getReplacementText()};
}

std::vector<TextEdit> replacementsToEdits(llvm::StringRef Code,
                                          const tooling::Replacements &Repls) {
  std::vector<TextEdit> Edits;
  for (const auto &R : Repls)
    Edits.push_back(replacementToEdit(Code, R));
  return Edits;
}

llvm::Optional<std::string> getCanonicalPath(const FileEntry *F,
                                             const SourceManager &SourceMgr) {
  if (!F)
    return None;

  llvm::SmallString<128> FilePath = F->getName();
  if (!llvm::sys::path::is_absolute(FilePath)) {
    if (auto EC =
            SourceMgr.getFileManager().getVirtualFileSystem().makeAbsolute(
                FilePath)) {
      elog("Could not turn relative path '{0}' to absolute: {1}", FilePath,
           EC.message());
      return None;
    }
  }

  // Handle the symbolic link path case where the current working directory
  // (getCurrentWorkingDirectory) is a symlink. We always want to the real
  // file path (instead of the symlink path) for the  C++ symbols.
  //
  // Consider the following example:
  //
  //   src dir: /project/src/foo.h
  //   current working directory (symlink): /tmp/build -> /project/src/
  //
  //  The file path of Symbol is "/project/src/foo.h" instead of
  //  "/tmp/build/foo.h"
  if (auto Dir = SourceMgr.getFileManager().getDirectory(
          llvm::sys::path::parent_path(FilePath))) {
    llvm::SmallString<128> RealPath;
    llvm::StringRef DirName = SourceMgr.getFileManager().getCanonicalName(*Dir);
    llvm::sys::path::append(RealPath, DirName,
                            llvm::sys::path::filename(FilePath));
    return RealPath.str().str();
  }

  return FilePath.str().str();
}

TextEdit toTextEdit(const FixItHint &FixIt, const SourceManager &M,
                    const LangOptions &L) {
  TextEdit Result;
  Result.range =
      halfOpenToRange(M, Lexer::makeFileCharRange(FixIt.RemoveRange, M, L));
  Result.newText = FixIt.CodeToInsert;
  return Result;
}

bool isRangeConsecutive(const Range &Left, const Range &Right) {
  return Left.end.line == Right.start.line &&
         Left.end.character == Right.start.character;
}

FileDigest digest(llvm::StringRef Content) {
  uint64_t Hash{llvm::xxHash64(Content)};
  FileDigest Result;
  for (unsigned I = 0; I < Result.size(); ++I) {
    Result[I] = uint8_t(Hash);
    Hash >>= 8;
  }
  return Result;
}

llvm::Optional<FileDigest> digestFile(const SourceManager &SM, FileID FID) {
  bool Invalid = false;
  llvm::StringRef Content = SM.getBufferData(FID, &Invalid);
  if (Invalid)
    return None;
  return digest(Content);
}

format::FormatStyle getFormatStyleForFile(llvm::StringRef File,
                                          llvm::StringRef Content,
                                          llvm::vfs::FileSystem *FS) {
  auto Style = format::getStyle(format::DefaultFormatStyle, File,
                                format::DefaultFallbackStyle, Content, FS);
  if (!Style) {
    log("getStyle() failed for file {0}: {1}. Fallback is LLVM style.", File,
        Style.takeError());
    Style = format::getLLVMStyle();
  }
  return *Style;
}

llvm::Expected<tooling::Replacements>
cleanupAndFormat(StringRef Code, const tooling::Replacements &Replaces,
                 const format::FormatStyle &Style) {
  auto CleanReplaces = cleanupAroundReplacements(Code, Replaces, Style);
  if (!CleanReplaces)
    return CleanReplaces;
  return formatReplacements(Code, std::move(*CleanReplaces), Style);
}

static void
lex(llvm::StringRef Code, const LangOptions &LangOpts,
    llvm::function_ref<void(const clang::Token &, const SourceManager &SM)>
        Action) {
  // FIXME: InMemoryFileAdapter crashes unless the buffer is null terminated!
  std::string NullTerminatedCode = Code.str();
  SourceManagerForFile FileSM("dummy.cpp", NullTerminatedCode);
  auto &SM = FileSM.get();
  auto FID = SM.getMainFileID();
  // Create a raw lexer (with no associated preprocessor object).
  Lexer Lex(FID, SM.getBuffer(FID), SM, LangOpts);
  Token Tok;

  while (!Lex.LexFromRawLexer(Tok))
    Action(Tok, SM);
  // LexFromRawLexer returns true after it lexes last token, so we still have
  // one more token to report.
  Action(Tok, SM);
}

llvm::StringMap<unsigned> collectIdentifiers(llvm::StringRef Content,
                                             const format::FormatStyle &Style) {
  llvm::StringMap<unsigned> Identifiers;
  auto LangOpt = format::getFormattingLangOpts(Style);
  lex(Content, LangOpt, [&](const clang::Token &Tok, const SourceManager &) {
    if (Tok.getKind() == tok::raw_identifier)
      ++Identifiers[Tok.getRawIdentifier()];
  });
  return Identifiers;
}

std::vector<Range> collectIdentifierRanges(llvm::StringRef Identifier,
                                           llvm::StringRef Content,
                                           const LangOptions &LangOpts) {
  std::vector<Range> Ranges;
  lex(Content, LangOpts, [&](const clang::Token &Tok, const SourceManager &SM) {
    if (Tok.getKind() != tok::raw_identifier)
      return;
    if (Tok.getRawIdentifier() != Identifier)
      return;
    auto Range = getTokenRange(SM, LangOpts, Tok.getLocation());
    if (!Range)
      return;
    Ranges.push_back(*Range);
  });
  return Ranges;
}

namespace {
struct NamespaceEvent {
  enum {
    BeginNamespace, // namespace <ns> {.     Payload is resolved <ns>.
    EndNamespace,   // } // namespace <ns>.  Payload is resolved *outer*
                    // namespace.
    UsingDirective  // using namespace <ns>. Payload is unresolved <ns>.
  } Trigger;
  std::string Payload;
  Position Pos;
};
// Scans C++ source code for constructs that change the visible namespaces.
void parseNamespaceEvents(llvm::StringRef Code,
                          const format::FormatStyle &Style,
                          llvm::function_ref<void(NamespaceEvent)> Callback) {

  // Stack of enclosing namespaces, e.g. {"clang", "clangd"}
  std::vector<std::string> Enclosing; // Contains e.g. "clang", "clangd"
  // Stack counts open braces. true if the brace opened a namespace.
  std::vector<bool> BraceStack;

  enum {
    Default,
    Namespace,          // just saw 'namespace'
    NamespaceName,      // just saw 'namespace' NSName
    Using,              // just saw 'using'
    UsingNamespace,     // just saw 'using namespace'
    UsingNamespaceName, // just saw 'using namespace' NSName
  } State = Default;
  std::string NSName;

  NamespaceEvent Event;
  lex(Code, format::getFormattingLangOpts(Style),
      [&](const clang::Token &Tok,const SourceManager &SM) {
    Event.Pos = sourceLocToPosition(SM, Tok.getLocation());
    switch (Tok.getKind()) {
    case tok::raw_identifier:
      // In raw mode, this could be a keyword or a name.
      switch (State) {
      case UsingNamespace:
      case UsingNamespaceName:
        NSName.append(Tok.getRawIdentifier());
        State = UsingNamespaceName;
        break;
      case Namespace:
      case NamespaceName:
        NSName.append(Tok.getRawIdentifier());
        State = NamespaceName;
        break;
      case Using:
        State =
            (Tok.getRawIdentifier() == "namespace") ? UsingNamespace : Default;
        break;
      case Default:
        NSName.clear();
        if (Tok.getRawIdentifier() == "namespace")
          State = Namespace;
        else if (Tok.getRawIdentifier() == "using")
          State = Using;
        break;
      }
      break;
    case tok::coloncolon:
      // This can come at the beginning or in the middle of a namespace name.
      switch (State) {
      case UsingNamespace:
      case UsingNamespaceName:
        NSName.append("::");
        State = UsingNamespaceName;
        break;
      case NamespaceName:
        NSName.append("::");
        State = NamespaceName;
        break;
      case Namespace: // Not legal here.
      case Using:
      case Default:
        State = Default;
        break;
      }
      break;
    case tok::l_brace:
      // Record which { started a namespace, so we know when } ends one.
      if (State == NamespaceName) {
        // Parsed: namespace <name> {
        BraceStack.push_back(true);
        Enclosing.push_back(NSName);
        Event.Trigger = NamespaceEvent::BeginNamespace;
        Event.Payload = llvm::join(Enclosing, "::");
        Callback(Event);
      } else {
        // This case includes anonymous namespaces (State = Namespace).
        // For our purposes, they're not namespaces and we ignore them.
        BraceStack.push_back(false);
      }
      State = Default;
      break;
    case tok::r_brace:
      // If braces are unmatched, we're going to be confused, but don't crash.
      if (!BraceStack.empty()) {
        if (BraceStack.back()) {
          // Parsed: } // namespace
          Enclosing.pop_back();
          Event.Trigger = NamespaceEvent::EndNamespace;
          Event.Payload = llvm::join(Enclosing, "::");
          Callback(Event);
        }
        BraceStack.pop_back();
      }
      break;
    case tok::semi:
      if (State == UsingNamespaceName) {
        // Parsed: using namespace <name> ;
        Event.Trigger = NamespaceEvent::UsingDirective;
        Event.Payload = std::move(NSName);
        Callback(Event);
      }
      State = Default;
      break;
    default:
      State = Default;
      break;
    }
  });
}

// Returns the prefix namespaces of NS: {"" ... NS}.
llvm::SmallVector<llvm::StringRef, 8> ancestorNamespaces(llvm::StringRef NS) {
  llvm::SmallVector<llvm::StringRef, 8> Results;
  Results.push_back(NS.take_front(0));
  NS.split(Results, "::", /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (llvm::StringRef &R : Results)
    R = NS.take_front(R.end() - NS.begin());
  return Results;
}

} // namespace

std::vector<std::string> visibleNamespaces(llvm::StringRef Code,
                                           const format::FormatStyle &Style) {
  std::string Current;
  // Map from namespace to (resolved) namespaces introduced via using directive.
  llvm::StringMap<llvm::StringSet<>> UsingDirectives;

  parseNamespaceEvents(Code, Style, [&](NamespaceEvent Event) {
    llvm::StringRef NS = Event.Payload;
    switch (Event.Trigger) {
    case NamespaceEvent::BeginNamespace:
    case NamespaceEvent::EndNamespace:
      Current = std::move(Event.Payload);
      break;
    case NamespaceEvent::UsingDirective:
      if (NS.consume_front("::"))
        UsingDirectives[Current].insert(NS);
      else {
        for (llvm::StringRef Enclosing : ancestorNamespaces(Current)) {
          if (Enclosing.empty())
            UsingDirectives[Current].insert(NS);
          else
            UsingDirectives[Current].insert((Enclosing + "::" + NS).str());
        }
      }
      break;
    }
  });

  std::vector<std::string> Found;
  for (llvm::StringRef Enclosing : ancestorNamespaces(Current)) {
    Found.push_back(Enclosing);
    auto It = UsingDirectives.find(Enclosing);
    if (It != UsingDirectives.end())
      for (const auto &Used : It->second)
        Found.push_back(Used.getKey());
  }

  llvm::sort(Found, [&](const std::string &LHS, const std::string &RHS) {
    if (Current == RHS)
      return false;
    if (Current == LHS)
      return true;
    return LHS < RHS;
  });
  Found.erase(std::unique(Found.begin(), Found.end()), Found.end());
  return Found;
}

llvm::StringSet<> collectWords(llvm::StringRef Content) {
  // We assume short words are not significant.
  // We may want to consider other stopwords, e.g. language keywords.
  // (A very naive implementation showed no benefit, but lexing might do better)
  static constexpr int MinWordLength = 4;

  std::vector<CharRole> Roles(Content.size());
  calculateRoles(Content, Roles);

  llvm::StringSet<> Result;
  llvm::SmallString<256> Word;
  auto Flush = [&] {
    if (Word.size() >= MinWordLength) {
      for (char &C : Word)
        C = llvm::toLower(C);
      Result.insert(Word);
    }
    Word.clear();
  };
  for (unsigned I = 0; I < Content.size(); ++I) {
    switch (Roles[I]) {
    case Head:
      Flush();
      LLVM_FALLTHROUGH;
    case Tail:
      Word.push_back(Content[I]);
      break;
    case Unknown:
    case Separator:
      Flush();
      break;
    }
  }
  Flush();

  return Result;
}

llvm::Optional<DefinedMacro> locateMacroAt(SourceLocation Loc,
                                           Preprocessor &PP) {
  const auto &SM = PP.getSourceManager();
  const auto &LangOpts = PP.getLangOpts();
  Token Result;
  if (Lexer::getRawToken(SM.getSpellingLoc(Loc), Result, SM, LangOpts, false))
    return None;
  if (Result.is(tok::raw_identifier))
    PP.LookUpIdentifierInfo(Result);
  IdentifierInfo *IdentifierInfo = Result.getIdentifierInfo();
  if (!IdentifierInfo || !IdentifierInfo->hadMacroDefinition())
    return None;

  std::pair<FileID, unsigned int> DecLoc = SM.getDecomposedExpansionLoc(Loc);
  // Get the definition just before the searched location so that a macro
  // referenced in a '#undef MACRO' can still be found.
  SourceLocation BeforeSearchedLocation =
      SM.getMacroArgExpandedLocation(SM.getLocForStartOfFile(DecLoc.first)
                                         .getLocWithOffset(DecLoc.second - 1));
  MacroDefinition MacroDef =
      PP.getMacroDefinitionAtLoc(IdentifierInfo, BeforeSearchedLocation);
  if (auto *MI = MacroDef.getMacroInfo())
    return DefinedMacro{IdentifierInfo->getName(), MI};
  return None;
}

llvm::Expected<std::string> Edit::apply() const {
  return tooling::applyAllReplacements(InitialCode, Replacements);
}

std::vector<TextEdit> Edit::asTextEdits() const {
  return replacementsToEdits(InitialCode, Replacements);
}

bool Edit::canApplyTo(llvm::StringRef Code) const {
  // Create line iterators, since line numbers are important while applying our
  // edit we cannot skip blank lines.
  auto LHS = llvm::MemoryBuffer::getMemBuffer(Code);
  llvm::line_iterator LHSIt(*LHS, /*SkipBlanks=*/false);

  auto RHS = llvm::MemoryBuffer::getMemBuffer(InitialCode);
  llvm::line_iterator RHSIt(*RHS, /*SkipBlanks=*/false);

  // Compare the InitialCode we prepared the edit for with the Code we received
  // line by line to make sure there are no differences.
  // FIXME: This check is too conservative now, it should be enough to only
  // check lines around the replacements contained inside the Edit.
  while (!LHSIt.is_at_eof() && !RHSIt.is_at_eof()) {
    if (*LHSIt != *RHSIt)
      return false;
    ++LHSIt;
    ++RHSIt;
  }

  // After we reach EOF for any of the files we make sure the other one doesn't
  // contain any additional content except empty lines, they should not
  // interfere with the edit we produced.
  while (!LHSIt.is_at_eof()) {
    if (!LHSIt->empty())
      return false;
    ++LHSIt;
  }
  while (!RHSIt.is_at_eof()) {
    if (!RHSIt->empty())
      return false;
    ++RHSIt;
  }
  return true;
}

llvm::Error reformatEdit(Edit &E, const format::FormatStyle &Style) {
  if (auto NewEdits = cleanupAndFormat(E.InitialCode, E.Replacements, Style))
    E.Replacements = std::move(*NewEdits);
  else
    return NewEdits.takeError();
  return llvm::Error::success();
}

EligibleRegion getEligiblePoints(llvm::StringRef Code,
                                 llvm::StringRef FullyQualifiedName,
                                 const format::FormatStyle &Style) {
  EligibleRegion ER;
  // Start with global namespace.
  std::vector<std::string> Enclosing = {""};
  // FIXME: In addition to namespaces try to generate events for function
  // definitions as well. One might use a closing parantheses(")" followed by an
  // opening brace "{" to trigger the start.
  parseNamespaceEvents(Code, Style, [&](NamespaceEvent Event) {
    // Using Directives only introduces declarations to current scope, they do
    // not change the current namespace, so skip them.
    if (Event.Trigger == NamespaceEvent::UsingDirective)
      return;
    // Do not qualify the global namespace.
    if (!Event.Payload.empty())
      Event.Payload.append("::");

    std::string CurrentNamespace;
    if (Event.Trigger == NamespaceEvent::BeginNamespace) {
      Enclosing.emplace_back(std::move(Event.Payload));
      CurrentNamespace = Enclosing.back();
      // parseNameSpaceEvents reports the beginning position of a token; we want
      // to insert after '{', so increment by one.
      ++Event.Pos.character;
    } else {
      // Event.Payload points to outer namespace when exiting a scope, so use
      // the namespace we've last entered instead.
      CurrentNamespace = std::move(Enclosing.back());
      Enclosing.pop_back();
      assert(Enclosing.back() == Event.Payload);
    }

    // Ignore namespaces that are not a prefix of the target.
    if (!FullyQualifiedName.startswith(CurrentNamespace))
      return;

    // Prefer the namespace that shares the longest prefix with target.
    if (CurrentNamespace.size() > ER.EnclosingNamespace.size()) {
      ER.EligiblePoints.clear();
      ER.EnclosingNamespace = CurrentNamespace;
    }
    if (CurrentNamespace.size() == ER.EnclosingNamespace.size())
      ER.EligiblePoints.emplace_back(std::move(Event.Pos));
  });
  // If there were no shared namespaces just return EOF.
  if (ER.EligiblePoints.empty()) {
    assert(ER.EnclosingNamespace.empty());
    ER.EligiblePoints.emplace_back(offsetToPosition(Code, Code.size()));
  }
  return ER;
}

bool isHeaderFile(llvm::StringRef FileName,
                  llvm::Optional<LangOptions> LangOpts) {
  // Respect the langOpts, for non-file-extension cases, e.g. standard library
  // files.
  if (LangOpts && LangOpts->IsHeaderFile)
    return true;
  namespace types = clang::driver::types;
  auto Lang = types::lookupTypeForExtension(
      llvm::sys::path::extension(FileName).substr(1));
  return Lang != types::TY_INVALID && types::onlyPrecompileType(Lang);
}

} // namespace clangd
} // namespace clang
