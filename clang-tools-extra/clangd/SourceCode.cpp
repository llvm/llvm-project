//===--- SourceCode.h - Manipulating source code as strings -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "SourceCode.h"

#include "Logger.h"
#include "clang/AST/ASTContext.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Path.h"

using namespace llvm;
namespace clang {
namespace clangd {

// Here be dragons. LSP positions use columns measured in *UTF-16 code units*!
// Clangd uses UTF-8 and byte-offsets internally, so conversion is nontrivial.

// Iterates over unicode codepoints in the (UTF-8) string. For each,
// invokes CB(UTF-8 length, UTF-16 length), and breaks if it returns true.
// Returns true if CB returned true, false if we hit the end of string.
template <typename Callback>
static bool iterateCodepoints(StringRef U8, const Callback &CB) {
  for (size_t I = 0; I < U8.size();) {
    unsigned char C = static_cast<unsigned char>(U8[I]);
    if (LLVM_LIKELY(!(C & 0x80))) { // ASCII character.
      if (CB(1, 1))
        return true;
      ++I;
      continue;
    }
    // This convenient property of UTF-8 holds for all non-ASCII characters.
    size_t UTF8Length = countLeadingOnes(C);
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

// Returns the offset into the string that matches \p Units UTF-16 code units.
// Conceptually, this converts to UTF-16, truncates to CodeUnits, converts back
// to UTF-8, and returns the length in bytes.
static size_t measureUTF16(StringRef U8, int U16Units, bool &Valid) {
  size_t Result = 0;
  Valid = U16Units == 0 || iterateCodepoints(U8, [&](int U8Len, int U16Len) {
            Result += U8Len;
            U16Units -= U16Len;
            return U16Units <= 0;
          });
  if (U16Units < 0) // Offset was into the middle of a surrogate pair.
    Valid = false;
  // Don't return an out-of-range index if we overran.
  return std::min(Result, U8.size());
}

// Like most strings in clangd, the input is UTF-8 encoded.
size_t lspLength(StringRef Code) {
  // A codepoint takes two UTF-16 code unit if it's astral (outside BMP).
  // Astral codepoints are encoded as 4 bytes in UTF-8, starting with 11110xxx.
  size_t Count = 0;
  iterateCodepoints(Code, [&](int U8Len, int U16Len) {
    Count += U16Len;
    return false;
  });
  return Count;
}

Expected<size_t> positionToOffset(StringRef Code, Position P,
                                  bool AllowColumnsBeyondLineLength) {
  if (P.line < 0)
    return make_error<StringError>(
        formatv("Line value can't be negative ({0})", P.line),
        errc::invalid_argument);
  if (P.character < 0)
    return make_error<StringError>(
        formatv("Character value can't be negative ({0})", P.character),
        errc::invalid_argument);
  size_t StartOfLine = 0;
  for (int I = 0; I != P.line; ++I) {
    size_t NextNL = Code.find('\n', StartOfLine);
    if (NextNL == StringRef::npos)
      return make_error<StringError>(
          formatv("Line value is out of range ({0})", P.line),
          errc::invalid_argument);
    StartOfLine = NextNL + 1;
  }

  size_t NextNL = Code.find('\n', StartOfLine);
  if (NextNL == StringRef::npos)
    NextNL = Code.size();

  bool Valid;
  size_t ByteOffsetInLine = measureUTF16(
      Code.substr(StartOfLine, NextNL - StartOfLine), P.character, Valid);
  if (!Valid && !AllowColumnsBeyondLineLength)
    return make_error<StringError>(
        formatv("UTF-16 offset {0} is invalid for line {1}", P.character,
                P.line),
        errc::invalid_argument);
  return StartOfLine + ByteOffsetInLine;
}

Position offsetToPosition(StringRef Code, size_t Offset) {
  Offset = std::min(Code.size(), Offset);
  StringRef Before = Code.substr(0, Offset);
  int Lines = Before.count('\n');
  size_t PrevNL = Before.rfind('\n');
  size_t StartOfLine = (PrevNL == StringRef::npos) ? 0 : (PrevNL + 1);
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
  StringRef Code = SM.getBufferData(FID, &Invalid);
  if (!Invalid) {
    auto ColumnInBytes = SM.getColumnNumber(FID, Offset) - 1;
    auto LineSoFar = Code.substr(Offset - ColumnInBytes, ColumnInBytes);
    P.character = lspLength(LineSoFar);
  }
  return P;
}

Range halfOpenToRange(const SourceManager &SM, CharSourceRange R) {
  // Clang is 1-based, LSP uses 0-based indexes.
  Position Begin = sourceLocToPosition(SM, R.getBegin());
  Position End = sourceLocToPosition(SM, R.getEnd());

  return {Begin, End};
}

std::pair<size_t, size_t> offsetToClangLineColumn(StringRef Code,
                                                  size_t Offset) {
  Offset = std::min(Code.size(), Offset);
  StringRef Before = Code.substr(0, Offset);
  int Lines = Before.count('\n');
  size_t PrevNL = Before.rfind('\n');
  size_t StartOfLine = (PrevNL == StringRef::npos) ? 0 : (PrevNL + 1);
  return {Lines + 1, Offset - StartOfLine + 1};
}

std::pair<StringRef, StringRef> splitQualifiedName(StringRef QName) {
  size_t Pos = QName.rfind("::");
  if (Pos == StringRef::npos)
    return {StringRef(), QName};
  return {QName.substr(0, Pos + 2), QName.substr(Pos + 2)};
}

TextEdit replacementToEdit(StringRef Code, const tooling::Replacement &R) {
  Range ReplacementRange = {
      offsetToPosition(Code, R.getOffset()),
      offsetToPosition(Code, R.getOffset() + R.getLength())};
  return {ReplacementRange, R.getReplacementText()};
}

std::vector<TextEdit> replacementsToEdits(StringRef Code,
                                          const tooling::Replacements &Repls) {
  std::vector<TextEdit> Edits;
  for (const auto &R : Repls)
    Edits.push_back(replacementToEdit(Code, R));
  return Edits;
}

Optional<std::string> getCanonicalPath(const FileEntry *F,
                                       const SourceManager &SourceMgr) {
  if (!F)
    return None;
  // Ideally, we get the real path from the FileEntry object.
  SmallString<128> FilePath = F->tryGetRealPathName();
  if (!FilePath.empty() && sys::path::is_absolute(FilePath))
    return FilePath.str().str();

  // Otherwise, we try to compute ourselves.
  FilePath = F->getName();
  vlog("FileEntry for {0} did not contain the real path.", FilePath);

  if (!sys::path::is_absolute(FilePath)) {
    if (auto EC =
            SourceMgr.getFileManager().getVirtualFileSystem()->makeAbsolute(
                FilePath)) {
      elog("Could not turn relative path '{0}' to absolute: {1}", FilePath,
           EC.message());
      return None;
    }
  }

  // Handle the symbolic link path case where the current working directory
  // (getCurrentWorkingDirectory) is a symlink./ We always want to the real
  // file path (instead of the symlink path) for the  C++ symbols.
  //
  // Consider the following example:
  //
  //   src dir: /project/src/foo.h
  //   current working directory (symlink): /tmp/build -> /project/src/
  //
  //  The file path of Symbol is "/project/src/foo.h" instead of
  //  "/tmp/build/foo.h"
  if (const DirectoryEntry *Dir = SourceMgr.getFileManager().getDirectory(
          sys::path::parent_path(FilePath))) {
    SmallString<128> RealPath;
    StringRef DirName = SourceMgr.getFileManager().getCanonicalName(Dir);
    sys::path::append(RealPath, DirName, sys::path::filename(FilePath));
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

bool IsRangeConsecutive(const Range &Left, const Range &Right) {
  return Left.end.line == Right.start.line &&
         Left.end.character == Right.start.character;
}

FileDigest digest(StringRef Content) {
  return llvm::SHA1::hash({(const uint8_t *)Content.data(), Content.size()});
}

Optional<FileDigest> digestFile(const SourceManager &SM, FileID FID) {
  bool Invalid = false;
  StringRef Content = SM.getBufferData(FID, &Invalid);
  if (Invalid)
    return None;
  return digest(Content);
}

} // namespace clangd
} // namespace clang
