//===--- SourceCode.h - Manipulating source code as strings -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "SourceCode.h"

#include "clang/Basic/SourceManager.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"

namespace clang {
namespace clangd {
using namespace llvm;

llvm::Expected<size_t> positionToOffset(StringRef Code, Position P,
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
    if (NextNL == StringRef::npos)
      return llvm::make_error<llvm::StringError>(
          llvm::formatv("Line value is out of range ({0})", P.line),
          llvm::errc::invalid_argument);
    StartOfLine = NextNL + 1;
  }

  size_t NextNL = Code.find('\n', StartOfLine);
  if (NextNL == StringRef::npos)
    NextNL = Code.size();

  if (StartOfLine + P.character > NextNL && !AllowColumnsBeyondLineLength)
    return llvm::make_error<llvm::StringError>(
        llvm::formatv("Character value is out of range ({0})", P.character),
        llvm::errc::invalid_argument);
  // FIXME: officially P.character counts UTF-16 code units, not UTF-8 bytes!
  return std::min(NextNL, StartOfLine + P.character);
}

Position offsetToPosition(StringRef Code, size_t Offset) {
  Offset = std::min(Code.size(), Offset);
  StringRef Before = Code.substr(0, Offset);
  int Lines = Before.count('\n');
  size_t PrevNL = Before.rfind('\n');
  size_t StartOfLine = (PrevNL == StringRef::npos) ? 0 : (PrevNL + 1);
  // FIXME: officially character counts UTF-16 code units, not UTF-8 bytes!
  Position Pos;
  Pos.line = Lines;
  Pos.character = static_cast<int>(Offset - StartOfLine);
  return Pos;
}

Position sourceLocToPosition(const SourceManager &SM, SourceLocation Loc) {
  Position P;
  P.line = static_cast<int>(SM.getSpellingLineNumber(Loc)) - 1;
  P.character = static_cast<int>(SM.getSpellingColumnNumber(Loc)) - 1;
  return P;
}

Range halfOpenToRange(const SourceManager &SM, CharSourceRange R) {
  // Clang is 1-based, LSP uses 0-based indexes.
  Position Begin = sourceLocToPosition(SM, R.getBegin());
  Position End = sourceLocToPosition(SM, R.getEnd());

  return {Begin, End};
}

} // namespace clangd
} // namespace clang
