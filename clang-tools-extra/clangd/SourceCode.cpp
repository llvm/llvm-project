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

namespace clang {
namespace clangd {
using namespace llvm;

size_t positionToOffset(StringRef Code, Position P) {
  if (P.line < 0)
    return 0;
  size_t StartOfLine = 0;
  for (int I = 0; I != P.line; ++I) {
    size_t NextNL = Code.find('\n', StartOfLine);
    if (NextNL == StringRef::npos)
      return Code.size();
    StartOfLine = NextNL + 1;
  }
  // FIXME: officially P.character counts UTF-16 code units, not UTF-8 bytes!
  return std::min(Code.size(), StartOfLine + std::max(0, P.character));
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

} // namespace clangd
} // namespace clang
