//===--- SymbolLocation.h ----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_SYMBOLLOCATION_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_SYMBOLLOCATION_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>

namespace clang {
namespace clangd {

// Specify a position (Line, Column) of symbol. Using Line/Column allows us to
// build LSP responses without reading the file content.
//
// clangd uses the following definitions, which differ slightly from LSP:
//  - Line is the number of newline characters (\n) before the point.
//  - Column is (by default) the number of UTF-16 code between the last \n
//    (or start of file) and the point.
//    If the `offsetEncoding` protocol extension is used to negotiate UTF-8,
//    then it is instead the number of *bytes* since the last \n.
//
// Position is encoded into 32 bits to save space.
// If Line/Column overflow, the value will be their maximum value.
struct SymbolPosition {
public:
  void setLine(uint32_t Line);
  uint32_t line() const { return LineColumnPacked >> ColumnBits; }
  void setColumn(uint32_t Column);
  uint32_t column() const { return LineColumnPacked & MaxColumn; }
  uint32_t rep() const { return LineColumnPacked; }

  bool hasOverflow() const {
    return line() == MaxLine || column() == MaxColumn;
  }

  static constexpr unsigned ColumnBits = 12;
  static constexpr uint32_t MaxLine = (1 << (32 - ColumnBits)) - 1;
  static constexpr uint32_t MaxColumn = (1 << ColumnBits) - 1;

private:
  uint32_t LineColumnPacked = 0; // Top 20 bit line, bottom 12 bits column.
};

struct SymbolNameLocation {
  /// The symbol range, using half-open range [Start, End).
  SymbolPosition Start;
  SymbolPosition End;

  explicit operator bool() const { return !llvm::StringRef(FileURI).empty(); }

  // The URI of the source file where a symbol occurs.
  // The string must be null-terminated.
  //
  // We avoid using llvm::StringRef here to save memory.
  // WARNING: unless you know what you are doing, it is recommended to use it
  // via llvm::StringRef.
  const char *FileURI = "";
};

struct SymbolDeclDefLocation {
  SymbolNameLocation NameLocation;

  /// The range of the full declaration/definition.
  SymbolPosition DeclDefStart;
  SymbolPosition DeclDefEnd;

  explicit operator bool() const { return NameLocation.operator bool(); }

  const char *fileURI() const { return NameLocation.FileURI; }
};

inline bool operator==(const SymbolPosition &L, const SymbolPosition &R) {
  return std::make_tuple(L.line(), L.column()) ==
         std::make_tuple(R.line(), R.column());
}
inline bool operator<(const SymbolPosition &L, const SymbolPosition &R) {
  return std::make_tuple(L.line(), L.column()) <
         std::make_tuple(R.line(), R.column());
}
inline bool operator==(const SymbolNameLocation &L,
                       const SymbolNameLocation &R) {
  assert(L.FileURI && R.FileURI);
  return !std::strcmp(L.FileURI, R.FileURI) &&
         std::tie(L.Start, L.End) == std::tie(R.Start, R.End);
}
inline bool operator<(const SymbolNameLocation &L,
                      const SymbolNameLocation &R) {
  assert(L.FileURI && R.FileURI);
  int Cmp = std::strcmp(L.FileURI, R.FileURI);
  if (Cmp != 0)
    return Cmp < 0;
  return std::tie(L.Start, L.End) < std::tie(R.Start, R.End);
}
inline bool operator==(const SymbolDeclDefLocation &L,
                       const SymbolDeclDefLocation &R) {
  return std::tie(L.NameLocation, L.DeclDefStart, L.DeclDefEnd) ==
         std::tie(R.NameLocation, R.DeclDefStart, R.DeclDefEnd);
}
inline bool operator<(const SymbolDeclDefLocation &L,
                      const SymbolDeclDefLocation &R) {
  return L.NameLocation < R.NameLocation;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &, const SymbolNameLocation &);

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_SYMBOLLOCATION_H
