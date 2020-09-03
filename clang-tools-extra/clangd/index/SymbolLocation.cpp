//===--- SymbolLocation.cpp --------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SymbolLocation.h"

namespace clang {
namespace clangd {

constexpr uint32_t SymbolLocation::Position::MaxLine;
constexpr uint32_t SymbolLocation::Position::MaxColumn;

void SymbolLocation::Position::setLine(uint32_t L) {
  if (L > MaxLine)
    L = MaxLine;
  LineColumnPacked = (L << ColumnBits) | column();
}
void SymbolLocation::Position::setColumn(uint32_t Col) {
  if (Col > MaxColumn)
    Col = MaxColumn;
  LineColumnPacked = (LineColumnPacked & ~MaxColumn) | Col;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const SymbolLocation &L) {
  if (!L)
    return OS << "(none)";
  return OS << L.FileURI << "[" << L.Start.line() << ":" << L.Start.column()
            << "-" << L.End.line() << ":" << L.End.column() << ")";
}

} // namespace clangd
} // namespace clang
