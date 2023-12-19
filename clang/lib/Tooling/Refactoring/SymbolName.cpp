//===--- SymbolName.cpp - Clang refactoring library -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Refactoring/Rename/SymbolName.h"
#include "clang/Basic/LangOptions.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace tooling {

SymbolName::SymbolName(StringRef Name, const LangOptions &LangOpts)
    : SymbolName(Name, LangOpts.ObjC) {}

SymbolName::SymbolName(StringRef Name, bool IsObjectiveCSelector) {
  if (!IsObjectiveCSelector) {
    NamePieces.push_back(Name.str());
    return;
  }
  // Decompose an Objective-C selector name into multiple strings.
  do {
    auto StringAndName = Name.split(':');
    NamePieces.push_back(StringAndName.first.str());
    Name = StringAndName.second;
  } while (!Name.empty());
}

SymbolName::SymbolName(ArrayRef<StringRef> NamePieces) {
  for (const auto &Piece : NamePieces)
    this->NamePieces.push_back(Piece.str());
}

void SymbolName::print(raw_ostream &OS) const {
  for (size_t I = 0, E = NamePieces.size(); I != E; ++I) {
    if (I != 0)
      OS << ':';
    OS << NamePieces[I];
  }
}

} // end namespace tooling
} // end namespace clang
