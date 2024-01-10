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

SymbolName::SymbolName() : NamePieces({}) {}

SymbolName::SymbolName(const DeclarationName &DeclName)
    : SymbolName(DeclName.getAsString(),
                 /*IsObjectiveCSelector=*/DeclName.getNameKind() ==
                         DeclarationName::NameKind::ObjCMultiArgSelector ||
                     DeclName.getNameKind() ==
                         DeclarationName::NameKind::ObjCOneArgSelector) {}

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

SymbolName::SymbolName(ArrayRef<std::string> NamePieces) {
  for (const auto &Piece : NamePieces)
    this->NamePieces.push_back(Piece);
}

std::optional<std::string> SymbolName::getSinglePiece() const {
  if (getNamePieces().size() == 1) {
    return NamePieces.front();
  }
  return std::nullopt;
}

std::string SymbolName::getAsString() const {
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  this->print(OS);
  return Result;
}

void SymbolName::print(raw_ostream &OS) const {
  llvm::interleave(NamePieces, OS, ":");
}

} // end namespace tooling
} // end namespace clang
