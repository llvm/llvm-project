//===-- IndexHelpers.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestIndex.h"

namespace clang {
namespace clangd {

Symbol symbol(llvm::StringRef QName) {
  Symbol Sym;
  Sym.ID = SymbolID(QName.str());
  size_t Pos = QName.rfind("::");
  if (Pos == llvm::StringRef::npos) {
    Sym.Name = QName;
    Sym.Scope = "";
  } else {
    Sym.Name = QName.substr(Pos + 2);
    Sym.Scope = QName.substr(0, Pos + 2);
  }
  return Sym;
}

SymbolSlab generateSymbols(std::vector<std::string> QualifiedNames) {
  SymbolSlab::Builder Slab;
  for (llvm::StringRef QName : QualifiedNames)
    Slab.insert(symbol(QName));
  return std::move(Slab).build();
}

SymbolSlab generateNumSymbols(int Begin, int End) {
  std::vector<std::string> Names;
  for (int i = Begin; i <= End; i++)
    Names.push_back(std::to_string(i));
  return generateSymbols(Names);
}

std::string getQualifiedName(const Symbol &Sym) {
  return (Sym.Scope + Sym.Name).str();
}

std::vector<std::string> match(const SymbolIndex &I,
                               const FuzzyFindRequest &Req, bool *Incomplete) {
  std::vector<std::string> Matches;
  bool IsIncomplete = I.fuzzyFind(Req, [&](const Symbol &Sym) {
    Matches.push_back(clang::clangd::getQualifiedName(Sym));
  });
  if (Incomplete)
    *Incomplete = IsIncomplete;
  return Matches;
}

// Returns qualified names of symbols with any of IDs in the index.
std::vector<std::string> lookup(const SymbolIndex &I,
                                llvm::ArrayRef<SymbolID> IDs) {
  LookupRequest Req;
  Req.IDs.insert(IDs.begin(), IDs.end());
  std::vector<std::string> Results;
  I.lookup(Req, [&](const Symbol &Sym) {
    Results.push_back(getQualifiedName(Sym));
  });
  return Results;
}

} // namespace clangd
} // namespace clang
