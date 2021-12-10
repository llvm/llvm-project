//===-- TestIndex.cpp -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestIndex.h"
#include "clang/Index/IndexSymbol.h"
#include "llvm/Support/Regex.h"

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

static std::string replace(llvm::StringRef Haystack, llvm::StringRef Needle,
                           llvm::StringRef Repl) {
  llvm::SmallVector<llvm::StringRef> Parts;
  Haystack.split(Parts, Needle);
  return llvm::join(Parts, Repl);
}

// Helpers to produce fake index symbols for memIndex() or completions().
// USRFormat is a regex replacement string for the unqualified part of the USR.
Symbol sym(llvm::StringRef QName, index::SymbolKind Kind,
           llvm::StringRef USRFormat) {
  Symbol Sym;
  std::string USR = "c:"; // We synthesize a few simple cases of USRs by hand!
  size_t Pos = QName.rfind("::");
  if (Pos == llvm::StringRef::npos) {
    Sym.Name = QName;
    Sym.Scope = "";
  } else {
    Sym.Name = QName.substr(Pos + 2);
    Sym.Scope = QName.substr(0, Pos + 2);
    USR += "@N@" + replace(QName.substr(0, Pos), "::", "@N@"); // ns:: -> @N@ns
  }
  USR += llvm::Regex("^.*$").sub(USRFormat, Sym.Name); // e.g. func -> @F@func#
  Sym.ID = SymbolID(USR);
  Sym.SymInfo.Kind = Kind;
  Sym.Flags |= Symbol::IndexedForCodeCompletion;
  Sym.Origin = SymbolOrigin::Static;
  return Sym;
}

Symbol func(llvm::StringRef Name) { // Assumes the function has no args.
  return sym(Name, index::SymbolKind::Function, "@F@\\0#"); // no args
}

Symbol cls(llvm::StringRef Name) {
  return sym(Name, index::SymbolKind::Class, "@S@\\0");
}

Symbol enm(llvm::StringRef Name) {
  return sym(Name, index::SymbolKind::Enum, "@E@\\0");
}

Symbol var(llvm::StringRef Name) {
  return sym(Name, index::SymbolKind::Variable, "@\\0");
}

Symbol ns(llvm::StringRef Name) {
  return sym(Name, index::SymbolKind::Namespace, "@N@\\0");
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
  return (Sym.Scope + Sym.Name + Sym.TemplateSpecializationArgs).str();
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
