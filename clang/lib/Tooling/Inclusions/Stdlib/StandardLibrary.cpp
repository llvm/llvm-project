//===--- StandardLibrary.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Inclusions/StandardLibrary.h"
#include "clang/AST/Decl.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

namespace clang {
namespace tooling {
namespace stdlib {

static llvm::StringRef *HeaderNames;
static std::pair<llvm::StringRef, llvm::StringRef> *SymbolNames;
static unsigned *SymbolHeaderIDs;
static llvm::DenseMap<llvm::StringRef, unsigned> *HeaderIDs;
// Maps symbol name -> Symbol::ID, within a namespace.
using NSSymbolMap = llvm::DenseMap<llvm::StringRef, unsigned>;
static llvm::DenseMap<llvm::StringRef, NSSymbolMap *> *NamespaceSymbols;

static int initialize() {
  unsigned SymCount = 0;
#define SYMBOL(Name, NS, Header) ++SymCount;
#include "clang/Tooling/Inclusions/CSymbolMap.inc"
#include "clang/Tooling/Inclusions/StdSymbolMap.inc"
#undef SYMBOL
  SymbolNames = new std::remove_reference_t<decltype(*SymbolNames)>[SymCount];
  SymbolHeaderIDs =
      new std::remove_reference_t<decltype(*SymbolHeaderIDs)>[SymCount];
  NamespaceSymbols = new std::remove_reference_t<decltype(*NamespaceSymbols)>;
  HeaderIDs = new std::remove_reference_t<decltype(*HeaderIDs)>;

  auto AddNS = [&](llvm::StringRef NS) -> NSSymbolMap & {
    auto R = NamespaceSymbols->try_emplace(NS, nullptr);
    if (R.second)
      R.first->second = new NSSymbolMap();
    return *R.first->second;
  };

  auto AddHeader = [&](llvm::StringRef Header) -> unsigned {
    return HeaderIDs->try_emplace(Header, HeaderIDs->size()).first->second;
  };

  auto Add = [&, SymIndex(0)](llvm::StringRef Name, llvm::StringRef NS,
                              llvm::StringRef HeaderName) mutable {
    if (NS == "None")
      NS = "";

    SymbolNames[SymIndex] = {NS, Name};
    SymbolHeaderIDs[SymIndex] = AddHeader(HeaderName);

    NSSymbolMap &NSSymbols = AddNS(NS);
    NSSymbols.try_emplace(Name, SymIndex);

    ++SymIndex;
  };
#define SYMBOL(Name, NS, Header) Add(#Name, #NS, #Header);
#include "clang/Tooling/Inclusions/CSymbolMap.inc"
#include "clang/Tooling/Inclusions/StdSymbolMap.inc"
#undef SYMBOL

  HeaderNames = new llvm::StringRef[HeaderIDs->size()];
  for (const auto &E : *HeaderIDs)
    HeaderNames[E.second] = E.first;

  return 0;
}

static void ensureInitialized() {
  static int Dummy = initialize();
  (void)Dummy;
}

std::optional<Header> Header::named(llvm::StringRef Name) {
  ensureInitialized();
  auto It = HeaderIDs->find(Name);
  if (It == HeaderIDs->end())
    return std::nullopt;
  return Header(It->second);
}
llvm::StringRef Header::name() const { return HeaderNames[ID]; }
llvm::StringRef Symbol::scope() const { return SymbolNames[ID].first; }
llvm::StringRef Symbol::name() const { return SymbolNames[ID].second; }
std::optional<Symbol> Symbol::named(llvm::StringRef Scope,
                                     llvm::StringRef Name) {
  ensureInitialized();
  if (NSSymbolMap *NSSymbols = NamespaceSymbols->lookup(Scope)) {
    auto It = NSSymbols->find(Name);
    if (It != NSSymbols->end())
      return Symbol(It->second);
  }
  return std::nullopt;
}
Header Symbol::header() const { return Header(SymbolHeaderIDs[ID]); }
llvm::SmallVector<Header> Symbol::headers() const {
  return {header()}; // FIXME: multiple in case of ambiguity
}

Recognizer::Recognizer() { ensureInitialized(); }

NSSymbolMap *Recognizer::namespaceSymbols(const NamespaceDecl *D) {
  auto It = NamespaceCache.find(D);
  if (It != NamespaceCache.end())
    return It->second;

  NSSymbolMap *Result = [&]() -> NSSymbolMap * {
    if (D && D->isAnonymousNamespace())
      return nullptr;
    // Print the namespace and its parents ommitting inline scopes.
    std::string Scope;
    for (const auto *ND = D; ND;
         ND = llvm::dyn_cast_or_null<NamespaceDecl>(ND->getParent()))
      if (!ND->isInlineNamespace() && !ND->isAnonymousNamespace())
        Scope = ND->getName().str() + "::" + Scope;
    return NamespaceSymbols->lookup(Scope);
  }();
  NamespaceCache.try_emplace(D, Result);
  return Result;
}

std::optional<Symbol> Recognizer::operator()(const Decl *D) {
  // If D is std::vector::iterator, `vector` is the outer symbol to look up.
  // We keep all the candidate DCs as some may turn out to be anon enums.
  // Do this resolution lazily as we may turn out not to have a std namespace.
  llvm::SmallVector<const DeclContext *> IntermediateDecl;
  const DeclContext *DC = D->getDeclContext();
  while (DC && !DC->isNamespace()) {
    if (NamedDecl::classofKind(DC->getDeclKind()))
      IntermediateDecl.push_back(DC);
    DC = DC->getParent();
  }
  NSSymbolMap *Symbols = namespaceSymbols(cast_or_null<NamespaceDecl>(DC));
  if (!Symbols)
    return std::nullopt;

  llvm::StringRef Name = [&]() -> llvm::StringRef {
    for (const auto *SymDC : llvm::reverse(IntermediateDecl)) {
      DeclarationName N = cast<NamedDecl>(SymDC)->getDeclName();
      if (const auto *II = N.getAsIdentifierInfo())
        return II->getName();
      if (!N.isEmpty())
        return ""; // e.g. operator<: give up
    }
    if (const auto *ND = llvm::dyn_cast<NamedDecl>(D))
      if (const auto *II = ND->getIdentifier())
        return II->getName();
    return "";
  }();
  if (Name.empty())
    return std::nullopt;

  auto It = Symbols->find(Name);
  if (It == Symbols->end())
    return std::nullopt;
  return Symbol(It->second);
}

} // namespace stdlib
} // namespace tooling
} // namespace clang
