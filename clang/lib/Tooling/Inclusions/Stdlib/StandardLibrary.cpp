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

// Header::ID => header name
static llvm::StringRef *HeaderNames;
// Header name => Header::ID
static llvm::DenseMap<llvm::StringRef, unsigned> *HeaderIDs;

static unsigned SymbolCount = 0;
// Symbol::ID => symbol qualified_name/name/scope
static struct SymbolName {
  const char *Data;  // std::vector
  unsigned ScopeLen; // ~~~~~
  unsigned NameLen;  //      ~~~~~~
} *SymbolNames;
// Symbol name -> Symbol::ID, within a namespace.
using NSSymbolMap = llvm::DenseMap<llvm::StringRef, unsigned>;
static llvm::DenseMap<llvm::StringRef, NSSymbolMap *> *NamespaceSymbols;
// Symbol::ID => Header::ID
static unsigned *SymbolHeaderIDs;

static int initialize() {
  SymbolCount = 0;
#define SYMBOL(Name, NS, Header) ++SymbolCount;
#include "clang/Tooling/Inclusions/CSymbolMap.inc"
#include "clang/Tooling/Inclusions/StdSymbolMap.inc"
#undef SYMBOL
  SymbolNames =
      new std::remove_reference_t<decltype(*SymbolNames)>[SymbolCount];
  SymbolHeaderIDs =
      new std::remove_reference_t<decltype(*SymbolHeaderIDs)>[SymbolCount];
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

  auto Add = [&, SymIndex(0)](llvm::StringRef QName, unsigned NSLen,
                              llvm::StringRef HeaderName) mutable {
    // Correct "Nonefoo" => foo.
    // FIXME: get rid of "None" from the generated mapping files.
    if (QName.take_front(NSLen) == "None") {
      QName = QName.drop_front(NSLen);
      NSLen = 0;
    }

    SymbolNames[SymIndex] = {QName.data(), NSLen,
                             static_cast<unsigned int>(QName.size() - NSLen)};
    SymbolHeaderIDs[SymIndex] = AddHeader(HeaderName);

    NSSymbolMap &NSSymbols = AddNS(QName.take_front(NSLen));
    NSSymbols.try_emplace(QName.drop_front(NSLen), SymIndex);

    ++SymIndex;
  };
#define SYMBOL(Name, NS, Header) Add(#NS #Name, strlen(#NS), #Header);
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

std::vector<Header> Header::all() {
  ensureInitialized();
  std::vector<Header> Result;
  Result.reserve(HeaderIDs->size());
  for (unsigned I = 0, E = HeaderIDs->size(); I < E; ++I)
    Result.push_back(Header(I));
  return Result;
}
std::optional<Header> Header::named(llvm::StringRef Name) {
  ensureInitialized();
  auto It = HeaderIDs->find(Name);
  if (It == HeaderIDs->end())
    return std::nullopt;
  return Header(It->second);
}
llvm::StringRef Header::name() const { return HeaderNames[ID]; }

std::vector<Symbol> Symbol::all() {
  ensureInitialized();
  std::vector<Symbol> Result;
  Result.reserve(SymbolCount);
  for (unsigned I = 0, E = SymbolCount; I < E; ++I)
    Result.push_back(Symbol(I));
  return Result;
}
llvm::StringRef Symbol::scope() const {
  SymbolName &S = SymbolNames[ID];
  return StringRef(S.Data, S.ScopeLen);
}
llvm::StringRef Symbol::name() const {
  SymbolName &S = SymbolNames[ID];
  return StringRef(S.Data + S.ScopeLen, S.NameLen);
}
llvm::StringRef Symbol::qualified_name() const {
  SymbolName &S = SymbolNames[ID];
  return StringRef(S.Data, S.ScopeLen + S.NameLen);
}
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
