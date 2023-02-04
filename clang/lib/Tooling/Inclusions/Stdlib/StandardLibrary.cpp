//===--- StandardLibrary.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Inclusions/StandardLibrary.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/LangOptions.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

namespace clang {
namespace tooling {
namespace stdlib {

namespace {
// Symbol name -> Symbol::ID, within a namespace.
using NSSymbolMap = llvm::DenseMap<llvm::StringRef, unsigned>;

// A Mapping per language.
struct SymbolHeaderMapping {
  llvm::StringRef *HeaderNames = nullptr;
  // Header name => Header::ID
  llvm::DenseMap<llvm::StringRef, unsigned> *HeaderIDs;

  unsigned SymbolCount = 0;
  // Symbol::ID => symbol qualified_name/name/scope
  struct SymbolName {
    const char *Data;  // std::vector
    unsigned ScopeLen; // ~~~~~
    unsigned NameLen;  //      ~~~~~~
    StringRef scope() const { return StringRef(Data, ScopeLen); }
    StringRef name() const { return StringRef(Data + ScopeLen, NameLen); }
    StringRef qualifiedName() const {
      return StringRef(Data, ScopeLen + NameLen);
    }
  } *SymbolNames = nullptr;
  // Symbol name -> Symbol::ID, within a namespace.
  llvm::DenseMap<llvm::StringRef, NSSymbolMap *> *NamespaceSymbols = nullptr;
  // Symbol::ID => Header::ID
  unsigned *SymbolHeaderIDs = nullptr;
};
} // namespace
static SymbolHeaderMapping
    *LanguageMappings[static_cast<unsigned>(Lang::LastValue) + 1];
static const SymbolHeaderMapping *getMappingPerLang(Lang L) {
  return LanguageMappings[static_cast<unsigned>(L)];
}

static int countSymbols(Lang Language) {
  unsigned SymCount = 0;
#define SYMBOL(Name, NS, Header) ++SymCount;
  switch (Language) {
  case Lang::C:
#include "clang/Tooling/Inclusions/CSymbolMap.inc"
    break;
  case Lang::CXX:
#include "clang/Tooling/Inclusions/StdSymbolMap.inc"
    break;
  }
#undef SYMBOL
  return SymCount;
}

static int initialize(Lang Language) {
  SymbolHeaderMapping *Mapping = new SymbolHeaderMapping();
  LanguageMappings[static_cast<unsigned>(Language)] = Mapping;

  unsigned SymCount = countSymbols(Language);
  Mapping->SymbolCount = SymCount;
  Mapping->SymbolNames =
      new std::remove_reference_t<decltype(*Mapping->SymbolNames)>[SymCount];
  Mapping->SymbolHeaderIDs = new std::remove_reference_t<
      decltype(*Mapping->SymbolHeaderIDs)>[SymCount];
  Mapping->NamespaceSymbols =
      new std::remove_reference_t<decltype(*Mapping->NamespaceSymbols)>;
  Mapping->HeaderIDs =
      new std::remove_reference_t<decltype(*Mapping->HeaderIDs)>;
  auto AddNS = [&](llvm::StringRef NS) -> NSSymbolMap & {
    auto R = Mapping->NamespaceSymbols->try_emplace(NS, nullptr);
    if (R.second)
      R.first->second = new NSSymbolMap();
    return *R.first->second;
  };

  auto AddHeader = [&](llvm::StringRef Header) -> unsigned {
    return Mapping->HeaderIDs->try_emplace(Header, Mapping->HeaderIDs->size())
        .first->second;
  };

  auto Add = [&, SymIndex(0)](llvm::StringRef QName, unsigned NSLen,
                              llvm::StringRef HeaderName) mutable {
    // Correct "Nonefoo" => foo.
    // FIXME: get rid of "None" from the generated mapping files.
    if (QName.take_front(NSLen) == "None") {
      QName = QName.drop_front(NSLen);
      NSLen = 0;
    }

    Mapping->SymbolNames[SymIndex] = {
        QName.data(), NSLen, static_cast<unsigned int>(QName.size() - NSLen)};
    Mapping->SymbolHeaderIDs[SymIndex] = AddHeader(HeaderName);

    NSSymbolMap &NSSymbols = AddNS(QName.take_front(NSLen));
    NSSymbols.try_emplace(QName.drop_front(NSLen), SymIndex);

    ++SymIndex;
  };
#define SYMBOL(Name, NS, Header) Add(#NS #Name, strlen(#NS), #Header);
  switch (Language) {
  case Lang::C:
#include "clang/Tooling/Inclusions/CSymbolMap.inc"
    break;
  case Lang::CXX:
#include "clang/Tooling/Inclusions/StdSymbolMap.inc"
    break;
  }
#undef SYMBOL

  Mapping->HeaderNames = new llvm::StringRef[Mapping->HeaderIDs->size()];
  for (const auto &E : *Mapping->HeaderIDs)
    Mapping->HeaderNames[E.second] = E.first;

  return 0;
}

static void ensureInitialized() {
  static int Dummy = []() {
    for (unsigned L = 0; L <= static_cast<unsigned>(Lang::LastValue); ++L)
      initialize(static_cast<Lang>(L));
    return 0;
  }();
  (void)Dummy;
}

std::vector<Header> Header::all(Lang L) {
  ensureInitialized();
  std::vector<Header> Result;
  const auto *Mapping = getMappingPerLang(L);
  Result.reserve(Mapping->HeaderIDs->size());
  for (unsigned I = 0, E = Mapping->HeaderIDs->size(); I < E; ++I)
    Result.push_back(Header(I, L));
  return Result;
}
std::optional<Header> Header::named(llvm::StringRef Name, Lang L) {
  ensureInitialized();
  const auto *Mapping = getMappingPerLang(L);
  auto It = Mapping->HeaderIDs->find(Name);
  if (It == Mapping->HeaderIDs->end())
    return std::nullopt;
  return Header(It->second, L);
}
llvm::StringRef Header::name() const {
  return getMappingPerLang(Language)->HeaderNames[ID];
}

std::vector<Symbol> Symbol::all(Lang L) {
  ensureInitialized();
  std::vector<Symbol> Result;
  const auto *Mapping = getMappingPerLang(L);
  Result.reserve(Mapping->SymbolCount);
  for (unsigned I = 0, E = Mapping->SymbolCount; I < E; ++I)
    Result.push_back(Symbol(I, L));
  return Result;
}
llvm::StringRef Symbol::scope() const {
  return getMappingPerLang(Language)->SymbolNames[ID].scope();
}
llvm::StringRef Symbol::name() const {
  return getMappingPerLang(Language)->SymbolNames[ID].name();
}
llvm::StringRef Symbol::qualifiedName() const {
  return getMappingPerLang(Language)->SymbolNames[ID].qualifiedName();
}
std::optional<Symbol> Symbol::named(llvm::StringRef Scope, llvm::StringRef Name,
                                    Lang L) {
  ensureInitialized();

  if (NSSymbolMap *NSSymbols =
          getMappingPerLang(L)->NamespaceSymbols->lookup(Scope)) {
    auto It = NSSymbols->find(Name);
    if (It != NSSymbols->end())
      return Symbol(It->second, L);
  }
  return std::nullopt;
}
Header Symbol::header() const {
  return Header(getMappingPerLang(Language)->SymbolHeaderIDs[ID], Language);
}
llvm::SmallVector<Header> Symbol::headers() const {
  return {header()}; // FIXME: multiple in case of ambiguity
}

Recognizer::Recognizer() { ensureInitialized(); }

NSSymbolMap *Recognizer::namespaceSymbols(const NamespaceDecl *D) {
  if (!D)
    return nullptr;
  Lang Language;
  if (D->getLangOpts().CPlusPlus)
    Language = Lang::CXX;
  else if (D->getLangOpts().C11)
    Language = Lang::C;
  else
    return nullptr;

  auto It = NamespaceCache.find(D);
  if (It != NamespaceCache.end())
    return It->second;

  NSSymbolMap *Result = [&]() -> NSSymbolMap * {
    if (D->isAnonymousNamespace())
      return nullptr;
    // Print the namespace and its parents ommitting inline scopes.
    std::string Scope;
    for (const auto *ND = D; ND;
         ND = llvm::dyn_cast_or_null<NamespaceDecl>(ND->getParent()))
      if (!ND->isInlineNamespace() && !ND->isAnonymousNamespace())
        Scope = ND->getName().str() + "::" + Scope;
    return getMappingPerLang(Language)->NamespaceSymbols->lookup(Scope);
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
  return Symbol(It->second, D->getLangOpts().CPlusPlus? Lang::CXX : Lang::C);
}

} // namespace stdlib
} // namespace tooling
} // namespace clang
