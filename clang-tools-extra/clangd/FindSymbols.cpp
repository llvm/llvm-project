//===--- FindSymbols.cpp ------------------------------------*- C++-*------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "FindSymbols.h"

#include "Logger.h"
#include "FuzzyMatch.h"
#include "SourceCode.h"
#include "Quality.h"
#include "index/Index.h"
#include "clang/Index/IndexSymbol.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"

#define DEBUG_TYPE "FindSymbols"

namespace clang {
namespace clangd {

namespace {

// Convert a index::SymbolKind to clangd::SymbolKind (LSP)
// Note, some are not perfect matches and should be improved when this LSP
// issue is addressed:
// https://github.com/Microsoft/language-server-protocol/issues/344
SymbolKind indexSymbolKindToSymbolKind(index::SymbolKind Kind) {
  switch (Kind) {
  case index::SymbolKind::Unknown:
    return SymbolKind::Variable;
  case index::SymbolKind::Module:
    return SymbolKind::Module;
  case index::SymbolKind::Namespace:
    return SymbolKind::Namespace;
  case index::SymbolKind::NamespaceAlias:
    return SymbolKind::Namespace;
  case index::SymbolKind::Macro:
    return SymbolKind::String;
  case index::SymbolKind::Enum:
    return SymbolKind::Enum;
  case index::SymbolKind::Struct:
    return SymbolKind::Struct;
  case index::SymbolKind::Class:
    return SymbolKind::Class;
  case index::SymbolKind::Protocol:
    return SymbolKind::Interface;
  case index::SymbolKind::Extension:
    return SymbolKind::Interface;
  case index::SymbolKind::Union:
    return SymbolKind::Class;
  case index::SymbolKind::TypeAlias:
    return SymbolKind::Class;
  case index::SymbolKind::Function:
    return SymbolKind::Function;
  case index::SymbolKind::Variable:
    return SymbolKind::Variable;
  case index::SymbolKind::Field:
    return SymbolKind::Field;
  case index::SymbolKind::EnumConstant:
    return SymbolKind::EnumMember;
  case index::SymbolKind::InstanceMethod:
  case index::SymbolKind::ClassMethod:
  case index::SymbolKind::StaticMethod:
    return SymbolKind::Method;
  case index::SymbolKind::InstanceProperty:
  case index::SymbolKind::ClassProperty:
  case index::SymbolKind::StaticProperty:
    return SymbolKind::Property;
  case index::SymbolKind::Constructor:
  case index::SymbolKind::Destructor:
    return SymbolKind::Method;
  case index::SymbolKind::ConversionFunction:
    return SymbolKind::Function;
  case index::SymbolKind::Parameter:
    return SymbolKind::Variable;
  case index::SymbolKind::Using:
    return SymbolKind::Namespace;
  }
  llvm_unreachable("invalid symbol kind");
}

using ScoredSymbolInfo = std::pair<float, SymbolInformation>;
struct ScoredSymbolGreater {
  bool operator()(const ScoredSymbolInfo &L, const ScoredSymbolInfo &R) {
    if (L.first != R.first)
      return L.first > R.first;
    return L.second.name < R.second.name; // Earlier name is better.
  }
};

} // namespace

llvm::Expected<std::vector<SymbolInformation>>
getWorkspaceSymbols(StringRef Query, int Limit,
                    const SymbolIndex *const Index) {
  std::vector<SymbolInformation> Result;
  if (Query.empty() || !Index)
    return Result;

  auto Names = splitQualifiedName(Query);

  FuzzyFindRequest Req;
  Req.Query = Names.second;

  // FuzzyFind doesn't want leading :: qualifier
  bool IsGlobalQuery = Names.first.consume_front("::");
  // Restrict results to the scope in the query string if present (global or
  // not).
  if (IsGlobalQuery || !Names.first.empty())
    Req.Scopes = {Names.first};
  if (Limit)
    Req.MaxCandidateCount = Limit;
  TopN<ScoredSymbolInfo, ScoredSymbolGreater> Top(Req.MaxCandidateCount);
  FuzzyMatcher Filter(Req.Query);
  Index->fuzzyFind(Req, [&Top, &Filter](const Symbol &Sym) {
    // Prefer the definition over e.g. a function declaration in a header
    auto &CD = Sym.Definition ? Sym.Definition : Sym.CanonicalDeclaration;
    auto Uri = URI::parse(CD.FileURI);
    if (!Uri) {
      log(llvm::formatv(
          "Workspace symbol: Could not parse URI '{0}' for symbol '{1}'.",
          CD.FileURI, Sym.Name));
      return;
    }
    // FIXME: Passing no HintPath here will work for "file" and "test" schemes
    // because they don't use it but this might not work for other custom ones.
    auto Path = URI::resolve(*Uri);
    if (!Path) {
      log(llvm::formatv("Workspace symbol: Could not resolve path for URI "
                        "'{0}' for symbol '{1}'.",
                        (*Uri).toString(), Sym.Name.str()));
      return;
    }
    Location L;
    L.uri = URIForFile((*Path));
    Position Start, End;
    Start.line = CD.Start.Line;
    Start.character = CD.Start.Column;
    End.line = CD.End.Line;
    End.character = CD.End.Column;
    L.range = {Start, End};
    SymbolKind SK = indexSymbolKindToSymbolKind(Sym.SymInfo.Kind);
    std::string Scope = Sym.Scope;
    StringRef ScopeRef = Scope;
    ScopeRef.consume_back("::");
    SymbolInformation Info = {Sym.Name, SK, L, ScopeRef};

    SymbolQualitySignals Quality;
    Quality.merge(Sym);
    SymbolRelevanceSignals Relevance;
    Relevance.Query = SymbolRelevanceSignals::Generic;
    if (auto NameMatch = Filter.match(Sym.Name))
      Relevance.NameMatch = *NameMatch;
    else {
      log(llvm::formatv("Workspace symbol: {0} didn't match query {1}",
                        Sym.Name, Filter.pattern()));
      return;
    }
    Relevance.merge(Sym);
    auto Score =
        evaluateSymbolAndRelevance(Quality.evaluate(), Relevance.evaluate());
    LLVM_DEBUG(llvm::dbgs() << "FindSymbols: " << Sym.Scope << Sym.Name << " = "
                            << Score << "\n"
                            << Quality << Relevance << "\n");

    Top.push({Score, std::move(Info)});
  });
  for (auto &R : std::move(Top).items())
    Result.push_back(std::move(R.second));
  return Result;
}

} // namespace clangd
} // namespace clang
