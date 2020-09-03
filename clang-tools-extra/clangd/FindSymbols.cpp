//===--- FindSymbols.cpp ------------------------------------*- C++-*------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "FindSymbols.h"

#include "AST.h"
#include "FuzzyMatch.h"
#include "ParsedAST.h"
#include "Quality.h"
#include "SourceCode.h"
#include "index/Index.h"
#include "support/Logger.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/Index/IndexDataConsumer.h"
#include "clang/Index/IndexSymbol.h"
#include "clang/Index/IndexingAction.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ScopedPrinter.h"

#define DEBUG_TYPE "FindSymbols"

namespace clang {
namespace clangd {

namespace {
using ScoredSymbolInfo = std::pair<float, SymbolInformation>;
struct ScoredSymbolGreater {
  bool operator()(const ScoredSymbolInfo &L, const ScoredSymbolInfo &R) {
    if (L.first != R.first)
      return L.first > R.first;
    return L.second.name < R.second.name; // Earlier name is better.
  }
};

} // namespace

llvm::Expected<Location> indexToLSPLocation(const SymbolLocation &Loc,
                                            llvm::StringRef TUPath) {
  auto Path = URI::resolve(Loc.FileURI, TUPath);
  if (!Path) {
    return llvm::make_error<llvm::StringError>(
        llvm::formatv("Could not resolve path for file '{0}': {1}", Loc.FileURI,
                      llvm::toString(Path.takeError())),
        llvm::inconvertibleErrorCode());
  }
  Location L;
  L.uri = URIForFile::canonicalize(*Path, TUPath);
  Position Start, End;
  Start.line = Loc.Start.line();
  Start.character = Loc.Start.column();
  End.line = Loc.End.line();
  End.character = Loc.End.column();
  L.range = {Start, End};
  return L;
}

llvm::Expected<Location> symbolToLocation(const Symbol &Sym,
                                          llvm::StringRef TUPath) {
  // Prefer the definition over e.g. a function declaration in a header
  return indexToLSPLocation(
      Sym.Definition ? Sym.Definition : Sym.CanonicalDeclaration, TUPath);
}

llvm::Expected<std::vector<SymbolInformation>>
getWorkspaceSymbols(llvm::StringRef Query, int Limit,
                    const SymbolIndex *const Index, llvm::StringRef HintPath) {
  std::vector<SymbolInformation> Result;
  if (Query.empty() || !Index)
    return Result;

  auto Names = splitQualifiedName(Query);

  FuzzyFindRequest Req;
  Req.Query = std::string(Names.second);

  // FuzzyFind doesn't want leading :: qualifier
  bool IsGlobalQuery = Names.first.consume_front("::");
  // Restrict results to the scope in the query string if present (global or
  // not).
  if (IsGlobalQuery || !Names.first.empty())
    Req.Scopes = {std::string(Names.first)};
  else
    Req.AnyScope = true;
  if (Limit)
    Req.Limit = Limit;
  TopN<ScoredSymbolInfo, ScoredSymbolGreater> Top(
      Req.Limit ? *Req.Limit : std::numeric_limits<size_t>::max());
  FuzzyMatcher Filter(Req.Query);
  Index->fuzzyFind(Req, [HintPath, &Top, &Filter](const Symbol &Sym) {
    auto Loc = symbolToLocation(Sym, HintPath);
    if (!Loc) {
      log("Workspace symbols: {0}", Loc.takeError());
      return;
    }

    SymbolKind SK = indexSymbolKindToSymbolKind(Sym.SymInfo.Kind);
    std::string Scope = std::string(Sym.Scope);
    llvm::StringRef ScopeRef = Scope;
    ScopeRef.consume_back("::");
    SymbolInformation Info = {(Sym.Name + Sym.TemplateSpecializationArgs).str(),
                              SK, *Loc, std::string(ScopeRef)};

    SymbolQualitySignals Quality;
    Quality.merge(Sym);
    SymbolRelevanceSignals Relevance;
    Relevance.Name = Sym.Name;
    Relevance.Query = SymbolRelevanceSignals::Generic;
    if (auto NameMatch = Filter.match(Sym.Name))
      Relevance.NameMatch = *NameMatch;
    else {
      log("Workspace symbol: {0} didn't match query {1}", Sym.Name,
          Filter.pattern());
      return;
    }
    Relevance.merge(Sym);
    auto Score =
        evaluateSymbolAndRelevance(Quality.evaluate(), Relevance.evaluate());
    dlog("FindSymbols: {0}{1} = {2}\n{3}{4}\n", Sym.Scope, Sym.Name, Score,
         Quality, Relevance);

    Top.push({Score, std::move(Info)});
  });
  for (auto &R : std::move(Top).items())
    Result.push_back(std::move(R.second));
  return Result;
}

namespace {
llvm::Optional<DocumentSymbol> declToSym(ASTContext &Ctx, const NamedDecl &ND) {
  auto &SM = Ctx.getSourceManager();

  SourceLocation NameLoc = nameLocation(ND, SM);
  SourceLocation BeginLoc = SM.getSpellingLoc(SM.getFileLoc(ND.getBeginLoc()));
  SourceLocation EndLoc = SM.getSpellingLoc(SM.getFileLoc(ND.getEndLoc()));
  const auto SymbolRange =
      toHalfOpenFileRange(SM, Ctx.getLangOpts(), {BeginLoc, EndLoc});
  if (!SymbolRange)
    return llvm::None;

  Position NameBegin = sourceLocToPosition(SM, NameLoc);
  Position NameEnd = sourceLocToPosition(
      SM, Lexer::getLocForEndOfToken(NameLoc, 0, SM, Ctx.getLangOpts()));

  index::SymbolInfo SymInfo = index::getSymbolInfo(&ND);
  // FIXME: this is not classifying constructors, destructors and operators
  //        correctly (they're all "methods").
  SymbolKind SK = indexSymbolKindToSymbolKind(SymInfo.Kind);

  DocumentSymbol SI;
  SI.name = printName(Ctx, ND);
  SI.kind = SK;
  SI.deprecated = ND.isDeprecated();
  SI.range = Range{sourceLocToPosition(SM, SymbolRange->getBegin()),
                   sourceLocToPosition(SM, SymbolRange->getEnd())};
  SI.selectionRange = Range{NameBegin, NameEnd};
  if (!SI.range.contains(SI.selectionRange)) {
    // 'selectionRange' must be contained in 'range', so in cases where clang
    // reports unrelated ranges we need to reconcile somehow.
    SI.range = SI.selectionRange;
  }
  return SI;
}

/// A helper class to build an outline for the parse AST. It traverses the AST
/// directly instead of using RecursiveASTVisitor (RAV) for three main reasons:
///    - there is no way to keep RAV from traversing subtrees we are not
///      interested in. E.g. not traversing function locals or implicit template
///      instantiations.
///    - it's easier to combine results of recursive passes,
///    - visiting decls is actually simple, so we don't hit the complicated
///      cases that RAV mostly helps with (types, expressions, etc.)
class DocumentOutline {
public:
  DocumentOutline(ParsedAST &AST) : AST(AST) {}

  /// Builds the document outline for the generated AST.
  std::vector<DocumentSymbol> build() {
    std::vector<DocumentSymbol> Results;
    for (auto &TopLevel : AST.getLocalTopLevelDecls())
      traverseDecl(TopLevel, Results);
    return Results;
  }

private:
  enum class VisitKind { No, OnlyDecl, DeclAndChildren };

  void traverseDecl(Decl *D, std::vector<DocumentSymbol> &Results) {
    if (auto *Templ = llvm::dyn_cast<TemplateDecl>(D)) {
      // TemplatedDecl might be null, e.g. concepts.
      if (auto *TD = Templ->getTemplatedDecl())
        D = TD;
    }
    auto *ND = llvm::dyn_cast<NamedDecl>(D);
    if (!ND)
      return;
    VisitKind Visit = shouldVisit(ND);
    if (Visit == VisitKind::No)
      return;
    llvm::Optional<DocumentSymbol> Sym = declToSym(AST.getASTContext(), *ND);
    if (!Sym)
      return;
    if (Visit == VisitKind::DeclAndChildren)
      traverseChildren(D, Sym->children);
    Results.push_back(std::move(*Sym));
  }

  void traverseChildren(Decl *D, std::vector<DocumentSymbol> &Results) {
    auto *Scope = llvm::dyn_cast<DeclContext>(D);
    if (!Scope)
      return;
    for (auto *C : Scope->decls())
      traverseDecl(C, Results);
  }

  VisitKind shouldVisit(NamedDecl *D) {
    if (D->isImplicit())
      return VisitKind::No;

    if (auto Func = llvm::dyn_cast<FunctionDecl>(D)) {
      // Some functions are implicit template instantiations, those should be
      // ignored.
      if (auto *Info = Func->getTemplateSpecializationInfo()) {
        if (!Info->isExplicitInstantiationOrSpecialization())
          return VisitKind::No;
      }
      // Only visit the function itself, do not visit the children (i.e.
      // function parameters, etc.)
      return VisitKind::OnlyDecl;
    }
    // Handle template instantiations. We have three cases to consider:
    //   - explicit instantiations, e.g. 'template class std::vector<int>;'
    //     Visit the decl itself (it's present in the code), but not the
    //     children.
    //   - implicit instantiations, i.e. not written by the user.
    //     Do not visit at all, they are not present in the code.
    //   - explicit specialization, e.g. 'template <> class vector<bool> {};'
    //     Visit both the decl and its children, both are written in the code.
    if (auto *TemplSpec = llvm::dyn_cast<ClassTemplateSpecializationDecl>(D)) {
      if (TemplSpec->isExplicitInstantiationOrSpecialization())
        return TemplSpec->isExplicitSpecialization()
                   ? VisitKind::DeclAndChildren
                   : VisitKind::OnlyDecl;
      return VisitKind::No;
    }
    if (auto *TemplSpec = llvm::dyn_cast<VarTemplateSpecializationDecl>(D)) {
      if (TemplSpec->isExplicitInstantiationOrSpecialization())
        return TemplSpec->isExplicitSpecialization()
                   ? VisitKind::DeclAndChildren
                   : VisitKind::OnlyDecl;
      return VisitKind::No;
    }
    // For all other cases, visit both the children and the decl.
    return VisitKind::DeclAndChildren;
  }

  ParsedAST &AST;
};

std::vector<DocumentSymbol> collectDocSymbols(ParsedAST &AST) {
  return DocumentOutline(AST).build();
}
} // namespace

llvm::Expected<std::vector<DocumentSymbol>> getDocumentSymbols(ParsedAST &AST) {
  return collectDocSymbols(AST);
}

} // namespace clangd
} // namespace clang
