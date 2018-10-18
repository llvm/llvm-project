//===--- FindSymbols.cpp ------------------------------------*- C++-*------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "FindSymbols.h"

#include "AST.h"
#include "ClangdUnit.h"
#include "FuzzyMatch.h"
#include "Logger.h"
#include "Quality.h"
#include "SourceCode.h"
#include "index/Index.h"
#include "clang/Index/IndexDataConsumer.h"
#include "clang/Index/IndexSymbol.h"
#include "clang/Index/IndexingAction.h"
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
getWorkspaceSymbols(StringRef Query, int Limit, const SymbolIndex *const Index,
                    StringRef HintPath) {
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
    Req.Limit = Limit;
  TopN<ScoredSymbolInfo, ScoredSymbolGreater> Top(
      Req.Limit ? *Req.Limit : std::numeric_limits<size_t>::max());
  FuzzyMatcher Filter(Req.Query);
  Index->fuzzyFind(Req, [HintPath, &Top, &Filter](const Symbol &Sym) {
    // Prefer the definition over e.g. a function declaration in a header
    auto &CD = Sym.Definition ? Sym.Definition : Sym.CanonicalDeclaration;
    auto Uri = URI::parse(CD.FileURI);
    if (!Uri) {
      log("Workspace symbol: Could not parse URI '{0}' for symbol '{1}'.",
          CD.FileURI, Sym.Name);
      return;
    }
    auto Path = URI::resolve(*Uri, HintPath);
    if (!Path) {
      log("Workspace symbol: Could not resolve path for URI '{0}' for symbol "
          "'{1}'.",
          Uri->toString(), Sym.Name);
      return;
    }
    Location L;
    L.uri = URIForFile((*Path));
    Position Start, End;
    Start.line = CD.Start.line();
    Start.character = CD.Start.column();
    End.line = CD.End.line();
    End.character = CD.End.column();
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
/// Finds document symbols in the main file of the AST.
class DocumentSymbolsConsumer : public index::IndexDataConsumer {
  ASTContext &AST;
  std::vector<SymbolInformation> Symbols;
  // We are always list document for the same file, so cache the value.
  llvm::Optional<URIForFile> MainFileUri;

public:
  DocumentSymbolsConsumer(ASTContext &AST) : AST(AST) {}
  std::vector<SymbolInformation> takeSymbols() { return std::move(Symbols); }

  void initialize(ASTContext &Ctx) override {
    // Compute the absolute path of the main file which we will use for all
    // results.
    const SourceManager &SM = AST.getSourceManager();
    const FileEntry *F = SM.getFileEntryForID(SM.getMainFileID());
    if (!F)
      return;
    auto FilePath = getRealPath(F, SM);
    if (FilePath)
      MainFileUri = URIForFile(*FilePath);
  }

  bool shouldIncludeSymbol(const NamedDecl *ND) {
    if (!ND || ND->isImplicit())
      return false;
    // Skip anonymous declarations, e.g (anonymous enum/class/struct).
    if (ND->getDeclName().isEmpty())
      return false;
    return true;
  }

  bool
  handleDeclOccurence(const Decl *, index::SymbolRoleSet Roles,
                      ArrayRef<index::SymbolRelation> Relations,
                      SourceLocation Loc,
                      index::IndexDataConsumer::ASTNodeInfo ASTNode) override {
    assert(ASTNode.OrigD);
    // No point in continuing the index consumer if we could not get the
    // absolute path of the main file.
    if (!MainFileUri)
      return false;
    // We only want declarations and definitions, i.e. no references.
    if (!(Roles & static_cast<unsigned>(index::SymbolRole::Declaration) ||
          Roles & static_cast<unsigned>(index::SymbolRole::Definition)))
      return true;
    SourceLocation NameLoc = findNameLoc(ASTNode.OrigD);
    const SourceManager &SourceMgr = AST.getSourceManager();
    // We should be only be looking at "local" decls in the main file.
    if (!SourceMgr.isWrittenInMainFile(NameLoc)) {
      // Even thought we are visiting only local (non-preamble) decls,
      // we can get here when in the presence of "extern" decls.
      return true;
    }
    const NamedDecl *ND = llvm::dyn_cast<NamedDecl>(ASTNode.OrigD);
    if (!shouldIncludeSymbol(ND))
      return true;

    SourceLocation EndLoc =
        Lexer::getLocForEndOfToken(NameLoc, 0, SourceMgr, AST.getLangOpts());
    Position Begin = sourceLocToPosition(SourceMgr, NameLoc);
    Position End = sourceLocToPosition(SourceMgr, EndLoc);
    Range R = {Begin, End};
    Location L;
    L.uri = *MainFileUri;
    L.range = R;

    std::string QName = printQualifiedName(*ND);
    StringRef Scope, Name;
    std::tie(Scope, Name) = splitQualifiedName(QName);
    Scope.consume_back("::");

    index::SymbolInfo SymInfo = index::getSymbolInfo(ND);
    SymbolKind SK = indexSymbolKindToSymbolKind(SymInfo.Kind);

    SymbolInformation SI;
    SI.name = Name;
    SI.kind = SK;
    SI.location = L;
    SI.containerName = Scope;
    Symbols.push_back(std::move(SI));
    return true;
  }
};
} // namespace

llvm::Expected<std::vector<SymbolInformation>>
getDocumentSymbols(ParsedAST &AST) {
  DocumentSymbolsConsumer DocumentSymbolsCons(AST.getASTContext());

  index::IndexingOptions IndexOpts;
  IndexOpts.SystemSymbolFilter =
      index::IndexingOptions::SystemSymbolFilterKind::DeclarationsOnly;
  IndexOpts.IndexFunctionLocals = false;
  indexTopLevelDecls(AST.getASTContext(), AST.getPreprocessor(),
                     AST.getLocalTopLevelDecls(), DocumentSymbolsCons,
                     IndexOpts);

  return DocumentSymbolsCons.takeSymbols();
}

} // namespace clangd
} // namespace clang
