//===--- XRefs.cpp -----------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "XRefs.h"
#include "AST.h"
#include "CodeCompletionStrings.h"
#include "FindSymbols.h"
#include "FindTarget.h"
#include "Logger.h"
#include "ParsedAST.h"
#include "Protocol.h"
#include "Selection.h"
#include "SourceCode.h"
#include "URI.h"
#include "index/Index.h"
#include "index/Merge.h"
#include "index/Relation.h"
#include "index/SymbolLocation.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Type.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Index/IndexDataConsumer.h"
#include "clang/Index/IndexSymbol.h"
#include "clang/Index/IndexingAction.h"
#include "clang/Index/IndexingOptions.h"
#include "clang/Index/USRGeneration.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace clangd {
namespace {

// Returns the single definition of the entity declared by D, if visible.
// In particular:
// - for non-redeclarable kinds (e.g. local vars), return D
// - for kinds that allow multiple definitions (e.g. namespaces), return nullptr
// Kinds of nodes that always return nullptr here will not have definitions
// reported by locateSymbolAt().
const Decl *getDefinition(const Decl *D) {
  assert(D);
  // Decl has one definition that we can find.
  if (const auto *TD = dyn_cast<TagDecl>(D))
    return TD->getDefinition();
  if (const auto *VD = dyn_cast<VarDecl>(D))
    return VD->getDefinition();
  if (const auto *FD = dyn_cast<FunctionDecl>(D))
    return FD->getDefinition();
  // Only a single declaration is allowed.
  if (isa<ValueDecl>(D) || isa<TemplateTypeParmDecl>(D) ||
      isa<TemplateTemplateParmDecl>(D)) // except cases above
    return D;
  // Multiple definitions are allowed.
  return nullptr; // except cases above
}

void logIfOverflow(const SymbolLocation &Loc) {
  if (Loc.Start.hasOverflow() || Loc.End.hasOverflow())
    log("Possible overflow in symbol location: {0}", Loc);
}

// Convert a SymbolLocation to LSP's Location.
// TUPath is used to resolve the path of URI.
// FIXME: figure out a good home for it, and share the implementation with
// FindSymbols.
llvm::Optional<Location> toLSPLocation(const SymbolLocation &Loc,
                                       llvm::StringRef TUPath) {
  if (!Loc)
    return None;
  auto Uri = URI::parse(Loc.FileURI);
  if (!Uri) {
    elog("Could not parse URI {0}: {1}", Loc.FileURI, Uri.takeError());
    return None;
  }
  auto U = URIForFile::fromURI(*Uri, TUPath);
  if (!U) {
    elog("Could not resolve URI {0}: {1}", Loc.FileURI, U.takeError());
    return None;
  }

  Location LSPLoc;
  LSPLoc.uri = std::move(*U);
  LSPLoc.range.start.line = Loc.Start.line();
  LSPLoc.range.start.character = Loc.Start.column();
  LSPLoc.range.end.line = Loc.End.line();
  LSPLoc.range.end.character = Loc.End.column();
  logIfOverflow(Loc);
  return LSPLoc;
}

SymbolLocation toIndexLocation(const Location &Loc, std::string &URIStorage) {
  SymbolLocation SymLoc;
  URIStorage = Loc.uri.uri();
  SymLoc.FileURI = URIStorage.c_str();
  SymLoc.Start.setLine(Loc.range.start.line);
  SymLoc.Start.setColumn(Loc.range.start.character);
  SymLoc.End.setLine(Loc.range.end.line);
  SymLoc.End.setColumn(Loc.range.end.character);
  return SymLoc;
}

// Returns the preferred location between an AST location and an index location.
SymbolLocation getPreferredLocation(const Location &ASTLoc,
                                    const SymbolLocation &IdxLoc,
                                    std::string &Scratch) {
  // Also use a dummy symbol for the index location so that other fields (e.g.
  // definition) are not factored into the preference.
  Symbol ASTSym, IdxSym;
  ASTSym.ID = IdxSym.ID = SymbolID("dummy_id");
  ASTSym.CanonicalDeclaration = toIndexLocation(ASTLoc, Scratch);
  IdxSym.CanonicalDeclaration = IdxLoc;
  auto Merged = mergeSymbol(ASTSym, IdxSym);
  return Merged.CanonicalDeclaration;
}

std::vector<const Decl *> getDeclAtPosition(ParsedAST &AST, SourceLocation Pos,
                                            DeclRelationSet Relations) {
  FileID FID;
  unsigned Offset;
  std::tie(FID, Offset) = AST.getSourceManager().getDecomposedSpellingLoc(Pos);
  SelectionTree Selection(AST.getASTContext(), AST.getTokens(), Offset);
  std::vector<const Decl *> Result;
  if (const SelectionTree::Node *N = Selection.commonAncestor()) {
    auto Decls = targetDecl(N->ASTNode, Relations);
    Result.assign(Decls.begin(), Decls.end());
  }
  return Result;
}

llvm::Optional<Location> makeLocation(ASTContext &AST, SourceLocation TokLoc,
                                      llvm::StringRef TUPath) {
  const SourceManager &SourceMgr = AST.getSourceManager();
  const FileEntry *F = SourceMgr.getFileEntryForID(SourceMgr.getFileID(TokLoc));
  if (!F)
    return None;
  auto FilePath = getCanonicalPath(F, SourceMgr);
  if (!FilePath) {
    log("failed to get path!");
    return None;
  }
  if (auto Range =
          getTokenRange(AST.getSourceManager(), AST.getLangOpts(), TokLoc)) {
    Location L;
    L.uri = URIForFile::canonicalize(*FilePath, TUPath);
    L.range = *Range;
    return L;
  }
  return None;
}

} // namespace

std::vector<LocatedSymbol> locateSymbolAt(ParsedAST &AST, Position Pos,
                                          const SymbolIndex *Index) {
  const auto &SM = AST.getSourceManager();
  auto MainFilePath =
      getCanonicalPath(SM.getFileEntryForID(SM.getMainFileID()), SM);
  if (!MainFilePath) {
    elog("Failed to get a path for the main file, so no references");
    return {};
  }

  // Treat #included files as symbols, to enable go-to-definition on them.
  for (auto &Inc : AST.getIncludeStructure().MainFileIncludes) {
    if (!Inc.Resolved.empty() && Inc.R.start.line == Pos.line) {
      LocatedSymbol File;
      File.Name = llvm::sys::path::filename(Inc.Resolved);
      File.PreferredDeclaration = {
          URIForFile::canonicalize(Inc.Resolved, *MainFilePath), Range{}};
      File.Definition = File.PreferredDeclaration;
      // We're not going to find any further symbols on #include lines.
      return {std::move(File)};
    }
  }

  // Macros are simple: there's no declaration/definition distinction.
  // As a consequence, there's no need to look them up in the index either.
  SourceLocation IdentStartLoc = SM.getMacroArgExpandedLocation(
      getBeginningOfIdentifier(Pos, AST.getSourceManager(), AST.getLangOpts()));
  std::vector<LocatedSymbol> Result;
  if (auto M = locateMacroAt(IdentStartLoc, AST.getPreprocessor())) {
    if (auto Loc = makeLocation(AST.getASTContext(),
                                M->Info->getDefinitionLoc(), *MainFilePath)) {
      LocatedSymbol Macro;
      Macro.Name = M->Name;
      Macro.PreferredDeclaration = *Loc;
      Macro.Definition = Loc;
      Result.push_back(std::move(Macro));

      // Don't look at the AST or index if we have a macro result.
      // (We'd just return declarations referenced from the macro's
      // expansion.)
      return Result;
    }
  }

  // Decls are more complicated.
  // The AST contains at least a declaration, maybe a definition.
  // These are up-to-date, and so generally preferred over index results.
  // We perform a single batch index lookup to find additional definitions.

  // Results follow the order of Symbols.Decls.
  // Keep track of SymbolID -> index mapping, to fill in index data later.
  llvm::DenseMap<SymbolID, size_t> ResultIndex;

  SourceLocation SourceLoc;
  if (auto L = sourceLocationInMainFile(SM, Pos)) {
    SourceLoc = *L;
  } else {
    elog("locateSymbolAt failed to convert position to source location: {0}",
         L.takeError());
    return Result;
  }

  // Emit all symbol locations (declaration or definition) from AST.
  DeclRelationSet Relations =
      DeclRelation::TemplatePattern | DeclRelation::Alias;
  for (const Decl *D : getDeclAtPosition(AST, SourceLoc, Relations)) {
    const Decl *Def = getDefinition(D);
    const Decl *Preferred = Def ? Def : D;

    // If we're at the point of declaration of a template specialization,
    // it's more useful to navigate to the template declaration.
    if (SM.getMacroArgExpandedLocation(Preferred->getLocation()) ==
        IdentStartLoc) {
      if (auto *CTSD = dyn_cast<ClassTemplateSpecializationDecl>(Preferred)) {
        D = CTSD->getSpecializedTemplate();
        Def = getDefinition(D);
        Preferred = Def ? Def : D;
      }
    }

    auto Loc = makeLocation(AST.getASTContext(), nameLocation(*Preferred, SM),
                            *MainFilePath);
    if (!Loc)
      continue;

    Result.emplace_back();
    if (auto *ND = dyn_cast<NamedDecl>(Preferred))
      Result.back().Name = printName(AST.getASTContext(), *ND);
    Result.back().PreferredDeclaration = *Loc;
    // Preferred is always a definition if possible, so this check works.
    if (Def == Preferred)
      Result.back().Definition = *Loc;

    // Record SymbolID for index lookup later.
    if (auto ID = getSymbolID(Preferred))
      ResultIndex[*ID] = Result.size() - 1;
  }

  // Now query the index for all Symbol IDs we found in the AST.
  if (Index && !ResultIndex.empty()) {
    LookupRequest QueryRequest;
    for (auto It : ResultIndex)
      QueryRequest.IDs.insert(It.first);
    std::string Scratch;
    Index->lookup(QueryRequest, [&](const Symbol &Sym) {
      auto &R = Result[ResultIndex.lookup(Sym.ID)];

      if (R.Definition) { // from AST
        // Special case: if the AST yielded a definition, then it may not be
        // the right *declaration*. Prefer the one from the index.
        if (auto Loc = toLSPLocation(Sym.CanonicalDeclaration, *MainFilePath))
          R.PreferredDeclaration = *Loc;

        // We might still prefer the definition from the index, e.g. for
        // generated symbols.
        if (auto Loc = toLSPLocation(
                getPreferredLocation(*R.Definition, Sym.Definition, Scratch),
                *MainFilePath))
          R.Definition = *Loc;
      } else {
        R.Definition = toLSPLocation(Sym.Definition, *MainFilePath);

        // Use merge logic to choose AST or index declaration.
        if (auto Loc = toLSPLocation(
                getPreferredLocation(R.PreferredDeclaration,
                                     Sym.CanonicalDeclaration, Scratch),
                *MainFilePath))
          R.PreferredDeclaration = *Loc;
      }
    });
  }

  return Result;
}

namespace {

/// Collects references to symbols within the main file.
class ReferenceFinder : public index::IndexDataConsumer {
public:
  struct Reference {
    SourceLocation Loc;
    index::SymbolRoleSet Role;
  };

  ReferenceFinder(ASTContext &AST, Preprocessor &PP,
                  const std::vector<const Decl *> &TargetDecls)
      : AST(AST) {
    for (const Decl *D : TargetDecls)
      CanonicalTargets.insert(D->getCanonicalDecl());
  }

  std::vector<Reference> take() && {
    llvm::sort(References, [](const Reference &L, const Reference &R) {
      return std::tie(L.Loc, L.Role) < std::tie(R.Loc, R.Role);
    });
    // We sometimes see duplicates when parts of the AST get traversed twice.
    References.erase(std::unique(References.begin(), References.end(),
                                 [](const Reference &L, const Reference &R) {
                                   return std::tie(L.Loc, L.Role) ==
                                          std::tie(R.Loc, R.Role);
                                 }),
                     References.end());
    return std::move(References);
  }

  bool
  handleDeclOccurrence(const Decl *D, index::SymbolRoleSet Roles,
                       llvm::ArrayRef<index::SymbolRelation> Relations,
                       SourceLocation Loc,
                       index::IndexDataConsumer::ASTNodeInfo ASTNode) override {
    assert(D->isCanonicalDecl() && "expect D to be a canonical declaration");
    const SourceManager &SM = AST.getSourceManager();
    Loc = SM.getFileLoc(Loc);
    if (isInsideMainFile(Loc, SM) && CanonicalTargets.count(D))
      References.push_back({Loc, Roles});
    return true;
  }

private:
  llvm::SmallSet<const Decl *, 4> CanonicalTargets;
  std::vector<Reference> References;
  const ASTContext &AST;
};

std::vector<ReferenceFinder::Reference>
findRefs(const std::vector<const Decl *> &Decls, ParsedAST &AST) {
  ReferenceFinder RefFinder(AST.getASTContext(), AST.getPreprocessor(), Decls);
  index::IndexingOptions IndexOpts;
  IndexOpts.SystemSymbolFilter =
      index::IndexingOptions::SystemSymbolFilterKind::All;
  IndexOpts.IndexFunctionLocals = true;
  IndexOpts.IndexParametersInDeclarations = true;
  IndexOpts.IndexTemplateParameters = true;
  indexTopLevelDecls(AST.getASTContext(), AST.getPreprocessor(),
                     AST.getLocalTopLevelDecls(), RefFinder, IndexOpts);
  return std::move(RefFinder).take();
}

} // namespace

std::vector<DocumentHighlight> findDocumentHighlights(ParsedAST &AST,
                                                      Position Pos) {
  const SourceManager &SM = AST.getSourceManager();
  // FIXME: show references to macro within file?
  DeclRelationSet Relations =
      DeclRelation::TemplatePattern | DeclRelation::Alias;
  auto References = findRefs(
      getDeclAtPosition(AST,
                        SM.getMacroArgExpandedLocation(getBeginningOfIdentifier(
                            Pos, SM, AST.getLangOpts())),
                        Relations),
      AST);

  // FIXME: we may get multiple DocumentHighlights with the same location and
  // different kinds, deduplicate them.
  std::vector<DocumentHighlight> Result;
  for (const auto &Ref : References) {
    if (auto Range =
            getTokenRange(AST.getSourceManager(), AST.getLangOpts(), Ref.Loc)) {
      DocumentHighlight DH;
      DH.range = *Range;
      if (Ref.Role & index::SymbolRoleSet(index::SymbolRole::Write))
        DH.kind = DocumentHighlightKind::Write;
      else if (Ref.Role & index::SymbolRoleSet(index::SymbolRole::Read))
        DH.kind = DocumentHighlightKind::Read;
      else
        DH.kind = DocumentHighlightKind::Text;
      Result.push_back(std::move(DH));
    }
  }
  return Result;
}

ReferencesResult findReferences(ParsedAST &AST, Position Pos, uint32_t Limit,
                                const SymbolIndex *Index) {
  if (!Limit)
    Limit = std::numeric_limits<uint32_t>::max();
  ReferencesResult Results;
  const SourceManager &SM = AST.getSourceManager();
  auto MainFilePath =
      getCanonicalPath(SM.getFileEntryForID(SM.getMainFileID()), SM);
  if (!MainFilePath) {
    elog("Failed to get a path for the main file, so no references");
    return Results;
  }
  auto Loc = SM.getMacroArgExpandedLocation(
      getBeginningOfIdentifier(Pos, SM, AST.getLangOpts()));
  // TODO: should we handle macros, too?
  // We also show references to the targets of using-decls, so we include
  // DeclRelation::Underlying.
  DeclRelationSet Relations = DeclRelation::TemplatePattern |
                              DeclRelation::Alias | DeclRelation::Underlying;
  auto Decls = getDeclAtPosition(AST, Loc, Relations);

  // We traverse the AST to find references in the main file.
  auto MainFileRefs = findRefs(Decls, AST);
  // We may get multiple refs with the same location and different Roles, as
  // cross-reference is only interested in locations, we deduplicate them
  // by the location to avoid emitting duplicated locations.
  MainFileRefs.erase(std::unique(MainFileRefs.begin(), MainFileRefs.end(),
                                 [](const ReferenceFinder::Reference &L,
                                    const ReferenceFinder::Reference &R) {
                                   return L.Loc == R.Loc;
                                 }),
                     MainFileRefs.end());
  for (const auto &Ref : MainFileRefs) {
    if (auto Range = getTokenRange(SM, AST.getLangOpts(), Ref.Loc)) {
      Location Result;
      Result.range = *Range;
      Result.uri = URIForFile::canonicalize(*MainFilePath, *MainFilePath);
      Results.References.push_back(std::move(Result));
    }
  }
  // Now query the index for references from other files.
  if (Index && Results.References.size() <= Limit) {
    RefsRequest Req;
    Req.Limit = Limit;

    for (const Decl *D : Decls) {
      // Not all symbols can be referenced from outside (e.g. function-locals).
      // TODO: we could skip TU-scoped symbols here (e.g. static functions) if
      // we know this file isn't a header. The details might be tricky.
      if (D->getParentFunctionOrMethod())
        continue;
      if (auto ID = getSymbolID(D))
        Req.IDs.insert(*ID);
    }
    if (Req.IDs.empty())
      return Results;
    Results.HasMore |= Index->refs(Req, [&](const Ref &R) {
      // no need to continue process if we reach the limit.
      if (Results.References.size() > Limit)
        return;
      auto LSPLoc = toLSPLocation(R.Location, *MainFilePath);
      // Avoid indexed results for the main file - the AST is authoritative.
      if (!LSPLoc || LSPLoc->uri.file() == *MainFilePath)
        return;

      Results.References.push_back(std::move(*LSPLoc));
    });
  }
  if (Results.References.size() > Limit) {
    Results.HasMore = true;
    Results.References.resize(Limit);
  }
  return Results;
}

std::vector<SymbolDetails> getSymbolInfo(ParsedAST &AST, Position Pos) {
  const SourceManager &SM = AST.getSourceManager();
  auto Loc = SM.getMacroArgExpandedLocation(
      getBeginningOfIdentifier(Pos, SM, AST.getLangOpts()));

  std::vector<SymbolDetails> Results;

  // We also want the targets of using-decls, so we include
  // DeclRelation::Underlying.
  DeclRelationSet Relations = DeclRelation::TemplatePattern |
                              DeclRelation::Alias | DeclRelation::Underlying;
  for (const Decl *D : getDeclAtPosition(AST, Loc, Relations)) {
    SymbolDetails NewSymbol;
    if (const NamedDecl *ND = dyn_cast<NamedDecl>(D)) {
      std::string QName = printQualifiedName(*ND);
      std::tie(NewSymbol.containerName, NewSymbol.name) =
          splitQualifiedName(QName);

      if (NewSymbol.containerName.empty()) {
        if (const auto *ParentND =
                dyn_cast_or_null<NamedDecl>(ND->getDeclContext()))
          NewSymbol.containerName = printQualifiedName(*ParentND);
      }
    }
    llvm::SmallString<32> USR;
    if (!index::generateUSRForDecl(D, USR)) {
      NewSymbol.USR = USR.str();
      NewSymbol.ID = SymbolID(NewSymbol.USR);
    }
    Results.push_back(std::move(NewSymbol));
  }

  if (auto M = locateMacroAt(Loc, AST.getPreprocessor())) {
    SymbolDetails NewMacro;
    NewMacro.name = M->Name;
    llvm::SmallString<32> USR;
    if (!index::generateUSRForMacro(NewMacro.name, M->Info->getDefinitionLoc(),
                                    SM, USR)) {
      NewMacro.USR = USR.str();
      NewMacro.ID = SymbolID(NewMacro.USR);
    }
    Results.push_back(std::move(NewMacro));
  }

  return Results;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const LocatedSymbol &S) {
  OS << S.Name << ": " << S.PreferredDeclaration;
  if (S.Definition)
    OS << " def=" << *S.Definition;
  return OS;
}

// FIXME(nridge): Reduce duplication between this function and declToSym().
static llvm::Optional<TypeHierarchyItem>
declToTypeHierarchyItem(ASTContext &Ctx, const NamedDecl &ND) {
  auto &SM = Ctx.getSourceManager();

  SourceLocation NameLoc = nameLocation(ND, Ctx.getSourceManager());
  // getFileLoc is a good choice for us, but we also need to make sure
  // sourceLocToPosition won't switch files, so we call getSpellingLoc on top of
  // that to make sure it does not switch files.
  // FIXME: sourceLocToPosition should not switch files!
  SourceLocation BeginLoc = SM.getSpellingLoc(SM.getFileLoc(ND.getBeginLoc()));
  SourceLocation EndLoc = SM.getSpellingLoc(SM.getFileLoc(ND.getEndLoc()));
  if (NameLoc.isInvalid() || BeginLoc.isInvalid() || EndLoc.isInvalid())
    return llvm::None;

  Position NameBegin = sourceLocToPosition(SM, NameLoc);
  Position NameEnd = sourceLocToPosition(
      SM, Lexer::getLocForEndOfToken(NameLoc, 0, SM, Ctx.getLangOpts()));

  index::SymbolInfo SymInfo = index::getSymbolInfo(&ND);
  // FIXME: this is not classifying constructors, destructors and operators
  //        correctly (they're all "methods").
  SymbolKind SK = indexSymbolKindToSymbolKind(SymInfo.Kind);

  TypeHierarchyItem THI;
  THI.name = printName(Ctx, ND);
  THI.kind = SK;
  THI.deprecated = ND.isDeprecated();
  THI.range =
      Range{sourceLocToPosition(SM, BeginLoc), sourceLocToPosition(SM, EndLoc)};
  THI.selectionRange = Range{NameBegin, NameEnd};
  if (!THI.range.contains(THI.selectionRange)) {
    // 'selectionRange' must be contained in 'range', so in cases where clang
    // reports unrelated ranges we need to reconcile somehow.
    THI.range = THI.selectionRange;
  }

  auto FilePath =
      getCanonicalPath(SM.getFileEntryForID(SM.getFileID(BeginLoc)), SM);
  auto TUPath = getCanonicalPath(SM.getFileEntryForID(SM.getMainFileID()), SM);
  if (!FilePath || !TUPath)
    return llvm::None; // Not useful without a uri.
  THI.uri = URIForFile::canonicalize(*FilePath, *TUPath);

  return THI;
}

static Optional<TypeHierarchyItem>
symbolToTypeHierarchyItem(const Symbol &S, const SymbolIndex *Index,
                          PathRef TUPath) {
  auto Loc = symbolToLocation(S, TUPath);
  if (!Loc) {
    log("Type hierarchy: {0}", Loc.takeError());
    return llvm::None;
  }
  TypeHierarchyItem THI;
  THI.name = S.Name;
  THI.kind = indexSymbolKindToSymbolKind(S.SymInfo.Kind);
  THI.deprecated = (S.Flags & Symbol::Deprecated);
  THI.selectionRange = Loc->range;
  // FIXME: Populate 'range' correctly
  // (https://github.com/clangd/clangd/issues/59).
  THI.range = THI.selectionRange;
  THI.uri = Loc->uri;
  // Store the SymbolID in the 'data' field. The client will
  // send this back in typeHierarchy/resolve, allowing us to
  // continue resolving additional levels of the type hierarchy.
  THI.data = S.ID.str();

  return std::move(THI);
}

static void fillSubTypes(const SymbolID &ID,
                         std::vector<TypeHierarchyItem> &SubTypes,
                         const SymbolIndex *Index, int Levels, PathRef TUPath) {
  RelationsRequest Req;
  Req.Subjects.insert(ID);
  Req.Predicate = RelationKind::BaseOf;
  Index->relations(Req, [&](const SymbolID &Subject, const Symbol &Object) {
    if (Optional<TypeHierarchyItem> ChildSym =
            symbolToTypeHierarchyItem(Object, Index, TUPath)) {
      if (Levels > 1) {
        ChildSym->children.emplace();
        fillSubTypes(Object.ID, *ChildSym->children, Index, Levels - 1, TUPath);
      }
      SubTypes.emplace_back(std::move(*ChildSym));
    }
  });
}

using RecursionProtectionSet = llvm::SmallSet<const CXXRecordDecl *, 4>;

static void fillSuperTypes(const CXXRecordDecl &CXXRD, ASTContext &ASTCtx,
                           std::vector<TypeHierarchyItem> &SuperTypes,
                           RecursionProtectionSet &RPSet) {
  // typeParents() will replace dependent template specializations
  // with their class template, so to avoid infinite recursion for
  // certain types of hierarchies, keep the templates encountered
  // along the parent chain in a set, and stop the recursion if one
  // starts to repeat.
  auto *Pattern = CXXRD.getDescribedTemplate() ? &CXXRD : nullptr;
  if (Pattern) {
    if (!RPSet.insert(Pattern).second) {
      return;
    }
  }

  for (const CXXRecordDecl *ParentDecl : typeParents(&CXXRD)) {
    if (Optional<TypeHierarchyItem> ParentSym =
            declToTypeHierarchyItem(ASTCtx, *ParentDecl)) {
      ParentSym->parents.emplace();
      fillSuperTypes(*ParentDecl, ASTCtx, *ParentSym->parents, RPSet);
      SuperTypes.emplace_back(std::move(*ParentSym));
    }
  }

  if (Pattern) {
    RPSet.erase(Pattern);
  }
}

const CXXRecordDecl *findRecordTypeAt(ParsedAST &AST, Position Pos) {
  const SourceManager &SM = AST.getSourceManager();
  SourceLocation SourceLocationBeg = SM.getMacroArgExpandedLocation(
      getBeginningOfIdentifier(Pos, SM, AST.getLangOpts()));
  DeclRelationSet Relations =
      DeclRelation::TemplatePattern | DeclRelation::Underlying;
  auto Decls = getDeclAtPosition(AST, SourceLocationBeg, Relations);
  if (Decls.empty())
    return nullptr;

  const Decl *D = Decls[0];

  if (const VarDecl *VD = dyn_cast<VarDecl>(D)) {
    // If this is a variable, use the type of the variable.
    return VD->getType().getTypePtr()->getAsCXXRecordDecl();
  }

  if (const CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(D)) {
    // If this is a method, use the type of the class.
    return Method->getParent();
  }

  // We don't handle FieldDecl because it's not clear what behaviour
  // the user would expect: the enclosing class type (as with a
  // method), or the field's type (as with a variable).

  return dyn_cast<CXXRecordDecl>(D);
}

std::vector<const CXXRecordDecl *> typeParents(const CXXRecordDecl *CXXRD) {
  std::vector<const CXXRecordDecl *> Result;

  for (auto Base : CXXRD->bases()) {
    const CXXRecordDecl *ParentDecl = nullptr;

    const Type *Type = Base.getType().getTypePtr();
    if (const RecordType *RT = Type->getAs<RecordType>()) {
      ParentDecl = RT->getAsCXXRecordDecl();
    }

    if (!ParentDecl) {
      // Handle a dependent base such as "Base<T>" by using the primary
      // template.
      if (const TemplateSpecializationType *TS =
              Type->getAs<TemplateSpecializationType>()) {
        TemplateName TN = TS->getTemplateName();
        if (TemplateDecl *TD = TN.getAsTemplateDecl()) {
          ParentDecl = dyn_cast<CXXRecordDecl>(TD->getTemplatedDecl());
        }
      }
    }

    if (ParentDecl)
      Result.push_back(ParentDecl);
  }

  return Result;
}

llvm::Optional<TypeHierarchyItem>
getTypeHierarchy(ParsedAST &AST, Position Pos, int ResolveLevels,
                 TypeHierarchyDirection Direction, const SymbolIndex *Index,
                 PathRef TUPath) {
  const CXXRecordDecl *CXXRD = findRecordTypeAt(AST, Pos);
  if (!CXXRD)
    return llvm::None;

  Optional<TypeHierarchyItem> Result =
      declToTypeHierarchyItem(AST.getASTContext(), *CXXRD);
  if (!Result)
    return Result;

  if (Direction == TypeHierarchyDirection::Parents ||
      Direction == TypeHierarchyDirection::Both) {
    Result->parents.emplace();

    RecursionProtectionSet RPSet;
    fillSuperTypes(*CXXRD, AST.getASTContext(), *Result->parents, RPSet);
  }

  if ((Direction == TypeHierarchyDirection::Children ||
       Direction == TypeHierarchyDirection::Both) &&
      ResolveLevels > 0) {
    Result->children.emplace();

    if (Index) {
      if (Optional<SymbolID> ID = getSymbolID(CXXRD))
        fillSubTypes(*ID, *Result->children, Index, ResolveLevels, TUPath);
    }
  }

  return Result;
}

void resolveTypeHierarchy(TypeHierarchyItem &Item, int ResolveLevels,
                          TypeHierarchyDirection Direction,
                          const SymbolIndex *Index) {
  // We only support typeHierarchy/resolve for children, because for parents
  // we ignore ResolveLevels and return all levels of parents eagerly.
  if (Direction == TypeHierarchyDirection::Parents || ResolveLevels == 0)
    return;

  Item.children.emplace();

  if (Index && Item.data) {
    // We store the item's SymbolID in the 'data' field, and the client
    // passes it back to us in typeHierarchy/resolve.
    if (Expected<SymbolID> ID = SymbolID::fromStr(*Item.data)) {
      fillSubTypes(*ID, *Item.children, Index, ResolveLevels, Item.uri.file());
    }
  }
}

llvm::DenseSet<const Decl *> getNonLocalDeclRefs(ParsedAST &AST,
                                                 const FunctionDecl *FD) {
  if (!FD->hasBody())
    return {};
  llvm::DenseSet<const Decl *> DeclRefs;
  findExplicitReferences(FD, [&](ReferenceLoc Ref) {
    for (const Decl *D : Ref.Targets) {
      if (!index::isFunctionLocalSymbol(D) && !D->isTemplateParameter() &&
          !Ref.IsDecl)
        DeclRefs.insert(D);
    }
  });
  return DeclRefs;
}
} // namespace clangd
} // namespace clang
