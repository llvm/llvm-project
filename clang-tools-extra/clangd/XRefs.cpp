//===--- XRefs.cpp -----------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "XRefs.h"
#include "AST.h"
#include "FindSymbols.h"
#include "FindTarget.h"
#include "HeuristicResolver.h"
#include "ParsedAST.h"
#include "Protocol.h"
#include "Quality.h"
#include "Selection.h"
#include "SourceCode.h"
#include "URI.h"
#include "index/Index.h"
#include "index/Merge.h"
#include "index/Relation.h"
#include "index/SymbolID.h"
#include "index/SymbolLocation.h"
#include "support/Logger.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Attrs.inc"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/Type.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Index/IndexDataConsumer.h"
#include "clang/Index/IndexSymbol.h"
#include "clang/Index/IndexingAction.h"
#include "clang/Index/IndexingOptions.h"
#include "clang/Index/USRGeneration.h"
#include "clang/Tooling/Syntax/Tokens.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <vector>

namespace clang {
namespace clangd {
namespace {

// Returns the single definition of the entity declared by D, if visible.
// In particular:
// - for non-redeclarable kinds (e.g. local vars), return D
// - for kinds that allow multiple definitions (e.g. namespaces), return nullptr
// Kinds of nodes that always return nullptr here will not have definitions
// reported by locateSymbolAt().
const NamedDecl *getDefinition(const NamedDecl *D) {
  assert(D);
  // Decl has one definition that we can find.
  if (const auto *TD = dyn_cast<TagDecl>(D))
    return TD->getDefinition();
  if (const auto *VD = dyn_cast<VarDecl>(D))
    return VD->getDefinition();
  if (const auto *FD = dyn_cast<FunctionDecl>(D))
    return FD->getDefinition();
  if (const auto *CTD = dyn_cast<ClassTemplateDecl>(D))
    if (const auto *RD = CTD->getTemplatedDecl())
      return RD->getDefinition();
  if (const auto *MD = dyn_cast<ObjCMethodDecl>(D)) {
    if (MD->isThisDeclarationADefinition())
      return MD;
    // Look for the method definition inside the implementation decl.
    auto *DeclCtx = cast<Decl>(MD->getDeclContext());
    if (DeclCtx->isInvalidDecl())
      return nullptr;

    if (const auto *CD = dyn_cast<ObjCContainerDecl>(DeclCtx))
      if (const auto *Impl = getCorrespondingObjCImpl(CD))
        return Impl->getMethod(MD->getSelector(), MD->isInstanceMethod());
  }
  if (const auto *CD = dyn_cast<ObjCContainerDecl>(D))
    return getCorrespondingObjCImpl(CD);
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
  // Also use a mock symbol for the index location so that other fields (e.g.
  // definition) are not factored into the preference.
  Symbol ASTSym, IdxSym;
  ASTSym.ID = IdxSym.ID = SymbolID("mock_symbol_id");
  ASTSym.CanonicalDeclaration = toIndexLocation(ASTLoc, Scratch);
  IdxSym.CanonicalDeclaration = IdxLoc;
  auto Merged = mergeSymbol(ASTSym, IdxSym);
  return Merged.CanonicalDeclaration;
}

std::vector<std::pair<const NamedDecl *, DeclRelationSet>>
getDeclAtPositionWithRelations(ParsedAST &AST, SourceLocation Pos,
                               DeclRelationSet Relations,
                               ASTNodeKind *NodeKind = nullptr) {
  unsigned Offset = AST.getSourceManager().getDecomposedSpellingLoc(Pos).second;
  std::vector<std::pair<const NamedDecl *, DeclRelationSet>> Result;
  auto ResultFromTree = [&](SelectionTree ST) {
    if (const SelectionTree::Node *N = ST.commonAncestor()) {
      if (NodeKind)
        *NodeKind = N->ASTNode.getNodeKind();
      // Attributes don't target decls, look at the
      // thing it's attached to.
      // We still report the original NodeKind!
      // This makes the `override` hack work.
      if (N->ASTNode.get<Attr>() && N->Parent)
        N = N->Parent;
      llvm::copy_if(allTargetDecls(N->ASTNode, AST.getHeuristicResolver()),
                    std::back_inserter(Result),
                    [&](auto &Entry) { return !(Entry.second & ~Relations); });
    }
    return !Result.empty();
  };
  SelectionTree::createEach(AST.getASTContext(), AST.getTokens(), Offset,
                            Offset, ResultFromTree);
  return Result;
}

std::vector<const NamedDecl *>
getDeclAtPosition(ParsedAST &AST, SourceLocation Pos, DeclRelationSet Relations,
                  ASTNodeKind *NodeKind = nullptr) {
  std::vector<const NamedDecl *> Result;
  for (auto &Entry :
       getDeclAtPositionWithRelations(AST, Pos, Relations, NodeKind))
    Result.push_back(Entry.first);
  return Result;
}

// Expects Loc to be a SpellingLocation, will bail out otherwise as it can't
// figure out a filename.
llvm::Optional<Location> makeLocation(const ASTContext &AST, SourceLocation Loc,
                                      llvm::StringRef TUPath) {
  const auto &SM = AST.getSourceManager();
  const FileEntry *F = SM.getFileEntryForID(SM.getFileID(Loc));
  if (!F)
    return None;
  auto FilePath = getCanonicalPath(F, SM);
  if (!FilePath) {
    log("failed to get path!");
    return None;
  }
  Location L;
  L.uri = URIForFile::canonicalize(*FilePath, TUPath);
  // We call MeasureTokenLength here as TokenBuffer doesn't store spelled tokens
  // outside the main file.
  auto TokLen = Lexer::MeasureTokenLength(Loc, SM, AST.getLangOpts());
  L.range = halfOpenToRange(
      SM, CharSourceRange::getCharRange(Loc, Loc.getLocWithOffset(TokLen)));
  return L;
}

// Treat #included files as symbols, to enable go-to-definition on them.
llvm::Optional<LocatedSymbol> locateFileReferent(const Position &Pos,
                                                 ParsedAST &AST,
                                                 llvm::StringRef MainFilePath) {
  for (auto &Inc : AST.getIncludeStructure().MainFileIncludes) {
    if (!Inc.Resolved.empty() && Inc.HashLine == Pos.line) {
      LocatedSymbol File;
      File.Name = std::string(llvm::sys::path::filename(Inc.Resolved));
      File.PreferredDeclaration = {
          URIForFile::canonicalize(Inc.Resolved, MainFilePath), Range{}};
      File.Definition = File.PreferredDeclaration;
      // We're not going to find any further symbols on #include lines.
      return File;
    }
  }
  return llvm::None;
}

// Macros are simple: there's no declaration/definition distinction.
// As a consequence, there's no need to look them up in the index either.
llvm::Optional<LocatedSymbol>
locateMacroReferent(const syntax::Token &TouchedIdentifier, ParsedAST &AST,
                    llvm::StringRef MainFilePath) {
  if (auto M = locateMacroAt(TouchedIdentifier, AST.getPreprocessor())) {
    if (auto Loc =
            makeLocation(AST.getASTContext(), M->NameLoc, MainFilePath)) {
      LocatedSymbol Macro;
      Macro.Name = std::string(M->Name);
      Macro.PreferredDeclaration = *Loc;
      Macro.Definition = Loc;
      Macro.ID = getSymbolID(M->Name, M->Info, AST.getSourceManager());
      return Macro;
    }
  }
  return llvm::None;
}

// A wrapper around `Decl::getCanonicalDecl` to support cases where Clang's
// definition of a canonical declaration doesn't match up to what a programmer
// would expect. For example, Objective-C classes can have three types of
// declarations:
//
// - forward declaration(s): @class MyClass;
// - true declaration (interface definition): @interface MyClass ... @end
// - true definition (implementation): @implementation MyClass ... @end
//
// Clang will consider the forward declaration to be the canonical declaration
// because it is first. We actually want the class definition if it is
// available since that is what a programmer would consider the primary
// declaration to be.
const NamedDecl *getPreferredDecl(const NamedDecl *D) {
  // FIXME: Canonical declarations of some symbols might refer to built-in
  // decls with possibly-invalid source locations (e.g. global new operator).
  // In such cases we should pick up a redecl with valid source location
  // instead of failing.
  D = llvm::cast<NamedDecl>(D->getCanonicalDecl());

  // Prefer Objective-C class/protocol definitions over the forward declaration.
  if (const auto *ID = dyn_cast<ObjCInterfaceDecl>(D))
    if (const auto *DefinitionID = ID->getDefinition())
      return DefinitionID;
  if (const auto *PD = dyn_cast<ObjCProtocolDecl>(D))
    if (const auto *DefinitionID = PD->getDefinition())
      return DefinitionID;

  return D;
}

std::vector<LocatedSymbol> findImplementors(llvm::DenseSet<SymbolID> IDs,
                                            RelationKind Predicate,
                                            const SymbolIndex *Index,
                                            llvm::StringRef MainFilePath) {
  if (IDs.empty() || !Index)
    return {};
  static constexpr trace::Metric FindImplementorsMetric(
      "find_implementors", trace::Metric::Counter, "case");
  switch (Predicate) {
  case RelationKind::BaseOf:
    FindImplementorsMetric.record(1, "find-base");
    break;
  case RelationKind::OverriddenBy:
    FindImplementorsMetric.record(1, "find-override");
    break;
  }

  RelationsRequest Req;
  Req.Predicate = Predicate;
  Req.Subjects = std::move(IDs);
  std::vector<LocatedSymbol> Results;
  Index->relations(Req, [&](const SymbolID &Subject, const Symbol &Object) {
    auto DeclLoc =
        indexToLSPLocation(Object.CanonicalDeclaration, MainFilePath);
    if (!DeclLoc) {
      elog("Find overrides: {0}", DeclLoc.takeError());
      return;
    }
    Results.emplace_back();
    Results.back().Name = Object.Name.str();
    Results.back().PreferredDeclaration = *DeclLoc;
    auto DefLoc = indexToLSPLocation(Object.Definition, MainFilePath);
    if (!DefLoc) {
      elog("Failed to convert location: {0}", DefLoc.takeError());
      return;
    }
    Results.back().Definition = *DefLoc;
  });
  return Results;
}

// Decls are more complicated.
// The AST contains at least a declaration, maybe a definition.
// These are up-to-date, and so generally preferred over index results.
// We perform a single batch index lookup to find additional definitions.
std::vector<LocatedSymbol>
locateASTReferent(SourceLocation CurLoc, const syntax::Token *TouchedIdentifier,
                  ParsedAST &AST, llvm::StringRef MainFilePath,
                  const SymbolIndex *Index, ASTNodeKind &NodeKind) {
  const SourceManager &SM = AST.getSourceManager();
  // Results follow the order of Symbols.Decls.
  std::vector<LocatedSymbol> Result;
  // Keep track of SymbolID -> index mapping, to fill in index data later.
  llvm::DenseMap<SymbolID, size_t> ResultIndex;

  static constexpr trace::Metric LocateASTReferentMetric(
      "locate_ast_referent", trace::Metric::Counter, "case");
  auto AddResultDecl = [&](const NamedDecl *D) {
    D = getPreferredDecl(D);
    auto Loc =
        makeLocation(AST.getASTContext(), nameLocation(*D, SM), MainFilePath);
    if (!Loc)
      return;

    Result.emplace_back();
    Result.back().Name = printName(AST.getASTContext(), *D);
    Result.back().PreferredDeclaration = *Loc;
    Result.back().ID = getSymbolID(D);
    if (const NamedDecl *Def = getDefinition(D))
      Result.back().Definition = makeLocation(
          AST.getASTContext(), nameLocation(*Def, SM), MainFilePath);

    // Record SymbolID for index lookup later.
    if (auto ID = getSymbolID(D))
      ResultIndex[ID] = Result.size() - 1;
  };

  // Emit all symbol locations (declaration or definition) from AST.
  DeclRelationSet Relations =
      DeclRelation::TemplatePattern | DeclRelation::Alias;
  auto Candidates =
      getDeclAtPositionWithRelations(AST, CurLoc, Relations, &NodeKind);
  llvm::DenseSet<SymbolID> VirtualMethods;
  for (const auto &E : Candidates) {
    const NamedDecl *D = E.first;
    if (const auto *CMD = llvm::dyn_cast<CXXMethodDecl>(D)) {
      // Special case: virtual void ^method() = 0: jump to all overrides.
      // FIXME: extend it to ^virtual, unfortunately, virtual location is not
      // saved in the AST.
      if (CMD->isPure()) {
        if (TouchedIdentifier && SM.getSpellingLoc(CMD->getLocation()) ==
                                     TouchedIdentifier->location()) {
          VirtualMethods.insert(getSymbolID(CMD));
          LocateASTReferentMetric.record(1, "method-to-override");
        }
      }
      // Special case: void foo() ^override: jump to the overridden method.
      if (NodeKind.isSame(ASTNodeKind::getFromNodeKind<OverrideAttr>()) ||
          NodeKind.isSame(ASTNodeKind::getFromNodeKind<FinalAttr>())) {
        // We may be overridding multiple methods - offer them all.
        for (const NamedDecl *ND : CMD->overridden_methods())
          AddResultDecl(ND);
        continue;
      }
    }

    // Special case: the cursor is on an alias, prefer other results.
    // This targets "using ns::^Foo", where the target is more interesting.
    // This does not trigger on renaming aliases:
    //   `using Foo = ^Bar` already targets Bar via a TypeLoc
    //   `using ^Foo = Bar` has no other results, as Underlying is filtered.
    if (E.second & DeclRelation::Alias && Candidates.size() > 1 &&
        // beginLoc/endLoc are a token range, so rewind the identifier we're in.
        SM.isPointWithin(TouchedIdentifier ? TouchedIdentifier->location()
                                           : CurLoc,
                         D->getBeginLoc(), D->getEndLoc()))
      continue;

    // Special case: the point of declaration of a template specialization,
    // it's more useful to navigate to the template declaration.
    if (auto *CTSD = dyn_cast<ClassTemplateSpecializationDecl>(D)) {
      if (TouchedIdentifier &&
          D->getLocation() == TouchedIdentifier->location()) {
        LocateASTReferentMetric.record(1, "template-specialization-to-primary");
        AddResultDecl(CTSD->getSpecializedTemplate());
        continue;
      }
    }

    // Special case: if the class name is selected, also map Objective-C
    // categories and category implementations back to their class interface.
    //
    // Since `TouchedIdentifier` might refer to the `ObjCCategoryImplDecl`
    // instead of the `ObjCCategoryDecl` we intentionally check the contents
    // of the locs when checking for class name equivalence.
    if (const auto *CD = dyn_cast<ObjCCategoryDecl>(D))
      if (const auto *ID = CD->getClassInterface())
        if (TouchedIdentifier &&
            (CD->getLocation() == TouchedIdentifier->location() ||
             ID->getName() == TouchedIdentifier->text(SM))) {
          LocateASTReferentMetric.record(1, "objc-category-to-class");
          AddResultDecl(ID);
        }

    LocateASTReferentMetric.record(1, "regular");
    // Otherwise the target declaration is the right one.
    AddResultDecl(D);
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
        if (auto Loc = toLSPLocation(Sym.CanonicalDeclaration, MainFilePath))
          R.PreferredDeclaration = *Loc;

        // We might still prefer the definition from the index, e.g. for
        // generated symbols.
        if (auto Loc = toLSPLocation(
                getPreferredLocation(*R.Definition, Sym.Definition, Scratch),
                MainFilePath))
          R.Definition = *Loc;
      } else {
        R.Definition = toLSPLocation(Sym.Definition, MainFilePath);

        // Use merge logic to choose AST or index declaration.
        if (auto Loc = toLSPLocation(
                getPreferredLocation(R.PreferredDeclaration,
                                     Sym.CanonicalDeclaration, Scratch),
                MainFilePath))
          R.PreferredDeclaration = *Loc;
      }
    });
  }

  auto Overrides = findImplementors(VirtualMethods, RelationKind::OverriddenBy,
                                    Index, MainFilePath);
  Result.insert(Result.end(), Overrides.begin(), Overrides.end());
  return Result;
}

std::vector<LocatedSymbol> locateSymbolForType(const ParsedAST &AST,
                                               const QualType &Type) {
  const auto &SM = AST.getSourceManager();
  auto MainFilePath = AST.tuPath();

  // FIXME: this sends unique_ptr<Foo> to unique_ptr<T>.
  // Likely it would be better to send it to Foo (heuristically) or to both.
  auto Decls = targetDecl(DynTypedNode::create(Type.getNonReferenceType()),
                          DeclRelation::TemplatePattern | DeclRelation::Alias,
                          AST.getHeuristicResolver());
  if (Decls.empty())
    return {};

  std::vector<LocatedSymbol> Results;
  const auto &ASTContext = AST.getASTContext();

  for (const NamedDecl *D : Decls) {
    D = getPreferredDecl(D);

    auto Loc = makeLocation(ASTContext, nameLocation(*D, SM), MainFilePath);
    if (!Loc)
      continue;

    Results.emplace_back();
    Results.back().Name = printName(ASTContext, *D);
    Results.back().PreferredDeclaration = *Loc;
    Results.back().ID = getSymbolID(D);
    if (const NamedDecl *Def = getDefinition(D))
      Results.back().Definition =
          makeLocation(ASTContext, nameLocation(*Def, SM), MainFilePath);
  }

  return Results;
}

bool tokenSpelledAt(SourceLocation SpellingLoc, const syntax::TokenBuffer &TB) {
  auto ExpandedTokens = TB.expandedTokens(
      TB.sourceManager().getMacroArgExpandedLocation(SpellingLoc));
  return !ExpandedTokens.empty();
}

llvm::StringRef sourcePrefix(SourceLocation Loc, const SourceManager &SM) {
  auto D = SM.getDecomposedLoc(Loc);
  bool Invalid = false;
  llvm::StringRef Buf = SM.getBufferData(D.first, &Invalid);
  if (Invalid || D.second > Buf.size())
    return "";
  return Buf.substr(0, D.second);
}

bool isDependentName(ASTNodeKind NodeKind) {
  return NodeKind.isSame(ASTNodeKind::getFromNodeKind<OverloadExpr>()) ||
         NodeKind.isSame(
             ASTNodeKind::getFromNodeKind<CXXDependentScopeMemberExpr>()) ||
         NodeKind.isSame(
             ASTNodeKind::getFromNodeKind<DependentScopeDeclRefExpr>());
}

} // namespace

std::vector<LocatedSymbol> locateSymbolTextually(const SpelledWord &Word,
                                                 ParsedAST &AST,
                                                 const SymbolIndex *Index,
                                                 llvm::StringRef MainFilePath,
                                                 ASTNodeKind NodeKind) {
  // Don't use heuristics if this is a real identifier, or not an
  // identifier.
  // Exception: dependent names, because those may have useful textual
  // matches that AST-based heuristics cannot find.
  if ((Word.ExpandedToken && !isDependentName(NodeKind)) ||
      !Word.LikelyIdentifier || !Index)
    return {};
  // We don't want to handle words in string literals. (It'd be nice to list
  // *allowed* token kinds explicitly, but comment Tokens aren't retained).
  if (Word.PartOfSpelledToken &&
      isStringLiteral(Word.PartOfSpelledToken->kind()))
    return {};

  const auto &SM = AST.getSourceManager();
  // Look up the selected word in the index.
  FuzzyFindRequest Req;
  Req.Query = Word.Text.str();
  Req.ProximityPaths = {MainFilePath.str()};
  // Find the namespaces to query by lexing the file.
  Req.Scopes =
      visibleNamespaces(sourcePrefix(Word.Location, SM), AST.getLangOpts());
  // FIXME: For extra strictness, consider AnyScope=false.
  Req.AnyScope = true;
  // We limit the results to 3 further below. This limit is to avoid fetching
  // too much data, while still likely having enough for 3 results to remain
  // after additional filtering.
  Req.Limit = 10;
  bool TooMany = false;
  using ScoredLocatedSymbol = std::pair<float, LocatedSymbol>;
  std::vector<ScoredLocatedSymbol> ScoredResults;
  Index->fuzzyFind(Req, [&](const Symbol &Sym) {
    // Only consider exact name matches, including case.
    // This is to avoid too many false positives.
    // We could relax this in the future (e.g. to allow for typos) if we make
    // the query more accurate by other means.
    if (Sym.Name != Word.Text)
      return;

    // Exclude constructor results. They have the same name as the class,
    // but we don't have enough context to prefer them over the class.
    if (Sym.SymInfo.Kind == index::SymbolKind::Constructor)
      return;

    auto MaybeDeclLoc =
        indexToLSPLocation(Sym.CanonicalDeclaration, MainFilePath);
    if (!MaybeDeclLoc) {
      log("locateSymbolNamedTextuallyAt: {0}", MaybeDeclLoc.takeError());
      return;
    }
    LocatedSymbol Located;
    Located.PreferredDeclaration = *MaybeDeclLoc;
    Located.Name = (Sym.Name + Sym.TemplateSpecializationArgs).str();
    Located.ID = Sym.ID;
    if (Sym.Definition) {
      auto MaybeDefLoc = indexToLSPLocation(Sym.Definition, MainFilePath);
      if (!MaybeDefLoc) {
        log("locateSymbolNamedTextuallyAt: {0}", MaybeDefLoc.takeError());
        return;
      }
      Located.PreferredDeclaration = *MaybeDefLoc;
      Located.Definition = *MaybeDefLoc;
    }

    if (ScoredResults.size() >= 5) {
      // If we have more than 5 results, don't return anything,
      // as confidence is too low.
      // FIXME: Alternatively, try a stricter query?
      TooMany = true;
      return;
    }

    SymbolQualitySignals Quality;
    Quality.merge(Sym);
    SymbolRelevanceSignals Relevance;
    Relevance.Name = Sym.Name;
    Relevance.Query = SymbolRelevanceSignals::Generic;
    Relevance.merge(Sym);
    auto Score = evaluateSymbolAndRelevance(Quality.evaluateHeuristics(),
                                            Relevance.evaluateHeuristics());
    dlog("locateSymbolNamedTextuallyAt: {0}{1} = {2}\n{3}{4}\n", Sym.Scope,
         Sym.Name, Score, Quality, Relevance);

    ScoredResults.push_back({Score, std::move(Located)});
  });

  if (TooMany) {
    vlog("Heuristic index lookup for {0} returned too many candidates, ignored",
         Word.Text);
    return {};
  }

  llvm::sort(ScoredResults,
             [](const ScoredLocatedSymbol &A, const ScoredLocatedSymbol &B) {
               return A.first > B.first;
             });
  std::vector<LocatedSymbol> Results;
  for (auto &Res : std::move(ScoredResults))
    Results.push_back(std::move(Res.second));
  if (Results.empty())
    vlog("No heuristic index definition for {0}", Word.Text);
  else
    log("Found definition heuristically in index for {0}", Word.Text);
  return Results;
}

const syntax::Token *findNearbyIdentifier(const SpelledWord &Word,
                                          const syntax::TokenBuffer &TB) {
  // Don't use heuristics if this is a real identifier.
  // Unlikely identifiers are OK if they were used as identifiers nearby.
  if (Word.ExpandedToken)
    return nullptr;
  // We don't want to handle words in string literals. (It'd be nice to list
  // *allowed* token kinds explicitly, but comment Tokens aren't retained).
  if (Word.PartOfSpelledToken &&
      isStringLiteral(Word.PartOfSpelledToken->kind()))
    return {};

  const SourceManager &SM = TB.sourceManager();
  // We prefer the closest possible token, line-wise. Backwards is penalized.
  // Ties are implicitly broken by traversal order (first-one-wins).
  auto File = SM.getFileID(Word.Location);
  unsigned WordLine = SM.getSpellingLineNumber(Word.Location);
  auto Cost = [&](SourceLocation Loc) -> unsigned {
    assert(SM.getFileID(Loc) == File && "spelled token in wrong file?");
    unsigned Line = SM.getSpellingLineNumber(Loc);
    return Line >= WordLine ? Line - WordLine : 2 * (WordLine - Line);
  };
  const syntax::Token *BestTok = nullptr;
  unsigned BestCost = -1;
  // Search bounds are based on word length:
  // - forward: 2^N lines
  // - backward: 2^(N-1) lines.
  unsigned MaxDistance =
      1U << std::min<unsigned>(Word.Text.size(),
                               std::numeric_limits<unsigned>::digits - 1);
  // Line number for SM.translateLineCol() should be one-based, also
  // SM.translateLineCol() can handle line number greater than
  // number of lines in the file.
  // - LineMin = max(1, WordLine + 1 - 2^(N-1))
  // - LineMax = WordLine + 1 + 2^N
  unsigned LineMin =
      WordLine + 1 <= MaxDistance / 2 ? 1 : WordLine + 1 - MaxDistance / 2;
  unsigned LineMax = WordLine + 1 + MaxDistance;
  SourceLocation LocMin = SM.translateLineCol(File, LineMin, 1);
  assert(LocMin.isValid());
  SourceLocation LocMax = SM.translateLineCol(File, LineMax, 1);
  assert(LocMax.isValid());

  // Updates BestTok and BestCost if Tok is a good candidate.
  // May return true if the cost is too high for this token.
  auto Consider = [&](const syntax::Token &Tok) {
    if (Tok.location() < LocMin || Tok.location() > LocMax)
      return true; // we are too far from the word, break the outer loop.
    if (!(Tok.kind() == tok::identifier && Tok.text(SM) == Word.Text))
      return false;
    // No point guessing the same location we started with.
    if (Tok.location() == Word.Location)
      return false;
    // We've done cheap checks, compute cost so we can break the caller's loop.
    unsigned TokCost = Cost(Tok.location());
    if (TokCost >= BestCost)
      return true; // causes the outer loop to break.
    // Allow locations that might be part of the AST, and macros (even if empty)
    // but not things like disabled preprocessor sections.
    if (!(tokenSpelledAt(Tok.location(), TB) || TB.expansionStartingAt(&Tok)))
      return false;
    // We already verified this token is an improvement.
    BestCost = TokCost;
    BestTok = &Tok;
    return false;
  };
  auto SpelledTokens = TB.spelledTokens(File);
  // Find where the word occurred in the token stream, to search forward & back.
  auto *I = llvm::partition_point(SpelledTokens, [&](const syntax::Token &T) {
    assert(SM.getFileID(T.location()) == SM.getFileID(Word.Location));
    return T.location() < Word.Location; // Comparison OK: same file.
  });
  // Search for matches after the cursor.
  for (const syntax::Token &Tok : llvm::makeArrayRef(I, SpelledTokens.end()))
    if (Consider(Tok))
      break; // costs of later tokens are greater...
  // Search for matches before the cursor.
  for (const syntax::Token &Tok :
       llvm::reverse(llvm::makeArrayRef(SpelledTokens.begin(), I)))
    if (Consider(Tok))
      break;

  if (BestTok)
    vlog(
        "Word {0} under cursor {1} isn't a token (after PP), trying nearby {2}",
        Word.Text, Word.Location.printToString(SM),
        BestTok->location().printToString(SM));

  return BestTok;
}

std::vector<LocatedSymbol> locateSymbolAt(ParsedAST &AST, Position Pos,
                                          const SymbolIndex *Index) {
  const auto &SM = AST.getSourceManager();
  auto MainFilePath = AST.tuPath();

  if (auto File = locateFileReferent(Pos, AST, MainFilePath))
    return {std::move(*File)};

  auto CurLoc = sourceLocationInMainFile(SM, Pos);
  if (!CurLoc) {
    elog("locateSymbolAt failed to convert position to source location: {0}",
         CurLoc.takeError());
    return {};
  }

  const syntax::Token *TouchedIdentifier = nullptr;
  auto TokensTouchingCursor =
      syntax::spelledTokensTouching(*CurLoc, AST.getTokens());
  for (const syntax::Token &Tok : TokensTouchingCursor) {
    if (Tok.kind() == tok::identifier) {
      if (auto Macro = locateMacroReferent(Tok, AST, MainFilePath))
        // Don't look at the AST or index if we have a macro result.
        // (We'd just return declarations referenced from the macro's
        // expansion.)
        return {*std::move(Macro)};

      TouchedIdentifier = &Tok;
      break;
    }

    if (Tok.kind() == tok::kw_auto || Tok.kind() == tok::kw_decltype) {
      // go-to-definition on auto should find the definition of the deduced
      // type, if possible
      if (auto Deduced = getDeducedType(AST.getASTContext(), Tok.location())) {
        auto LocSym = locateSymbolForType(AST, *Deduced);
        if (!LocSym.empty())
          return LocSym;
      }
    }
  }

  ASTNodeKind NodeKind;
  auto ASTResults = locateASTReferent(*CurLoc, TouchedIdentifier, AST,
                                      MainFilePath, Index, NodeKind);
  if (!ASTResults.empty())
    return ASTResults;

  // If the cursor can't be resolved directly, try fallback strategies.
  auto Word =
      SpelledWord::touching(*CurLoc, AST.getTokens(), AST.getLangOpts());
  if (Word) {
    // Is the same word nearby a real identifier that might refer to something?
    if (const syntax::Token *NearbyIdent =
            findNearbyIdentifier(*Word, AST.getTokens())) {
      if (auto Macro = locateMacroReferent(*NearbyIdent, AST, MainFilePath)) {
        log("Found macro definition heuristically using nearby identifier {0}",
            Word->Text);
        return {*std::move(Macro)};
      }
      ASTResults = locateASTReferent(NearbyIdent->location(), NearbyIdent, AST,
                                     MainFilePath, Index, NodeKind);
      if (!ASTResults.empty()) {
        log("Found definition heuristically using nearby identifier {0}",
            NearbyIdent->text(SM));
        return ASTResults;
      }
      vlog("No definition found using nearby identifier {0} at {1}", Word->Text,
           Word->Location.printToString(SM));
    }
    // No nearby word, or it didn't refer to anything either. Try the index.
    auto TextualResults =
        locateSymbolTextually(*Word, AST, Index, MainFilePath, NodeKind);
    if (!TextualResults.empty())
      return TextualResults;
  }

  return {};
}

std::vector<DocumentLink> getDocumentLinks(ParsedAST &AST) {
  const auto &SM = AST.getSourceManager();

  std::vector<DocumentLink> Result;
  for (auto &Inc : AST.getIncludeStructure().MainFileIncludes) {
    if (Inc.Resolved.empty())
      continue;
    auto HashLoc = SM.getComposedLoc(SM.getMainFileID(), Inc.HashOffset);
    const auto *HashTok = AST.getTokens().spelledTokenAt(HashLoc);
    assert(HashTok && "got inclusion at wrong offset");
    const auto *IncludeTok = std::next(HashTok);
    const auto *FileTok = std::next(IncludeTok);
    // FileTok->range is not sufficient here, as raw lexing wouldn't yield
    // correct tokens for angled filenames. Hence we explicitly use
    // Inc.Written's length.
    auto FileRange =
        syntax::FileRange(SM, FileTok->location(), Inc.Written.length())
            .toCharRange(SM);

    Result.push_back(
        DocumentLink({halfOpenToRange(SM, FileRange),
                      URIForFile::canonicalize(Inc.Resolved, AST.tuPath())}));
  }

  return Result;
}

namespace {

/// Collects references to symbols within the main file.
class ReferenceFinder : public index::IndexDataConsumer {
public:
  struct Reference {
    syntax::Token SpelledTok;
    index::SymbolRoleSet Role;
    SymbolID Target;

    Range range(const SourceManager &SM) const {
      return halfOpenToRange(SM, SpelledTok.range(SM).toCharRange(SM));
    }
  };

  ReferenceFinder(const ParsedAST &AST,
                  const llvm::ArrayRef<const NamedDecl *> Targets,
                  bool PerToken)
      : PerToken(PerToken), AST(AST) {
    for (const NamedDecl *ND : Targets) {
      const Decl *CD = ND->getCanonicalDecl();
      TargetDeclToID[CD] = getSymbolID(CD);
    }
  }

  std::vector<Reference> take() && {
    llvm::sort(References, [](const Reference &L, const Reference &R) {
      auto LTok = L.SpelledTok.location();
      auto RTok = R.SpelledTok.location();
      return std::tie(LTok, L.Role) < std::tie(RTok, R.Role);
    });
    // We sometimes see duplicates when parts of the AST get traversed twice.
    References.erase(std::unique(References.begin(), References.end(),
                                 [](const Reference &L, const Reference &R) {
                                   auto LTok = L.SpelledTok.location();
                                   auto RTok = R.SpelledTok.location();
                                   return std::tie(LTok, L.Role) ==
                                          std::tie(RTok, R.Role);
                                 }),
                     References.end());
    return std::move(References);
  }

  bool
  handleDeclOccurrence(const Decl *D, index::SymbolRoleSet Roles,
                       llvm::ArrayRef<index::SymbolRelation> Relations,
                       SourceLocation Loc,
                       index::IndexDataConsumer::ASTNodeInfo ASTNode) override {
    auto DeclID = TargetDeclToID.find(D->getCanonicalDecl());
    if (DeclID == TargetDeclToID.end())
      return true;
    const SourceManager &SM = AST.getSourceManager();
    if (!isInsideMainFile(Loc, SM))
      return true;
    const auto &TB = AST.getTokens();

    llvm::SmallVector<SourceLocation, 1> Locs;
    if (PerToken) {
      // Check whether this is one of the few constructs where the reference
      // can be split over several tokens.
      if (auto *OME = llvm::dyn_cast_or_null<ObjCMessageExpr>(ASTNode.OrigE)) {
        OME->getSelectorLocs(Locs);
      } else if (auto *OMD =
                     llvm::dyn_cast_or_null<ObjCMethodDecl>(ASTNode.OrigD)) {
        OMD->getSelectorLocs(Locs);
      }
      // Sanity check: we expect the *first* token to match the reported loc.
      // Otherwise, maybe it was e.g. some other kind of reference to a Decl.
      if (!Locs.empty() && Locs.front() != Loc)
        Locs.clear(); // First token doesn't match, assume our guess was wrong.
    }
    if (Locs.empty())
      Locs.push_back(Loc);

    for (SourceLocation L : Locs) {
      L = SM.getFileLoc(L);
      if (const auto *Tok = TB.spelledTokenAt(L))
        References.push_back({*Tok, Roles, DeclID->getSecond()});
    }
    return true;
  }

private:
  bool PerToken; // If true, report 3 references for split ObjC selector names.
  std::vector<Reference> References;
  const ParsedAST &AST;
  llvm::DenseMap<const Decl *, SymbolID> TargetDeclToID;
};

std::vector<ReferenceFinder::Reference>
findRefs(const llvm::ArrayRef<const NamedDecl *> TargetDecls, ParsedAST &AST,
         bool PerToken) {
  ReferenceFinder RefFinder(AST, TargetDecls, PerToken);
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

const Stmt *getFunctionBody(DynTypedNode N) {
  if (const auto *FD = N.get<FunctionDecl>())
    return FD->getBody();
  if (const auto *FD = N.get<BlockDecl>())
    return FD->getBody();
  if (const auto *FD = N.get<LambdaExpr>())
    return FD->getBody();
  if (const auto *FD = N.get<ObjCMethodDecl>())
    return FD->getBody();
  return nullptr;
}

const Stmt *getLoopBody(DynTypedNode N) {
  if (const auto *LS = N.get<ForStmt>())
    return LS->getBody();
  if (const auto *LS = N.get<CXXForRangeStmt>())
    return LS->getBody();
  if (const auto *LS = N.get<WhileStmt>())
    return LS->getBody();
  if (const auto *LS = N.get<DoStmt>())
    return LS->getBody();
  return nullptr;
}

// AST traversal to highlight control flow statements under some root.
// Once we hit further control flow we prune the tree (or at least restrict
// what we highlight) so we capture e.g. breaks from the outer loop only.
class FindControlFlow : public RecursiveASTVisitor<FindControlFlow> {
  // Types of control-flow statements we might highlight.
  enum Target {
    Break = 1,
    Continue = 2,
    Return = 4,
    Case = 8,
    Throw = 16,
    Goto = 32,
    All = Break | Continue | Return | Case | Throw | Goto,
  };
  int Ignore = 0;     // bitmask of Target - what are we *not* highlighting?
  SourceRange Bounds; // Half-open, restricts reported targets.
  std::vector<SourceLocation> &Result;
  const SourceManager &SM;

  // Masks out targets for a traversal into D.
  // Traverses the subtree using Delegate() if any targets remain.
  template <typename Func>
  bool filterAndTraverse(DynTypedNode D, const Func &Delegate) {
    auto RestoreIgnore = llvm::make_scope_exit(
        [OldIgnore(Ignore), this] { Ignore = OldIgnore; });
    if (getFunctionBody(D))
      Ignore = All;
    else if (getLoopBody(D))
      Ignore |= Continue | Break;
    else if (D.get<SwitchStmt>())
      Ignore |= Break | Case;
    // Prune tree if we're not looking for anything.
    return (Ignore == All) ? true : Delegate();
  }

  void found(Target T, SourceLocation Loc) {
    if (T & Ignore)
      return;
    if (SM.isBeforeInTranslationUnit(Loc, Bounds.getBegin()) ||
        SM.isBeforeInTranslationUnit(Bounds.getEnd(), Loc))
      return;
    Result.push_back(Loc);
  }

public:
  FindControlFlow(SourceRange Bounds, std::vector<SourceLocation> &Result,
                  const SourceManager &SM)
      : Bounds(Bounds), Result(Result), SM(SM) {}

  // When traversing function or loops, limit targets to those that still
  // refer to the original root.
  bool TraverseDecl(Decl *D) {
    return !D || filterAndTraverse(DynTypedNode::create(*D), [&] {
      return RecursiveASTVisitor::TraverseDecl(D);
    });
  }
  bool TraverseStmt(Stmt *S) {
    return !S || filterAndTraverse(DynTypedNode::create(*S), [&] {
      return RecursiveASTVisitor::TraverseStmt(S);
    });
  }

  // Add leaves that we found and want.
  bool VisitReturnStmt(ReturnStmt *R) {
    found(Return, R->getReturnLoc());
    return true;
  }
  bool VisitBreakStmt(BreakStmt *B) {
    found(Break, B->getBreakLoc());
    return true;
  }
  bool VisitContinueStmt(ContinueStmt *C) {
    found(Continue, C->getContinueLoc());
    return true;
  }
  bool VisitSwitchCase(SwitchCase *C) {
    found(Case, C->getKeywordLoc());
    return true;
  }
  bool VisitCXXThrowExpr(CXXThrowExpr *T) {
    found(Throw, T->getThrowLoc());
    return true;
  }
  bool VisitGotoStmt(GotoStmt *G) {
    // Goto is interesting if its target is outside the root.
    if (const auto *LD = G->getLabel()) {
      if (SM.isBeforeInTranslationUnit(LD->getLocation(), Bounds.getBegin()) ||
          SM.isBeforeInTranslationUnit(Bounds.getEnd(), LD->getLocation()))
        found(Goto, G->getGotoLoc());
    }
    return true;
  }
};

// Given a location within a switch statement, return the half-open range that
// covers the case it's contained in.
// We treat `case X: case Y: ...` as one case, and assume no other fallthrough.
SourceRange findCaseBounds(const SwitchStmt &Switch, SourceLocation Loc,
                           const SourceManager &SM) {
  // Cases are not stored in order, sort them first.
  // (In fact they seem to be stored in reverse order, don't rely on this)
  std::vector<const SwitchCase *> Cases;
  for (const SwitchCase *Case = Switch.getSwitchCaseList(); Case;
       Case = Case->getNextSwitchCase())
    Cases.push_back(Case);
  llvm::sort(Cases, [&](const SwitchCase *L, const SwitchCase *R) {
    return SM.isBeforeInTranslationUnit(L->getKeywordLoc(), R->getKeywordLoc());
  });

  // Find the first case after the target location, the end of our range.
  auto CaseAfter = llvm::partition_point(Cases, [&](const SwitchCase *C) {
    return !SM.isBeforeInTranslationUnit(Loc, C->getKeywordLoc());
  });
  SourceLocation End = CaseAfter == Cases.end() ? Switch.getEndLoc()
                                                : (*CaseAfter)->getKeywordLoc();

  // Our target can be before the first case - cases are optional!
  if (CaseAfter == Cases.begin())
    return SourceRange(Switch.getBeginLoc(), End);
  // The start of our range is usually the previous case, but...
  auto CaseBefore = std::prev(CaseAfter);
  // ... rewind CaseBefore to the first in a `case A: case B: ...` sequence.
  while (CaseBefore != Cases.begin() &&
         (*std::prev(CaseBefore))->getSubStmt() == *CaseBefore)
    --CaseBefore;
  return SourceRange((*CaseBefore)->getKeywordLoc(), End);
}

// Returns the locations of control flow statements related to N. e.g.:
//   for    => branches: break/continue/return/throw
//   break  => controlling loop (forwhile/do), and its related control flow
//   return => all returns/throws from the same function
// When an inner block is selected, we include branches bound to outer blocks
// as these are exits from the inner block. e.g. return in a for loop.
// FIXME: We don't analyze catch blocks, throw is treated the same as return.
std::vector<SourceLocation> relatedControlFlow(const SelectionTree::Node &N) {
  const SourceManager &SM =
      N.getDeclContext().getParentASTContext().getSourceManager();
  std::vector<SourceLocation> Result;

  // First, check if we're at a node that can resolve to a root.
  enum class Cur { None, Break, Continue, Return, Case, Throw } Cursor;
  if (N.ASTNode.get<BreakStmt>()) {
    Cursor = Cur::Break;
  } else if (N.ASTNode.get<ContinueStmt>()) {
    Cursor = Cur::Continue;
  } else if (N.ASTNode.get<ReturnStmt>()) {
    Cursor = Cur::Return;
  } else if (N.ASTNode.get<CXXThrowExpr>()) {
    Cursor = Cur::Throw;
  } else if (N.ASTNode.get<SwitchCase>()) {
    Cursor = Cur::Case;
  } else if (const GotoStmt *GS = N.ASTNode.get<GotoStmt>()) {
    // We don't know what root to associate with, but highlight the goto/label.
    Result.push_back(GS->getGotoLoc());
    if (const auto *LD = GS->getLabel())
      Result.push_back(LD->getLocation());
    Cursor = Cur::None;
  } else {
    Cursor = Cur::None;
  }

  const Stmt *Root = nullptr; // Loop or function body to traverse.
  SourceRange Bounds;
  // Look up the tree for a root (or just at this node if we didn't find a leaf)
  for (const auto *P = &N; P; P = P->Parent) {
    // return associates with enclosing function
    if (const Stmt *FunctionBody = getFunctionBody(P->ASTNode)) {
      if (Cursor == Cur::Return || Cursor == Cur::Throw) {
        Root = FunctionBody;
      }
      break; // other leaves don't cross functions.
    }
    // break/continue associate with enclosing loop.
    if (const Stmt *LoopBody = getLoopBody(P->ASTNode)) {
      if (Cursor == Cur::None || Cursor == Cur::Break ||
          Cursor == Cur::Continue) {
        Root = LoopBody;
        // Highlight the loop keyword itself.
        // FIXME: for do-while, this only covers the `do`..
        Result.push_back(P->ASTNode.getSourceRange().getBegin());
        break;
      }
    }
    // For switches, users think of case statements as control flow blocks.
    // We highlight only occurrences surrounded by the same case.
    // We don't detect fallthrough (other than 'case X, case Y').
    if (const auto *SS = P->ASTNode.get<SwitchStmt>()) {
      if (Cursor == Cur::Break || Cursor == Cur::Case) {
        Result.push_back(SS->getSwitchLoc()); // Highlight the switch.
        Root = SS->getBody();
        // Limit to enclosing case, if there is one.
        Bounds = findCaseBounds(*SS, N.ASTNode.getSourceRange().getBegin(), SM);
        break;
      }
    }
    // If we didn't start at some interesting node, we're done.
    if (Cursor == Cur::None)
      break;
  }
  if (Root) {
    if (!Bounds.isValid())
      Bounds = Root->getSourceRange();
    FindControlFlow(Bounds, Result, SM).TraverseStmt(const_cast<Stmt *>(Root));
  }
  return Result;
}

DocumentHighlight toHighlight(const ReferenceFinder::Reference &Ref,
                              const SourceManager &SM) {
  DocumentHighlight DH;
  DH.range = Ref.range(SM);
  if (Ref.Role & index::SymbolRoleSet(index::SymbolRole::Write))
    DH.kind = DocumentHighlightKind::Write;
  else if (Ref.Role & index::SymbolRoleSet(index::SymbolRole::Read))
    DH.kind = DocumentHighlightKind::Read;
  else
    DH.kind = DocumentHighlightKind::Text;
  return DH;
}

llvm::Optional<DocumentHighlight> toHighlight(SourceLocation Loc,
                                              const syntax::TokenBuffer &TB) {
  Loc = TB.sourceManager().getFileLoc(Loc);
  if (const auto *Tok = TB.spelledTokenAt(Loc)) {
    DocumentHighlight Result;
    Result.range = halfOpenToRange(
        TB.sourceManager(),
        CharSourceRange::getCharRange(Tok->location(), Tok->endLocation()));
    return Result;
  }
  return llvm::None;
}

} // namespace

std::vector<DocumentHighlight> findDocumentHighlights(ParsedAST &AST,
                                                      Position Pos) {
  const SourceManager &SM = AST.getSourceManager();
  // FIXME: show references to macro within file?
  auto CurLoc = sourceLocationInMainFile(SM, Pos);
  if (!CurLoc) {
    llvm::consumeError(CurLoc.takeError());
    return {};
  }
  std::vector<DocumentHighlight> Result;
  auto TryTree = [&](SelectionTree ST) {
    if (const SelectionTree::Node *N = ST.commonAncestor()) {
      DeclRelationSet Relations =
          DeclRelation::TemplatePattern | DeclRelation::Alias;
      auto TargetDecls =
          targetDecl(N->ASTNode, Relations, AST.getHeuristicResolver());
      if (!TargetDecls.empty()) {
        // FIXME: we may get multiple DocumentHighlights with the same location
        // and different kinds, deduplicate them.
        for (const auto &Ref : findRefs(TargetDecls, AST, /*PerToken=*/true))
          Result.push_back(toHighlight(Ref, SM));
        return true;
      }
      auto ControlFlow = relatedControlFlow(*N);
      if (!ControlFlow.empty()) {
        for (SourceLocation Loc : ControlFlow)
          if (auto Highlight = toHighlight(Loc, AST.getTokens()))
            Result.push_back(std::move(*Highlight));
        return true;
      }
    }
    return false;
  };

  unsigned Offset =
      AST.getSourceManager().getDecomposedSpellingLoc(*CurLoc).second;
  SelectionTree::createEach(AST.getASTContext(), AST.getTokens(), Offset,
                            Offset, TryTree);
  return Result;
}

std::vector<LocatedSymbol> findImplementations(ParsedAST &AST, Position Pos,
                                               const SymbolIndex *Index) {
  // We rely on index to find the implementations in subclasses.
  // FIXME: Index can be stale, so we may loose some latest results from the
  // main file.
  if (!Index)
    return {};
  const SourceManager &SM = AST.getSourceManager();
  auto CurLoc = sourceLocationInMainFile(SM, Pos);
  if (!CurLoc) {
    elog("Failed to convert position to source location: {0}",
         CurLoc.takeError());
    return {};
  }
  DeclRelationSet Relations =
      DeclRelation::TemplatePattern | DeclRelation::Alias;
  llvm::DenseSet<SymbolID> IDs;
  RelationKind QueryKind = RelationKind::OverriddenBy;
  for (const NamedDecl *ND : getDeclAtPosition(AST, *CurLoc, Relations)) {
    if (const auto *CXXMD = llvm::dyn_cast<CXXMethodDecl>(ND)) {
      if (CXXMD->isVirtual()) {
        IDs.insert(getSymbolID(ND));
        QueryKind = RelationKind::OverriddenBy;
      }
    } else if (const auto *RD = dyn_cast<CXXRecordDecl>(ND)) {
      IDs.insert(getSymbolID(RD));
      QueryKind = RelationKind::BaseOf;
    }
  }
  return findImplementors(std::move(IDs), QueryKind, Index, AST.tuPath());
}

namespace {
// Recursively finds all the overridden methods of `CMD` in complete type
// hierarchy.
void getOverriddenMethods(const CXXMethodDecl *CMD,
                          llvm::DenseSet<SymbolID> &OverriddenMethods) {
  if (!CMD)
    return;
  for (const CXXMethodDecl *Base : CMD->overridden_methods()) {
    if (auto ID = getSymbolID(Base))
      OverriddenMethods.insert(ID);
    getOverriddenMethods(Base, OverriddenMethods);
  }
}
} // namespace

ReferencesResult findReferences(ParsedAST &AST, Position Pos, uint32_t Limit,
                                const SymbolIndex *Index) {
  ReferencesResult Results;
  const SourceManager &SM = AST.getSourceManager();
  auto MainFilePath = AST.tuPath();
  auto URIMainFile = URIForFile::canonicalize(MainFilePath, MainFilePath);
  auto CurLoc = sourceLocationInMainFile(SM, Pos);
  if (!CurLoc) {
    llvm::consumeError(CurLoc.takeError());
    return {};
  }

  llvm::DenseSet<SymbolID> IDsToQuery, OverriddenMethods;

  const auto *IdentifierAtCursor =
      syntax::spelledIdentifierTouching(*CurLoc, AST.getTokens());
  llvm::Optional<DefinedMacro> Macro;
  if (IdentifierAtCursor)
    Macro = locateMacroAt(*IdentifierAtCursor, AST.getPreprocessor());
  if (Macro) {
    // Handle references to macro.
    if (auto MacroSID = getSymbolID(Macro->Name, Macro->Info, SM)) {
      // Collect macro references from main file.
      const auto &IDToRefs = AST.getMacros().MacroRefs;
      auto Refs = IDToRefs.find(MacroSID);
      if (Refs != IDToRefs.end()) {
        for (const auto &Ref : Refs->second) {
          ReferencesResult::Reference Result;
          Result.Loc.range = Ref.Rng;
          Result.Loc.uri = URIMainFile;
          if (Ref.IsDefinition) {
            Result.Attributes |= ReferencesResult::Declaration;
            Result.Attributes |= ReferencesResult::Definition;
          }
          Results.References.push_back(std::move(Result));
        }
      }
      IDsToQuery.insert(MacroSID);
    }
  } else {
    // Handle references to Decls.

    DeclRelationSet Relations =
        DeclRelation::TemplatePattern | DeclRelation::Alias;
    std::vector<const NamedDecl *> Decls =
        getDeclAtPosition(AST, *CurLoc, Relations);
    llvm::SmallVector<const NamedDecl *> TargetsInMainFile;
    for (const NamedDecl *D : Decls) {
      auto ID = getSymbolID(D);
      if (!ID)
        continue;
      TargetsInMainFile.push_back(D);
      // Not all symbols can be referenced from outside (e.g. function-locals).
      // TODO: we could skip TU-scoped symbols here (e.g. static functions) if
      // we know this file isn't a header. The details might be tricky.
      if (D->getParentFunctionOrMethod())
        continue;
      IDsToQuery.insert(ID);
    }

    RelationsRequest OverriddenBy;
    if (Index) {
      OverriddenBy.Predicate = RelationKind::OverriddenBy;
      for (const NamedDecl *ND : Decls) {
        // Special case: For virtual methods, report decl/def of overrides and
        // references to all overridden methods in complete type hierarchy.
        if (const auto *CMD = llvm::dyn_cast<CXXMethodDecl>(ND)) {
          if (CMD->isVirtual()) {
            if (auto ID = getSymbolID(CMD))
              OverriddenBy.Subjects.insert(ID);
            getOverriddenMethods(CMD, OverriddenMethods);
          }
        }
      }
    }

    // We traverse the AST to find references in the main file.
    auto MainFileRefs = findRefs(TargetsInMainFile, AST, /*PerToken=*/false);
    // We may get multiple refs with the same location and different Roles, as
    // cross-reference is only interested in locations, we deduplicate them
    // by the location to avoid emitting duplicated locations.
    MainFileRefs.erase(std::unique(MainFileRefs.begin(), MainFileRefs.end(),
                                   [](const ReferenceFinder::Reference &L,
                                      const ReferenceFinder::Reference &R) {
                                     return L.SpelledTok.location() ==
                                            R.SpelledTok.location();
                                   }),
                       MainFileRefs.end());
    for (const auto &Ref : MainFileRefs) {
      ReferencesResult::Reference Result;
      Result.Loc.range = Ref.range(SM);
      Result.Loc.uri = URIMainFile;
      if (Ref.Role & static_cast<unsigned>(index::SymbolRole::Declaration))
        Result.Attributes |= ReferencesResult::Declaration;
      // clang-index doesn't report definitions as declarations, but they are.
      if (Ref.Role & static_cast<unsigned>(index::SymbolRole::Definition))
        Result.Attributes |=
            ReferencesResult::Definition | ReferencesResult::Declaration;
      Results.References.push_back(std::move(Result));
    }
    // Add decl/def of overridding methods.
    if (Index && !OverriddenBy.Subjects.empty()) {
      Index->relations(OverriddenBy, [&](const SymbolID &Subject,
                                         const Symbol &Object) {
        if (Limit && Results.References.size() >= Limit) {
          Results.HasMore = true;
          return;
        }
        const auto LSPLocDecl =
            toLSPLocation(Object.CanonicalDeclaration, MainFilePath);
        const auto LSPLocDef = toLSPLocation(Object.Definition, MainFilePath);
        if (LSPLocDecl && LSPLocDecl != LSPLocDef) {
          ReferencesResult::Reference Result;
          Result.Loc = std::move(*LSPLocDecl);
          Result.Attributes =
              ReferencesResult::Declaration | ReferencesResult::Override;
          Results.References.push_back(std::move(Result));
        }
        if (LSPLocDef) {
          ReferencesResult::Reference Result;
          Result.Loc = std::move(*LSPLocDef);
          Result.Attributes = ReferencesResult::Declaration |
                              ReferencesResult::Definition |
                              ReferencesResult::Override;
          Results.References.push_back(std::move(Result));
        }
      });
    }
  }
  // Now query the index for references from other files.
  auto QueryIndex = [&](llvm::DenseSet<SymbolID> IDs, bool AllowAttributes,
                        bool AllowMainFileSymbols) {
    if (IDs.empty() || !Index || Results.HasMore)
      return;
    RefsRequest Req;
    Req.IDs = std::move(IDs);
    if (Limit) {
      if (Limit < Results.References.size()) {
        // We've already filled our quota, still check the index to correctly
        // return the `HasMore` info.
        Req.Limit = 0;
      } else {
        // Query index only for the remaining size.
        Req.Limit = Limit - Results.References.size();
      }
    }
    Results.HasMore |= Index->refs(Req, [&](const Ref &R) {
      auto LSPLoc = toLSPLocation(R.Location, MainFilePath);
      // Avoid indexed results for the main file - the AST is authoritative.
      if (!LSPLoc ||
          (!AllowMainFileSymbols && LSPLoc->uri.file() == MainFilePath))
        return;
      ReferencesResult::Reference Result;
      Result.Loc = std::move(*LSPLoc);
      if (AllowAttributes) {
        if ((R.Kind & RefKind::Declaration) == RefKind::Declaration)
          Result.Attributes |= ReferencesResult::Declaration;
        // FIXME: our index should definitely store def | decl separately!
        if ((R.Kind & RefKind::Definition) == RefKind::Definition)
          Result.Attributes |=
              ReferencesResult::Declaration | ReferencesResult::Definition;
      }
      Results.References.push_back(std::move(Result));
    });
  };
  QueryIndex(std::move(IDsToQuery), /*AllowAttributes=*/true,
             /*AllowMainFileSymbols=*/false);
  // For a virtual method: Occurrences of BaseMethod should be treated as refs
  // and not as decl/def. Allow symbols from main file since AST does not report
  // these.
  QueryIndex(std::move(OverriddenMethods), /*AllowAttributes=*/false,
             /*AllowMainFileSymbols=*/true);
  return Results;
}

std::vector<SymbolDetails> getSymbolInfo(ParsedAST &AST, Position Pos) {
  const SourceManager &SM = AST.getSourceManager();
  auto CurLoc = sourceLocationInMainFile(SM, Pos);
  if (!CurLoc) {
    llvm::consumeError(CurLoc.takeError());
    return {};
  }
  auto MainFilePath = AST.tuPath();
  std::vector<SymbolDetails> Results;

  // We also want the targets of using-decls, so we include
  // DeclRelation::Underlying.
  DeclRelationSet Relations = DeclRelation::TemplatePattern |
                              DeclRelation::Alias | DeclRelation::Underlying;
  for (const NamedDecl *D : getDeclAtPosition(AST, *CurLoc, Relations)) {
    D = getPreferredDecl(D);

    SymbolDetails NewSymbol;
    std::string QName = printQualifiedName(*D);
    auto SplitQName = splitQualifiedName(QName);
    NewSymbol.containerName = std::string(SplitQName.first);
    NewSymbol.name = std::string(SplitQName.second);

    if (NewSymbol.containerName.empty()) {
      if (const auto *ParentND =
              dyn_cast_or_null<NamedDecl>(D->getDeclContext()))
        NewSymbol.containerName = printQualifiedName(*ParentND);
    }
    llvm::SmallString<32> USR;
    if (!index::generateUSRForDecl(D, USR)) {
      NewSymbol.USR = std::string(USR.str());
      NewSymbol.ID = SymbolID(NewSymbol.USR);
    }
    if (const NamedDecl *Def = getDefinition(D))
      NewSymbol.definitionRange = makeLocation(
          AST.getASTContext(), nameLocation(*Def, SM), MainFilePath);
    NewSymbol.declarationRange =
        makeLocation(AST.getASTContext(), nameLocation(*D, SM), MainFilePath);

    Results.push_back(std::move(NewSymbol));
  }

  const auto *IdentifierAtCursor =
      syntax::spelledIdentifierTouching(*CurLoc, AST.getTokens());
  if (!IdentifierAtCursor)
    return Results;

  if (auto M = locateMacroAt(*IdentifierAtCursor, AST.getPreprocessor())) {
    SymbolDetails NewMacro;
    NewMacro.name = std::string(M->Name);
    llvm::SmallString<32> USR;
    if (!index::generateUSRForMacro(NewMacro.name, M->Info->getDefinitionLoc(),
                                    SM, USR)) {
      NewMacro.USR = std::string(USR.str());
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

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              const ReferencesResult::Reference &R) {
  OS << R.Loc;
  if (R.Attributes & ReferencesResult::Declaration)
    OS << " [decl]";
  if (R.Attributes & ReferencesResult::Definition)
    OS << " [def]";
  if (R.Attributes & ReferencesResult::Override)
    OS << " [override]";
  return OS;
}

template <typename HierarchyItem>
static llvm::Optional<HierarchyItem>
declToHierarchyItem(const NamedDecl &ND, llvm::StringRef TUPath) {
  ASTContext &Ctx = ND.getASTContext();
  auto &SM = Ctx.getSourceManager();
  SourceLocation NameLoc = nameLocation(ND, Ctx.getSourceManager());
  SourceLocation BeginLoc = SM.getSpellingLoc(SM.getFileLoc(ND.getBeginLoc()));
  SourceLocation EndLoc = SM.getSpellingLoc(SM.getFileLoc(ND.getEndLoc()));
  const auto DeclRange =
      toHalfOpenFileRange(SM, Ctx.getLangOpts(), {BeginLoc, EndLoc});
  if (!DeclRange)
    return llvm::None;
  auto FilePath =
      getCanonicalPath(SM.getFileEntryForID(SM.getFileID(NameLoc)), SM);
  if (!FilePath)
    return llvm::None; // Not useful without a uri.

  Position NameBegin = sourceLocToPosition(SM, NameLoc);
  Position NameEnd = sourceLocToPosition(
      SM, Lexer::getLocForEndOfToken(NameLoc, 0, SM, Ctx.getLangOpts()));

  index::SymbolInfo SymInfo = index::getSymbolInfo(&ND);
  // FIXME: This is not classifying constructors, destructors and operators
  // correctly.
  SymbolKind SK = indexSymbolKindToSymbolKind(SymInfo.Kind);

  HierarchyItem HI;
  HI.name = printName(Ctx, ND);
  HI.kind = SK;
  HI.range = Range{sourceLocToPosition(SM, DeclRange->getBegin()),
                   sourceLocToPosition(SM, DeclRange->getEnd())};
  HI.selectionRange = Range{NameBegin, NameEnd};
  if (!HI.range.contains(HI.selectionRange)) {
    // 'selectionRange' must be contained in 'range', so in cases where clang
    // reports unrelated ranges we need to reconcile somehow.
    HI.range = HI.selectionRange;
  }

  HI.uri = URIForFile::canonicalize(*FilePath, TUPath);

  return HI;
}

static llvm::Optional<TypeHierarchyItem>
declToTypeHierarchyItem(const NamedDecl &ND, llvm::StringRef TUPath) {
  auto Result = declToHierarchyItem<TypeHierarchyItem>(ND, TUPath);
  if (Result) {
    Result->deprecated = ND.isDeprecated();
    // Compute the SymbolID and store it in the 'data' field.
    // This allows typeHierarchy/resolve to be used to
    // resolve children of items returned in a previous request
    // for parents.
    Result->data.symbolID = getSymbolID(&ND);
  }
  return Result;
}

static llvm::Optional<CallHierarchyItem>
declToCallHierarchyItem(const NamedDecl &ND, llvm::StringRef TUPath) {
  auto Result = declToHierarchyItem<CallHierarchyItem>(ND, TUPath);
  if (!Result)
    return Result;
  if (ND.isDeprecated())
    Result->tags.push_back(SymbolTag::Deprecated);
  if (auto ID = getSymbolID(&ND))
    Result->data = ID.str();
  return Result;
}

template <typename HierarchyItem>
static llvm::Optional<HierarchyItem> symbolToHierarchyItem(const Symbol &S,
                                                           PathRef TUPath) {
  auto Loc = symbolToLocation(S, TUPath);
  if (!Loc) {
    elog("Failed to convert symbol to hierarchy item: {0}", Loc.takeError());
    return llvm::None;
  }
  HierarchyItem HI;
  HI.name = std::string(S.Name);
  HI.kind = indexSymbolKindToSymbolKind(S.SymInfo.Kind);
  HI.selectionRange = Loc->range;
  // FIXME: Populate 'range' correctly
  // (https://github.com/clangd/clangd/issues/59).
  HI.range = HI.selectionRange;
  HI.uri = Loc->uri;

  return HI;
}

static llvm::Optional<TypeHierarchyItem>
symbolToTypeHierarchyItem(const Symbol &S, PathRef TUPath) {
  auto Result = symbolToHierarchyItem<TypeHierarchyItem>(S, TUPath);
  if (Result) {
    Result->deprecated = (S.Flags & Symbol::Deprecated);
    Result->data.symbolID = S.ID;
  }
  return Result;
}

static llvm::Optional<CallHierarchyItem>
symbolToCallHierarchyItem(const Symbol &S, PathRef TUPath) {
  auto Result = symbolToHierarchyItem<CallHierarchyItem>(S, TUPath);
  if (!Result)
    return Result;
  Result->data = S.ID.str();
  if (S.Flags & Symbol::Deprecated)
    Result->tags.push_back(SymbolTag::Deprecated);
  return Result;
}

static void fillSubTypes(const SymbolID &ID,
                         std::vector<TypeHierarchyItem> &SubTypes,
                         const SymbolIndex *Index, int Levels, PathRef TUPath) {
  RelationsRequest Req;
  Req.Subjects.insert(ID);
  Req.Predicate = RelationKind::BaseOf;
  Index->relations(Req, [&](const SymbolID &Subject, const Symbol &Object) {
    if (Optional<TypeHierarchyItem> ChildSym =
            symbolToTypeHierarchyItem(Object, TUPath)) {
      if (Levels > 1) {
        ChildSym->children.emplace();
        fillSubTypes(Object.ID, *ChildSym->children, Index, Levels - 1, TUPath);
      }
      SubTypes.emplace_back(std::move(*ChildSym));
    }
  });
}

using RecursionProtectionSet = llvm::SmallSet<const CXXRecordDecl *, 4>;

// Extracts parents from AST and populates the type hierarchy item.
static void fillSuperTypes(const CXXRecordDecl &CXXRD, llvm::StringRef TUPath,
                           TypeHierarchyItem &Item,
                           RecursionProtectionSet &RPSet) {
  Item.parents.emplace();
  Item.data.parents.emplace();
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
            declToTypeHierarchyItem(*ParentDecl, TUPath)) {
      fillSuperTypes(*ParentDecl, TUPath, *ParentSym, RPSet);
      Item.data.parents->emplace_back(ParentSym->data);
      Item.parents->emplace_back(std::move(*ParentSym));
    }
  }

  if (Pattern) {
    RPSet.erase(Pattern);
  }
}

std::vector<const CXXRecordDecl *> findRecordTypeAt(ParsedAST &AST,
                                                    Position Pos) {
  auto RecordFromNode = [&AST](const SelectionTree::Node *N) {
    std::vector<const CXXRecordDecl *> Records;
    if (!N)
      return Records;

    // Note: explicitReferenceTargets() will search for both template
    // instantiations and template patterns, and prefer the former if available
    // (generally, one will be available for non-dependent specializations of a
    // class template).
    auto Decls = explicitReferenceTargets(N->ASTNode, DeclRelation::Underlying,
                                          AST.getHeuristicResolver());
    for (const NamedDecl *D : Decls) {

      if (const VarDecl *VD = dyn_cast<VarDecl>(D)) {
        // If this is a variable, use the type of the variable.
        Records.push_back(VD->getType().getTypePtr()->getAsCXXRecordDecl());
        continue;
      }

      if (const CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(D)) {
        // If this is a method, use the type of the class.
        Records.push_back(Method->getParent());
        continue;
      }

      // We don't handle FieldDecl because it's not clear what behaviour
      // the user would expect: the enclosing class type (as with a
      // method), or the field's type (as with a variable).

      if (auto *RD = dyn_cast<CXXRecordDecl>(D))
        Records.push_back(RD);
    }
    return Records;
  };

  const SourceManager &SM = AST.getSourceManager();
  std::vector<const CXXRecordDecl *> Result;
  auto Offset = positionToOffset(SM.getBufferData(SM.getMainFileID()), Pos);
  if (!Offset) {
    llvm::consumeError(Offset.takeError());
    return Result;
  }
  SelectionTree::createEach(AST.getASTContext(), AST.getTokens(), *Offset,
                            *Offset, [&](SelectionTree ST) {
                              Result = RecordFromNode(ST.commonAncestor());
                              return !Result.empty();
                            });
  return Result;
}

// Return the type most associated with an AST node.
// This isn't precisely defined: we want "go to type" to do something useful.
static QualType typeForNode(const SelectionTree::Node *N) {
  // If we're looking at a namespace qualifier, walk up to what it's qualifying.
  // (If we're pointing at a *class* inside a NNS, N will be a TypeLoc).
  while (N && N->ASTNode.get<NestedNameSpecifierLoc>())
    N = N->Parent;
  if (!N)
    return QualType();

  // If we're pointing at a type => return it.
  if (const TypeLoc *TL = N->ASTNode.get<TypeLoc>()) {
    if (llvm::isa<DeducedType>(TL->getTypePtr()))
      if (auto Deduced = getDeducedType(
              N->getDeclContext().getParentASTContext(), TL->getBeginLoc()))
        return *Deduced;
    // Exception: an alias => underlying type.
    if (llvm::isa<TypedefType>(TL->getTypePtr()))
      return TL->getTypePtr()->getLocallyUnqualifiedSingleStepDesugaredType();
    return TL->getType();
  }

  // Constructor initializers => the type of thing being initialized.
  if (const auto *CCI = N->ASTNode.get<CXXCtorInitializer>()) {
    if (const FieldDecl *FD = CCI->getAnyMember())
      return FD->getType();
    if (const Type *Base = CCI->getBaseClass())
      return QualType(Base, 0);
  }

  // Base specifier => the base type.
  if (const auto *CBS = N->ASTNode.get<CXXBaseSpecifier>())
    return CBS->getType();

  if (const Decl *D = N->ASTNode.get<Decl>()) {
    struct Visitor : ConstDeclVisitor<Visitor, QualType> {
      QualType VisitValueDecl(const ValueDecl *D) { return D->getType(); }
      // Declaration of a type => that type.
      QualType VisitTypeDecl(const TypeDecl *D) {
        return QualType(D->getTypeForDecl(), 0);
      }
      // Exception: alias declaration => the underlying type, not the alias.
      QualType VisitTypedefNameDecl(const TypedefNameDecl *D) {
        return D->getUnderlyingType();
      }
      // Look inside templates.
      QualType VisitTemplateDecl(const TemplateDecl *D) {
        return Visit(D->getTemplatedDecl());
      }
    } V;
    return V.Visit(D);
  }

  if (const Stmt *S = N->ASTNode.get<Stmt>()) {
    struct Visitor : ConstStmtVisitor<Visitor, QualType> {
      // Null-safe version of visit simplifies recursive calls below.
      QualType type(const Stmt *S) { return S ? Visit(S) : QualType(); }

      // In general, expressions => type of expression.
      QualType VisitExpr(const Expr *S) {
        return S->IgnoreImplicitAsWritten()->getType();
      }
      QualType VisitMemberExpr(const MemberExpr *S) {
        // The `foo` in `s.foo()` pretends not to have a real type!
        if (S->getType()->isSpecificBuiltinType(BuiltinType::BoundMember))
          return Expr::findBoundMemberType(S);
        return VisitExpr(S);
      }
      // Exceptions for void expressions that operate on a type in some way.
      QualType VisitCXXDeleteExpr(const CXXDeleteExpr *S) {
        return S->getDestroyedType();
      }
      QualType VisitCXXPseudoDestructorExpr(const CXXPseudoDestructorExpr *S) {
        return S->getDestroyedType();
      }
      QualType VisitCXXThrowExpr(const CXXThrowExpr *S) {
        return S->getSubExpr()->getType();
      }
      QualType VisitCoyieldExpr(const CoyieldExpr *S) {
        return type(S->getOperand());
      }
      // Treat a designated initializer like a reference to the field.
      QualType VisitDesignatedInitExpr(const DesignatedInitExpr *S) {
        // In .foo.bar we want to jump to bar's type, so find *last* field.
        for (auto &D : llvm::reverse(S->designators()))
          if (D.isFieldDesignator())
            if (const auto *FD = D.getField())
              return FD->getType();
        return QualType();
      }

      // Control flow statements that operate on data: use the data type.
      QualType VisitSwitchStmt(const SwitchStmt *S) {
        return type(S->getCond());
      }
      QualType VisitWhileStmt(const WhileStmt *S) { return type(S->getCond()); }
      QualType VisitDoStmt(const DoStmt *S) { return type(S->getCond()); }
      QualType VisitIfStmt(const IfStmt *S) { return type(S->getCond()); }
      QualType VisitCaseStmt(const CaseStmt *S) { return type(S->getLHS()); }
      QualType VisitCXXForRangeStmt(const CXXForRangeStmt *S) {
        return S->getLoopVariable()->getType();
      }
      QualType VisitReturnStmt(const ReturnStmt *S) {
        return type(S->getRetValue());
      }
      QualType VisitCoreturnStmt(const CoreturnStmt *S) {
        return type(S->getOperand());
      }
      QualType VisitCXXCatchStmt(const CXXCatchStmt *S) {
        return S->getCaughtType();
      }
      QualType VisitObjCAtThrowStmt(const ObjCAtThrowStmt *S) {
        return type(S->getThrowExpr());
      }
      QualType VisitObjCAtCatchStmt(const ObjCAtCatchStmt *S) {
        return S->getCatchParamDecl() ? S->getCatchParamDecl()->getType()
                                      : QualType();
      }
    } V;
    return V.Visit(S);
  }

  return QualType();
}

// Given a type targeted by the cursor, return one or more types that are more interesting
// to target.
static void unwrapFindType(
    QualType T, const HeuristicResolver* H, llvm::SmallVector<QualType>& Out) {
  if (T.isNull())
    return;

  // If there's a specific type alias, point at that rather than unwrapping.
  if (const auto* TDT = T->getAs<TypedefType>())
    return Out.push_back(QualType(TDT, 0));

  // Pointers etc => pointee type.
  if (const auto *PT = T->getAs<PointerType>())
    return unwrapFindType(PT->getPointeeType(), H, Out);
  if (const auto *RT = T->getAs<ReferenceType>())
    return unwrapFindType(RT->getPointeeType(), H, Out);
  if (const auto *AT = T->getAsArrayTypeUnsafe())
    return unwrapFindType(AT->getElementType(), H, Out);

  // Function type => return type.
  if (auto *FT = T->getAs<FunctionType>())
    return unwrapFindType(FT->getReturnType(), H, Out);
  if (auto *CRD = T->getAsCXXRecordDecl()) {
    if (CRD->isLambda())
      return unwrapFindType(CRD->getLambdaCallOperator()->getReturnType(), H, Out);
    // FIXME: more cases we'd prefer the return type of the call operator?
    //        std::function etc?
  }

  // For smart pointer types, add the underlying type
  if (H)
    if (const auto* PointeeType = H->getPointeeType(T.getNonReferenceType().getTypePtr())) {
        unwrapFindType(QualType(PointeeType, 0), H, Out);
        return Out.push_back(T);
    }

  return Out.push_back(T);
}

// Convenience overload, to allow calling this without the out-parameter
static llvm::SmallVector<QualType> unwrapFindType(
    QualType T, const HeuristicResolver* H) {
    llvm::SmallVector<QualType> Result;
    unwrapFindType(T, H, Result);
    return Result;
}


std::vector<LocatedSymbol> findType(ParsedAST &AST, Position Pos) {
  const SourceManager &SM = AST.getSourceManager();
  auto Offset = positionToOffset(SM.getBufferData(SM.getMainFileID()), Pos);
  std::vector<LocatedSymbol> Result;
  if (!Offset) {
    elog("failed to convert position {0} for findTypes: {1}", Pos,
         Offset.takeError());
    return Result;
  }
  // The general scheme is: position -> AST node -> type -> declaration.
  auto SymbolsFromNode =
      [&AST](const SelectionTree::Node *N) -> std::vector<LocatedSymbol> {
    std::vector<LocatedSymbol> LocatedSymbols;

    // NOTE: unwrapFindType might return duplicates for something like
    // unique_ptr<unique_ptr<T>>. Let's *not* remove them, because it gives you some
    // information about the type you may have not known before
    // (since unique_ptr<unique_ptr<T>> != unique_ptr<T>).
    for (const QualType& Type : unwrapFindType(typeForNode(N), AST.getHeuristicResolver()))
        llvm::copy(locateSymbolForType(AST, Type), std::back_inserter(LocatedSymbols));

    return LocatedSymbols;
  };
  SelectionTree::createEach(AST.getASTContext(), AST.getTokens(), *Offset,
                            *Offset, [&](SelectionTree ST) {
                              Result = SymbolsFromNode(ST.commonAncestor());
                              return !Result.empty();
                            });
  return Result;
}

std::vector<const CXXRecordDecl *> typeParents(const CXXRecordDecl *CXXRD) {
  std::vector<const CXXRecordDecl *> Result;

  // If this is an invalid instantiation, instantiation of the bases
  // may not have succeeded, so fall back to the template pattern.
  if (auto *CTSD = dyn_cast<ClassTemplateSpecializationDecl>(CXXRD)) {
    if (CTSD->isInvalidDecl())
      CXXRD = CTSD->getSpecializedTemplate()->getTemplatedDecl();
  }

  // Can't query bases without a definition.
  if (!CXXRD->hasDefinition())
    return Result;

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

std::vector<TypeHierarchyItem>
getTypeHierarchy(ParsedAST &AST, Position Pos, int ResolveLevels,
                 TypeHierarchyDirection Direction, const SymbolIndex *Index,
                 PathRef TUPath) {
  std::vector<TypeHierarchyItem> Results;
  for (const auto *CXXRD : findRecordTypeAt(AST, Pos)) {

    bool WantChildren = Direction == TypeHierarchyDirection::Children ||
                        Direction == TypeHierarchyDirection::Both;

    // If we're looking for children, we're doing the lookup in the index.
    // The index does not store relationships between implicit
    // specializations, so if we have one, use the template pattern instead.
    // Note that this needs to be done before the declToTypeHierarchyItem(),
    // otherwise the type hierarchy item would misleadingly contain the
    // specialization parameters, while the children would involve classes
    // that derive from other specializations of the template.
    if (WantChildren) {
      if (auto *CTSD = dyn_cast<ClassTemplateSpecializationDecl>(CXXRD))
        CXXRD = CTSD->getTemplateInstantiationPattern();
    }

    Optional<TypeHierarchyItem> Result =
        declToTypeHierarchyItem(*CXXRD, AST.tuPath());
    if (!Result)
      continue;

    RecursionProtectionSet RPSet;
    fillSuperTypes(*CXXRD, AST.tuPath(), *Result, RPSet);

    if (WantChildren && ResolveLevels > 0) {
      Result->children.emplace();

      if (Index) {
        if (auto ID = getSymbolID(CXXRD))
          fillSubTypes(ID, *Result->children, Index, ResolveLevels, TUPath);
      }
    }
    Results.emplace_back(std::move(*Result));
  }

  return Results;
}

llvm::Optional<std::vector<TypeHierarchyItem>>
superTypes(const TypeHierarchyItem &Item, const SymbolIndex *Index) {
  std::vector<TypeHierarchyItem> Results;
  if (!Item.data.parents)
    return llvm::None;
  if (Item.data.parents->empty())
    return Results;
  LookupRequest Req;
  llvm::DenseMap<SymbolID, const TypeHierarchyItem::ResolveParams *> IDToData;
  for (const auto &Parent : *Item.data.parents) {
    Req.IDs.insert(Parent.symbolID);
    IDToData[Parent.symbolID] = &Parent;
  }
  Index->lookup(Req, [&Item, &Results, &IDToData](const Symbol &S) {
    if (auto THI = symbolToTypeHierarchyItem(S, Item.uri.file())) {
      THI->data = *IDToData.lookup(S.ID);
      Results.emplace_back(std::move(*THI));
    }
  });
  return Results;
}

std::vector<TypeHierarchyItem> subTypes(const TypeHierarchyItem &Item,
                                        const SymbolIndex *Index) {
  std::vector<TypeHierarchyItem> Results;
  fillSubTypes(Item.data.symbolID, Results, Index, 1, Item.uri.file());
  for (auto &ChildSym : Results)
    ChildSym.data.parents = {Item.data};
  return Results;
}

void resolveTypeHierarchy(TypeHierarchyItem &Item, int ResolveLevels,
                          TypeHierarchyDirection Direction,
                          const SymbolIndex *Index) {
  // We only support typeHierarchy/resolve for children, because for parents
  // we ignore ResolveLevels and return all levels of parents eagerly.
  if (!Index || Direction == TypeHierarchyDirection::Parents ||
      ResolveLevels == 0)
    return;

  Item.children.emplace();
  fillSubTypes(Item.data.symbolID, *Item.children, Index, ResolveLevels,
               Item.uri.file());
}

std::vector<CallHierarchyItem>
prepareCallHierarchy(ParsedAST &AST, Position Pos, PathRef TUPath) {
  std::vector<CallHierarchyItem> Result;
  const auto &SM = AST.getSourceManager();
  auto Loc = sourceLocationInMainFile(SM, Pos);
  if (!Loc) {
    elog("prepareCallHierarchy failed to convert position to source location: "
         "{0}",
         Loc.takeError());
    return Result;
  }
  for (const NamedDecl *Decl : getDeclAtPosition(AST, *Loc, {})) {
    if (!(isa<DeclContext>(Decl) &&
          cast<DeclContext>(Decl)->isFunctionOrMethod()) &&
        Decl->getKind() != Decl::Kind::FunctionTemplate)
      continue;
    if (auto CHI = declToCallHierarchyItem(*Decl, AST.tuPath()))
      Result.emplace_back(std::move(*CHI));
  }
  return Result;
}

std::vector<CallHierarchyIncomingCall>
incomingCalls(const CallHierarchyItem &Item, const SymbolIndex *Index) {
  std::vector<CallHierarchyIncomingCall> Results;
  if (!Index || Item.data.empty())
    return Results;
  auto ID = SymbolID::fromStr(Item.data);
  if (!ID) {
    elog("incomingCalls failed to find symbol: {0}", ID.takeError());
    return Results;
  }
  // In this function, we find incoming calls based on the index only.
  // In principle, the AST could have more up-to-date information about
  // occurrences within the current file. However, going from a SymbolID
  // to an AST node isn't cheap, particularly when the declaration isn't
  // in the main file.
  // FIXME: Consider also using AST information when feasible.
  RefsRequest Request;
  Request.IDs.insert(*ID);
  Request.WantContainer = true;
  // We could restrict more specifically to calls by introducing a new RefKind,
  // but non-call references (such as address-of-function) can still be
  // interesting as they can indicate indirect calls.
  Request.Filter = RefKind::Reference;
  // Initially store the ranges in a map keyed by SymbolID of the caller.
  // This allows us to group different calls with the same caller
  // into the same CallHierarchyIncomingCall.
  llvm::DenseMap<SymbolID, std::vector<Range>> CallsIn;
  // We can populate the ranges based on a refs request only. As we do so, we
  // also accumulate the container IDs into a lookup request.
  LookupRequest ContainerLookup;
  Index->refs(Request, [&](const Ref &R) {
    auto Loc = indexToLSPLocation(R.Location, Item.uri.file());
    if (!Loc) {
      elog("incomingCalls failed to convert location: {0}", Loc.takeError());
      return;
    }
    auto It = CallsIn.try_emplace(R.Container, std::vector<Range>{}).first;
    It->second.push_back(Loc->range);

    ContainerLookup.IDs.insert(R.Container);
  });
  // Perform the lookup request and combine its results with CallsIn to
  // get complete CallHierarchyIncomingCall objects.
  Index->lookup(ContainerLookup, [&](const Symbol &Caller) {
    auto It = CallsIn.find(Caller.ID);
    assert(It != CallsIn.end());
    if (auto CHI = symbolToCallHierarchyItem(Caller, Item.uri.file()))
      Results.push_back(
          CallHierarchyIncomingCall{std::move(*CHI), std::move(It->second)});
  });
  // Sort results by name of container.
  llvm::sort(Results, [](const CallHierarchyIncomingCall &A,
                         const CallHierarchyIncomingCall &B) {
    return A.from.name < B.from.name;
  });
  return Results;
}

llvm::DenseSet<const Decl *> getNonLocalDeclRefs(ParsedAST &AST,
                                                 const FunctionDecl *FD) {
  if (!FD->hasBody())
    return {};
  llvm::DenseSet<const Decl *> DeclRefs;
  findExplicitReferences(
      FD,
      [&](ReferenceLoc Ref) {
        for (const Decl *D : Ref.Targets) {
          if (!index::isFunctionLocalSymbol(D) && !D->isTemplateParameter() &&
              !Ref.IsDecl)
            DeclRefs.insert(D);
        }
      },
      AST.getHeuristicResolver());
  return DeclRefs;
}

} // namespace clangd
} // namespace clang
