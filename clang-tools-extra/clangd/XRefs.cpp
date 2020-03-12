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
#include "Quality.h"
#include "Selection.h"
#include "SourceCode.h"
#include "URI.h"
#include "index/Index.h"
#include "index/Merge.h"
#include "index/Relation.h"
#include "index/SymbolLocation.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Attrs.inc"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Type.h"
#include "clang/Basic/CharInfo.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Index/IndexDataConsumer.h"
#include "clang/Index/IndexSymbol.h"
#include "clang/Index/IndexingAction.h"
#include "clang/Index/IndexingOptions.h"
#include "clang/Index/USRGeneration.h"
#include "clang/Tooling/Syntax/Tokens.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
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
const NamedDecl *getDefinition(const NamedDecl *D) {
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

std::vector<const NamedDecl *> getDeclAtPosition(ParsedAST &AST,
                                                 SourceLocation Pos,
                                                 DeclRelationSet Relations) {
  unsigned Offset = AST.getSourceManager().getDecomposedSpellingLoc(Pos).second;
  std::vector<const NamedDecl *> Result;
  SelectionTree::createEach(AST.getASTContext(), AST.getTokens(), Offset,
                            Offset, [&](SelectionTree ST) {
                              if (const SelectionTree::Node *N =
                                      ST.commonAncestor())
                                llvm::copy(targetDecl(N->ASTNode, Relations),
                                           std::back_inserter(Result));
                              return !Result.empty();
                            });
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
    if (!Inc.Resolved.empty() && Inc.R.start.line == Pos.line) {
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
    if (auto Loc = makeLocation(AST.getASTContext(),
                                M->Info->getDefinitionLoc(), MainFilePath)) {
      LocatedSymbol Macro;
      Macro.Name = std::string(M->Name);
      Macro.PreferredDeclaration = *Loc;
      Macro.Definition = Loc;
      return Macro;
    }
  }
  return llvm::None;
}

// Decls are more complicated.
// The AST contains at least a declaration, maybe a definition.
// These are up-to-date, and so generally preferred over index results.
// We perform a single batch index lookup to find additional definitions.
std::vector<LocatedSymbol>
locateASTReferent(SourceLocation CurLoc, const syntax::Token *TouchedIdentifier,
                  ParsedAST &AST, llvm::StringRef MainFilePath,
                  const SymbolIndex *Index) {
  const SourceManager &SM = AST.getSourceManager();
  // Results follow the order of Symbols.Decls.
  std::vector<LocatedSymbol> Result;
  // Keep track of SymbolID -> index mapping, to fill in index data later.
  llvm::DenseMap<SymbolID, size_t> ResultIndex;

  auto AddResultDecl = [&](const NamedDecl *D) {
    const NamedDecl *Def = getDefinition(D);
    const NamedDecl *Preferred = Def ? Def : D;

    auto Loc = makeLocation(AST.getASTContext(), nameLocation(*Preferred, SM),
                            MainFilePath);
    if (!Loc)
      return;

    Result.emplace_back();
    Result.back().Name = printName(AST.getASTContext(), *Preferred);
    Result.back().PreferredDeclaration = *Loc;
    // Preferred is always a definition if possible, so this check works.
    if (Def == Preferred)
      Result.back().Definition = *Loc;

    // Record SymbolID for index lookup later.
    if (auto ID = getSymbolID(Preferred))
      ResultIndex[*ID] = Result.size() - 1;
  };

  // Emit all symbol locations (declaration or definition) from AST.
  DeclRelationSet Relations =
      DeclRelation::TemplatePattern | DeclRelation::Alias;
  for (const NamedDecl *D : getDeclAtPosition(AST, CurLoc, Relations)) {
    // Special case: void foo() ^override: jump to the overridden method.
    if (const auto *CMD = llvm::dyn_cast<CXXMethodDecl>(D)) {
      const InheritableAttr *Attr = D->getAttr<OverrideAttr>();
      if (!Attr)
        Attr = D->getAttr<FinalAttr>();
      if (Attr && TouchedIdentifier &&
          SM.getSpellingLoc(Attr->getLocation()) ==
              TouchedIdentifier->location()) {
        // We may be overridding multiple methods - offer them all.
        for (const NamedDecl *ND : CMD->overridden_methods())
          AddResultDecl(ND);
        continue;
      }
    }

    // Special case: the point of declaration of a template specialization,
    // it's more useful to navigate to the template declaration.
    if (auto *CTSD = dyn_cast<ClassTemplateSpecializationDecl>(D)) {
      if (TouchedIdentifier &&
          D->getLocation() == TouchedIdentifier->location()) {
        AddResultDecl(CTSD->getSpecializedTemplate());
        continue;
      }
    }

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

  return Result;
}

llvm::StringRef wordTouching(llvm::StringRef Code, unsigned Offset) {
  unsigned B = Offset, E = Offset;
  while (B > 0 && isIdentifierBody(Code[B - 1]))
    --B;
  while (E < Code.size() && isIdentifierBody(Code[E]))
    ++E;
  return Code.slice(B, E);
}

bool isLikelyToBeIdentifier(StringRef Word) {
  // Word contains underscore.
  // This handles things like snake_case and MACRO_CASE.
  if (Word.contains('_')) {
    return true;
  }
  // Word contains capital letter other than at beginning.
  // This handles things like lowerCamel and UpperCamel.
  // The check for also containing a lowercase letter is to rule out
  // initialisms like "HTTP".
  bool HasLower = Word.find_if(clang::isLowercase) != StringRef::npos;
  bool HasUpper = Word.substr(1).find_if(clang::isUppercase) != StringRef::npos;
  if (HasLower && HasUpper) {
    return true;
  }
  // FIXME: There are other signals we could listen for.
  // Some of these require inspecting the surroundings of the word as well.
  //   - mid-sentence Capitalization
  //   - markup like quotes / backticks / brackets / "\p"
  //   - word has a qualifier (foo::bar)
  return false;
}

bool tokenSurvivedPreprocessing(SourceLocation Loc,
                                const syntax::TokenBuffer &TB) {
  auto WordExpandedTokens =
      TB.expandedTokens(TB.sourceManager().getMacroArgExpandedLocation(Loc));
  return !WordExpandedTokens.empty();
}

} // namespace

std::vector<LocatedSymbol>
locateSymbolNamedTextuallyAt(ParsedAST &AST, const SymbolIndex *Index,
                             SourceLocation Loc,
                             const std::string &MainFilePath) {
  const auto &SM = AST.getSourceManager();

  // Get the raw word at the specified location.
  unsigned Pos;
  FileID File;
  std::tie(File, Pos) = SM.getDecomposedLoc(Loc);
  llvm::StringRef Code = SM.getBufferData(File);
  llvm::StringRef Word = wordTouching(Code, Pos);
  if (Word.empty())
    return {};
  unsigned WordOffset = Word.data() - Code.data();
  SourceLocation WordStart = SM.getComposedLoc(File, WordOffset);

  // Do not consider tokens that survived preprocessing.
  // We are erring on the safe side here, as a user may expect to get
  // accurate (as opposed to textual-heuristic) results for such tokens.
  // FIXME: Relax this for dependent code.
  if (tokenSurvivedPreprocessing(WordStart, AST.getTokens()))
    return {};

  // Additionally filter for signals that the word is likely to be an
  // identifier. This avoids triggering on e.g. random words in a comment.
  if (!isLikelyToBeIdentifier(Word))
    return {};

  // Look up the selected word in the index.
  FuzzyFindRequest Req;
  Req.Query = Word.str();
  Req.ProximityPaths = {MainFilePath};
  Req.Scopes = visibleNamespaces(Code.take_front(Pos), AST.getLangOpts());
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
    if (Sym.Name != Word)
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
    Location DeclLoc = *MaybeDeclLoc;
    Location DefLoc;
    if (Sym.Definition) {
      auto MaybeDefLoc = indexToLSPLocation(Sym.Definition, MainFilePath);
      if (!MaybeDefLoc) {
        log("locateSymbolNamedTextuallyAt: {0}", MaybeDefLoc.takeError());
        return;
      }
      DefLoc = *MaybeDefLoc;
    }

    if (ScoredResults.size() >= 3) {
      // If we have more than 3 results, don't return anything,
      // as confidence is too low.
      // FIXME: Alternatively, try a stricter query?
      TooMany = true;
      return;
    }

    LocatedSymbol Located;
    Located.Name = (Sym.Name + Sym.TemplateSpecializationArgs).str();
    Located.PreferredDeclaration = bool(Sym.Definition) ? DefLoc : DeclLoc;
    Located.Definition = DefLoc;

    SymbolQualitySignals Quality;
    Quality.merge(Sym);
    SymbolRelevanceSignals Relevance;
    Relevance.Name = Sym.Name;
    Relevance.Query = SymbolRelevanceSignals::Generic;
    Relevance.merge(Sym);
    auto Score =
        evaluateSymbolAndRelevance(Quality.evaluate(), Relevance.evaluate());
    dlog("locateSymbolNamedTextuallyAt: {0}{1} = {2}\n{3}{4}\n", Sym.Scope,
         Sym.Name, Score, Quality, Relevance);

    ScoredResults.push_back({Score, std::move(Located)});
  });

  if (TooMany)
    return {};

  llvm::sort(ScoredResults,
             [](const ScoredLocatedSymbol &A, const ScoredLocatedSymbol &B) {
               return A.first > B.first;
             });
  std::vector<LocatedSymbol> Results;
  for (auto &Res : std::move(ScoredResults))
    Results.push_back(std::move(Res.second));
  return Results;
}

std::vector<LocatedSymbol> locateSymbolAt(ParsedAST &AST, Position Pos,
                                          const SymbolIndex *Index) {
  const auto &SM = AST.getSourceManager();
  auto MainFilePath =
      getCanonicalPath(SM.getFileEntryForID(SM.getMainFileID()), SM);
  if (!MainFilePath) {
    elog("Failed to get a path for the main file, so no references");
    return {};
  }

  if (auto File = locateFileReferent(Pos, AST, *MainFilePath))
    return {std::move(*File)};

  auto CurLoc = sourceLocationInMainFile(SM, Pos);
  if (!CurLoc) {
    elog("locateSymbolAt failed to convert position to source location: {0}",
         CurLoc.takeError());
    return {};
  }

  const syntax::Token *TouchedIdentifier =
      syntax::spelledIdentifierTouching(*CurLoc, AST.getTokens());
  if (TouchedIdentifier)
    if (auto Macro =
            locateMacroReferent(*TouchedIdentifier, AST, *MainFilePath))
      // Don't look at the AST or index if we have a macro result.
      // (We'd just return declarations referenced from the macro's
      // expansion.)
      return {*std::move(Macro)};

  auto ASTResults =
      locateASTReferent(*CurLoc, TouchedIdentifier, AST, *MainFilePath, Index);
  if (!ASTResults.empty())
    return ASTResults;

  return locateSymbolNamedTextuallyAt(AST, Index, *CurLoc, *MainFilePath);
}

std::vector<DocumentLink> getDocumentLinks(ParsedAST &AST) {
  const auto &SM = AST.getSourceManager();
  auto MainFilePath =
      getCanonicalPath(SM.getFileEntryForID(SM.getMainFileID()), SM);
  if (!MainFilePath) {
    elog("Failed to get a path for the main file, so no links");
    return {};
  }

  std::vector<DocumentLink> Result;
  for (auto &Inc : AST.getIncludeStructure().MainFileIncludes) {
    if (!Inc.Resolved.empty()) {
      Result.push_back(DocumentLink(
          {Inc.R, URIForFile::canonicalize(Inc.Resolved, *MainFilePath)}));
    }
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

    Range range(const SourceManager &SM) const {
      return halfOpenToRange(SM, SpelledTok.range(SM).toCharRange(SM));
    }
  };

  ReferenceFinder(const ParsedAST &AST,
                  const std::vector<const NamedDecl *> &TargetDecls)
      : AST(AST) {
    for (const NamedDecl *D : TargetDecls)
      CanonicalTargets.insert(D->getCanonicalDecl());
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
    assert(D->isCanonicalDecl() && "expect D to be a canonical declaration");
    if (!CanonicalTargets.count(D))
      return true;
    const auto &TB = AST.getTokens();
    const SourceManager &SM = AST.getSourceManager();
    Loc = SM.getFileLoc(Loc);
    // We are only traversing decls *inside* the main file, so this should hold.
    assert(isInsideMainFile(Loc, SM));
    if (const auto *Tok = TB.spelledTokenAt(Loc))
      References.push_back({*Tok, Roles});
    return true;
  }

private:
  llvm::SmallSet<const Decl *, 4> CanonicalTargets;
  std::vector<Reference> References;
  const ParsedAST &AST;
};

std::vector<ReferenceFinder::Reference>
findRefs(const std::vector<const NamedDecl *> &Decls, ParsedAST &AST) {
  ReferenceFinder RefFinder(AST, Decls);
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
  auto CurLoc = sourceLocationInMainFile(SM, Pos);
  if (!CurLoc) {
    llvm::consumeError(CurLoc.takeError());
    return {};
  }
  auto References = findRefs(getDeclAtPosition(AST, *CurLoc, Relations), AST);

  // FIXME: we may get multiple DocumentHighlights with the same location and
  // different kinds, deduplicate them.
  std::vector<DocumentHighlight> Result;
  for (const auto &Ref : References) {
    DocumentHighlight DH;
    DH.range = Ref.range(SM);
    if (Ref.Role & index::SymbolRoleSet(index::SymbolRole::Write))
      DH.kind = DocumentHighlightKind::Write;
    else if (Ref.Role & index::SymbolRoleSet(index::SymbolRole::Read))
      DH.kind = DocumentHighlightKind::Read;
    else
      DH.kind = DocumentHighlightKind::Text;
    Result.push_back(std::move(DH));
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
  auto URIMainFile = URIForFile::canonicalize(*MainFilePath, *MainFilePath);
  auto CurLoc = sourceLocationInMainFile(SM, Pos);
  if (!CurLoc) {
    llvm::consumeError(CurLoc.takeError());
    return {};
  }
  llvm::Optional<DefinedMacro> Macro;
  if (const auto *IdentifierAtCursor =
          syntax::spelledIdentifierTouching(*CurLoc, AST.getTokens())) {
    Macro = locateMacroAt(*IdentifierAtCursor, AST.getPreprocessor());
  }

  RefsRequest Req;
  if (Macro) {
    // Handle references to macro.
    if (auto MacroSID = getSymbolID(Macro->Name, Macro->Info, SM)) {
      // Collect macro references from main file.
      const auto &IDToRefs = AST.getMacros().MacroRefs;
      auto Refs = IDToRefs.find(*MacroSID);
      if (Refs != IDToRefs.end()) {
        for (const auto Ref : Refs->second) {
          Location Result;
          Result.range = Ref;
          Result.uri = URIMainFile;
          Results.References.push_back(std::move(Result));
        }
      }
      Req.IDs.insert(*MacroSID);
    }
  } else {
    // Handle references to Decls.

    // We also show references to the targets of using-decls, so we include
    // DeclRelation::Underlying.
    DeclRelationSet Relations = DeclRelation::TemplatePattern |
                                DeclRelation::Alias | DeclRelation::Underlying;
    auto Decls = getDeclAtPosition(AST, *CurLoc, Relations);

    // We traverse the AST to find references in the main file.
    auto MainFileRefs = findRefs(Decls, AST);
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
      Location Result;
      Result.range = Ref.range(SM);
      Result.uri = URIMainFile;
      Results.References.push_back(std::move(Result));
    }
    if (Index && Results.References.size() <= Limit) {
      for (const Decl *D : Decls) {
        // Not all symbols can be referenced from outside (e.g.
        // function-locals).
        // TODO: we could skip TU-scoped symbols here (e.g. static functions) if
        // we know this file isn't a header. The details might be tricky.
        if (D->getParentFunctionOrMethod())
          continue;
        if (auto ID = getSymbolID(D))
          Req.IDs.insert(*ID);
      }
    }
  }
  // Now query the index for references from other files.
  if (!Req.IDs.empty() && Index && Results.References.size() <= Limit) {
    Req.Limit = Limit;
    Results.HasMore |= Index->refs(Req, [&](const Ref &R) {
      // No need to continue process if we reach the limit.
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
  auto CurLoc = sourceLocationInMainFile(SM, Pos);
  if (!CurLoc) {
    llvm::consumeError(CurLoc.takeError());
    return {};
  }

  std::vector<SymbolDetails> Results;

  // We also want the targets of using-decls, so we include
  // DeclRelation::Underlying.
  DeclRelationSet Relations = DeclRelation::TemplatePattern |
                              DeclRelation::Alias | DeclRelation::Underlying;
  for (const NamedDecl *D : getDeclAtPosition(AST, *CurLoc, Relations)) {
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

// FIXME(nridge): Reduce duplication between this function and declToSym().
static llvm::Optional<TypeHierarchyItem>
declToTypeHierarchyItem(ASTContext &Ctx, const NamedDecl &ND,
                        const syntax::TokenBuffer &TB) {
  auto &SM = Ctx.getSourceManager();
  SourceLocation NameLoc = nameLocation(ND, Ctx.getSourceManager());
  auto FilePath =
      getCanonicalPath(SM.getFileEntryForID(SM.getFileID(NameLoc)), SM);
  auto TUPath = getCanonicalPath(SM.getFileEntryForID(SM.getMainFileID()), SM);
  if (!FilePath || !TUPath)
    return llvm::None; // Not useful without a uri.

  auto DeclToks = TB.spelledForExpanded(TB.expandedTokens(ND.getSourceRange()));
  if (!DeclToks || DeclToks->empty())
    return llvm::None;

  auto NameToks = TB.spelledForExpanded(TB.expandedTokens(NameLoc));
  if (!NameToks || NameToks->empty())
    return llvm::None;

  index::SymbolInfo SymInfo = index::getSymbolInfo(&ND);
  // FIXME: this is not classifying constructors, destructors and operators
  //        correctly (they're all "methods").
  SymbolKind SK = indexSymbolKindToSymbolKind(SymInfo.Kind);

  TypeHierarchyItem THI;
  THI.name = printName(Ctx, ND);
  THI.kind = SK;
  THI.deprecated = ND.isDeprecated();
  THI.range = halfOpenToRange(
      SM, syntax::Token::range(SM, DeclToks->front(), DeclToks->back())
              .toCharRange(SM));
  THI.selectionRange = halfOpenToRange(
      SM, syntax::Token::range(SM, NameToks->front(), NameToks->back())
              .toCharRange(SM));
  if (!THI.range.contains(THI.selectionRange)) {
    // 'selectionRange' must be contained in 'range', so in cases where clang
    // reports unrelated ranges we need to reconcile somehow.
    THI.range = THI.selectionRange;
  }

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
  THI.name = std::string(S.Name);
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
                           RecursionProtectionSet &RPSet,
                           const syntax::TokenBuffer &TB) {
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
            declToTypeHierarchyItem(ASTCtx, *ParentDecl, TB)) {
      ParentSym->parents.emplace();
      fillSuperTypes(*ParentDecl, ASTCtx, *ParentSym->parents, RPSet, TB);
      SuperTypes.emplace_back(std::move(*ParentSym));
    }
  }

  if (Pattern) {
    RPSet.erase(Pattern);
  }
}

const CXXRecordDecl *findRecordTypeAt(ParsedAST &AST, Position Pos) {
  auto RecordFromNode =
      [](const SelectionTree::Node *N) -> const CXXRecordDecl * {
    if (!N)
      return nullptr;

    // Note: explicitReferenceTargets() will search for both template
    // instantiations and template patterns, and prefer the former if available
    // (generally, one will be available for non-dependent specializations of a
    // class template).
    auto Decls = explicitReferenceTargets(N->ASTNode, DeclRelation::Underlying);
    if (Decls.empty())
      return nullptr;

    const NamedDecl *D = Decls[0];

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
  };

  const SourceManager &SM = AST.getSourceManager();
  const CXXRecordDecl *Result = nullptr;
  auto Offset = positionToOffset(SM.getBufferData(SM.getMainFileID()), Pos);
  if (!Offset) {
    llvm::consumeError(Offset.takeError());
    return Result;
  }
  SelectionTree::createEach(AST.getASTContext(), AST.getTokens(), *Offset,
                            *Offset, [&](SelectionTree ST) {
                              Result = RecordFromNode(ST.commonAncestor());
                              return Result != nullptr;
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
      declToTypeHierarchyItem(AST.getASTContext(), *CXXRD, AST.getTokens());
  if (!Result)
    return Result;

  if (Direction == TypeHierarchyDirection::Parents ||
      Direction == TypeHierarchyDirection::Both) {
    Result->parents.emplace();

    RecursionProtectionSet RPSet;
    fillSuperTypes(*CXXRD, AST.getASTContext(), *Result->parents, RPSet,
                   AST.getTokens());
  }

  if ((Direction == TypeHierarchyDirection::Children ||
       Direction == TypeHierarchyDirection::Both) &&
      ResolveLevels > 0) {
    Result->children.emplace();

    if (Index) {
      // The index does not store relationships between implicit
      // specializations, so if we have one, use the template pattern instead.
      if (auto *CTSD = dyn_cast<ClassTemplateSpecializationDecl>(CXXRD))
        CXXRD = CTSD->getTemplateInstantiationPattern();

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
