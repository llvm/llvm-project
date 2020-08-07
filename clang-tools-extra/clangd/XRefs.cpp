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
#include "support/Logger.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Attrs.inc"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/Type.h"
#include "clang/Basic/CharInfo.h"
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
#include "llvm/ADT/None.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MathExtras.h"
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

std::vector<const NamedDecl *>
getDeclAtPosition(ParsedAST &AST, SourceLocation Pos, DeclRelationSet Relations,
                  ASTNodeKind *NodeKind = nullptr) {
  unsigned Offset = AST.getSourceManager().getDecomposedSpellingLoc(Pos).second;
  std::vector<const NamedDecl *> Result;
  SelectionTree::createEach(AST.getASTContext(), AST.getTokens(), Offset,
                            Offset, [&](SelectionTree ST) {
                              if (const SelectionTree::Node *N =
                                      ST.commonAncestor()) {
                                if (NodeKind)
                                  *NodeKind = N->ASTNode.getNodeKind();
                                llvm::copy(targetDecl(N->ASTNode, Relations),
                                           std::back_inserter(Result));
                              }
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
                  const SymbolIndex *Index, ASTNodeKind *NodeKind) {
  const SourceManager &SM = AST.getSourceManager();
  // Results follow the order of Symbols.Decls.
  std::vector<LocatedSymbol> Result;
  // Keep track of SymbolID -> index mapping, to fill in index data later.
  llvm::DenseMap<SymbolID, size_t> ResultIndex;

  auto AddResultDecl = [&](const NamedDecl *D) {
    // FIXME: Canonical declarations of some symbols might refer to built-in
    // decls with possibly-invalid source locations (e.g. global new operator).
    // In such cases we should pick up a redecl with valid source location
    // instead of failing.
    D = llvm::cast<NamedDecl>(D->getCanonicalDecl());
    auto Loc =
        makeLocation(AST.getASTContext(), nameLocation(*D, SM), MainFilePath);
    if (!Loc)
      return;

    Result.emplace_back();
    Result.back().Name = printName(AST.getASTContext(), *D);
    Result.back().PreferredDeclaration = *Loc;
    if (const NamedDecl *Def = getDefinition(D))
      Result.back().Definition = makeLocation(
          AST.getASTContext(), nameLocation(*Def, SM), MainFilePath);

    // Record SymbolID for index lookup later.
    if (auto ID = getSymbolID(D))
      ResultIndex[*ID] = Result.size() - 1;
  };

  // Emit all symbol locations (declaration or definition) from AST.
  DeclRelationSet Relations =
      DeclRelation::TemplatePattern | DeclRelation::Alias;
  for (const NamedDecl *D :
       getDeclAtPosition(AST, CurLoc, Relations, NodeKind)) {
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

    // Give the underlying decl if navigation is triggered on a non-renaming
    // alias.
    if (llvm::isa<UsingDecl>(D)) {
      // FIXME: address more complicated cases. TargetDecl(... Underlying) gives
      // all overload candidates, we only want the targeted one if the cursor is
      // on an using-alias usage, workround it with getDeclAtPosition.
      llvm::for_each(
          getDeclAtPosition(AST, CurLoc, DeclRelation::Underlying, NodeKind),
          [&](const NamedDecl *UD) { AddResultDecl(UD); });
      continue;
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

std::vector<LocatedSymbol>
locateSymbolTextually(const SpelledWord &Word, ParsedAST &AST,
                      const SymbolIndex *Index, const std::string &MainFilePath,
                      ASTNodeKind NodeKind) {
  // Don't use heuristics if this is a real identifier, or not an
  // identifier.
  // Exception: dependent names, because those may have useful textual
  // matches that AST-based heuristics cannot find.
  if ((Word.ExpandedToken && !isDependentName(NodeKind)) ||
      !Word.LikelyIdentifier || !Index)
    return {};
  // We don't want to handle words in string literals. It'd be nice to include
  // comments, but they're not retained in TokenBuffer.
  if (Word.PartOfSpelledToken &&
      isStringLiteral(Word.PartOfSpelledToken->kind()))
    return {};

  const auto &SM = AST.getSourceManager();
  // Look up the selected word in the index.
  FuzzyFindRequest Req;
  Req.Query = Word.Text.str();
  Req.ProximityPaths = {MainFilePath};
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
    if (Sym.Definition) {
      auto MaybeDefLoc = indexToLSPLocation(Sym.Definition, MainFilePath);
      if (!MaybeDefLoc) {
        log("locateSymbolNamedTextuallyAt: {0}", MaybeDefLoc.takeError());
        return;
      }
      Located.PreferredDeclaration = *MaybeDefLoc;
      Located.Definition = *MaybeDefLoc;
    }

    if (ScoredResults.size() >= 3) {
      // If we have more than 3 results, don't return anything,
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
    auto Score =
        evaluateSymbolAndRelevance(Quality.evaluate(), Relevance.evaluate());
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
  // We don't want to handle words in string literals. It'd be nice to include
  // comments, but they're not retained in TokenBuffer.
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
    if (Line > WordLine)
      return 1 + llvm::Log2_64(Line - WordLine);
    if (Line < WordLine)
      return 2 + llvm::Log2_64(WordLine - Line);
    return 0;
  };
  const syntax::Token *BestTok = nullptr;
  // Search bounds are based on word length: 2^N lines forward.
  unsigned BestCost = Word.Text.size() + 1;

  // Updates BestTok and BestCost if Tok is a good candidate.
  // May return true if the cost is too high for this token.
  auto Consider = [&](const syntax::Token &Tok) {
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

  ASTNodeKind NodeKind;
  auto ASTResults = locateASTReferent(*CurLoc, TouchedIdentifier, AST,
                                      *MainFilePath, Index, &NodeKind);
  if (!ASTResults.empty())
    return ASTResults;

  // If the cursor can't be resolved directly, try fallback strategies.
  auto Word =
      SpelledWord::touching(*CurLoc, AST.getTokens(), AST.getLangOpts());
  if (Word) {
    // Is the same word nearby a real identifier that might refer to something?
    if (const syntax::Token *NearbyIdent =
            findNearbyIdentifier(*Word, AST.getTokens())) {
      if (auto Macro = locateMacroReferent(*NearbyIdent, AST, *MainFilePath)) {
        log("Found macro definition heuristically using nearby identifier {0}",
            Word->Text);
        return {*std::move(Macro)};
      }
      ASTResults =
          locateASTReferent(NearbyIdent->location(), NearbyIdent, AST,
                            *MainFilePath, Index, /*NodeKind=*/nullptr);
      if (!ASTResults.empty()) {
        log("Found definition heuristically using nearby identifier {0}",
            NearbyIdent->text(SM));
        return ASTResults;
      } else {
        vlog("No definition found using nearby identifier {0} at {1}",
             Word->Text, Word->Location.printToString(SM));
      }
    }
    // No nearby word, or it didn't refer to anything either. Try the index.
    auto TextualResults =
        locateSymbolTextually(*Word, AST, Index, *MainFilePath, NodeKind);
    if (!TextualResults.empty())
      return TextualResults;
  }

  return {};
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
                      URIForFile::canonicalize(Inc.Resolved, *MainFilePath)}));
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
    const SourceManager &SM = AST.getSourceManager();
    if (!CanonicalTargets.count(D) || !isInsideMainFile(Loc, SM))
      return true;
    const auto &TB = AST.getTokens();
    Loc = SM.getFileLoc(Loc);
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
      auto Decls = targetDecl(N->ASTNode, Relations);
      if (!Decls.empty()) {
        // FIXME: we may get multiple DocumentHighlights with the same location
        // and different kinds, deduplicate them.
        for (const auto &Ref : findRefs({Decls.begin(), Decls.end()}, AST))
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
        for (const auto &Ref : Refs->second) {
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
declToTypeHierarchyItem(ASTContext &Ctx, const NamedDecl &ND) {
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
  auto TUPath = getCanonicalPath(SM.getFileEntryForID(SM.getMainFileID()), SM);
  if (!FilePath || !TUPath)
    return llvm::None; // Not useful without a uri.

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
  THI.range = Range{sourceLocToPosition(SM, DeclRange->getBegin()),
                    sourceLocToPosition(SM, DeclRange->getEnd())};
  THI.selectionRange = Range{NameBegin, NameEnd};
  if (!THI.range.contains(THI.selectionRange)) {
    // 'selectionRange' must be contained in 'range', so in cases where clang
    // reports unrelated ranges we need to reconcile somehow.
    THI.range = THI.selectionRange;
  }

  THI.uri = URIForFile::canonicalize(*FilePath, *TUPath);

  // Compute the SymbolID and store it in the 'data' field.
  // This allows typeHierarchy/resolve to be used to
  // resolve children of items returned in a previous request
  // for parents.
  if (auto ID = getSymbolID(&ND)) {
    THI.data = ID->str();
  }

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
