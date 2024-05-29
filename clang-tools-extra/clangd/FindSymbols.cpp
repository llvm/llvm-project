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
#include "clang/Index/IndexSymbol.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <limits>
#include <optional>
#include <tuple>

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

// Returns true if \p Query can be found as a sub-sequence inside \p Scope.
bool approximateScopeMatch(llvm::StringRef Scope, llvm::StringRef Query) {
  assert(Scope.empty() || Scope.ends_with("::"));
  assert(Query.empty() || Query.ends_with("::"));
  while (!Scope.empty() && !Query.empty()) {
    auto Colons = Scope.find("::");
    assert(Colons != llvm::StringRef::npos);

    llvm::StringRef LeadingSpecifier = Scope.slice(0, Colons + 2);
    Scope = Scope.slice(Colons + 2, llvm::StringRef::npos);
    Query.consume_front(LeadingSpecifier);
  }
  return Query.empty();
}

} // namespace

llvm::Expected<Location> indexToLSPLocation(const SymbolLocation &Loc,
                                            llvm::StringRef TUPath) {
  auto Path = URI::resolve(Loc.FileURI, TUPath);
  if (!Path)
    return error("Could not resolve path for file '{0}': {1}", Loc.FileURI,
                 Path.takeError());
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
  if (!Index)
    return Result;

  // Lookup for qualified names are performed as:
  // - Exact namespaces are boosted by the index.
  // - Approximate matches are (sub-scope match) included via AnyScope logic.
  // - Non-matching namespaces (no sub-scope match) are post-filtered.
  auto Names = splitQualifiedName(Query);

  FuzzyFindRequest Req;
  Req.Query = std::string(Names.second);

  // FuzzyFind doesn't want leading :: qualifier.
  auto HasLeadingColons = Names.first.consume_front("::");
  // Limit the query to specific namespace if it is fully-qualified.
  Req.AnyScope = !HasLeadingColons;
  // Boost symbols from desired namespace.
  if (HasLeadingColons || !Names.first.empty())
    Req.Scopes = {std::string(Names.first)};
  if (Limit) {
    Req.Limit = Limit;
    // If we are boosting a specific scope allow more results to be retrieved,
    // since some symbols from preferred namespaces might not make the cut.
    if (Req.AnyScope && !Req.Scopes.empty())
      *Req.Limit *= 5;
  }
  TopN<ScoredSymbolInfo, ScoredSymbolGreater> Top(
      Req.Limit ? *Req.Limit : std::numeric_limits<size_t>::max());
  FuzzyMatcher Filter(Req.Query);

  Index->fuzzyFind(Req, [HintPath, &Top, &Filter, AnyScope = Req.AnyScope,
                         ReqScope = Names.first](const Symbol &Sym) {
    llvm::StringRef Scope = Sym.Scope;
    // Fuzzyfind might return symbols from irrelevant namespaces if query was
    // not fully-qualified, drop those.
    if (AnyScope && !approximateScopeMatch(Scope, ReqScope))
      return;

    auto Loc = symbolToLocation(Sym, HintPath);
    if (!Loc) {
      log("Workspace symbols: {0}", Loc.takeError());
      return;
    }

    SymbolQualitySignals Quality;
    Quality.merge(Sym);
    SymbolRelevanceSignals Relevance;
    Relevance.Name = Sym.Name;
    Relevance.Query = SymbolRelevanceSignals::Generic;
    // If symbol and request scopes do not match exactly, apply a penalty.
    Relevance.InBaseClass = AnyScope && Scope != ReqScope;
    if (auto NameMatch = Filter.match(Sym.Name))
      Relevance.NameMatch = *NameMatch;
    else {
      log("Workspace symbol: {0} didn't match query {1}", Sym.Name,
          Filter.pattern());
      return;
    }
    Relevance.merge(Sym);
    auto QualScore = Quality.evaluateHeuristics();
    auto RelScore = Relevance.evaluateHeuristics();
    auto Score = evaluateSymbolAndRelevance(QualScore, RelScore);
    dlog("FindSymbols: {0}{1} = {2}\n{3}{4}\n", Sym.Scope, Sym.Name, Score,
         Quality, Relevance);

    SymbolInformation Info;
    Info.name = (Sym.Name + Sym.TemplateSpecializationArgs).str();
    Info.kind = indexSymbolKindToSymbolKind(Sym.SymInfo.Kind);
    Info.location = *Loc;
    Scope.consume_back("::");
    Info.containerName = Scope.str();

    // Exposed score excludes fuzzy-match component, for client-side re-ranking.
    Info.score = Relevance.NameMatch > std::numeric_limits<float>::epsilon()
                     ? Score / Relevance.NameMatch
                     : QualScore;
    Top.push({Score, std::move(Info)});
  });
  for (auto &R : std::move(Top).items())
    Result.push_back(std::move(R.second));
  return Result;
}

namespace {
std::string getSymbolName(ASTContext &Ctx, const NamedDecl &ND) {
  // Print `MyClass(Category)` instead of `Category` and `MyClass()` instead
  // of `anonymous`.
  if (const auto *Container = dyn_cast<ObjCContainerDecl>(&ND))
    return printObjCContainer(*Container);
  // Differentiate between class and instance methods: print `-foo` instead of
  // `foo` and `+sharedInstance` instead of `sharedInstance`.
  if (const auto *Method = dyn_cast<ObjCMethodDecl>(&ND)) {
    std::string Name;
    llvm::raw_string_ostream OS(Name);

    OS << (Method->isInstanceMethod() ? '-' : '+');
    Method->getSelector().print(OS);

    OS.flush();
    return Name;
  }
  return printName(Ctx, ND);
}

std::string getSymbolDetail(ASTContext &Ctx, const NamedDecl &ND) {
  PrintingPolicy P(Ctx.getPrintingPolicy());
  P.SuppressScope = true;
  P.SuppressUnwrittenScope = true;
  P.AnonymousTagLocations = false;
  P.PolishForDeclaration = true;
  std::string Detail;
  llvm::raw_string_ostream OS(Detail);
  if (ND.getDescribedTemplateParams()) {
    OS << "template ";
  }
  if (const auto *VD = dyn_cast<ValueDecl>(&ND)) {
    // FIXME: better printing for dependent type
    if (isa<CXXConstructorDecl>(VD)) {
      std::string ConstructorType = VD->getType().getAsString(P);
      // Print constructor type as "(int)" instead of "void (int)".
      llvm::StringRef WithoutVoid = ConstructorType;
      WithoutVoid.consume_front("void ");
      OS << WithoutVoid;
    } else if (!isa<CXXDestructorDecl>(VD)) {
      VD->getType().print(OS, P);
    }
  } else if (const auto *TD = dyn_cast<TagDecl>(&ND)) {
    OS << TD->getKindName();
  } else if (isa<TypedefNameDecl>(&ND)) {
    OS << "type alias";
  } else if (isa<ConceptDecl>(&ND)) {
    OS << "concept";
  }
  return std::move(OS.str());
}

std::optional<DocumentSymbol> declToSym(ASTContext &Ctx, const NamedDecl &ND) {
  auto &SM = Ctx.getSourceManager();

  SourceLocation BeginLoc = ND.getBeginLoc();
  SourceLocation EndLoc = ND.getEndLoc();
  const auto SymbolRange =
      toHalfOpenFileRange(SM, Ctx.getLangOpts(), {BeginLoc, EndLoc});
  if (!SymbolRange)
    return std::nullopt;

  index::SymbolInfo SymInfo = index::getSymbolInfo(&ND);
  // FIXME: This is not classifying constructors, destructors and operators
  // correctly.
  SymbolKind SK = indexSymbolKindToSymbolKind(SymInfo.Kind);

  DocumentSymbol SI;
  SI.name = getSymbolName(Ctx, ND);
  SI.kind = SK;
  SI.deprecated = ND.isDeprecated();
  SI.range = Range{sourceLocToPosition(SM, SymbolRange->getBegin()),
                   sourceLocToPosition(SM, SymbolRange->getEnd())};
  SI.detail = getSymbolDetail(Ctx, ND);

  SourceLocation NameLoc = ND.getLocation();
  SourceLocation FallbackNameLoc;
  if (NameLoc.isMacroID()) {
    if (isSpelledInSource(NameLoc, SM)) {
      // Prefer the spelling loc, but save the expansion loc as a fallback.
      FallbackNameLoc = SM.getExpansionLoc(NameLoc);
      NameLoc = SM.getSpellingLoc(NameLoc);
    } else {
      NameLoc = SM.getExpansionLoc(NameLoc);
    }
  }
  auto ComputeSelectionRange = [&](SourceLocation L) -> Range {
    Position NameBegin = sourceLocToPosition(SM, L);
    Position NameEnd = sourceLocToPosition(
        SM, Lexer::getLocForEndOfToken(L, 0, SM, Ctx.getLangOpts()));
    return Range{NameBegin, NameEnd};
  };

  SI.selectionRange = ComputeSelectionRange(NameLoc);
  if (!SI.range.contains(SI.selectionRange) && FallbackNameLoc.isValid()) {
    // 'selectionRange' must be contained in 'range'. In cases where clang
    // reports unrelated ranges, we first try falling back to the expansion
    // loc for the selection range.
    SI.selectionRange = ComputeSelectionRange(FallbackNameLoc);
  }
  if (!SI.range.contains(SI.selectionRange)) {
    // If the containment relationship still doesn't hold, throw away
    // 'range' and use 'selectionRange' for both.
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
  // A DocumentSymbol we're constructing.
  // We use this instead of DocumentSymbol directly so that we can keep track
  // of the nodes we insert for macros.
  class SymBuilder {
    std::vector<SymBuilder> Children;
    DocumentSymbol Symbol; // Symbol.children is empty, use Children instead.
    // Macro expansions that this node or its parents are associated with.
    // (Thus we will never create further children for these expansions).
    llvm::SmallVector<SourceLocation> EnclosingMacroLoc;

  public:
    DocumentSymbol build() && {
      for (SymBuilder &C : Children) {
        Symbol.children.push_back(std::move(C).build());
        // Expand range to ensure children nest properly, which editors expect.
        // This can fix some edge-cases in the AST, but is vital for macros.
        // A macro expansion "contains" AST node if it covers the node's primary
        // location, but it may not span the node's whole range.
        Symbol.range.start =
            std::min(Symbol.range.start, Symbol.children.back().range.start);
        Symbol.range.end =
            std::max(Symbol.range.end, Symbol.children.back().range.end);
      }
      return std::move(Symbol);
    }

    // Add a symbol as a child of the current one.
    SymBuilder &addChild(DocumentSymbol S) {
      Children.emplace_back();
      Children.back().EnclosingMacroLoc = EnclosingMacroLoc;
      Children.back().Symbol = std::move(S);
      return Children.back();
    }

    // Get an appropriate container for children of this symbol that were
    // expanded from a macro (whose spelled name is Tok).
    //
    // This may return:
    //  - a macro symbol child of this (either new or previously created)
    //  - this scope itself, if it *is* the macro symbol or is nested within it
    SymBuilder &inMacro(const syntax::Token &Tok, const SourceManager &SM,
                        std::optional<syntax::TokenBuffer::Expansion> Exp) {
      if (llvm::is_contained(EnclosingMacroLoc, Tok.location()))
        return *this;
      // If there's an existing child for this macro, we expect it to be last.
      if (!Children.empty() && !Children.back().EnclosingMacroLoc.empty() &&
          Children.back().EnclosingMacroLoc.back() == Tok.location())
        return Children.back();

      DocumentSymbol Sym;
      Sym.name = Tok.text(SM).str();
      Sym.kind = SymbolKind::Null; // There's no suitable kind!
      Sym.range = Sym.selectionRange =
          halfOpenToRange(SM, Tok.range(SM).toCharRange(SM));

      // FIXME: Exp is currently unavailable for nested expansions.
      if (Exp) {
        // Full range covers the macro args.
        Sym.range = halfOpenToRange(SM, CharSourceRange::getCharRange(
                                            Exp->Spelled.front().location(),
                                            Exp->Spelled.back().endLocation()));
        // Show macro args as detail.
        llvm::raw_string_ostream OS(Sym.detail);
        const syntax::Token *Prev = nullptr;
        for (const auto &Tok : Exp->Spelled.drop_front()) {
          // Don't dump arbitrarily long macro args.
          if (OS.tell() > 80) {
            OS << " ...)";
            break;
          }
          if (Prev && Prev->endLocation() != Tok.location())
            OS << ' ';
          OS << Tok.text(SM);
          Prev = &Tok;
        }
      }
      SymBuilder &Child = addChild(std::move(Sym));
      Child.EnclosingMacroLoc.push_back(Tok.location());
      return Child;
    }
  };

public:
  DocumentOutline(ParsedAST &AST) : AST(AST) {}

  /// Builds the document outline for the generated AST.
  std::vector<DocumentSymbol> build() {
    SymBuilder Root;
    for (auto &TopLevel : AST.getLocalTopLevelDecls())
      traverseDecl(TopLevel, Root);
    return std::move(std::move(Root).build().children);
  }

private:
  enum class VisitKind { No, OnlyDecl, OnlyChildren, DeclAndChildren };

  void traverseDecl(Decl *D, SymBuilder &Parent) {
    // Skip symbols which do not originate from the main file.
    if (!isInsideMainFile(D->getLocation(), AST.getSourceManager()))
      return;

    if (auto *Templ = llvm::dyn_cast<TemplateDecl>(D)) {
      // TemplatedDecl might be null, e.g. concepts.
      if (auto *TD = Templ->getTemplatedDecl())
        D = TD;
    }

    VisitKind Visit = shouldVisit(D);
    if (Visit == VisitKind::No)
      return;

    if (Visit == VisitKind::OnlyChildren)
      return traverseChildren(D, Parent);

    auto *ND = llvm::cast<NamedDecl>(D);
    auto Sym = declToSym(AST.getASTContext(), *ND);
    if (!Sym)
      return;
    SymBuilder &MacroParent = possibleMacroContainer(D->getLocation(), Parent);
    SymBuilder &Child = MacroParent.addChild(std::move(*Sym));

    if (Visit == VisitKind::OnlyDecl)
      return;

    assert(Visit == VisitKind::DeclAndChildren && "Unexpected VisitKind");
    traverseChildren(ND, Child);
  }

  // Determines where a decl should appear in the DocumentSymbol hierarchy.
  //
  // This is usually a direct child of the relevant AST parent.
  // But we may also insert nodes for macros. Given:
  //   #define DECLARE_INT(V) int v;
  //   namespace a { DECLARE_INT(x) }
  // We produce:
  //   Namespace a
  //     Macro DECLARE_INT(x)
  //       Variable x
  //
  // In the absence of macros, this method simply returns Parent.
  // Otherwise it may return a macro expansion node instead.
  // Each macro only has at most one node in the hierarchy, even if it expands
  // to multiple decls.
  SymBuilder &possibleMacroContainer(SourceLocation TargetLoc,
                                     SymBuilder &Parent) {
    const auto &SM = AST.getSourceManager();
    // Look at the path of macro-callers from the token to the main file.
    // Note that along these paths we see the "outer" macro calls first.
    SymBuilder *CurParent = &Parent;
    for (SourceLocation Loc = TargetLoc; Loc.isMacroID();
         Loc = SM.getImmediateMacroCallerLoc(Loc)) {
      // Find the virtual macro body that our token is being substituted into.
      FileID MacroBody;
      if (SM.isMacroArgExpansion(Loc)) {
        // Loc is part of a macro arg being substituted into a macro body.
        MacroBody = SM.getFileID(SM.getImmediateExpansionRange(Loc).getBegin());
      } else {
        // Loc is already in the macro body.
        MacroBody = SM.getFileID(Loc);
      }
      // The macro body is being substituted for a macro expansion, whose
      // first token is the name of the macro.
      SourceLocation MacroName =
          SM.getSLocEntry(MacroBody).getExpansion().getExpansionLocStart();
      // Only include the macro expansion in the outline if it was written
      // directly in the main file, rather than expanded from another macro.
      if (!MacroName.isValid() || !MacroName.isFileID())
        continue;
      // All conditions satisfied, add the macro.
      if (auto *Tok = AST.getTokens().spelledTokenAt(MacroName))
        CurParent = &CurParent->inMacro(
            *Tok, SM, AST.getTokens().expansionStartingAt(Tok));
    }
    return *CurParent;
  }

  void traverseChildren(Decl *D, SymBuilder &Builder) {
    auto *Scope = llvm::dyn_cast<DeclContext>(D);
    if (!Scope)
      return;
    for (auto *C : Scope->decls())
      traverseDecl(C, Builder);
  }

  VisitKind shouldVisit(Decl *D) {
    if (D->isImplicit())
      return VisitKind::No;

    if (llvm::isa<LinkageSpecDecl>(D) || llvm::isa<ExportDecl>(D))
      return VisitKind::OnlyChildren;

    if (!llvm::isa<NamedDecl>(D))
      return VisitKind::No;

    if (auto *Func = llvm::dyn_cast<FunctionDecl>(D)) {
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

struct PragmaMarkSymbol {
  DocumentSymbol DocSym;
  bool IsGroup;
};

/// Merge in `PragmaMarkSymbols`, sorted ascending by range, into the given
/// `DocumentSymbol` tree.
void mergePragmas(DocumentSymbol &Root, ArrayRef<PragmaMarkSymbol> Pragmas) {
  while (!Pragmas.empty()) {
    // We'll figure out where the Pragmas.front() should go.
    PragmaMarkSymbol P = std::move(Pragmas.front());
    Pragmas = Pragmas.drop_front();
    DocumentSymbol *Cur = &Root;
    while (Cur->range.contains(P.DocSym.range)) {
      bool Swapped = false;
      for (auto &C : Cur->children) {
        // We assume at most 1 child can contain the pragma (as pragmas are on
        // a single line, and children have disjoint ranges).
        if (C.range.contains(P.DocSym.range)) {
          Cur = &C;
          Swapped = true;
          break;
        }
      }
      // Cur is the parent of P since none of the children contain P.
      if (!Swapped)
        break;
    }
    // Pragma isn't a group so we can just insert it and we are done.
    if (!P.IsGroup) {
      Cur->children.emplace_back(std::move(P.DocSym));
      continue;
    }
    // Pragma is a group, so we need to figure out where it terminates:
    // - If the next Pragma is not contained in Cur, P owns all of its
    //   parent's children which occur after P.
    // - If the next pragma is contained in Cur but actually belongs to one
    //   of the parent's children, we temporarily skip over it and look at
    //   the next pragma to decide where we end.
    // - Otherwise nest all of its parent's children which occur after P but
    //   before the next pragma.
    bool TerminatedByNextPragma = false;
    for (auto &NextPragma : Pragmas) {
      // If we hit a pragma outside of Cur, the rest will be outside as well.
      if (!Cur->range.contains(NextPragma.DocSym.range))
        break;

      // NextPragma cannot terminate P if it is nested inside a child, look for
      // the next one.
      if (llvm::any_of(Cur->children, [&NextPragma](const auto &Child) {
            return Child.range.contains(NextPragma.DocSym.range);
          }))
        continue;

      // Pragma owns all the children between P and NextPragma
      auto It = llvm::partition(Cur->children,
                                [&P, &NextPragma](const auto &S) -> bool {
                                  return !(P.DocSym.range < S.range &&
                                           S.range < NextPragma.DocSym.range);
                                });
      P.DocSym.children.assign(make_move_iterator(It),
                               make_move_iterator(Cur->children.end()));
      Cur->children.erase(It, Cur->children.end());
      TerminatedByNextPragma = true;
      break;
    }
    if (!TerminatedByNextPragma) {
      // P is terminated by the end of current symbol, hence it owns all the
      // children after P.
      auto It = llvm::partition(Cur->children, [&P](const auto &S) -> bool {
        return !(P.DocSym.range < S.range);
      });
      P.DocSym.children.assign(make_move_iterator(It),
                               make_move_iterator(Cur->children.end()));
      Cur->children.erase(It, Cur->children.end());
    }
    // Update the range for P to cover children and append to Cur.
    for (DocumentSymbol &Sym : P.DocSym.children)
      unionRanges(P.DocSym.range, Sym.range);
    Cur->children.emplace_back(std::move(P.DocSym));
  }
}

PragmaMarkSymbol markToSymbol(const PragmaMark &P) {
  StringRef Name = StringRef(P.Trivia).trim();
  bool IsGroup = false;
  // "-\s+<group name>" or "<name>" after an initial trim. The former is
  // considered a group, the latter just a mark. Like Xcode, we don't consider
  // `-Foo` to be a group (space(s) after the `-` is required).
  //
  // We need to include a name here, otherwise editors won't properly render the
  // symbol.
  StringRef MaybeGroupName = Name;
  if (MaybeGroupName.consume_front("-") &&
      (MaybeGroupName.ltrim() != MaybeGroupName || MaybeGroupName.empty())) {
    Name = MaybeGroupName.empty() ? "(unnamed group)" : MaybeGroupName.ltrim();
    IsGroup = true;
  } else if (Name.empty()) {
    Name = "(unnamed mark)";
  }
  DocumentSymbol Sym;
  Sym.name = Name.str();
  Sym.kind = SymbolKind::File;
  Sym.range = P.Rng;
  Sym.selectionRange = P.Rng;
  return {Sym, IsGroup};
}

std::vector<DocumentSymbol> collectDocSymbols(ParsedAST &AST) {
  std::vector<DocumentSymbol> Syms = DocumentOutline(AST).build();

  const auto &PragmaMarks = AST.getMarks();
  if (PragmaMarks.empty())
    return Syms;

  std::vector<PragmaMarkSymbol> Pragmas;
  Pragmas.reserve(PragmaMarks.size());
  for (const auto &P : PragmaMarks)
    Pragmas.push_back(markToSymbol(P));
  Range EntireFile = {
      {0, 0},
      {std::numeric_limits<int>::max(), std::numeric_limits<int>::max()}};
  DocumentSymbol Root;
  Root.children = std::move(Syms);
  Root.range = EntireFile;
  mergePragmas(Root, llvm::ArrayRef(Pragmas));
  return Root.children;
}

} // namespace

llvm::Expected<std::vector<DocumentSymbol>> getDocumentSymbols(ParsedAST &AST) {
  return collectDocSymbols(AST);
}

} // namespace clangd
} // namespace clang
