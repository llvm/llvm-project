//===--- XRefs.cpp ----------------------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//
#include "XRefs.h"
#include "AST.h"
#include "Logger.h"
#include "SourceCode.h"
#include "URI.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Index/IndexDataConsumer.h"
#include "clang/Index/IndexingAction.h"
#include "clang/Index/USRGeneration.h"
#include "llvm/Support/Path.h"
namespace clang {
namespace clangd {
using namespace llvm;
namespace {

// Get the definition from a given declaration `D`.
// Return nullptr if no definition is found, or the declaration type of `D` is
// not supported.
const Decl *GetDefinition(const Decl *D) {
  assert(D);
  if (const auto *TD = dyn_cast<TagDecl>(D))
    return TD->getDefinition();
  else if (const auto *VD = dyn_cast<VarDecl>(D))
    return VD->getDefinition();
  else if (const auto *FD = dyn_cast<FunctionDecl>(D))
    return FD->getDefinition();
  return nullptr;
}

// Convert a SymbolLocation to LSP's Location.
// HintPath is used to resolve the path of URI.
// FIXME: figure out a good home for it, and share the implementation with
// FindSymbols.
llvm::Optional<Location> ToLSPLocation(const SymbolLocation &Loc,
                                       llvm::StringRef HintPath) {
  if (!Loc)
    return llvm::None;
  auto Uri = URI::parse(Loc.FileURI);
  if (!Uri) {
    log("Could not parse URI: " + Loc.FileURI);
    return llvm::None;
  }
  auto Path = URI::resolve(*Uri, HintPath);
  if (!Path) {
    log("Could not resolve URI: " + Loc.FileURI);
    return llvm::None;
  }
  Location LSPLoc;
  LSPLoc.uri = URIForFile(*Path);
  LSPLoc.range.start.line = Loc.Start.Line;
  LSPLoc.range.start.character = Loc.Start.Column;
  LSPLoc.range.end.line = Loc.End.Line;
  LSPLoc.range.end.character = Loc.End.Column;
  return LSPLoc;
}

struct MacroDecl {
  StringRef Name;
  const MacroInfo *Info;
};

/// Finds declarations locations that a given source location refers to.
class DeclarationAndMacrosFinder : public index::IndexDataConsumer {
  std::vector<const Decl *> Decls;
  std::vector<MacroDecl> MacroInfos;
  const SourceLocation &SearchedLocation;
  const ASTContext &AST;
  Preprocessor &PP;

public:
  DeclarationAndMacrosFinder(raw_ostream &OS,
                             const SourceLocation &SearchedLocation,
                             ASTContext &AST, Preprocessor &PP)
      : SearchedLocation(SearchedLocation), AST(AST), PP(PP) {}

  std::vector<const Decl *> takeDecls() {
    // Don't keep the same declaration multiple times.
    // This can happen when nodes in the AST are visited twice.
    std::sort(Decls.begin(), Decls.end());
    auto Last = std::unique(Decls.begin(), Decls.end());
    Decls.erase(Last, Decls.end());
    return std::move(Decls);
  }

  std::vector<MacroDecl> takeMacroInfos() {
    // Don't keep the same Macro info multiple times.
    std::sort(MacroInfos.begin(), MacroInfos.end(),
              [](const MacroDecl &Left, const MacroDecl &Right) {
                return Left.Info < Right.Info;
              });

    auto Last = std::unique(MacroInfos.begin(), MacroInfos.end(),
                            [](const MacroDecl &Left, const MacroDecl &Right) {
                              return Left.Info == Right.Info;
                            });
    MacroInfos.erase(Last, MacroInfos.end());
    return std::move(MacroInfos);
  }

  bool
  handleDeclOccurence(const Decl *D, index::SymbolRoleSet Roles,
                      ArrayRef<index::SymbolRelation> Relations,
                      SourceLocation Loc,
                      index::IndexDataConsumer::ASTNodeInfo ASTNode) override {
    if (Loc == SearchedLocation) {
      // Find and add definition declarations (for GoToDefinition).
      // We don't use parameter `D`, as Parameter `D` is the canonical
      // declaration, which is the first declaration of a redeclarable
      // declaration, and it could be a forward declaration.
      if (const auto *Def = GetDefinition(D)) {
        Decls.push_back(Def);
      } else {
        // Couldn't find a definition, fall back to use `D`.
        Decls.push_back(D);
      }
    }
    return true;
  }

private:
  void finish() override {
    // Also handle possible macro at the searched location.
    Token Result;
    auto &Mgr = AST.getSourceManager();
    if (!Lexer::getRawToken(Mgr.getSpellingLoc(SearchedLocation), Result, Mgr,
                            AST.getLangOpts(), false)) {
      if (Result.is(tok::raw_identifier)) {
        PP.LookUpIdentifierInfo(Result);
      }
      IdentifierInfo *IdentifierInfo = Result.getIdentifierInfo();
      if (IdentifierInfo && IdentifierInfo->hadMacroDefinition()) {
        std::pair<FileID, unsigned int> DecLoc =
            Mgr.getDecomposedExpansionLoc(SearchedLocation);
        // Get the definition just before the searched location so that a macro
        // referenced in a '#undef MACRO' can still be found.
        SourceLocation BeforeSearchedLocation = Mgr.getMacroArgExpandedLocation(
            Mgr.getLocForStartOfFile(DecLoc.first)
                .getLocWithOffset(DecLoc.second - 1));
        MacroDefinition MacroDef =
            PP.getMacroDefinitionAtLoc(IdentifierInfo, BeforeSearchedLocation);
        MacroInfo *MacroInf = MacroDef.getMacroInfo();
        if (MacroInf) {
          MacroInfos.push_back(MacroDecl{IdentifierInfo->getName(), MacroInf});
          assert(Decls.empty());
        }
      }
    }
  }
};

struct IdentifiedSymbol {
  std::vector<const Decl *> Decls;
  std::vector<MacroDecl> Macros;
};

IdentifiedSymbol getSymbolAtPosition(ParsedAST &AST, SourceLocation Pos) {
  auto DeclMacrosFinder = DeclarationAndMacrosFinder(
      llvm::errs(), Pos, AST.getASTContext(), AST.getPreprocessor());
  index::IndexingOptions IndexOpts;
  IndexOpts.SystemSymbolFilter =
      index::IndexingOptions::SystemSymbolFilterKind::All;
  IndexOpts.IndexFunctionLocals = true;
  indexTopLevelDecls(AST.getASTContext(), AST.getLocalTopLevelDecls(),
                     DeclMacrosFinder, IndexOpts);

  return {DeclMacrosFinder.takeDecls(), DeclMacrosFinder.takeMacroInfos()};
}

llvm::Optional<std::string>
getAbsoluteFilePath(const FileEntry *F, const SourceManager &SourceMgr) {
  SmallString<64> FilePath = F->tryGetRealPathName();
  if (FilePath.empty())
    FilePath = F->getName();
  if (!llvm::sys::path::is_absolute(FilePath)) {
    if (!SourceMgr.getFileManager().makeAbsolutePath(FilePath)) {
      log("Could not turn relative path to absolute: " + FilePath);
      return llvm::None;
    }
  }
  return FilePath.str().str();
}

llvm::Optional<Location>
makeLocation(ParsedAST &AST, const SourceRange &ValSourceRange) {
  const SourceManager &SourceMgr = AST.getASTContext().getSourceManager();
  const LangOptions &LangOpts = AST.getASTContext().getLangOpts();
  SourceLocation LocStart = ValSourceRange.getBegin();

  const FileEntry *F =
      SourceMgr.getFileEntryForID(SourceMgr.getFileID(LocStart));
  if (!F)
    return llvm::None;
  SourceLocation LocEnd = Lexer::getLocForEndOfToken(ValSourceRange.getEnd(), 0,
                                                     SourceMgr, LangOpts);
  Position Begin = sourceLocToPosition(SourceMgr, LocStart);
  Position End = sourceLocToPosition(SourceMgr, LocEnd);
  Range R = {Begin, End};
  Location L;

  auto FilePath = getAbsoluteFilePath(F, SourceMgr);
  if (!FilePath) {
    log("failed to get path!");
    return llvm::None;
  }
  L.uri = URIForFile(*FilePath);
  L.range = R;
  return L;
}

// Get the symbol ID for a declaration, if possible.
llvm::Optional<SymbolID> getSymbolID(const Decl *D) {
  llvm::SmallString<128> USR;
  if (index::generateUSRForDecl(D, USR)) {
    return None;
  }
  return SymbolID(USR);
}

} // namespace

std::vector<Location> findDefinitions(ParsedAST &AST, Position Pos,
                                      const SymbolIndex *Index) {
  const SourceManager &SourceMgr = AST.getASTContext().getSourceManager();

  std::vector<Location> Result;
  // Handle goto definition for #include.
  for (auto &Inc : AST.getIncludeStructure().MainFileIncludes) {
    if (!Inc.Resolved.empty() && Inc.R.contains(Pos))
      Result.push_back(Location{URIForFile{Inc.Resolved}, {}});
  }
  if (!Result.empty())
    return Result;

  // Identified symbols at a specific position.
  SourceLocation SourceLocationBeg =
      getBeginningOfIdentifier(AST, Pos, SourceMgr.getMainFileID());
  auto Symbols = getSymbolAtPosition(AST, SourceLocationBeg);

  for (auto Item : Symbols.Macros) {
    auto Loc = Item.Info->getDefinitionLoc();
    auto L = makeLocation(AST, SourceRange(Loc, Loc));
    if (L)
      Result.push_back(*L);
  }

  // Declaration and definition are different terms in C-family languages, and
  // LSP only defines the "GoToDefinition" specification, so we try to perform
  // the "most sensible" GoTo operation:
  //
  //  - We use the location from AST and index (if available) to provide the
  //    final results. When there are duplicate results, we prefer AST over
  //    index because AST is more up-to-date.
  //
  //  - For each symbol, we will return a location of the canonical declaration
  //    (e.g. function declaration in header), and a location of definition if
  //    they are available.
  //
  // So the work flow:
  //
  //   1. Identify the symbols being search for by traversing the AST.
  //   2. Populate one of the locations with the AST location.
  //   3. Use the AST information to query the index, and populate the index
  //      location (if available).
  //   4. Return all populated locations for all symbols, definition first (
  //      which  we think is the users wants most often).
  struct CandidateLocation {
    llvm::Optional<Location> Def;
    llvm::Optional<Location> Decl;
  };
  llvm::DenseMap<SymbolID, CandidateLocation> ResultCandidates;

  // Emit all symbol locations (declaration or definition) from AST.
  for (const auto *D : Symbols.Decls) {
    // Fake key for symbols don't have USR (no SymbolID).
    // Ideally, there should be a USR for each identified symbols. Symbols
    // without USR are rare and unimportant cases, we use the a fake holder to
    // minimize the invasiveness of these cases.
    SymbolID Key("");
    if (auto ID = getSymbolID(D))
      Key = *ID;

    auto &Candidate = ResultCandidates[Key];
    auto Loc = findNameLoc(D);
    auto L = makeLocation(AST, SourceRange(Loc, Loc));
    // The declaration in the identified symbols is a definition if possible
    // otherwise it is declaration.
    bool IsDef = GetDefinition(D) == D;
    // Populate one of the slots with location for the AST.
    if (!IsDef)
      Candidate.Decl = L;
    else
      Candidate.Def = L;
  }

  if (Index) {
    LookupRequest QueryRequest;
    // Build request for index query, using SymbolID.
    for (auto It : ResultCandidates)
      QueryRequest.IDs.insert(It.first);
    std::string HintPath;
    const FileEntry *FE =
        SourceMgr.getFileEntryForID(SourceMgr.getMainFileID());
    if (auto Path = getAbsoluteFilePath(FE, SourceMgr))
      HintPath = *Path;
    // Query the index and populate the empty slot.
    Index->lookup(
        QueryRequest, [&HintPath, &ResultCandidates](const Symbol &Sym) {
          auto It = ResultCandidates.find(Sym.ID);
          assert(It != ResultCandidates.end());
          auto &Value = It->second;

          if (!Value.Def)
            Value.Def = ToLSPLocation(Sym.Definition, HintPath);
          if (!Value.Decl)
            Value.Decl = ToLSPLocation(Sym.CanonicalDeclaration, HintPath);
        });
  }

  // Populate the results, definition first.
  for (auto It : ResultCandidates) {
    const auto &Candidate = It.second;
    if (Candidate.Def)
      Result.push_back(*Candidate.Def);
    if (Candidate.Decl &&
        Candidate.Decl != Candidate.Def) // Decl and Def might be the same
      Result.push_back(*Candidate.Decl);
  }

  return Result;
}

namespace {

/// Finds document highlights that a given list of declarations refers to.
class DocumentHighlightsFinder : public index::IndexDataConsumer {
  std::vector<const Decl *> &Decls;
  std::vector<DocumentHighlight> DocumentHighlights;
  const ASTContext &AST;

public:
  DocumentHighlightsFinder(raw_ostream &OS, ASTContext &AST, Preprocessor &PP,
                           std::vector<const Decl *> &Decls)
      : Decls(Decls), AST(AST) {}
  std::vector<DocumentHighlight> takeHighlights() {
    // Don't keep the same highlight multiple times.
    // This can happen when nodes in the AST are visited twice.
    std::sort(DocumentHighlights.begin(), DocumentHighlights.end());
    auto Last =
        std::unique(DocumentHighlights.begin(), DocumentHighlights.end());
    DocumentHighlights.erase(Last, DocumentHighlights.end());
    return std::move(DocumentHighlights);
  }

  bool
  handleDeclOccurence(const Decl *D, index::SymbolRoleSet Roles,
                      ArrayRef<index::SymbolRelation> Relations,
                      SourceLocation Loc,
                      index::IndexDataConsumer::ASTNodeInfo ASTNode) override {
    const SourceManager &SourceMgr = AST.getSourceManager();
    SourceLocation HighlightStartLoc = SourceMgr.getFileLoc(Loc);
    if (SourceMgr.getMainFileID() != SourceMgr.getFileID(HighlightStartLoc) ||
        std::find(Decls.begin(), Decls.end(), D) == Decls.end()) {
      return true;
    }
    SourceLocation End;
    const LangOptions &LangOpts = AST.getLangOpts();
    End = Lexer::getLocForEndOfToken(HighlightStartLoc, 0, SourceMgr, LangOpts);
    SourceRange SR(HighlightStartLoc, End);

    DocumentHighlightKind Kind = DocumentHighlightKind::Text;
    if (static_cast<index::SymbolRoleSet>(index::SymbolRole::Write) & Roles)
      Kind = DocumentHighlightKind::Write;
    else if (static_cast<index::SymbolRoleSet>(index::SymbolRole::Read) & Roles)
      Kind = DocumentHighlightKind::Read;

    DocumentHighlights.push_back(getDocumentHighlight(SR, Kind));
    return true;
  }

private:
  DocumentHighlight getDocumentHighlight(SourceRange SR,
                                         DocumentHighlightKind Kind) {
    const SourceManager &SourceMgr = AST.getSourceManager();
    Position Begin = sourceLocToPosition(SourceMgr, SR.getBegin());
    Position End = sourceLocToPosition(SourceMgr, SR.getEnd());
    Range R = {Begin, End};
    DocumentHighlight DH;
    DH.range = R;
    DH.kind = Kind;
    return DH;
  }
};

} // namespace

std::vector<DocumentHighlight> findDocumentHighlights(ParsedAST &AST,
                                                      Position Pos) {
  const SourceManager &SourceMgr = AST.getASTContext().getSourceManager();
  SourceLocation SourceLocationBeg =
      getBeginningOfIdentifier(AST, Pos, SourceMgr.getMainFileID());

  auto Symbols = getSymbolAtPosition(AST, SourceLocationBeg);
  std::vector<const Decl *> SelectedDecls = Symbols.Decls;

  DocumentHighlightsFinder DocHighlightsFinder(
      llvm::errs(), AST.getASTContext(), AST.getPreprocessor(), SelectedDecls);

  index::IndexingOptions IndexOpts;
  IndexOpts.SystemSymbolFilter =
      index::IndexingOptions::SystemSymbolFilterKind::All;
  IndexOpts.IndexFunctionLocals = true;
  indexTopLevelDecls(AST.getASTContext(), AST.getLocalTopLevelDecls(),
                     DocHighlightsFinder, IndexOpts);

  return DocHighlightsFinder.takeHighlights();
}

static PrintingPolicy PrintingPolicyForDecls(PrintingPolicy Base) {
  PrintingPolicy Policy(Base);

  Policy.AnonymousTagLocations = false;
  Policy.TerseOutput = true;
  Policy.PolishForDeclaration = true;
  Policy.ConstantsAsWritten = true;
  Policy.SuppressTagKeyword = false;

  return Policy;
}

/// Return a string representation (e.g. "class MyNamespace::MyClass") of
/// the type declaration \p TD.
static std::string TypeDeclToString(const TypeDecl *TD) {
  QualType Type = TD->getASTContext().getTypeDeclType(TD);

  PrintingPolicy Policy =
      PrintingPolicyForDecls(TD->getASTContext().getPrintingPolicy());

  std::string Name;
  llvm::raw_string_ostream Stream(Name);
  Type.print(Stream, Policy);

  return Stream.str();
}

/// Return a string representation (e.g. "namespace ns1::ns2") of
/// the named declaration \p ND.
static std::string NamedDeclQualifiedName(const NamedDecl *ND,
                                          StringRef Prefix) {
  PrintingPolicy Policy =
      PrintingPolicyForDecls(ND->getASTContext().getPrintingPolicy());

  std::string Name;
  llvm::raw_string_ostream Stream(Name);
  Stream << Prefix << ' ';
  ND->printQualifiedName(Stream, Policy);

  return Stream.str();
}

/// Given a declaration \p D, return a human-readable string representing the
/// scope in which it is declared.  If the declaration is in the global scope,
/// return the string "global namespace".
static llvm::Optional<std::string> getScopeName(const Decl *D) {
  const DeclContext *DC = D->getDeclContext();

  if (isa<TranslationUnitDecl>(DC))
    return std::string("global namespace");
  if (const TypeDecl *TD = dyn_cast<TypeDecl>(DC))
    return TypeDeclToString(TD);
  else if (const NamespaceDecl *ND = dyn_cast<NamespaceDecl>(DC))
    return NamedDeclQualifiedName(ND, "namespace");
  else if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(DC))
    return NamedDeclQualifiedName(FD, "function");

  return llvm::None;
}

/// Generate a \p Hover object given the declaration \p D.
static Hover getHoverContents(const Decl *D) {
  Hover H;
  llvm::Optional<std::string> NamedScope = getScopeName(D);

  // Generate the "Declared in" section.
  if (NamedScope) {
    assert(!NamedScope->empty());

    H.contents.value += "Declared in ";
    H.contents.value += *NamedScope;
    H.contents.value += "\n\n";
  }

  // We want to include the template in the Hover.
  if (TemplateDecl *TD = D->getDescribedTemplate())
    D = TD;

  std::string DeclText;
  llvm::raw_string_ostream OS(DeclText);

  PrintingPolicy Policy =
      PrintingPolicyForDecls(D->getASTContext().getPrintingPolicy());

  D->print(OS, Policy);

  OS.flush();

  H.contents.value += DeclText;
  return H;
}

/// Generate a \p Hover object given the type \p T.
static Hover getHoverContents(QualType T, ASTContext &ASTCtx) {
  Hover H;
  std::string TypeText;
  llvm::raw_string_ostream OS(TypeText);
  PrintingPolicy Policy = PrintingPolicyForDecls(ASTCtx.getPrintingPolicy());
  T.print(OS, Policy);
  OS.flush();
  H.contents.value += TypeText;
  return H;
}

/// Generate a \p Hover object given the macro \p MacroInf.
static Hover getHoverContents(StringRef MacroName) {
  Hover H;

  H.contents.value = "#define ";
  H.contents.value += MacroName;

  return H;
}

namespace {
/// Computes the deduced type at a given location by visiting the relevant
/// nodes. We use this to display the actual type when hovering over an "auto"
/// keyword or "decltype()" expression.
/// FIXME: This could have been a lot simpler by visiting AutoTypeLocs but it
/// seems that the AutoTypeLocs that can be visited along with their AutoType do
/// not have the deduced type set. Instead, we have to go to the appropriate
/// DeclaratorDecl/FunctionDecl and work our back to the AutoType that does have
/// a deduced type set. The AST should be improved to simplify this scenario.
class DeducedTypeVisitor : public RecursiveASTVisitor<DeducedTypeVisitor> {
  SourceLocation SearchedLocation;
  llvm::Optional<QualType> DeducedType;

public:
  DeducedTypeVisitor(SourceLocation SearchedLocation)
      : SearchedLocation(SearchedLocation) {}

  llvm::Optional<QualType> getDeducedType() { return DeducedType; }

  // Handle auto initializers:
  //- auto i = 1;
  //- decltype(auto) i = 1;
  //- auto& i = 1;
  bool VisitDeclaratorDecl(DeclaratorDecl *D) {
    if (!D->getTypeSourceInfo() ||
        D->getTypeSourceInfo()->getTypeLoc().getLocStart() != SearchedLocation)
      return true;

    auto DeclT = D->getType();
    // "auto &" is represented as a ReferenceType containing an AutoType
    if (const ReferenceType *RT = dyn_cast<ReferenceType>(DeclT.getTypePtr()))
      DeclT = RT->getPointeeType();

    const AutoType *AT = dyn_cast<AutoType>(DeclT.getTypePtr());
    if (AT && !AT->getDeducedType().isNull()) {
      // For auto, use the underlying type because the const& would be
      // represented twice: written in the code and in the hover.
      // Example: "const auto I = 1", we only want "int" when hovering on auto,
      // not "const int".
      //
      // For decltype(auto), take the type as is because it cannot be written
      // with qualifiers or references but its decuded type can be const-ref.
      DeducedType = AT->isDecltypeAuto() ? DeclT : DeclT.getUnqualifiedType();
    }
    return true;
  }

  // Handle auto return types:
  //- auto foo() {}
  //- auto& foo() {}
  //- auto foo() -> decltype(1+1) {}
  //- operator auto() const { return 10; }
  bool VisitFunctionDecl(FunctionDecl *D) {
    if (!D->getTypeSourceInfo())
      return true;
    // Loc of auto in return type (c++14).
    auto CurLoc = D->getReturnTypeSourceRange().getBegin();
    // Loc of "auto" in operator auto()
    if (CurLoc.isInvalid() && dyn_cast<CXXConversionDecl>(D))
      CurLoc = D->getTypeSourceInfo()->getTypeLoc().getBeginLoc();
    // Loc of "auto" in function with traling return type (c++11).
    if (CurLoc.isInvalid())
      CurLoc = D->getSourceRange().getBegin();
    if (CurLoc != SearchedLocation)
      return true;

    auto T = D->getReturnType();
    // "auto &" is represented as a ReferenceType containing an AutoType.
    if (const ReferenceType *RT = dyn_cast<ReferenceType>(T.getTypePtr()))
      T = RT->getPointeeType();

    const AutoType *AT = dyn_cast<AutoType>(T.getTypePtr());
    if (AT && !AT->getDeducedType().isNull()) {
      DeducedType = T.getUnqualifiedType();
    } else { // auto in a trailing return type just points to a DecltypeType.
      const DecltypeType *DT = dyn_cast<DecltypeType>(T.getTypePtr());
      if (!DT->getUnderlyingType().isNull())
        DeducedType = DT->getUnderlyingType();
    }
    return true;
  }

  // Handle non-auto decltype, e.g.:
  // - auto foo() -> decltype(expr) {}
  // - decltype(expr);
  bool VisitDecltypeTypeLoc(DecltypeTypeLoc TL) {
    if (TL.getBeginLoc() != SearchedLocation)
      return true;

    // A DecltypeType's underlying type can be another DecltypeType! E.g.
    //  int I = 0;
    //  decltype(I) J = I;
    //  decltype(J) K = J;
    const DecltypeType *DT = dyn_cast<DecltypeType>(TL.getTypePtr());
    while (DT && !DT->getUnderlyingType().isNull()) {
      DeducedType = DT->getUnderlyingType();
      DT = dyn_cast<DecltypeType>(DeducedType->getTypePtr());
    }
    return true;
  }
};
} // namespace

/// Retrieves the deduced type at a given location (auto, decltype).
llvm::Optional<QualType> getDeducedType(ParsedAST &AST,
                                        SourceLocation SourceLocationBeg) {
  Token Tok;
  auto &ASTCtx = AST.getASTContext();
  // Only try to find a deduced type if the token is auto or decltype.
  if (!SourceLocationBeg.isValid() ||
      Lexer::getRawToken(SourceLocationBeg, Tok, ASTCtx.getSourceManager(),
                         ASTCtx.getLangOpts(), false) ||
      !Tok.is(tok::raw_identifier)) {
    return {};
  }
  AST.getPreprocessor().LookUpIdentifierInfo(Tok);
  if (!(Tok.is(tok::kw_auto) || Tok.is(tok::kw_decltype)))
    return {};

  DeducedTypeVisitor V(SourceLocationBeg);
  for (Decl *D : AST.getLocalTopLevelDecls())
    V.TraverseDecl(D);
  return V.getDeducedType();
}

Optional<Hover> getHover(ParsedAST &AST, Position Pos) {
  const SourceManager &SourceMgr = AST.getASTContext().getSourceManager();
  SourceLocation SourceLocationBeg =
      getBeginningOfIdentifier(AST, Pos, SourceMgr.getMainFileID());
  // Identified symbols at a specific position.
  auto Symbols = getSymbolAtPosition(AST, SourceLocationBeg);

  if (!Symbols.Macros.empty())
    return getHoverContents(Symbols.Macros[0].Name);

  if (!Symbols.Decls.empty())
    return getHoverContents(Symbols.Decls[0]);

  auto DeducedType = getDeducedType(AST, SourceLocationBeg);
  if (DeducedType && !DeducedType->isNull())
    return getHoverContents(*DeducedType, AST.getASTContext());

  return None;
}

} // namespace clangd
} // namespace clang
