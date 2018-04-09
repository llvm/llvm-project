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
#include "clang/Index/IndexDataConsumer.h"
#include "clang/Index/IndexingAction.h"
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

  SmallString<64> FilePath = F->tryGetRealPathName();
  if (FilePath.empty())
    FilePath = F->getName();
  if (!llvm::sys::path::is_absolute(FilePath)) {
    if (!SourceMgr.getFileManager().makeAbsolutePath(FilePath)) {
      log("Could not turn relative path to absolute: " + FilePath);
      return llvm::None;
    }
  }

  L.uri = URIForFile(FilePath.str());
  L.range = R;
  return L;
}

} // namespace

std::vector<Location> findDefinitions(ParsedAST &AST, Position Pos) {
  const SourceManager &SourceMgr = AST.getASTContext().getSourceManager();
  const FileEntry *FE = SourceMgr.getFileEntryForID(SourceMgr.getMainFileID());
  if (!FE)
    return {};

  SourceLocation SourceLocationBeg = getBeginningOfIdentifier(AST, Pos, FE);

  std::vector<Location> Result;
  // Handle goto definition for #include.
  for (auto &IncludeLoc : AST.getInclusionLocations()) {
    Range R = IncludeLoc.first;
    Position Pos = sourceLocToPosition(SourceMgr, SourceLocationBeg);

    if (R.contains(Pos))
      Result.push_back(Location{URIForFile{IncludeLoc.second}, {}});
  }
  if (!Result.empty())
    return Result;

  auto DeclMacrosFinder = std::make_shared<DeclarationAndMacrosFinder>(
      llvm::errs(), SourceLocationBeg, AST.getASTContext(),
      AST.getPreprocessor());
  index::IndexingOptions IndexOpts;
  IndexOpts.SystemSymbolFilter =
      index::IndexingOptions::SystemSymbolFilterKind::All;
  IndexOpts.IndexFunctionLocals = true;

  indexTopLevelDecls(AST.getASTContext(), AST.getTopLevelDecls(),
                     DeclMacrosFinder, IndexOpts);

  std::vector<const Decl *> Decls = DeclMacrosFinder->takeDecls();
  std::vector<MacroDecl> MacroInfos = DeclMacrosFinder->takeMacroInfos();

  for (auto D : Decls) {
    auto Loc = findNameLoc(D);
    auto L = makeLocation(AST, SourceRange(Loc, Loc));
    if (L)
      Result.push_back(*L);
  }

  for (auto Item : MacroInfos) {
    auto Loc = Item.Info->getDefinitionLoc();
    auto L = makeLocation(AST, SourceRange(Loc, Loc));
    if (L)
      Result.push_back(*L);
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
  const FileEntry *FE = SourceMgr.getFileEntryForID(SourceMgr.getMainFileID());
  if (!FE)
    return {};

  SourceLocation SourceLocationBeg = getBeginningOfIdentifier(AST, Pos, FE);

  auto DeclMacrosFinder = std::make_shared<DeclarationAndMacrosFinder>(
      llvm::errs(), SourceLocationBeg, AST.getASTContext(),
      AST.getPreprocessor());
  index::IndexingOptions IndexOpts;
  IndexOpts.SystemSymbolFilter =
      index::IndexingOptions::SystemSymbolFilterKind::All;
  IndexOpts.IndexFunctionLocals = true;

  // Macro occurences are not currently handled.
  indexTopLevelDecls(AST.getASTContext(), AST.getTopLevelDecls(),
                     DeclMacrosFinder, IndexOpts);

  std::vector<const Decl *> SelectedDecls = DeclMacrosFinder->takeDecls();

  auto DocHighlightsFinder = std::make_shared<DocumentHighlightsFinder>(
      llvm::errs(), AST.getASTContext(), AST.getPreprocessor(), SelectedDecls);

  indexTopLevelDecls(AST.getASTContext(), AST.getTopLevelDecls(),
                     DocHighlightsFinder, IndexOpts);

  return DocHighlightsFinder->takeHighlights();
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

/// Generate a \p Hover object given the macro \p MacroInf.
static Hover getHoverContents(StringRef MacroName) {
  Hover H;

  H.contents.value = "#define ";
  H.contents.value += MacroName;

  return H;
}

Hover getHover(ParsedAST &AST, Position Pos) {
  const SourceManager &SourceMgr = AST.getASTContext().getSourceManager();
  const FileEntry *FE = SourceMgr.getFileEntryForID(SourceMgr.getMainFileID());
  if (FE == nullptr)
    return Hover();

  SourceLocation SourceLocationBeg = getBeginningOfIdentifier(AST, Pos, FE);
  auto DeclMacrosFinder = std::make_shared<DeclarationAndMacrosFinder>(
      llvm::errs(), SourceLocationBeg, AST.getASTContext(),
      AST.getPreprocessor());

  index::IndexingOptions IndexOpts;
  IndexOpts.SystemSymbolFilter =
      index::IndexingOptions::SystemSymbolFilterKind::All;
  IndexOpts.IndexFunctionLocals = true;

  indexTopLevelDecls(AST.getASTContext(), AST.getTopLevelDecls(),
                     DeclMacrosFinder, IndexOpts);

  std::vector<MacroDecl> Macros = DeclMacrosFinder->takeMacroInfos();
  if (!Macros.empty())
    return getHoverContents(Macros[0].Name);

  std::vector<const Decl *> Decls = DeclMacrosFinder->takeDecls();
  if (!Decls.empty())
    return getHoverContents(Decls[0]);

  return Hover();
}

} // namespace clangd
} // namespace clang
