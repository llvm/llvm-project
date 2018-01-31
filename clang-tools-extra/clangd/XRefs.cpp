//===--- XRefs.cpp ----------------------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//
#include "XRefs.h"
#include "Logger.h"
#include "URI.h"
#include "clang/Index/IndexDataConsumer.h"
#include "clang/Index/IndexingAction.h"
namespace clang {
namespace clangd {
using namespace llvm;
namespace {

// Get the definition from a given declaration `D`.
// Return nullptr if no definition is found, or the declaration type of `D` is
// not supported.
const Decl* GetDefinition(const Decl* D) {
  assert(D);
  if (const auto *TD = dyn_cast<TagDecl>(D))
    return TD->getDefinition();
  else if (const auto *VD = dyn_cast<VarDecl>(D))
    return VD->getDefinition();
  else if (const auto *FD = dyn_cast<FunctionDecl>(D))
    return FD->getDefinition();
  return nullptr;
}

/// Finds declarations locations that a given source location refers to.
class DeclarationAndMacrosFinder : public index::IndexDataConsumer {
  std::vector<const Decl *> Decls;
  std::vector<const MacroInfo *> MacroInfos;
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

  std::vector<const MacroInfo *> takeMacroInfos() {
    // Don't keep the same Macro info multiple times.
    std::sort(MacroInfos.begin(), MacroInfos.end());
    auto Last = std::unique(MacroInfos.begin(), MacroInfos.end());
    MacroInfos.erase(Last, MacroInfos.end());
    return std::move(MacroInfos);
  }

  bool
  handleDeclOccurence(const Decl *D, index::SymbolRoleSet Roles,
                      ArrayRef<index::SymbolRelation> Relations, FileID FID,
                      unsigned Offset,
                      index::IndexDataConsumer::ASTNodeInfo ASTNode) override {
    if (isSearchedLocation(FID, Offset)) {
      // Find and add definition declarations (for GoToDefinition).
      // We don't use parameter `D`, as Parameter `D` is the canonical
      // declaration, which is the first declaration of a redeclarable
      // declaration, and it could be a forward declaration.
      if (const auto* Def = GetDefinition(D)) {
        Decls.push_back(Def);
      } else {
        // Couldn't find a definition, fall back to use `D`.
        Decls.push_back(D);
      }
    }
    return true;
  }

private:
  bool isSearchedLocation(FileID FID, unsigned Offset) const {
    const SourceManager &SourceMgr = AST.getSourceManager();
    return SourceMgr.getFileOffset(SearchedLocation) == Offset &&
           SourceMgr.getFileID(SearchedLocation) == FID;
  }

  void finish() override {
    // Also handle possible macro at the searched location.
    Token Result;
    auto &Mgr = AST.getSourceManager();
    if (!Lexer::getRawToken(SearchedLocation, Result, Mgr, AST.getLangOpts(),
                            false)) {
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
          MacroInfos.push_back(MacroInf);
        }
      }
    }
  }
};

llvm::Optional<Location>
getDeclarationLocation(ParsedAST &AST, const SourceRange &ValSourceRange) {
  const SourceManager &SourceMgr = AST.getASTContext().getSourceManager();
  const LangOptions &LangOpts = AST.getASTContext().getLangOpts();
  SourceLocation LocStart = ValSourceRange.getBegin();

  const FileEntry *F =
      SourceMgr.getFileEntryForID(SourceMgr.getFileID(LocStart));
  if (!F)
    return llvm::None;
  SourceLocation LocEnd = Lexer::getLocForEndOfToken(ValSourceRange.getEnd(), 0,
                                                     SourceMgr, LangOpts);
  Position Begin;
  Begin.line = SourceMgr.getSpellingLineNumber(LocStart) - 1;
  Begin.character = SourceMgr.getSpellingColumnNumber(LocStart) - 1;
  Position End;
  End.line = SourceMgr.getSpellingLineNumber(LocEnd) - 1;
  End.character = SourceMgr.getSpellingColumnNumber(LocEnd) - 1;
  Range R = {Begin, End};
  Location L;

  StringRef FilePath = F->tryGetRealPathName();
  if (FilePath.empty())
    FilePath = F->getName();
  L.uri.file = FilePath;
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
  std::vector<const MacroInfo *> MacroInfos =
      DeclMacrosFinder->takeMacroInfos();
  std::vector<Location> Result;

  for (auto Item : Decls) {
    auto L = getDeclarationLocation(AST, Item->getSourceRange());
    if (L)
      Result.push_back(*L);
  }

  for (auto Item : MacroInfos) {
    SourceRange SR(Item->getDefinitionLoc(), Item->getDefinitionEndLoc());
    auto L = getDeclarationLocation(AST, SR);
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
                      ArrayRef<index::SymbolRelation> Relations, FileID FID,
                      unsigned Offset,
                      index::IndexDataConsumer::ASTNodeInfo ASTNode) override {
    const SourceManager &SourceMgr = AST.getSourceManager();
    if (SourceMgr.getMainFileID() != FID ||
        std::find(Decls.begin(), Decls.end(), D) == Decls.end()) {
      return true;
    }
    SourceLocation End;
    const LangOptions &LangOpts = AST.getLangOpts();
    SourceLocation StartOfFileLoc = SourceMgr.getLocForStartOfFile(FID);
    SourceLocation HightlightStartLoc = StartOfFileLoc.getLocWithOffset(Offset);
    End =
        Lexer::getLocForEndOfToken(HightlightStartLoc, 0, SourceMgr, LangOpts);
    SourceRange SR(HightlightStartLoc, End);

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
    SourceLocation LocStart = SR.getBegin();
    Position Begin;
    Begin.line = SourceMgr.getSpellingLineNumber(LocStart) - 1;
    Begin.character = SourceMgr.getSpellingColumnNumber(LocStart) - 1;
    Position End;
    End.line = SourceMgr.getSpellingLineNumber(SR.getEnd()) - 1;
    End.character = SourceMgr.getSpellingColumnNumber(SR.getEnd()) - 1;
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

} // namespace clangd
} // namespace clang
