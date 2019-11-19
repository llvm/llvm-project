//===--- Rename.cpp - Symbol-rename refactorings -----------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "refactor/Rename.h"
#include "AST.h"
#include "Logger.h"
#include "ParsedAST.h"
#include "SourceCode.h"
#include "index/SymbolCollector.h"
#include "clang/Tooling/Refactoring/Rename/RenamingAction.h"
#include "clang/Tooling/Refactoring/Rename/USRFinder.h"
#include "clang/Tooling/Refactoring/Rename/USRFindingAction.h"
#include "clang/Tooling/Refactoring/Rename/USRLocFinder.h"

namespace clang {
namespace clangd {
namespace {

llvm::Optional<std::string> filePath(const SymbolLocation &Loc,
                                     llvm::StringRef HintFilePath) {
  if (!Loc)
    return None;
  auto Path = URI::resolve(Loc.FileURI, HintFilePath);
  if (!Path) {
    elog("Could not resolve URI {0}: {1}", Loc.FileURI, Path.takeError());
    return None;
  }

  return *Path;
}

// Query the index to find some other files where the Decl is referenced.
llvm::Optional<std::string> getOtherRefFile(const Decl &D, StringRef MainFile,
                                            const SymbolIndex &Index) {
  RefsRequest Req;
  // We limit the number of results, this is a correctness/performance
  // tradeoff. We expect the number of symbol references in the current file
  // is smaller than the limit.
  Req.Limit = 100;
  if (auto ID = getSymbolID(&D))
    Req.IDs.insert(*ID);
  llvm::Optional<std::string> OtherFile;
  Index.refs(Req, [&](const Ref &R) {
    if (OtherFile)
      return;
    if (auto RefFilePath = filePath(R.Location, /*HintFilePath=*/MainFile)) {
      if (*RefFilePath != MainFile)
        OtherFile = *RefFilePath;
    }
  });
  return OtherFile;
}

enum ReasonToReject {
  NoSymbolFound,
  NoIndexProvided,
  NonIndexable,
  UsedOutsideFile,
  UnsupportedSymbol,
};

// Check the symbol Decl is renameable (per the index) within the file.
llvm::Optional<ReasonToReject> renamableWithinFile(const Decl &RenameDecl,
                                                   StringRef MainFile,
                                                   const SymbolIndex *Index) {
  if (llvm::isa<NamespaceDecl>(&RenameDecl))
    return ReasonToReject::UnsupportedSymbol;
  if (const auto *FD = llvm::dyn_cast<FunctionDecl>(&RenameDecl)) {
    if (FD->isOverloadedOperator())
      return ReasonToReject::UnsupportedSymbol;
  }
  auto &ASTCtx = RenameDecl.getASTContext();
  const auto &SM = ASTCtx.getSourceManager();
  bool MainFileIsHeader = isHeaderFile(MainFile, ASTCtx.getLangOpts());
  bool DeclaredInMainFile = isInsideMainFile(RenameDecl.getBeginLoc(), SM);

  if (!DeclaredInMainFile)
    // We are sure the symbol is used externally, bail out early.
    return UsedOutsideFile;

  // If the symbol is declared in the main file (which is not a header), we
  // rename it.
  if (!MainFileIsHeader)
    return None;

  // Below are cases where the symbol is declared in the header.
  // If the symbol is function-local, we rename it.
  if (RenameDecl.getParentFunctionOrMethod())
    return None;

  if (!Index)
    return ReasonToReject::NoIndexProvided;

  bool IsIndexable = isa<NamedDecl>(RenameDecl) &&
                     SymbolCollector::shouldCollectSymbol(
                         cast<NamedDecl>(RenameDecl), ASTCtx, {}, false);
  // If the symbol is not indexable, we disallow rename.
  if (!IsIndexable)
    return ReasonToReject::NonIndexable;
  auto OtherFile = getOtherRefFile(RenameDecl, MainFile, *Index);
  // If the symbol is indexable and has no refs from other files in the index,
  // we rename it.
  if (!OtherFile)
    return None;
  // If the symbol is indexable and has refs from other files in the index,
  // we disallow rename.
  return ReasonToReject::UsedOutsideFile;
}

llvm::Error makeError(ReasonToReject Reason) {
  auto Message = [](ReasonToReject Reason) {
    switch (Reason) {
    case NoSymbolFound:
      return "there is no symbol at the given location";
    case NoIndexProvided:
      return "symbol may be used in other files (no index available)";
    case UsedOutsideFile:
      return "the symbol is used outside main file";
    case NonIndexable:
      return "symbol may be used in other files (not eligible for indexing)";
    case UnsupportedSymbol:
      return "symbol is not a supported kind (e.g. namespace, macro)";
    }
    llvm_unreachable("unhandled reason kind");
  };
  return llvm::make_error<llvm::StringError>(
      llvm::formatv("Cannot rename symbol: {0}", Message(Reason)),
      llvm::inconvertibleErrorCode());
}

// Return all rename occurrences in the main file.
tooling::SymbolOccurrences
findOccurrencesWithinFile(ParsedAST &AST, const NamedDecl *RenameDecl) {
  const NamedDecl *CanonicalRenameDecl =
      tooling::getCanonicalSymbolDeclaration(RenameDecl);
  assert(CanonicalRenameDecl && "RenameDecl must be not null");
  std::vector<std::string> RenameUSRs =
      tooling::getUSRsForDeclaration(CanonicalRenameDecl, AST.getASTContext());
  std::string OldName = CanonicalRenameDecl->getNameAsString();
  tooling::SymbolOccurrences Result;
  for (Decl *TopLevelDecl : AST.getLocalTopLevelDecls()) {
    tooling::SymbolOccurrences RenameInDecl =
        tooling::getOccurrencesOfUSRs(RenameUSRs, OldName, TopLevelDecl);
    Result.insert(Result.end(), std::make_move_iterator(RenameInDecl.begin()),
                  std::make_move_iterator(RenameInDecl.end()));
  }
  return Result;
}

} // namespace

llvm::Expected<tooling::Replacements>
renameWithinFile(ParsedAST &AST, llvm::StringRef File, Position Pos,
                 llvm::StringRef NewName, const SymbolIndex *Index) {
  const SourceManager &SM = AST.getSourceManager();
  SourceLocation SourceLocationBeg = SM.getMacroArgExpandedLocation(
      getBeginningOfIdentifier(Pos, SM, AST.getASTContext().getLangOpts()));
  // FIXME: renaming macros is not supported yet, the macro-handling code should
  // be moved to rename tooling library.
  if (locateMacroAt(SourceLocationBeg, AST.getPreprocessor()))
    return makeError(UnsupportedSymbol);

  const auto *RenameDecl =
      tooling::getNamedDeclAt(AST.getASTContext(), SourceLocationBeg);
  if (!RenameDecl)
    return makeError(NoSymbolFound);

  if (auto Reject =
          renamableWithinFile(*RenameDecl->getCanonicalDecl(), File, Index))
    return makeError(*Reject);

  // Rename sometimes returns duplicate edits (which is a bug). A side-effect of
  // adding them to a single Replacements object is these are deduplicated.
  tooling::Replacements FilteredChanges;
  for (const tooling::SymbolOccurrence &Rename :
       findOccurrencesWithinFile(AST, RenameDecl)) {
    // Currently, we only support normal rename (one range) for C/C++.
    // FIXME: support multiple-range rename for objective-c methods.
    if (Rename.getNameRanges().size() > 1)
      continue;
    // We shouldn't have conflicting replacements. If there are conflicts, it
    // means that we have bugs either in clangd or in Rename library, therefore
    // we refuse to perform the rename.
    if (auto Err = FilteredChanges.add(tooling::Replacement(
            AST.getASTContext().getSourceManager(),
            CharSourceRange::getCharRange(Rename.getNameRanges()[0]), NewName)))
      return std::move(Err);
  }
  return FilteredChanges;
}

} // namespace clangd
} // namespace clang
