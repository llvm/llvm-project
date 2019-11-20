//===--- Rename.cpp - Symbol-rename refactorings -----------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "refactor/Rename.h"
#include "AST.h"
#include "FindTarget.h"
#include "Logger.h"
#include "ParsedAST.h"
#include "Selection.h"
#include "SourceCode.h"
#include "index/SymbolCollector.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Tooling/Refactoring/Rename/USRFindingAction.h"

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

// Returns true if the given location is expanded from any macro body.
bool isInMacroBody(const SourceManager &SM, SourceLocation Loc) {
  while (Loc.isMacroID()) {
    if (SM.isMacroBodyExpansion(Loc))
      return true;
    Loc = SM.getImmediateMacroCallerLoc(Loc);
  }

  return false;
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

llvm::DenseSet<const Decl *> locateDeclAt(ParsedAST &AST,
                                          SourceLocation TokenStartLoc) {
  unsigned Offset =
      AST.getSourceManager().getDecomposedSpellingLoc(TokenStartLoc).second;

  SelectionTree Selection(AST.getASTContext(), AST.getTokens(), Offset);
  const SelectionTree::Node *SelectedNode = Selection.commonAncestor();
  if (!SelectedNode)
    return {};

  // If the location points to a Decl, we check it is actually on the name
  // range of the Decl. This would avoid allowing rename on unrelated tokens.
  //   ^class Foo {} // SelectionTree returns CXXRecordDecl,
  //                 // we don't attempt to trigger rename on this position.
  // FIXME: make this work on destructors, e.g. "~F^oo()".
  if (const auto *D = SelectedNode->ASTNode.get<Decl>()) {
    if (D->getLocation() != TokenStartLoc)
      return {};
  }

  llvm::DenseSet<const Decl *> Result;
  for (const auto *D :
       targetDecl(SelectedNode->ASTNode,
                  DeclRelation::Alias | DeclRelation::TemplatePattern))
    Result.insert(D);
  return Result;
}

enum ReasonToReject {
  NoSymbolFound,
  NoIndexProvided,
  NonIndexable,
  UsedOutsideFile,
  UnsupportedSymbol,
  AmbiguousSymbol,
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
    case AmbiguousSymbol:
      return "there are multiple symbols at the given location";
    }
    llvm_unreachable("unhandled reason kind");
  };
  return llvm::make_error<llvm::StringError>(
      llvm::formatv("Cannot rename symbol: {0}", Message(Reason)),
      llvm::inconvertibleErrorCode());
}

// Return all rename occurrences in the main file.
std::vector<SourceLocation> findOccurrencesWithinFile(ParsedAST &AST,
                                                      const NamedDecl &ND) {
  // In theory, locateDeclAt should return the primary template. However, if the
  // cursor is under the underlying CXXRecordDecl of the ClassTemplateDecl, ND
  // will be the CXXRecordDecl, for this case, we need to get the primary
  // template maunally.
  const auto &RenameDecl =
      ND.getDescribedTemplate() ? *ND.getDescribedTemplate() : ND;
  // getUSRsForDeclaration will find other related symbols, e.g. virtual and its
  // overriddens, primary template and all explicit specializations.
  // FIXME: get rid of the remaining tooling APIs.
  std::vector<std::string> RenameUSRs = tooling::getUSRsForDeclaration(
      tooling::getCanonicalSymbolDeclaration(&RenameDecl), AST.getASTContext());
  llvm::DenseSet<SymbolID> TargetIDs;
  for (auto &USR : RenameUSRs)
    TargetIDs.insert(SymbolID(USR));

  std::vector<SourceLocation> Results;
  for (Decl *TopLevelDecl : AST.getLocalTopLevelDecls()) {
    findExplicitReferences(TopLevelDecl, [&](ReferenceLoc Ref) {
      if (Ref.Targets.empty())
        return;
      for (const auto *Target : Ref.Targets) {
        auto ID = getSymbolID(Target);
        if (!ID || TargetIDs.find(*ID) == TargetIDs.end())
          return;
      }
      Results.push_back(Ref.NameLoc);
    });
  }

  return Results;
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

  auto DeclsUnderCursor = locateDeclAt(AST, SourceLocationBeg);
  if (DeclsUnderCursor.empty())
    return makeError(NoSymbolFound);
  if (DeclsUnderCursor.size() > 1)
    return makeError(AmbiguousSymbol);

  const auto *RenameDecl = llvm::dyn_cast<NamedDecl>(*DeclsUnderCursor.begin());
  if (!RenameDecl)
    return makeError(UnsupportedSymbol);

  if (auto Reject =
          renamableWithinFile(*RenameDecl->getCanonicalDecl(), File, Index))
    return makeError(*Reject);

  tooling::Replacements FilteredChanges;
  for (SourceLocation Loc : findOccurrencesWithinFile(AST, *RenameDecl)) {
    SourceLocation RenameLoc = Loc;
    // We don't rename in any macro bodies, but we allow rename the symbol
    // spelled in a top-level macro argument in the main file.
    if (RenameLoc.isMacroID()) {
      if (isInMacroBody(SM, RenameLoc))
        continue;
      RenameLoc = SM.getSpellingLoc(Loc);
    }
    // Filter out locations not from main file.
    // We traverse only main file decls, but locations could come from an
    // non-preamble #include file e.g.
    //   void test() {
    //     int f^oo;
    //     #include "use_foo.inc"
    //   }
    if (!isInsideMainFile(RenameLoc, SM))
      continue;
    if (auto Err = FilteredChanges.add(tooling::Replacement(
            SM, CharSourceRange::getTokenRange(RenameLoc), NewName)))
      return std::move(Err);
  }
  return FilteredChanges;
}

} // namespace clangd
} // namespace clang
