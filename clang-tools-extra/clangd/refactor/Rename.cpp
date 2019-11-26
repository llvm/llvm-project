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
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"

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
  Req.IDs.insert(*getSymbolID(&D));
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
  UsedOutsideFile, // for within-file rename only.
  UnsupportedSymbol,
  AmbiguousSymbol,
};

llvm::Optional<ReasonToReject> renameable(const Decl &RenameDecl,
                                          StringRef MainFilePath,
                                          const SymbolIndex *Index,
                                          bool CrossFile) {
  // Filter out symbols that are unsupported in both rename modes.
  if (llvm::isa<NamespaceDecl>(&RenameDecl))
    return ReasonToReject::UnsupportedSymbol;
  if (const auto *FD = llvm::dyn_cast<FunctionDecl>(&RenameDecl)) {
    if (FD->isOverloadedOperator())
      return ReasonToReject::UnsupportedSymbol;
  }
  // function-local symbols is safe to rename.
  if (RenameDecl.getParentFunctionOrMethod())
    return None;

  bool IsIndexable =
      isa<NamedDecl>(RenameDecl) &&
      SymbolCollector::shouldCollectSymbol(
          cast<NamedDecl>(RenameDecl), RenameDecl.getASTContext(),
          SymbolCollector::Options(), CrossFile);
  if (!IsIndexable) // If the symbol is not indexable, we disallow rename.
    return ReasonToReject::NonIndexable;

  if (!CrossFile) {
    auto &ASTCtx = RenameDecl.getASTContext();
    const auto &SM = ASTCtx.getSourceManager();
    bool MainFileIsHeader = isHeaderFile(MainFilePath, ASTCtx.getLangOpts());
    bool DeclaredInMainFile = isInsideMainFile(RenameDecl.getBeginLoc(), SM);

    if (!DeclaredInMainFile)
      // We are sure the symbol is used externally, bail out early.
      return ReasonToReject::UsedOutsideFile;

    // If the symbol is declared in the main file (which is not a header), we
    // rename it.
    if (!MainFileIsHeader)
      return None;

    if (!Index)
      return ReasonToReject::NoIndexProvided;

    auto OtherFile = getOtherRefFile(RenameDecl, MainFilePath, *Index);
    // If the symbol is indexable and has no refs from other files in the index,
    // we rename it.
    if (!OtherFile)
      return None;
    // If the symbol is indexable and has refs from other files in the index,
    // we disallow rename.
    return ReasonToReject::UsedOutsideFile;
  }

  assert(CrossFile);
  if (!Index)
    return ReasonToReject::NoIndexProvided;

  // Blacklist symbols that are not supported yet in cross-file mode due to the
  // limitations of our index.
  // FIXME: renaming templates requries to rename all related specializations,
  //        our index doesn't have this information.
  if (RenameDecl.getDescribedTemplate())
    return ReasonToReject::UnsupportedSymbol;

  // FIXME: renaming virtual methods requires to rename all overridens in
  //        subclasses, our index doesn't have this information.
  // Note: within-file rename does support this through the AST.
  if (const auto *S = llvm::dyn_cast<CXXMethodDecl>(&RenameDecl)) {
    if (S->isVirtual())
      return ReasonToReject::UnsupportedSymbol;
  }
  return None;
}

llvm::Error makeError(ReasonToReject Reason) {
  auto Message = [](ReasonToReject Reason) {
    switch (Reason) {
    case ReasonToReject::NoSymbolFound:
      return "there is no symbol at the given location";
    case ReasonToReject::NoIndexProvided:
      return "symbol may be used in other files (no index available)";
    case ReasonToReject::UsedOutsideFile:
      return "the symbol is used outside main file";
    case ReasonToReject::NonIndexable:
      return "symbol may be used in other files (not eligible for indexing)";
    case ReasonToReject::UnsupportedSymbol:
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

// AST-based rename, it renames all occurrences in the main file.
llvm::Expected<tooling::Replacements>
renameWithinFile(ParsedAST &AST, const NamedDecl &RenameDecl,
                 llvm::StringRef NewName) {
  const SourceManager &SM = AST.getSourceManager();

  tooling::Replacements FilteredChanges;
  for (SourceLocation Loc : findOccurrencesWithinFile(AST, RenameDecl)) {
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

Range toRange(const SymbolLocation &L) {
  Range R;
  R.start.line = L.Start.line();
  R.start.character = L.Start.column();
  R.end.line = L.End.line();
  R.end.character = L.End.column();
  return R;
};

// Return all rename occurrences (per the index) outside of the main file,
// grouped by the absolute file path.
llvm::StringMap<std::vector<Range>>
findOccurrencesOutsideFile(const NamedDecl &RenameDecl,
                           llvm::StringRef MainFile, const SymbolIndex &Index) {
  RefsRequest RQuest;
  RQuest.IDs.insert(*getSymbolID(&RenameDecl));

  // Absolute file path => rename ocurrences in that file.
  llvm::StringMap<std::vector<Range>> AffectedFiles;
  Index.refs(RQuest, [&](const Ref &R) {
    if (auto RefFilePath = filePath(R.Location, /*HintFilePath=*/MainFile)) {
      if (*RefFilePath != MainFile)
        AffectedFiles[*RefFilePath].push_back(toRange(R.Location));
    }
  });
  return AffectedFiles;
}

// Index-based rename, it renames all occurrences outside of the main file.
//
// The cross-file rename is purely based on the index, as we don't want to
// build all ASTs for affected files, which may cause a performance hit.
// We choose to trade off some correctness for performance and scalability.
//
// Clangd builds a dynamic index for all opened files on top of the static
// index of the whole codebase. Dynamic index is up-to-date (respects dirty
// buffers) as long as clangd finishes processing opened files, while static
// index (background index) is relatively stale. We choose the dirty buffers
// as the file content we rename on, and fallback to file content on disk if
// there is no dirty buffer.
//
// FIXME: add range patching heuristics to detect staleness of the index, and
//        report to users.
// FIXME: our index may return implicit references, which are non-eligitble
//        for rename, we should filter out these references.
llvm::Expected<FileEdits> renameOutsideFile(
    const NamedDecl &RenameDecl, llvm::StringRef MainFilePath,
    llvm::StringRef NewName, const SymbolIndex &Index,
    llvm::function_ref<llvm::Expected<std::string>(PathRef)> GetFileContent) {
  auto AffectedFiles =
      findOccurrencesOutsideFile(RenameDecl, MainFilePath, Index);
  // FIXME: make the limit customizable.
  static constexpr size_t MaxLimitFiles = 50;
  if (AffectedFiles.size() >= MaxLimitFiles)
    return llvm::make_error<llvm::StringError>(
        llvm::formatv(
            "The number of affected files exceeds the max limit {0}: {1}",
            MaxLimitFiles, AffectedFiles.size()),
        llvm::inconvertibleErrorCode());

  FileEdits Results;
  for (auto &FileAndOccurrences : AffectedFiles) {
    llvm::StringRef FilePath = FileAndOccurrences.first();

    auto AffectedFileCode = GetFileContent(FilePath);
    if (!AffectedFileCode) {
      elog("Fail to read file content: {0}", AffectedFileCode.takeError());
      continue;
    }
    auto RenameEdit = buildRenameEdit(
        *AffectedFileCode, std::move(FileAndOccurrences.second), NewName);
    if (!RenameEdit) {
      return llvm::make_error<llvm::StringError>(
          llvm::formatv("fail to build rename edit for file {0}: {1}", FilePath,
                        llvm::toString(RenameEdit.takeError())),
          llvm::inconvertibleErrorCode());
    }
    if (!RenameEdit->Replacements.empty())
      Results.insert({FilePath, std::move(*RenameEdit)});
  }
  return Results;
}

} // namespace

llvm::Expected<FileEdits> rename(const RenameInputs &RInputs) {
  ParsedAST &AST = RInputs.AST;
  const SourceManager &SM = AST.getSourceManager();
  llvm::StringRef MainFileCode = SM.getBufferData(SM.getMainFileID());
  auto GetFileContent = [&RInputs,
                         &SM](PathRef AbsPath) -> llvm::Expected<std::string> {
    llvm::Optional<std::string> DirtyBuffer;
    if (RInputs.GetDirtyBuffer &&
        (DirtyBuffer = RInputs.GetDirtyBuffer(AbsPath)))
      return std::move(*DirtyBuffer);

    auto Content =
        SM.getFileManager().getVirtualFileSystem().getBufferForFile(AbsPath);
    if (!Content)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          llvm::formatv("Fail to open file {0}: {1}", AbsPath,
                        Content.getError().message()));
    if (!*Content)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          llvm::formatv("Got no buffer for file {0}", AbsPath));

    return (*Content)->getBuffer().str();
  };
  SourceLocation SourceLocationBeg =
      SM.getMacroArgExpandedLocation(getBeginningOfIdentifier(
          RInputs.Pos, SM, AST.getASTContext().getLangOpts()));
  // FIXME: renaming macros is not supported yet, the macro-handling code should
  // be moved to rename tooling library.
  if (locateMacroAt(SourceLocationBeg, AST.getPreprocessor()))
    return makeError(ReasonToReject::UnsupportedSymbol);

  auto DeclsUnderCursor = locateDeclAt(AST, SourceLocationBeg);
  if (DeclsUnderCursor.empty())
    return makeError(ReasonToReject::NoSymbolFound);
  if (DeclsUnderCursor.size() > 1)
    return makeError(ReasonToReject::AmbiguousSymbol);

  const auto *RenameDecl = llvm::dyn_cast<NamedDecl>(*DeclsUnderCursor.begin());
  if (!RenameDecl)
    return makeError(ReasonToReject::UnsupportedSymbol);

  auto Reject =
      renameable(*RenameDecl->getCanonicalDecl(), RInputs.MainFilePath,
                 RInputs.Index, RInputs.AllowCrossFile);
  if (Reject)
    return makeError(*Reject);

  // We have two implemenations of the rename:
  //   - AST-based rename: used for renaming local symbols, e.g. variables
  //     defined in a function body;
  //   - index-based rename: used for renaming non-local symbols, and not
  //     feasible for local symbols (as by design our index don't index these
  //     symbols by design;
  // To make cross-file rename work for local symbol, we use a hybrid solution:
  //   - run AST-based rename on the main file;
  //   - run index-based rename on other affected files;
  auto MainFileRenameEdit = renameWithinFile(AST, *RenameDecl, RInputs.NewName);
  if (!MainFileRenameEdit)
    return MainFileRenameEdit.takeError();

  if (!RInputs.AllowCrossFile) {
    // within-file rename, just return the main file results.
    return FileEdits(
        {std::make_pair(RInputs.MainFilePath,
                        Edit{MainFileCode, std::move(*MainFileRenameEdit)})});
  }

  FileEdits Results;
  // renameable safely guards us that at this point we are renaming a local
  // symbol if we don't have index,
  if (RInputs.Index) {
    auto OtherFilesEdits =
        renameOutsideFile(*RenameDecl, RInputs.MainFilePath, RInputs.NewName,
                          *RInputs.Index, GetFileContent);
    if (!OtherFilesEdits)
      return OtherFilesEdits.takeError();
    Results = std::move(*OtherFilesEdits);
  }
  // Attach the rename edits for the main file.
  Results.try_emplace(RInputs.MainFilePath, MainFileCode,
                      std::move(*MainFileRenameEdit));
  return Results;
}

llvm::Expected<Edit> buildRenameEdit(llvm::StringRef InitialCode,
                                     std::vector<Range> Occurrences,
                                     llvm::StringRef NewName) {
  llvm::sort(Occurrences);
  // These two always correspond to the same position.
  Position LastPos{0, 0};
  size_t LastOffset = 0;

  auto Offset = [&](const Position &P) -> llvm::Expected<size_t> {
    assert(LastPos <= P && "malformed input");
    Position Shifted = {
        P.line - LastPos.line,
        P.line > LastPos.line ? P.character : P.character - LastPos.character};
    auto ShiftedOffset =
        positionToOffset(InitialCode.substr(LastOffset), Shifted);
    if (!ShiftedOffset)
      return llvm::make_error<llvm::StringError>(
          llvm::formatv("fail to convert the position {0} to offset ({1})", P,
                        llvm::toString(ShiftedOffset.takeError())),
          llvm::inconvertibleErrorCode());
    LastPos = P;
    LastOffset += *ShiftedOffset;
    return LastOffset;
  };

  std::vector<std::pair</*start*/ size_t, /*end*/ size_t>> OccurrencesOffsets;
  for (const auto &R : Occurrences) {
    auto StartOffset = Offset(R.start);
    if (!StartOffset)
      return StartOffset.takeError();
    auto EndOffset = Offset(R.end);
    if (!EndOffset)
      return EndOffset.takeError();
    OccurrencesOffsets.push_back({*StartOffset, *EndOffset});
  }

  tooling::Replacements RenameEdit;
  for (const auto &R : OccurrencesOffsets) {
    auto ByteLength = R.second - R.first;
    if (auto Err = RenameEdit.add(
            tooling::Replacement(InitialCode, R.first, ByteLength, NewName)))
      return std::move(Err);
  }
  return Edit(InitialCode, std::move(RenameEdit));
}

} // namespace clangd
} // namespace clang
