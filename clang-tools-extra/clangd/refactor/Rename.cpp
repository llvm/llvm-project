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
#include "clang/Tooling/Syntax/Tokens.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include <algorithm>

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

  llvm::DenseSet<const Decl *> Result;
  for (const auto *D :
       targetDecl(SelectedNode->ASTNode,
                  DeclRelation::Alias | DeclRelation::TemplatePattern)) {
    const auto *ND = llvm::dyn_cast<NamedDecl>(D);
    if (!ND)
      continue;
    // Get to CXXRecordDecl from constructor or destructor.
    ND = tooling::getCanonicalSymbolDeclaration(ND);
    Result.insert(ND);
  }
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

  // Check whether the symbol being rename is indexable.
  auto &ASTCtx = RenameDecl.getASTContext();
  bool MainFileIsHeader = isHeaderFile(MainFilePath, ASTCtx.getLangOpts());
  bool DeclaredInMainFile =
      isInsideMainFile(RenameDecl.getBeginLoc(), ASTCtx.getSourceManager());
  bool IsMainFileOnly = true;
  if (MainFileIsHeader)
    // main file is a header, the symbol can't be main file only.
    IsMainFileOnly = false;
  else if (!DeclaredInMainFile)
    IsMainFileOnly = false;
  bool IsIndexable =
      isa<NamedDecl>(RenameDecl) &&
      SymbolCollector::shouldCollectSymbol(
          cast<NamedDecl>(RenameDecl), RenameDecl.getASTContext(),
          SymbolCollector::Options(), IsMainFileOnly);
  if (!IsIndexable) // If the symbol is not indexable, we disallow rename.
    return ReasonToReject::NonIndexable;

  if (!CrossFile) {
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
  // FIXME: Renaming templates requires to rename all related specializations,
  // our index doesn't have this information.
  if (RenameDecl.getDescribedTemplate())
    return ReasonToReject::UnsupportedSymbol;

  // FIXME: Renaming virtual methods requires to rename all overridens in
  // subclasses, our index doesn't have this information.
  // Note: Within-file rename does support this through the AST.
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
      return "no index provided";
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
  // If the cursor is at the underlying CXXRecordDecl of the
  // ClassTemplateDecl, ND will be the CXXRecordDecl. In this case, we need to
  // get the primary template maunally.
  // getUSRsForDeclaration will find other related symbols, e.g. virtual and its
  // overriddens, primary template and all explicit specializations.
  // FIXME: Get rid of the remaining tooling APIs.
  const auto RenameDecl =
      ND.getDescribedTemplate() ? ND.getDescribedTemplate() : &ND;
  std::vector<std::string> RenameUSRs =
      tooling::getUSRsForDeclaration(RenameDecl, AST.getASTContext());
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
}

// Return all rename occurrences (using the index) outside of the main file,
// grouped by the absolute file path.
llvm::Expected<llvm::StringMap<std::vector<Range>>>
findOccurrencesOutsideFile(const NamedDecl &RenameDecl,
                           llvm::StringRef MainFile, const SymbolIndex &Index) {
  RefsRequest RQuest;
  RQuest.IDs.insert(*getSymbolID(&RenameDecl));

  // Absolute file path => rename occurrences in that file.
  llvm::StringMap<std::vector<Range>> AffectedFiles;
  // FIXME: Make the limit customizable.
  static constexpr size_t MaxLimitFiles = 50;
  bool HasMore = Index.refs(RQuest, [&](const Ref &R) {
    if (AffectedFiles.size() > MaxLimitFiles)
      return;
    if (auto RefFilePath = filePath(R.Location, /*HintFilePath=*/MainFile)) {
      if (*RefFilePath != MainFile)
        AffectedFiles[*RefFilePath].push_back(toRange(R.Location));
    }
  });

  if (AffectedFiles.size() > MaxLimitFiles)
    return llvm::make_error<llvm::StringError>(
        llvm::formatv("The number of affected files exceeds the max limit {0}",
                      MaxLimitFiles),
        llvm::inconvertibleErrorCode());
  if (HasMore) {
    return llvm::make_error<llvm::StringError>(
        llvm::formatv("The symbol {0} has too many occurrences",
                      RenameDecl.getQualifiedNameAsString()),
        llvm::inconvertibleErrorCode());
  }
  // Sort and deduplicate the results, in case that index returns duplications.
  for (auto &FileAndOccurrences : AffectedFiles) {
    auto &Ranges = FileAndOccurrences.getValue();
    llvm::sort(Ranges);
    Ranges.erase(std::unique(Ranges.begin(), Ranges.end()), Ranges.end());
  }
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
// FIXME: Add range patching heuristics to detect staleness of the index, and
// report to users.
// FIXME: Our index may return implicit references, which are not eligible for
// rename, we should filter out these references.
llvm::Expected<FileEdits> renameOutsideFile(
    const NamedDecl &RenameDecl, llvm::StringRef MainFilePath,
    llvm::StringRef NewName, const SymbolIndex &Index,
    llvm::function_ref<llvm::Expected<std::string>(PathRef)> GetFileContent) {
  auto AffectedFiles =
      findOccurrencesOutsideFile(RenameDecl, MainFilePath, Index);
  if (!AffectedFiles)
    return AffectedFiles.takeError();
  FileEdits Results;
  for (auto &FileAndOccurrences : *AffectedFiles) {
    llvm::StringRef FilePath = FileAndOccurrences.first();

    auto AffectedFileCode = GetFileContent(FilePath);
    if (!AffectedFileCode) {
      elog("Fail to read file content: {0}", AffectedFileCode.takeError());
      continue;
    }
    auto RenameRanges =
        adjustRenameRanges(*AffectedFileCode, RenameDecl.getNameAsString(),
                           std::move(FileAndOccurrences.second),
                           RenameDecl.getASTContext().getLangOpts());
    if (!RenameRanges) {
      // Our heuristics fails to adjust rename ranges to the current state of
      // the file, it is most likely the index is stale, so we give up the
      // entire rename.
      return llvm::make_error<llvm::StringError>(
          llvm::formatv("Index results don't match the content of file {0} "
                        "(the index may be stale)",
                        FilePath),
          llvm::inconvertibleErrorCode());
    }
    auto RenameEdit =
        buildRenameEdit(FilePath, *AffectedFileCode, *RenameRanges, NewName);
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

// A simple edit is either changing line or column, but not both.
bool impliesSimpleEdit(const Position &LHS, const Position &RHS) {
  return LHS.line == RHS.line || LHS.character == RHS.character;
}

// Performs a DFS to enumerate all possible near-miss matches.
// It finds the locations where the indexed occurrences are now spelled in
// Lexed occurrences, a near miss is defined as:
//   - a near miss maps all of the **name** occurrences from the index onto a
//     *subset* of lexed occurrences (we allow a single name refers to more
//     than one symbol)
//   - all indexed occurrences must be mapped, and Result must be distinct and
//     preseve order (only support detecting simple edits to ensure a
//     robust mapping)
//   - each indexed -> lexed occurrences mapping correspondence may change the
//     *line* or *column*, but not both (increases chance of a robust mapping)
void findNearMiss(
    std::vector<size_t> &PartialMatch, ArrayRef<Range> IndexedRest,
    ArrayRef<Range> LexedRest, int LexedIndex, int &Fuel,
    llvm::function_ref<void(const std::vector<size_t> &)> MatchedCB) {
  if (--Fuel < 0)
    return;
  if (IndexedRest.size() > LexedRest.size())
    return;
  if (IndexedRest.empty()) {
    MatchedCB(PartialMatch);
    return;
  }
  if (impliesSimpleEdit(IndexedRest.front().start, LexedRest.front().start)) {
    PartialMatch.push_back(LexedIndex);
    findNearMiss(PartialMatch, IndexedRest.drop_front(), LexedRest.drop_front(),
                 LexedIndex + 1, Fuel, MatchedCB);
    PartialMatch.pop_back();
  }
  findNearMiss(PartialMatch, IndexedRest, LexedRest.drop_front(),
               LexedIndex + 1, Fuel, MatchedCB);
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
  // Try to find the tokens adjacent to the cursor position.
  auto Loc = sourceLocationInMainFile(SM, RInputs.Pos);
  if (!Loc)
    return Loc.takeError();
  const syntax::Token *IdentifierToken =
      spelledIdentifierTouching(*Loc, AST.getTokens());
  // Renames should only triggered on identifiers.
  if (!IdentifierToken)
    return makeError(ReasonToReject::NoSymbolFound);
  // FIXME: Renaming macros is not supported yet, the macro-handling code should
  // be moved to rename tooling library.
  if (locateMacroAt(IdentifierToken->location(), AST.getPreprocessor()))
    return makeError(ReasonToReject::UnsupportedSymbol);

  auto DeclsUnderCursor = locateDeclAt(AST, IdentifierToken->location());
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

  // We have two implementations of the rename:
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
    // Within-file rename: just return the main file results.
    return FileEdits(
        {std::make_pair(RInputs.MainFilePath,
                        Edit{MainFileCode, std::move(*MainFileRenameEdit)})});
  }

  FileEdits Results;
  // Renameable safely guards us that at this point we are renaming a local
  // symbol if we don't have index.
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

llvm::Expected<Edit> buildRenameEdit(llvm::StringRef AbsFilePath,
                                     llvm::StringRef InitialCode,
                                     std::vector<Range> Occurrences,
                                     llvm::StringRef NewName) {
  assert(std::is_sorted(Occurrences.begin(), Occurrences.end()));
  assert(std::unique(Occurrences.begin(), Occurrences.end()) ==
             Occurrences.end() &&
         "Occurrences must be unique");

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
            tooling::Replacement(AbsFilePath, R.first, ByteLength, NewName)))
      return std::move(Err);
  }
  return Edit(InitialCode, std::move(RenameEdit));
}

// Details:
//  - lex the draft code to get all rename candidates, this yields a superset
//    of candidates.
//  - apply range patching heuristics to generate "authoritative" occurrences,
//    cases we consider:
//      (a) index returns a subset of candidates, we use the indexed results.
//        - fully equal, we are sure the index is up-to-date
//        - proper subset, index is correct in most cases? there may be false
//          positives (e.g. candidates got appended), but rename is still safe
//      (b) index returns non-candidate results, we attempt to map the indexed
//          ranges onto candidates in a plausible way (e.g. guess that lines
//          were inserted). If such a "near miss" is found, the rename is still
//          possible
llvm::Optional<std::vector<Range>>
adjustRenameRanges(llvm::StringRef DraftCode, llvm::StringRef Identifier,
                   std::vector<Range> Indexed, const LangOptions &LangOpts) {
  assert(!Indexed.empty());
  assert(std::is_sorted(Indexed.begin(), Indexed.end()));
  std::vector<Range> Lexed =
      collectIdentifierRanges(Identifier, DraftCode, LangOpts);
  llvm::sort(Lexed);
  return getMappedRanges(Indexed, Lexed);
}

llvm::Optional<std::vector<Range>> getMappedRanges(ArrayRef<Range> Indexed,
                                                   ArrayRef<Range> Lexed) {
  assert(!Indexed.empty());
  assert(std::is_sorted(Indexed.begin(), Indexed.end()));
  assert(std::is_sorted(Lexed.begin(), Lexed.end()));

  if (Indexed.size() > Lexed.size()) {
    vlog("The number of lexed occurrences is less than indexed occurrences");
    return llvm::None;
  }
  // Fast check for the special subset case.
  if (std::includes(Indexed.begin(), Indexed.end(), Lexed.begin(), Lexed.end()))
    return Indexed.vec();

  std::vector<size_t> Best;
  size_t BestCost = std::numeric_limits<size_t>::max();
  bool HasMultiple = 0;
  std::vector<size_t> ResultStorage;
  int Fuel = 10000;
  findNearMiss(ResultStorage, Indexed, Lexed, 0, Fuel,
               [&](const std::vector<size_t> &Matched) {
                 size_t MCost =
                     renameRangeAdjustmentCost(Indexed, Lexed, Matched);
                 if (MCost < BestCost) {
                   BestCost = MCost;
                   Best = std::move(Matched);
                   HasMultiple = false; // reset
                   return;
                 }
                 if (MCost == BestCost)
                   HasMultiple = true;
               });
  if (HasMultiple) {
    vlog("The best near miss is not unique.");
    return llvm::None;
  }
  if (Best.empty()) {
    vlog("Didn't find a near miss.");
    return llvm::None;
  }
  std::vector<Range> Mapped;
  for (auto I : Best)
    Mapped.push_back(Lexed[I]);
  return Mapped;
}

// The cost is the sum of the implied edit sizes between successive diffs, only
// simple edits are considered:
//   - insert/remove a line (change line offset)
//   - insert/remove a character on an existing line (change column offset)
//
// Example I, total result is 1 + 1 = 2.
//   diff[0]: line + 1 <- insert a line before edit 0.
//   diff[1]: line + 1
//   diff[2]: line + 1
//   diff[3]: line + 2 <- insert a line before edits 2 and 3.
//
// Example II, total result is 1 + 1 + 1 = 3.
//   diff[0]: line + 1  <- insert a line before edit 0.
//   diff[1]: column + 1 <- remove a line between edits 0 and 1, and insert a
//   character on edit 1.
size_t renameRangeAdjustmentCost(ArrayRef<Range> Indexed, ArrayRef<Range> Lexed,
                                 ArrayRef<size_t> MappedIndex) {
  assert(Indexed.size() == MappedIndex.size());
  assert(std::is_sorted(Indexed.begin(), Indexed.end()));
  assert(std::is_sorted(Lexed.begin(), Lexed.end()));

  int LastLine = -1;
  int LastDLine = 0, LastDColumn = 0;
  int Cost = 0;
  for (size_t I = 0; I < Indexed.size(); ++I) {
    int DLine = Indexed[I].start.line - Lexed[MappedIndex[I]].start.line;
    int DColumn =
        Indexed[I].start.character - Lexed[MappedIndex[I]].start.character;
    int Line = Indexed[I].start.line;
    if (Line != LastLine)
      LastDColumn = 0; // colmun offsets don't carry cross lines.
    Cost += abs(DLine - LastDLine) + abs(DColumn - LastDColumn);
    std::tie(LastLine, LastDLine, LastDColumn) = std::tie(Line, DLine, DColumn);
  }
  return Cost;
}

} // namespace clangd
} // namespace clang
