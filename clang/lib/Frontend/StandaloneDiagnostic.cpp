//===--- StandaloneDiagnostic.h - Serializable Diagnostic ------------- ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/StandaloneDiagnostic.h"
#include "clang/Lex/Lexer.h"

namespace clang {

StandaloneDiagnostic::SourceOffsetRange::SourceOffsetRange(
    CharSourceRange Range, const SourceManager &SrcMgr,
    const LangOptions &LangOpts) {
  const auto FileRange = Lexer::makeFileCharRange(Range, SrcMgr, LangOpts);
  Begin = SrcMgr.getFileOffset(FileRange.getBegin());
  End = SrcMgr.getFileOffset(FileRange.getEnd());
}

StandaloneDiagnostic::StandaloneFixIt::StandaloneFixIt(
    const SourceManager &SrcMgr, const LangOptions &LangOpts,
    const FixItHint &FixIt)
    : RemoveRange(FixIt.RemoveRange, SrcMgr, LangOpts),
      InsertFromRange(FixIt.InsertFromRange, SrcMgr, LangOpts),
      CodeToInsert(FixIt.CodeToInsert),
      BeforePreviousInsertions(FixIt.BeforePreviousInsertions) {}

StandaloneDiagnostic::StandaloneDiagnostic(const LangOptions &LangOpts,
                                           const StoredDiagnostic &InDiag)
    : Level(InDiag.getLevel()), ID(InDiag.getID()),
      Message(InDiag.getMessage()) {
  const FullSourceLoc &FullLoc = InDiag.getLocation();
  // This is not an invalid diagnostic; invalid SourceLocations are used to
  // represent diagnostics without a specific SourceLocation.
  if (FullLoc.isInvalid())
    return;

  const auto &SrcMgr = FullLoc.getManager();
  FileKind = SrcMgr.getFileCharacteristic(static_cast<SourceLocation>(FullLoc));
  const auto FileLoc = SrcMgr.getFileLoc(static_cast<SourceLocation>(FullLoc));
  FileOffset = SrcMgr.getFileOffset(FileLoc);
  Filename = SrcMgr.getFilename(FileLoc);
  assert(!Filename.empty() && "diagnostic with location has no source file?");

  Ranges.reserve(InDiag.getRanges().size());
  for (const auto &Range : InDiag.getRanges())
    Ranges.emplace_back(Range, SrcMgr, LangOpts);

  FixIts.reserve(InDiag.getFixIts().size());
  for (const auto &FixIt : InDiag.getFixIts())
    FixIts.emplace_back(SrcMgr, LangOpts, FixIt);
}

StoredDiagnostic
translateStandaloneDiag(FileManager &FileMgr, SourceManager &SrcMgr,
                        StandaloneDiagnostic StandaloneDiag,
                        llvm::StringMap<SourceLocation> &SrcLocCache) {
  const auto FileRef = FileMgr.getOptionalFileRef(StandaloneDiag.Filename);
  if (!FileRef)
    return StoredDiagnostic(StandaloneDiag.Level, StandaloneDiag.ID,
                            StandaloneDiag.Message);

  // Try to get FileLoc from cache first
  SourceLocation FileLoc;
  auto It = SrcLocCache.find(StandaloneDiag.Filename);
  if (It != SrcLocCache.end()) {
    FileLoc = It->getValue();
  }

  // Cache miss - compute and cache the location
  if (FileLoc.isInvalid()) {
    const auto FileID =
        SrcMgr.getOrCreateFileID(*FileRef, StandaloneDiag.FileKind);
    FileLoc = SrcMgr.getLocForStartOfFile(FileID);

    if (FileLoc.isInvalid())
      return StoredDiagnostic(StandaloneDiag.Level, StandaloneDiag.ID,
                              std::move(StandaloneDiag.Message));

    SrcLocCache[StandaloneDiag.Filename] = FileLoc;
  }

  const auto DiagLoc = FileLoc.getLocWithOffset(StandaloneDiag.FileOffset);
  const FullSourceLoc Loc(DiagLoc, SrcMgr);

  auto ConvertOffsetRange =
      [&](const StandaloneDiagnostic::SourceOffsetRange &Range) {
        return CharSourceRange(
            SourceRange(FileLoc.getLocWithOffset(Range.Begin),
                        FileLoc.getLocWithOffset(Range.End)),
            /*IsTokenRange*/ false);
      };

  SmallVector<CharSourceRange, 4> TranslatedRanges;
  TranslatedRanges.reserve(StandaloneDiag.Ranges.size());
  transform(StandaloneDiag.Ranges, std::back_inserter(TranslatedRanges),
            ConvertOffsetRange);

  SmallVector<FixItHint, 2> TranslatedFixIts;
  TranslatedFixIts.reserve(StandaloneDiag.FixIts.size());
  for (auto &FixIt : StandaloneDiag.FixIts) {
    FixItHint TranslatedFixIt;
    TranslatedFixIt.CodeToInsert = std::move(FixIt.CodeToInsert);
    TranslatedFixIt.RemoveRange = ConvertOffsetRange(FixIt.RemoveRange);
    TranslatedFixIt.InsertFromRange = ConvertOffsetRange(FixIt.InsertFromRange);
    TranslatedFixIt.BeforePreviousInsertions = FixIt.BeforePreviousInsertions;
    TranslatedFixIts.push_back(std::move(TranslatedFixIt));
  }

  return StoredDiagnostic(StandaloneDiag.Level, StandaloneDiag.ID,
                          std::move(StandaloneDiag.Message), Loc,
                          TranslatedRanges, TranslatedFixIts);
}

} // namespace clang
