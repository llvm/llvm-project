//===--- CoverageMappingGen.cpp - Coverage mapping generation ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Instrumentation-based code coverage mapping generator
//
//===----------------------------------------------------------------------===//

#include "CoverageMappingGen.h"
#include "CodeGenFunction.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/FileManager.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ProfileData/Coverage/CoverageMapping.h"
#include "llvm/ProfileData/Coverage/CoverageMappingReader.h"
#include "llvm/ProfileData/Coverage/CoverageMappingWriter.h"
#include "llvm/ProfileData/InstrProfReader.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

// This selects the coverage mapping format defined when `InstrProfData.inc`
// is textually included.
#define COVMAP_V3

static llvm::cl::opt<bool> EmptyLineCommentCoverage(
    "emptyline-comment-coverage",
    llvm::cl::desc("Emit emptylines and comment lines as skipped regions (only "
                   "disable it on test)"),
    llvm::cl::init(true), llvm::cl::Hidden);

using namespace clang;
using namespace CodeGen;
using namespace llvm::coverage;

CoverageSourceInfo *
CoverageMappingModuleGen::setUpCoverageCallbacks(Preprocessor &PP) {
  CoverageSourceInfo *CoverageInfo =
      new CoverageSourceInfo(PP.getSourceManager());
  PP.addPPCallbacks(std::unique_ptr<PPCallbacks>(CoverageInfo));
  if (EmptyLineCommentCoverage) {
    PP.addCommentHandler(CoverageInfo);
    PP.setEmptylineHandler(CoverageInfo);
    PP.setPreprocessToken(true);
    PP.setTokenWatcher([CoverageInfo](clang::Token Tok) {
      // Update previous token location.
      CoverageInfo->PrevTokLoc = Tok.getLocation();
      if (Tok.getKind() != clang::tok::eod)
        CoverageInfo->updateNextTokLoc(Tok.getLocation());
    });
  }
  return CoverageInfo;
}

void CoverageSourceInfo::AddSkippedRange(SourceRange Range) {
  if (EmptyLineCommentCoverage && !SkippedRanges.empty() &&
      PrevTokLoc == SkippedRanges.back().PrevTokLoc &&
      SourceMgr.isWrittenInSameFile(SkippedRanges.back().Range.getEnd(),
                                    Range.getBegin()))
    SkippedRanges.back().Range.setEnd(Range.getEnd());
  else
    SkippedRanges.push_back({Range, PrevTokLoc});
}

void CoverageSourceInfo::SourceRangeSkipped(SourceRange Range, SourceLocation) {
  AddSkippedRange(Range);
}

void CoverageSourceInfo::HandleEmptyline(SourceRange Range) {
  AddSkippedRange(Range);
}

bool CoverageSourceInfo::HandleComment(Preprocessor &PP, SourceRange Range) {
  AddSkippedRange(Range);
  return false;
}

void CoverageSourceInfo::updateNextTokLoc(SourceLocation Loc) {
  if (!SkippedRanges.empty() && SkippedRanges.back().NextTokLoc.isInvalid())
    SkippedRanges.back().NextTokLoc = Loc;
}

namespace {

/// A region of source code that can be mapped to a counter.
class SourceMappingRegion {
  /// Primary Counter that is also used for Branch Regions for "True" branches.
  Counter Count;

  /// Secondary Counter used for Branch Regions for "False" branches.
  Optional<Counter> FalseCount;

  /// The region's starting location.
  Optional<SourceLocation> LocStart;

  /// The region's ending location.
  Optional<SourceLocation> LocEnd;

  /// Whether this region is a gap region. The count from a gap region is set
  /// as the line execution count if there are no other regions on the line.
  bool GapRegion;

public:
  SourceMappingRegion(Counter Count, Optional<SourceLocation> LocStart,
                      Optional<SourceLocation> LocEnd, bool GapRegion = false)
      : Count(Count), LocStart(LocStart), LocEnd(LocEnd), GapRegion(GapRegion) {
  }

  SourceMappingRegion(Counter Count, Optional<Counter> FalseCount,
                      Optional<SourceLocation> LocStart,
                      Optional<SourceLocation> LocEnd, bool GapRegion = false)
      : Count(Count), FalseCount(FalseCount), LocStart(LocStart),
        LocEnd(LocEnd), GapRegion(GapRegion) {}

  const Counter &getCounter() const { return Count; }

  const Counter &getFalseCounter() const {
    assert(FalseCount && "Region has no alternate counter");
    return *FalseCount;
  }

  void setCounter(Counter C) { Count = C; }

  bool hasStartLoc() const { return LocStart.hasValue(); }

  void setStartLoc(SourceLocation Loc) { LocStart = Loc; }

  SourceLocation getBeginLoc() const {
    assert(LocStart && "Region has no start location");
    return *LocStart;
  }

  bool hasEndLoc() const { return LocEnd.hasValue(); }

  void setEndLoc(SourceLocation Loc) {
    assert(Loc.isValid() && "Setting an invalid end location");
    LocEnd = Loc;
  }

  SourceLocation getEndLoc() const {
    assert(LocEnd && "Region has no end location");
    return *LocEnd;
  }

  bool isGap() const { return GapRegion; }

  void setGap(bool Gap) { GapRegion = Gap; }

  bool isBranch() const { return FalseCount.hasValue(); }
};

/// Spelling locations for the start and end of a source region.
struct SpellingRegion {
  /// The line where the region starts.
  unsigned LineStart;

  /// The column where the region starts.
  unsigned ColumnStart;

  /// The line where the region ends.
  unsigned LineEnd;

  /// The column where the region ends.
  unsigned ColumnEnd;

  SpellingRegion(SourceManager &SM, SourceLocation LocStart,
                 SourceLocation LocEnd) {
    LineStart = SM.getSpellingLineNumber(LocStart);
    ColumnStart = SM.getSpellingColumnNumber(LocStart);
    LineEnd = SM.getSpellingLineNumber(LocEnd);
    ColumnEnd = SM.getSpellingColumnNumber(LocEnd);
  }

  SpellingRegion(SourceManager &SM, SourceMappingRegion &R)
      : SpellingRegion(SM, R.getBeginLoc(), R.getEndLoc()) {}

  /// Check if the start and end locations appear in source order, i.e
  /// top->bottom, left->right.
  bool isInSourceOrder() const {
    return (LineStart < LineEnd) ||
           (LineStart == LineEnd && ColumnStart <= ColumnEnd);
  }
};

/// Provides the common functionality for the different
/// coverage mapping region builders.
class CoverageMappingBuilder {
public:
  CoverageMappingModuleGen &CVM;
  SourceManager &SM;
  const LangOptions &LangOpts;

private:
  /// Map of clang's FileIDs to IDs used for coverage mapping.
  llvm::SmallDenseMap<FileID, std::pair<unsigned, SourceLocation>, 8>
      FileIDMapping;

public:
  /// The coverage mapping regions for this function
  llvm::SmallVector<CounterMappingRegion, 32> MappingRegions;
  /// The source mapping regions for this function.
  std::vector<SourceMappingRegion> SourceRegions;

  /// A set of regions which can be used as a filter.
  ///
  /// It is produced by emitExpansionRegions() and is used in
  /// emitSourceRegions() to suppress producing code regions if
  /// the same area is covered by expansion regions.
  typedef llvm::SmallSet<std::pair<SourceLocation, SourceLocation>, 8>
      SourceRegionFilter;

  CoverageMappingBuilder(CoverageMappingModuleGen &CVM, SourceManager &SM,
                         const LangOptions &LangOpts)
      : CVM(CVM), SM(SM), LangOpts(LangOpts) {}

  /// Return the precise end location for the given token.
  SourceLocation getPreciseTokenLocEnd(SourceLocation Loc) {
    // We avoid getLocForEndOfToken here, because it doesn't do what we want for
    // macro locations, which we just treat as expanded files.
    unsigned TokLen =
        Lexer::MeasureTokenLength(SM.getSpellingLoc(Loc), SM, LangOpts);
    return Loc.getLocWithOffset(TokLen);
  }

  /// Return the start location of an included file or expanded macro.
  SourceLocation getStartOfFileOrMacro(SourceLocation Loc) {
    if (Loc.isMacroID())
      return Loc.getLocWithOffset(-SM.getFileOffset(Loc));
    return SM.getLocForStartOfFile(SM.getFileID(Loc));
  }

  /// Return the end location of an included file or expanded macro.
  SourceLocation getEndOfFileOrMacro(SourceLocation Loc) {
    if (Loc.isMacroID())
      return Loc.getLocWithOffset(SM.getFileIDSize(SM.getFileID(Loc)) -
                                  SM.getFileOffset(Loc));
    return SM.getLocForEndOfFile(SM.getFileID(Loc));
  }

  /// Find out where the current file is included or macro is expanded.
  SourceLocation getIncludeOrExpansionLoc(SourceLocation Loc) {
    return Loc.isMacroID() ? SM.getImmediateExpansionRange(Loc).getBegin()
                           : SM.getIncludeLoc(SM.getFileID(Loc));
  }

  /// Return true if \c Loc is a location in a built-in macro.
  bool isInBuiltin(SourceLocation Loc) {
    return SM.getBufferName(SM.getSpellingLoc(Loc)) == "<built-in>";
  }

  /// Check whether \c Loc is included or expanded from \c Parent.
  bool isNestedIn(SourceLocation Loc, FileID Parent) {
    do {
      Loc = getIncludeOrExpansionLoc(Loc);
      if (Loc.isInvalid())
        return false;
    } while (!SM.isInFileID(Loc, Parent));
    return true;
  }

  /// Get the start of \c S ignoring macro arguments and builtin macros.
  SourceLocation getStart(const Stmt *S) {
    SourceLocation Loc = S->getBeginLoc();
    while (SM.isMacroArgExpansion(Loc) || isInBuiltin(Loc))
      Loc = SM.getImmediateExpansionRange(Loc).getBegin();
    return Loc;
  }

  /// Get the end of \c S ignoring macro arguments and builtin macros.
  SourceLocation getEnd(const Stmt *S) {
    SourceLocation Loc = S->getEndLoc();
    while (SM.isMacroArgExpansion(Loc) || isInBuiltin(Loc))
      Loc = SM.getImmediateExpansionRange(Loc).getBegin();
    return getPreciseTokenLocEnd(Loc);
  }

  /// Find the set of files we have regions for and assign IDs
  ///
  /// Fills \c Mapping with the virtual file mapping needed to write out
  /// coverage and collects the necessary file information to emit source and
  /// expansion regions.
  void gatherFileIDs(SmallVectorImpl<unsigned> &Mapping) {
    FileIDMapping.clear();

    llvm::SmallSet<FileID, 8> Visited;
    SmallVector<std::pair<SourceLocation, unsigned>, 8> FileLocs;
    for (const auto &Region : SourceRegions) {
      SourceLocation Loc = Region.getBeginLoc();
      FileID File = SM.getFileID(Loc);
      if (!Visited.insert(File).second)
        continue;

      // Do not map FileID's associated with system headers.
      if (SM.isInSystemHeader(SM.getSpellingLoc(Loc)))
        continue;

      unsigned Depth = 0;
      for (SourceLocation Parent = getIncludeOrExpansionLoc(Loc);
           Parent.isValid(); Parent = getIncludeOrExpansionLoc(Parent))
        ++Depth;
      FileLocs.push_back(std::make_pair(Loc, Depth));
    }
    llvm::stable_sort(FileLocs, llvm::less_second());

    for (const auto &FL : FileLocs) {
      SourceLocation Loc = FL.first;
      FileID SpellingFile = SM.getDecomposedSpellingLoc(Loc).first;
      auto Entry = SM.getFileEntryForID(SpellingFile);
      if (!Entry)
        continue;

      FileIDMapping[SM.getFileID(Loc)] = std::make_pair(Mapping.size(), Loc);
      Mapping.push_back(CVM.getFileID(Entry));
    }
  }

  /// Get the coverage mapping file ID for \c Loc.
  ///
  /// If such file id doesn't exist, return None.
  Optional<unsigned> getCoverageFileID(SourceLocation Loc) {
    auto Mapping = FileIDMapping.find(SM.getFileID(Loc));
    if (Mapping != FileIDMapping.end())
      return Mapping->second.first;
    return None;
  }

  /// This shrinks the skipped range if it spans a line that contains a
  /// non-comment token. If shrinking the skipped range would make it empty,
  /// this returns None.
  Optional<SpellingRegion> adjustSkippedRange(SourceManager &SM,
                                              SourceLocation LocStart,
                                              SourceLocation LocEnd,
                                              SourceLocation PrevTokLoc,
                                              SourceLocation NextTokLoc) {
    SpellingRegion SR{SM, LocStart, LocEnd};
    SR.ColumnStart = 1;
    if (PrevTokLoc.isValid() && SM.isWrittenInSameFile(LocStart, PrevTokLoc) &&
        SR.LineStart == SM.getSpellingLineNumber(PrevTokLoc))
      SR.LineStart++;
    if (NextTokLoc.isValid() && SM.isWrittenInSameFile(LocEnd, NextTokLoc) &&
        SR.LineEnd == SM.getSpellingLineNumber(NextTokLoc)) {
      SR.LineEnd--;
      SR.ColumnEnd++;
    }
    if (SR.isInSourceOrder())
      return SR;
    return None;
  }

  /// Gather all the regions that were skipped by the preprocessor
  /// using the constructs like #if or comments.
  void gatherSkippedRegions() {
    /// An array of the minimum lineStarts and the maximum lineEnds
    /// for mapping regions from the appropriate source files.
    llvm::SmallVector<std::pair<unsigned, unsigned>, 8> FileLineRanges;
    FileLineRanges.resize(
        FileIDMapping.size(),
        std::make_pair(std::numeric_limits<unsigned>::max(), 0));
    for (const auto &R : MappingRegions) {
      FileLineRanges[R.FileID].first =
          std::min(FileLineRanges[R.FileID].first, R.LineStart);
      FileLineRanges[R.FileID].second =
          std::max(FileLineRanges[R.FileID].second, R.LineEnd);
    }

    auto SkippedRanges = CVM.getSourceInfo().getSkippedRanges();
    for (auto &I : SkippedRanges) {
      SourceRange Range = I.Range;
      auto LocStart = Range.getBegin();
      auto LocEnd = Range.getEnd();
      assert(SM.isWrittenInSameFile(LocStart, LocEnd) &&
             "region spans multiple files");

      auto CovFileID = getCoverageFileID(LocStart);
      if (!CovFileID)
        continue;
      Optional<SpellingRegion> SR =
          adjustSkippedRange(SM, LocStart, LocEnd, I.PrevTokLoc, I.NextTokLoc);
      if (!SR.hasValue())
        continue;
      auto Region = CounterMappingRegion::makeSkipped(
          *CovFileID, SR->LineStart, SR->ColumnStart, SR->LineEnd,
          SR->ColumnEnd);
      // Make sure that we only collect the regions that are inside
      // the source code of this function.
      if (Region.LineStart >= FileLineRanges[*CovFileID].first &&
          Region.LineEnd <= FileLineRanges[*CovFileID].second)
        MappingRegions.push_back(Region);
    }
  }

  /// Generate the coverage counter mapping regions from collected
  /// source regions.
  void emitSourceRegions(const SourceRegionFilter &Filter) {
    for (const auto &Region : SourceRegions) {
      assert(Region.hasEndLoc() && "incomplete region");

      SourceLocation LocStart = Region.getBeginLoc();
      assert(SM.getFileID(LocStart).isValid() && "region in invalid file");

      // Ignore regions from system headers.
      if (SM.isInSystemHeader(SM.getSpellingLoc(LocStart)))
        continue;

      auto CovFileID = getCoverageFileID(LocStart);
      // Ignore regions that don't have a file, such as builtin macros.
      if (!CovFileID)
        continue;

      SourceLocation LocEnd = Region.getEndLoc();
      assert(SM.isWrittenInSameFile(LocStart, LocEnd) &&
             "region spans multiple files");

      // Don't add code regions for the area covered by expansion regions.
      // This not only suppresses redundant regions, but sometimes prevents
      // creating regions with wrong counters if, for example, a statement's
      // body ends at the end of a nested macro.
      if (Filter.count(std::make_pair(LocStart, LocEnd)))
        continue;

      // Find the spelling locations for the mapping region.
      SpellingRegion SR{SM, LocStart, LocEnd};
      assert(SR.isInSourceOrder() && "region start and end out of order");

      if (Region.isGap()) {
        MappingRegions.push_back(CounterMappingRegion::makeGapRegion(
            Region.getCounter(), *CovFileID, SR.LineStart, SR.ColumnStart,
            SR.LineEnd, SR.ColumnEnd));
      } else if (Region.isBranch()) {
        MappingRegions.push_back(CounterMappingRegion::makeBranchRegion(
            Region.getCounter(), Region.getFalseCounter(), *CovFileID,
            SR.LineStart, SR.ColumnStart, SR.LineEnd, SR.ColumnEnd));
      } else {
        MappingRegions.push_back(CounterMappingRegion::makeRegion(
            Region.getCounter(), *CovFileID, SR.LineStart, SR.ColumnStart,
            SR.LineEnd, SR.ColumnEnd));
      }
    }
  }

  /// Generate expansion regions for each virtual file we've seen.
  SourceRegionFilter emitExpansionRegions() {
    SourceRegionFilter Filter;
    for (const auto &FM : FileIDMapping) {
      SourceLocation ExpandedLoc = FM.second.second;
      SourceLocation ParentLoc = getIncludeOrExpansionLoc(ExpandedLoc);
      if (ParentLoc.isInvalid())
        continue;

      auto ParentFileID = getCoverageFileID(ParentLoc);
      if (!ParentFileID)
        continue;
      auto ExpandedFileID = getCoverageFileID(ExpandedLoc);
      assert(ExpandedFileID && "expansion in uncovered file");

      SourceLocation LocEnd = getPreciseTokenLocEnd(ParentLoc);
      assert(SM.isWrittenInSameFile(ParentLoc, LocEnd) &&
             "region spans multiple files");
      Filter.insert(std::make_pair(ParentLoc, LocEnd));

      SpellingRegion SR{SM, ParentLoc, LocEnd};
      assert(SR.isInSourceOrder() && "region start and end out of order");
      MappingRegions.push_back(CounterMappingRegion::makeExpansion(
          *ParentFileID, *ExpandedFileID, SR.LineStart, SR.ColumnStart,
          SR.LineEnd, SR.ColumnEnd));
    }
    return Filter;
  }
};

/// Creates unreachable coverage regions for the functions that
/// are not emitted.
struct EmptyCoverageMappingBuilder : public CoverageMappingBuilder {
  EmptyCoverageMappingBuilder(CoverageMappingModuleGen &CVM, SourceManager &SM,
                              const LangOptions &LangOpts)
      : CoverageMappingBuilder(CVM, SM, LangOpts) {}

  void VisitDecl(const Decl *D) {
    if (!D->hasBody())
      return;
    auto Body = D->getBody();
    SourceLocation Start = getStart(Body);
    SourceLocation End = getEnd(Body);
    if (!SM.isWrittenInSameFile(Start, End)) {
      // Walk up to find the common ancestor.
      // Correct the locations accordingly.
      FileID StartFileID = SM.getFileID(Start);
      FileID EndFileID = SM.getFileID(End);
      while (StartFileID != EndFileID && !isNestedIn(End, StartFileID)) {
        Start = getIncludeOrExpansionLoc(Start);
        assert(Start.isValid() &&
               "Declaration start location not nested within a known region");
        StartFileID = SM.getFileID(Start);
      }
      while (StartFileID != EndFileID) {
        End = getPreciseTokenLocEnd(getIncludeOrExpansionLoc(End));
        assert(End.isValid() &&
               "Declaration end location not nested within a known region");
        EndFileID = SM.getFileID(End);
      }
    }
    SourceRegions.emplace_back(Counter(), Start, End);
  }

  /// Write the mapping data to the output stream
  void write(llvm::raw_ostream &OS) {
    SmallVector<unsigned, 16> FileIDMapping;
    gatherFileIDs(FileIDMapping);
    emitSourceRegions(SourceRegionFilter());

    if (MappingRegions.empty())
      return;

    CoverageMappingWriter Writer(FileIDMapping, None, MappingRegions);
    Writer.write(OS);
  }
};

/// A StmtVisitor that creates coverage mapping regions which map
/// from the source code locations to the PGO counters.
struct CounterCoverageMappingBuilder
    : public CoverageMappingBuilder,
      public ConstStmtVisitor<CounterCoverageMappingBuilder> {
  /// The map of statements to count values.
  llvm::DenseMap<const Stmt *, unsigned> &CounterMap;

  /// A stack of currently live regions.
  std::vector<SourceMappingRegion> RegionStack;

  CounterExpressionBuilder Builder;

  /// A location in the most recently visited file or macro.
  ///
  /// This is used to adjust the active source regions appropriately when
  /// expressions cross file or macro boundaries.
  SourceLocation MostRecentLocation;

  /// Whether the visitor at a terminate statement.
  bool HasTerminateStmt = false;

  /// Gap region counter after terminate statement.
  Counter GapRegionCounter;

  /// Return a counter for the subtraction of \c RHS from \c LHS
  Counter subtractCounters(Counter LHS, Counter RHS) {
    return Builder.subtract(LHS, RHS);
  }

  /// Return a counter for the sum of \c LHS and \c RHS.
  Counter addCounters(Counter LHS, Counter RHS) {
    return Builder.add(LHS, RHS);
  }

  Counter addCounters(Counter C1, Counter C2, Counter C3) {
    return addCounters(addCounters(C1, C2), C3);
  }

  /// Return the region counter for the given statement.
  ///
  /// This should only be called on statements that have a dedicated counter.
  Counter getRegionCounter(const Stmt *S) {
    return Counter::getCounter(CounterMap[S]);
  }

  /// Push a region onto the stack.
  ///
  /// Returns the index on the stack where the region was pushed. This can be
  /// used with popRegions to exit a "scope", ending the region that was pushed.
  size_t pushRegion(Counter Count, Optional<SourceLocation> StartLoc = None,
                    Optional<SourceLocation> EndLoc = None,
                    Optional<Counter> FalseCount = None) {

    if (StartLoc && !FalseCount.hasValue()) {
      MostRecentLocation = *StartLoc;
    }

    RegionStack.emplace_back(Count, FalseCount, StartLoc, EndLoc);

    return RegionStack.size() - 1;
  }

  size_t locationDepth(SourceLocation Loc) {
    size_t Depth = 0;
    while (Loc.isValid()) {
      Loc = getIncludeOrExpansionLoc(Loc);
      Depth++;
    }
    return Depth;
  }

  /// Pop regions from the stack into the function's list of regions.
  ///
  /// Adds all regions from \c ParentIndex to the top of the stack to the
  /// function's \c SourceRegions.
  void popRegions(size_t ParentIndex) {
    assert(RegionStack.size() >= ParentIndex && "parent not in stack");
    while (RegionStack.size() > ParentIndex) {
      SourceMappingRegion &Region = RegionStack.back();
      if (Region.hasStartLoc()) {
        SourceLocation StartLoc = Region.getBeginLoc();
        SourceLocation EndLoc = Region.hasEndLoc()
                                    ? Region.getEndLoc()
                                    : RegionStack[ParentIndex].getEndLoc();
        bool isBranch = Region.isBranch();
        size_t StartDepth = locationDepth(StartLoc);
        size_t EndDepth = locationDepth(EndLoc);
        while (!SM.isWrittenInSameFile(StartLoc, EndLoc)) {
          bool UnnestStart = StartDepth >= EndDepth;
          bool UnnestEnd = EndDepth >= StartDepth;
          if (UnnestEnd) {
            // The region ends in a nested file or macro expansion. If the
            // region is not a branch region, create a separate region for each
            // expansion, and for all regions, update the EndLoc. Branch
            // regions should not be split in order to keep a straightforward
            // correspondance between the region and its associated branch
            // condition, even if the condition spans multiple depths.
            SourceLocation NestedLoc = getStartOfFileOrMacro(EndLoc);
            assert(SM.isWrittenInSameFile(NestedLoc, EndLoc));

            if (!isBranch && !isRegionAlreadyAdded(NestedLoc, EndLoc))
              SourceRegions.emplace_back(Region.getCounter(), NestedLoc,
                                         EndLoc);

            EndLoc = getPreciseTokenLocEnd(getIncludeOrExpansionLoc(EndLoc));
            if (EndLoc.isInvalid())
              llvm::report_fatal_error(
                  "File exit not handled before popRegions");
            EndDepth--;
          }
          if (UnnestStart) {
            // The region ends in a nested file or macro expansion. If the
            // region is not a branch region, create a separate region for each
            // expansion, and for all regions, update the StartLoc. Branch
            // regions should not be split in order to keep a straightforward
            // correspondance between the region and its associated branch
            // condition, even if the condition spans multiple depths.
            SourceLocation NestedLoc = getEndOfFileOrMacro(StartLoc);
            assert(SM.isWrittenInSameFile(StartLoc, NestedLoc));

            if (!isBranch && !isRegionAlreadyAdded(StartLoc, NestedLoc))
              SourceRegions.emplace_back(Region.getCounter(), StartLoc,
                                         NestedLoc);

            StartLoc = getIncludeOrExpansionLoc(StartLoc);
            if (StartLoc.isInvalid())
              llvm::report_fatal_error(
                  "File exit not handled before popRegions");
            StartDepth--;
          }
        }
        Region.setStartLoc(StartLoc);
        Region.setEndLoc(EndLoc);

        if (!isBranch) {
          MostRecentLocation = EndLoc;
          // If this region happens to span an entire expansion, we need to
          // make sure we don't overlap the parent region with it.
          if (StartLoc == getStartOfFileOrMacro(StartLoc) &&
              EndLoc == getEndOfFileOrMacro(EndLoc))
            MostRecentLocation = getIncludeOrExpansionLoc(EndLoc);
        }

        assert(SM.isWrittenInSameFile(Region.getBeginLoc(), EndLoc));
        assert(SpellingRegion(SM, Region).isInSourceOrder());
        SourceRegions.push_back(Region);
        }
      RegionStack.pop_back();
    }
  }

  /// Return the currently active region.
  SourceMappingRegion &getRegion() {
    assert(!RegionStack.empty() && "statement has no region");
    return RegionStack.back();
  }

  /// Propagate counts through the children of \p S if \p VisitChildren is true.
  /// Otherwise, only emit a count for \p S itself.
  Counter propagateCounts(Counter TopCount, const Stmt *S,
                          bool VisitChildren = true) {
    SourceLocation StartLoc = getStart(S);
    SourceLocation EndLoc = getEnd(S);
    size_t Index = pushRegion(TopCount, StartLoc, EndLoc);
    if (VisitChildren)
      Visit(S);
    Counter ExitCount = getRegion().getCounter();
    popRegions(Index);

    // The statement may be spanned by an expansion. Make sure we handle a file
    // exit out of this expansion before moving to the next statement.
    if (SM.isBeforeInTranslationUnit(StartLoc, S->getBeginLoc()))
      MostRecentLocation = EndLoc;

    return ExitCount;
  }

  /// Determine whether the given condition can be constant folded.
  bool ConditionFoldsToBool(const Expr *Cond) {
    Expr::EvalResult Result;
    return (Cond->EvaluateAsInt(Result, CVM.getCodeGenModule().getContext()));
  }

  /// Create a Branch Region around an instrumentable condition for coverage
  /// and add it to the function's SourceRegions.  A branch region tracks a
  /// "True" counter and a "False" counter for boolean expressions that
  /// result in the generation of a branch.
  void createBranchRegion(const Expr *C, Counter TrueCnt, Counter FalseCnt) {
    // Check for NULL conditions.
    if (!C)
      return;

    // Ensure we are an instrumentable condition (i.e. no "&&" or "||").  Push
    // region onto RegionStack but immediately pop it (which adds it to the
    // function's SourceRegions) because it doesn't apply to any other source
    // code other than the Condition.
    if (CodeGenFunction::isInstrumentedCondition(C)) {
      // If a condition can fold to true or false, the corresponding branch
      // will be removed.  Create a region with both counters hard-coded to
      // zero. This allows us to visualize them in a special way.
      // Alternatively, we can prevent any optimization done via
      // constant-folding by ensuring that ConstantFoldsToSimpleInteger() in
      // CodeGenFunction.c always returns false, but that is very heavy-handed.
      if (ConditionFoldsToBool(C))
        popRegions(pushRegion(Counter::getZero(), getStart(C), getEnd(C),
                              Counter::getZero()));
      else
        // Otherwise, create a region with the True counter and False counter.
        popRegions(pushRegion(TrueCnt, getStart(C), getEnd(C), FalseCnt));
    }
  }

  /// Create a Branch Region around a SwitchCase for code coverage
  /// and add it to the function's SourceRegions.
  void createSwitchCaseRegion(const SwitchCase *SC, Counter TrueCnt,
                              Counter FalseCnt) {
    // Push region onto RegionStack but immediately pop it (which adds it to
    // the function's SourceRegions) because it doesn't apply to any other
    // source other than the SwitchCase.
    popRegions(pushRegion(TrueCnt, getStart(SC), SC->getColonLoc(), FalseCnt));
  }

  /// Check whether a region with bounds \c StartLoc and \c EndLoc
  /// is already added to \c SourceRegions.
  bool isRegionAlreadyAdded(SourceLocation StartLoc, SourceLocation EndLoc,
                            bool isBranch = false) {
    return llvm::any_of(
        llvm::reverse(SourceRegions), [&](const SourceMappingRegion &Region) {
          return Region.getBeginLoc() == StartLoc &&
                 Region.getEndLoc() == EndLoc && Region.isBranch() == isBranch;
        });
  }

  /// Adjust the most recently visited location to \c EndLoc.
  ///
  /// This should be used after visiting any statements in non-source order.
  void adjustForOutOfOrderTraversal(SourceLocation EndLoc) {
    MostRecentLocation = EndLoc;
    // The code region for a whole macro is created in handleFileExit() when
    // it detects exiting of the virtual file of that macro. If we visited
    // statements in non-source order, we might already have such a region
    // added, for example, if a body of a loop is divided among multiple
    // macros. Avoid adding duplicate regions in such case.
    if (getRegion().hasEndLoc() &&
        MostRecentLocation == getEndOfFileOrMacro(MostRecentLocation) &&
        isRegionAlreadyAdded(getStartOfFileOrMacro(MostRecentLocation),
                             MostRecentLocation, getRegion().isBranch()))
      MostRecentLocation = getIncludeOrExpansionLoc(MostRecentLocation);
  }

  /// Adjust regions and state when \c NewLoc exits a file.
  ///
  /// If moving from our most recently tracked location to \c NewLoc exits any
  /// files, this adjusts our current region stack and creates the file regions
  /// for the exited file.
  void handleFileExit(SourceLocation NewLoc) {
    if (NewLoc.isInvalid() ||
        SM.isWrittenInSameFile(MostRecentLocation, NewLoc))
      return;

    // If NewLoc is not in a file that contains MostRecentLocation, walk up to
    // find the common ancestor.
    SourceLocation LCA = NewLoc;
    FileID ParentFile = SM.getFileID(LCA);
    while (!isNestedIn(MostRecentLocation, ParentFile)) {
      LCA = getIncludeOrExpansionLoc(LCA);
      if (LCA.isInvalid() || SM.isWrittenInSameFile(LCA, MostRecentLocation)) {
        // Since there isn't a common ancestor, no file was exited. We just need
        // to adjust our location to the new file.
        MostRecentLocation = NewLoc;
        return;
      }
      ParentFile = SM.getFileID(LCA);
    }

    llvm::SmallSet<SourceLocation, 8> StartLocs;
    Optional<Counter> ParentCounter;
    for (SourceMappingRegion &I : llvm::reverse(RegionStack)) {
      if (!I.hasStartLoc())
        continue;
      SourceLocation Loc = I.getBeginLoc();
      if (!isNestedIn(Loc, ParentFile)) {
        ParentCounter = I.getCounter();
        break;
      }

      while (!SM.isInFileID(Loc, ParentFile)) {
        // The most nested region for each start location is the one with the
        // correct count. We avoid creating redundant regions by stopping once
        // we've seen this region.
        if (StartLocs.insert(Loc).second) {
          if (I.isBranch())
            SourceRegions.emplace_back(I.getCounter(), I.getFalseCounter(), Loc,
                                       getEndOfFileOrMacro(Loc), I.isBranch());
          else
            SourceRegions.emplace_back(I.getCounter(), Loc,
                                       getEndOfFileOrMacro(Loc));
        }
        Loc = getIncludeOrExpansionLoc(Loc);
      }
      I.setStartLoc(getPreciseTokenLocEnd(Loc));
    }

    if (ParentCounter) {
      // If the file is contained completely by another region and doesn't
      // immediately start its own region, the whole file gets a region
      // corresponding to the parent.
      SourceLocation Loc = MostRecentLocation;
      while (isNestedIn(Loc, ParentFile)) {
        SourceLocation FileStart = getStartOfFileOrMacro(Loc);
        if (StartLocs.insert(FileStart).second) {
          SourceRegions.emplace_back(*ParentCounter, FileStart,
                                     getEndOfFileOrMacro(Loc));
          assert(SpellingRegion(SM, SourceRegions.back()).isInSourceOrder());
        }
        Loc = getIncludeOrExpansionLoc(Loc);
      }
    }

    MostRecentLocation = NewLoc;
  }

  /// Ensure that \c S is included in the current region.
  void extendRegion(const Stmt *S) {
    SourceMappingRegion &Region = getRegion();
    SourceLocation StartLoc = getStart(S);

    handleFileExit(StartLoc);
    if (!Region.hasStartLoc())
      Region.setStartLoc(StartLoc);
  }

  /// Mark \c S as a terminator, starting a zero region.
  void terminateRegion(const Stmt *S) {
    extendRegion(S);
    SourceMappingRegion &Region = getRegion();
    SourceLocation EndLoc = getEnd(S);
    if (!Region.hasEndLoc())
      Region.setEndLoc(EndLoc);
    pushRegion(Counter::getZero());
    HasTerminateStmt = true;
  }

  /// Find a valid gap range between \p AfterLoc and \p BeforeLoc.
  Optional<SourceRange> findGapAreaBetween(SourceLocation AfterLoc,
                                           SourceLocation BeforeLoc) {
    // If AfterLoc is in function-like macro, use the right parenthesis
    // location.
    if (AfterLoc.isMacroID()) {
      FileID FID = SM.getFileID(AfterLoc);
      const SrcMgr::ExpansionInfo *EI = &SM.getSLocEntry(FID).getExpansion();
      if (EI->isFunctionMacroExpansion())
        AfterLoc = EI->getExpansionLocEnd();
    }

    size_t StartDepth = locationDepth(AfterLoc);
    size_t EndDepth = locationDepth(BeforeLoc);
    while (!SM.isWrittenInSameFile(AfterLoc, BeforeLoc)) {
      bool UnnestStart = StartDepth >= EndDepth;
      bool UnnestEnd = EndDepth >= StartDepth;
      if (UnnestEnd) {
        assert(SM.isWrittenInSameFile(getStartOfFileOrMacro(BeforeLoc),
                                      BeforeLoc));

        BeforeLoc = getIncludeOrExpansionLoc(BeforeLoc);
        assert(BeforeLoc.isValid());
        EndDepth--;
      }
      if (UnnestStart) {
        assert(SM.isWrittenInSameFile(AfterLoc,
                                      getEndOfFileOrMacro(AfterLoc)));

        AfterLoc = getIncludeOrExpansionLoc(AfterLoc);
        assert(AfterLoc.isValid());
        AfterLoc = getPreciseTokenLocEnd(AfterLoc);
        assert(AfterLoc.isValid());
        StartDepth--;
      }
    }
    AfterLoc = getPreciseTokenLocEnd(AfterLoc);
    // If the start and end locations of the gap are both within the same macro
    // file, the range may not be in source order.
    if (AfterLoc.isMacroID() || BeforeLoc.isMacroID())
      return None;
    if (!SM.isWrittenInSameFile(AfterLoc, BeforeLoc) ||
        !SpellingRegion(SM, AfterLoc, BeforeLoc).isInSourceOrder())
      return None;
    return {{AfterLoc, BeforeLoc}};
  }

  /// Emit a gap region between \p StartLoc and \p EndLoc with the given count.
  void fillGapAreaWithCount(SourceLocation StartLoc, SourceLocation EndLoc,
                            Counter Count) {
    if (StartLoc == EndLoc)
      return;
    assert(SpellingRegion(SM, StartLoc, EndLoc).isInSourceOrder());
    handleFileExit(StartLoc);
    size_t Index = pushRegion(Count, StartLoc, EndLoc);
    getRegion().setGap(true);
    handleFileExit(EndLoc);
    popRegions(Index);
  }

  /// Keep counts of breaks and continues inside loops.
  struct BreakContinue {
    Counter BreakCount;
    Counter ContinueCount;
  };
  SmallVector<BreakContinue, 8> BreakContinueStack;

  CounterCoverageMappingBuilder(
      CoverageMappingModuleGen &CVM,
      llvm::DenseMap<const Stmt *, unsigned> &CounterMap, SourceManager &SM,
      const LangOptions &LangOpts)
      : CoverageMappingBuilder(CVM, SM, LangOpts), CounterMap(CounterMap) {}

  /// Write the mapping data to the output stream
  void write(llvm::raw_ostream &OS) {
    llvm::SmallVector<unsigned, 8> VirtualFileMapping;
    gatherFileIDs(VirtualFileMapping);
    SourceRegionFilter Filter = emitExpansionRegions();
    emitSourceRegions(Filter);
    gatherSkippedRegions();

    if (MappingRegions.empty())
      return;

    CoverageMappingWriter Writer(VirtualFileMapping, Builder.getExpressions(),
                                 MappingRegions);
    Writer.write(OS);
  }

  void VisitStmt(const Stmt *S) {
    if (S->getBeginLoc().isValid())
      extendRegion(S);
    const Stmt *LastStmt = nullptr;
    bool SaveTerminateStmt = HasTerminateStmt;
    HasTerminateStmt = false;
    GapRegionCounter = Counter::getZero();
    for (const Stmt *Child : S->children())
      if (Child) {
        // If last statement contains terminate statements, add a gap area
        // between the two statements. Skipping attributed statements, because
        // they don't have valid start location.
        if (LastStmt && HasTerminateStmt && !isa<AttributedStmt>(Child)) {
          auto Gap = findGapAreaBetween(getEnd(LastStmt), getStart(Child));
          if (Gap)
            fillGapAreaWithCount(Gap->getBegin(), Gap->getEnd(),
                                 GapRegionCounter);
          SaveTerminateStmt = true;
          HasTerminateStmt = false;
        }
        this->Visit(Child);
        LastStmt = Child;
      }
    if (SaveTerminateStmt)
      HasTerminateStmt = true;
    handleFileExit(getEnd(S));
  }

  void VisitDecl(const Decl *D) {
    Stmt *Body = D->getBody();

    // Do not propagate region counts into system headers.
    if (Body && SM.isInSystemHeader(SM.getSpellingLoc(getStart(Body))))
      return;

    // Do not visit the artificial children nodes of defaulted methods. The
    // lexer may not be able to report back precise token end locations for
    // these children nodes (llvm.org/PR39822), and moreover users will not be
    // able to see coverage for them.
    bool Defaulted = false;
    if (auto *Method = dyn_cast<CXXMethodDecl>(D))
      Defaulted = Method->isDefaulted();

    propagateCounts(getRegionCounter(Body), Body,
                    /*VisitChildren=*/!Defaulted);
    assert(RegionStack.empty() && "Regions entered but never exited");
  }

  void VisitReturnStmt(const ReturnStmt *S) {
    extendRegion(S);
    if (S->getRetValue())
      Visit(S->getRetValue());
    terminateRegion(S);
  }

  void VisitCoroutineBodyStmt(const CoroutineBodyStmt *S) {
    extendRegion(S);
    Visit(S->getBody());
  }

  void VisitCoreturnStmt(const CoreturnStmt *S) {
    extendRegion(S);
    if (S->getOperand())
      Visit(S->getOperand());
    terminateRegion(S);
  }

  void VisitCXXThrowExpr(const CXXThrowExpr *E) {
    extendRegion(E);
    if (E->getSubExpr())
      Visit(E->getSubExpr());
    terminateRegion(E);
  }

  void VisitGotoStmt(const GotoStmt *S) { terminateRegion(S); }

  void VisitLabelStmt(const LabelStmt *S) {
    Counter LabelCount = getRegionCounter(S);
    SourceLocation Start = getStart(S);
    // We can't extendRegion here or we risk overlapping with our new region.
    handleFileExit(Start);
    pushRegion(LabelCount, Start);
    Visit(S->getSubStmt());
  }

  void VisitBreakStmt(const BreakStmt *S) {
    assert(!BreakContinueStack.empty() && "break not in a loop or switch!");
    BreakContinueStack.back().BreakCount = addCounters(
        BreakContinueStack.back().BreakCount, getRegion().getCounter());
    // FIXME: a break in a switch should terminate regions for all preceding
    // case statements, not just the most recent one.
    terminateRegion(S);
  }

  void VisitContinueStmt(const ContinueStmt *S) {
    assert(!BreakContinueStack.empty() && "continue stmt not in a loop!");
    BreakContinueStack.back().ContinueCount = addCounters(
        BreakContinueStack.back().ContinueCount, getRegion().getCounter());
    terminateRegion(S);
  }

  void VisitCallExpr(const CallExpr *E) {
    VisitStmt(E);

    // Terminate the region when we hit a noreturn function.
    // (This is helpful dealing with switch statements.)
    QualType CalleeType = E->getCallee()->getType();
    if (getFunctionExtInfo(*CalleeType).getNoReturn())
      terminateRegion(E);
  }

  void VisitWhileStmt(const WhileStmt *S) {
    extendRegion(S);

    Counter ParentCount = getRegion().getCounter();
    Counter BodyCount = getRegionCounter(S);

    // Handle the body first so that we can get the backedge count.
    BreakContinueStack.push_back(BreakContinue());
    extendRegion(S->getBody());
    Counter BackedgeCount = propagateCounts(BodyCount, S->getBody());
    BreakContinue BC = BreakContinueStack.pop_back_val();

    bool BodyHasTerminateStmt = HasTerminateStmt;
    HasTerminateStmt = false;

    // Go back to handle the condition.
    Counter CondCount =
        addCounters(ParentCount, BackedgeCount, BC.ContinueCount);
    propagateCounts(CondCount, S->getCond());
    adjustForOutOfOrderTraversal(getEnd(S));

    // The body count applies to the area immediately after the increment.
    auto Gap = findGapAreaBetween(S->getRParenLoc(), getStart(S->getBody()));
    if (Gap)
      fillGapAreaWithCount(Gap->getBegin(), Gap->getEnd(), BodyCount);

    Counter OutCount =
        addCounters(BC.BreakCount, subtractCounters(CondCount, BodyCount));
    if (OutCount != ParentCount) {
      pushRegion(OutCount);
      GapRegionCounter = OutCount;
      if (BodyHasTerminateStmt)
        HasTerminateStmt = true;
    }

    // Create Branch Region around condition.
    createBranchRegion(S->getCond(), BodyCount,
                       subtractCounters(CondCount, BodyCount));
  }

  void VisitDoStmt(const DoStmt *S) {
    extendRegion(S);

    Counter ParentCount = getRegion().getCounter();
    Counter BodyCount = getRegionCounter(S);

    BreakContinueStack.push_back(BreakContinue());
    extendRegion(S->getBody());
    Counter BackedgeCount =
        propagateCounts(addCounters(ParentCount, BodyCount), S->getBody());
    BreakContinue BC = BreakContinueStack.pop_back_val();

    bool BodyHasTerminateStmt = HasTerminateStmt;
    HasTerminateStmt = false;

    Counter CondCount = addCounters(BackedgeCount, BC.ContinueCount);
    propagateCounts(CondCount, S->getCond());

    Counter OutCount =
        addCounters(BC.BreakCount, subtractCounters(CondCount, BodyCount));
    if (OutCount != ParentCount) {
      pushRegion(OutCount);
      GapRegionCounter = OutCount;
    }

    // Create Branch Region around condition.
    createBranchRegion(S->getCond(), BodyCount,
                       subtractCounters(CondCount, BodyCount));

    if (BodyHasTerminateStmt)
      HasTerminateStmt = true;
  }

  void VisitForStmt(const ForStmt *S) {
    extendRegion(S);
    if (S->getInit())
      Visit(S->getInit());

    Counter ParentCount = getRegion().getCounter();
    Counter BodyCount = getRegionCounter(S);

    // The loop increment may contain a break or continue.
    if (S->getInc())
      BreakContinueStack.emplace_back();

    // Handle the body first so that we can get the backedge count.
    BreakContinueStack.emplace_back();
    extendRegion(S->getBody());
    Counter BackedgeCount = propagateCounts(BodyCount, S->getBody());
    BreakContinue BodyBC = BreakContinueStack.pop_back_val();

    bool BodyHasTerminateStmt = HasTerminateStmt;
    HasTerminateStmt = false;

    // The increment is essentially part of the body but it needs to include
    // the count for all the continue statements.
    BreakContinue IncrementBC;
    if (const Stmt *Inc = S->getInc()) {
      propagateCounts(addCounters(BackedgeCount, BodyBC.ContinueCount), Inc);
      IncrementBC = BreakContinueStack.pop_back_val();
    }

    // Go back to handle the condition.
    Counter CondCount = addCounters(
        addCounters(ParentCount, BackedgeCount, BodyBC.ContinueCount),
        IncrementBC.ContinueCount);
    if (const Expr *Cond = S->getCond()) {
      propagateCounts(CondCount, Cond);
      adjustForOutOfOrderTraversal(getEnd(S));
    }

    // The body count applies to the area immediately after the increment.
    auto Gap = findGapAreaBetween(S->getRParenLoc(), getStart(S->getBody()));
    if (Gap)
      fillGapAreaWithCount(Gap->getBegin(), Gap->getEnd(), BodyCount);

    Counter OutCount = addCounters(BodyBC.BreakCount, IncrementBC.BreakCount,
                                   subtractCounters(CondCount, BodyCount));
    if (OutCount != ParentCount) {
      pushRegion(OutCount);
      GapRegionCounter = OutCount;
      if (BodyHasTerminateStmt)
        HasTerminateStmt = true;
    }

    // Create Branch Region around condition.
    createBranchRegion(S->getCond(), BodyCount,
                       subtractCounters(CondCount, BodyCount));
  }

  void VisitCXXForRangeStmt(const CXXForRangeStmt *S) {
    extendRegion(S);
    if (S->getInit())
      Visit(S->getInit());
    Visit(S->getLoopVarStmt());
    Visit(S->getRangeStmt());

    Counter ParentCount = getRegion().getCounter();
    Counter BodyCount = getRegionCounter(S);

    BreakContinueStack.push_back(BreakContinue());
    extendRegion(S->getBody());
    Counter BackedgeCount = propagateCounts(BodyCount, S->getBody());
    BreakContinue BC = BreakContinueStack.pop_back_val();

    bool BodyHasTerminateStmt = HasTerminateStmt;
    HasTerminateStmt = false;

    // The body count applies to the area immediately after the range.
    auto Gap = findGapAreaBetween(S->getRParenLoc(), getStart(S->getBody()));
    if (Gap)
      fillGapAreaWithCount(Gap->getBegin(), Gap->getEnd(), BodyCount);

    Counter LoopCount =
        addCounters(ParentCount, BackedgeCount, BC.ContinueCount);
    Counter OutCount =
        addCounters(BC.BreakCount, subtractCounters(LoopCount, BodyCount));
    if (OutCount != ParentCount) {
      pushRegion(OutCount);
      GapRegionCounter = OutCount;
      if (BodyHasTerminateStmt)
        HasTerminateStmt = true;
    }

    // Create Branch Region around condition.
    createBranchRegion(S->getCond(), BodyCount,
                       subtractCounters(LoopCount, BodyCount));
  }

  void VisitObjCForCollectionStmt(const ObjCForCollectionStmt *S) {
    extendRegion(S);
    Visit(S->getElement());

    Counter ParentCount = getRegion().getCounter();
    Counter BodyCount = getRegionCounter(S);

    BreakContinueStack.push_back(BreakContinue());
    extendRegion(S->getBody());
    Counter BackedgeCount = propagateCounts(BodyCount, S->getBody());
    BreakContinue BC = BreakContinueStack.pop_back_val();

    // The body count applies to the area immediately after the collection.
    auto Gap = findGapAreaBetween(S->getRParenLoc(), getStart(S->getBody()));
    if (Gap)
      fillGapAreaWithCount(Gap->getBegin(), Gap->getEnd(), BodyCount);

    Counter LoopCount =
        addCounters(ParentCount, BackedgeCount, BC.ContinueCount);
    Counter OutCount =
        addCounters(BC.BreakCount, subtractCounters(LoopCount, BodyCount));
    if (OutCount != ParentCount) {
      pushRegion(OutCount);
      GapRegionCounter = OutCount;
    }
  }

  void VisitSwitchStmt(const SwitchStmt *S) {
    extendRegion(S);
    if (S->getInit())
      Visit(S->getInit());
    Visit(S->getCond());

    BreakContinueStack.push_back(BreakContinue());

    const Stmt *Body = S->getBody();
    extendRegion(Body);
    if (const auto *CS = dyn_cast<CompoundStmt>(Body)) {
      if (!CS->body_empty()) {
        // Make a region for the body of the switch.  If the body starts with
        // a case, that case will reuse this region; otherwise, this covers
        // the unreachable code at the beginning of the switch body.
        size_t Index = pushRegion(Counter::getZero(), getStart(CS));
        getRegion().setGap(true);
        Visit(Body);

        // Set the end for the body of the switch, if it isn't already set.
        for (size_t i = RegionStack.size(); i != Index; --i) {
          if (!RegionStack[i - 1].hasEndLoc())
            RegionStack[i - 1].setEndLoc(getEnd(CS->body_back()));
        }

        popRegions(Index);
      }
    } else
      propagateCounts(Counter::getZero(), Body);
    BreakContinue BC = BreakContinueStack.pop_back_val();

    if (!BreakContinueStack.empty())
      BreakContinueStack.back().ContinueCount = addCounters(
          BreakContinueStack.back().ContinueCount, BC.ContinueCount);

    Counter ParentCount = getRegion().getCounter();
    Counter ExitCount = getRegionCounter(S);
    SourceLocation ExitLoc = getEnd(S);
    pushRegion(ExitCount);
    GapRegionCounter = ExitCount;

    // Ensure that handleFileExit recognizes when the end location is located
    // in a different file.
    MostRecentLocation = getStart(S);
    handleFileExit(ExitLoc);

    // Create a Branch Region around each Case. Subtract the case's
    // counter from the Parent counter to track the "False" branch count.
    Counter CaseCountSum;
    bool HasDefaultCase = false;
    const SwitchCase *Case = S->getSwitchCaseList();
    for (; Case; Case = Case->getNextSwitchCase()) {
      HasDefaultCase = HasDefaultCase || isa<DefaultStmt>(Case);
      CaseCountSum = addCounters(CaseCountSum, getRegionCounter(Case));
      createSwitchCaseRegion(
          Case, getRegionCounter(Case),
          subtractCounters(ParentCount, getRegionCounter(Case)));
    }

    // If no explicit default case exists, create a branch region to represent
    // the hidden branch, which will be added later by the CodeGen. This region
    // will be associated with the switch statement's condition.
    if (!HasDefaultCase) {
      Counter DefaultTrue = subtractCounters(ParentCount, CaseCountSum);
      Counter DefaultFalse = subtractCounters(ParentCount, DefaultTrue);
      createBranchRegion(S->getCond(), DefaultTrue, DefaultFalse);
    }
  }

  void VisitSwitchCase(const SwitchCase *S) {
    extendRegion(S);

    SourceMappingRegion &Parent = getRegion();

    Counter Count = addCounters(Parent.getCounter(), getRegionCounter(S));
    // Reuse the existing region if it starts at our label. This is typical of
    // the first case in a switch.
    if (Parent.hasStartLoc() && Parent.getBeginLoc() == getStart(S))
      Parent.setCounter(Count);
    else
      pushRegion(Count, getStart(S));

    GapRegionCounter = Count;

    if (const auto *CS = dyn_cast<CaseStmt>(S)) {
      Visit(CS->getLHS());
      if (const Expr *RHS = CS->getRHS())
        Visit(RHS);
    }
    Visit(S->getSubStmt());
  }

  void VisitIfStmt(const IfStmt *S) {
    extendRegion(S);
    if (S->getInit())
      Visit(S->getInit());

    // Extend into the condition before we propagate through it below - this is
    // needed to handle macros that generate the "if" but not the condition.
    extendRegion(S->getCond());

    Counter ParentCount = getRegion().getCounter();
    Counter ThenCount = getRegionCounter(S);

    // Emitting a counter for the condition makes it easier to interpret the
    // counter for the body when looking at the coverage.
    propagateCounts(ParentCount, S->getCond());

    // The 'then' count applies to the area immediately after the condition.
    auto Gap = findGapAreaBetween(S->getRParenLoc(), getStart(S->getThen()));
    if (Gap)
      fillGapAreaWithCount(Gap->getBegin(), Gap->getEnd(), ThenCount);

    extendRegion(S->getThen());
    Counter OutCount = propagateCounts(ThenCount, S->getThen());

    Counter ElseCount = subtractCounters(ParentCount, ThenCount);
    if (const Stmt *Else = S->getElse()) {
      bool ThenHasTerminateStmt = HasTerminateStmt;
      HasTerminateStmt = false;

      // The 'else' count applies to the area immediately after the 'then'.
      Gap = findGapAreaBetween(getEnd(S->getThen()), getStart(Else));
      if (Gap)
        fillGapAreaWithCount(Gap->getBegin(), Gap->getEnd(), ElseCount);
      extendRegion(Else);
      OutCount = addCounters(OutCount, propagateCounts(ElseCount, Else));

      if (ThenHasTerminateStmt)
        HasTerminateStmt = true;
    } else
      OutCount = addCounters(OutCount, ElseCount);

    if (OutCount != ParentCount) {
      pushRegion(OutCount);
      GapRegionCounter = OutCount;
    }

    // Create Branch Region around condition.
    createBranchRegion(S->getCond(), ThenCount,
                       subtractCounters(ParentCount, ThenCount));
  }

  void VisitCXXTryStmt(const CXXTryStmt *S) {
    extendRegion(S);
    // Handle macros that generate the "try" but not the rest.
    extendRegion(S->getTryBlock());

    Counter ParentCount = getRegion().getCounter();
    propagateCounts(ParentCount, S->getTryBlock());

    for (unsigned I = 0, E = S->getNumHandlers(); I < E; ++I)
      Visit(S->getHandler(I));

    Counter ExitCount = getRegionCounter(S);
    pushRegion(ExitCount);
  }

  void VisitCXXCatchStmt(const CXXCatchStmt *S) {
    propagateCounts(getRegionCounter(S), S->getHandlerBlock());
  }

  void VisitAbstractConditionalOperator(const AbstractConditionalOperator *E) {
    extendRegion(E);

    Counter ParentCount = getRegion().getCounter();
    Counter TrueCount = getRegionCounter(E);

    propagateCounts(ParentCount, E->getCond());

    if (!isa<BinaryConditionalOperator>(E)) {
      // The 'then' count applies to the area immediately after the condition.
      auto Gap =
          findGapAreaBetween(E->getQuestionLoc(), getStart(E->getTrueExpr()));
      if (Gap)
        fillGapAreaWithCount(Gap->getBegin(), Gap->getEnd(), TrueCount);

      extendRegion(E->getTrueExpr());
      propagateCounts(TrueCount, E->getTrueExpr());
    }

    extendRegion(E->getFalseExpr());
    propagateCounts(subtractCounters(ParentCount, TrueCount),
                    E->getFalseExpr());

    // Create Branch Region around condition.
    createBranchRegion(E->getCond(), TrueCount,
                       subtractCounters(ParentCount, TrueCount));
  }

  void VisitBinLAnd(const BinaryOperator *E) {
    extendRegion(E->getLHS());
    propagateCounts(getRegion().getCounter(), E->getLHS());
    handleFileExit(getEnd(E->getLHS()));

    // Counter tracks the right hand side of a logical and operator.
    extendRegion(E->getRHS());
    propagateCounts(getRegionCounter(E), E->getRHS());

    // Extract the RHS's Execution Counter.
    Counter RHSExecCnt = getRegionCounter(E);

    // Extract the RHS's "True" Instance Counter.
    Counter RHSTrueCnt = getRegionCounter(E->getRHS());

    // Extract the Parent Region Counter.
    Counter ParentCnt = getRegion().getCounter();

    // Create Branch Region around LHS condition.
    createBranchRegion(E->getLHS(), RHSExecCnt,
                       subtractCounters(ParentCnt, RHSExecCnt));

    // Create Branch Region around RHS condition.
    createBranchRegion(E->getRHS(), RHSTrueCnt,
                       subtractCounters(RHSExecCnt, RHSTrueCnt));
  }

  void VisitBinLOr(const BinaryOperator *E) {
    extendRegion(E->getLHS());
    propagateCounts(getRegion().getCounter(), E->getLHS());
    handleFileExit(getEnd(E->getLHS()));

    // Counter tracks the right hand side of a logical or operator.
    extendRegion(E->getRHS());
    propagateCounts(getRegionCounter(E), E->getRHS());

    // Extract the RHS's Execution Counter.
    Counter RHSExecCnt = getRegionCounter(E);

    // Extract the RHS's "False" Instance Counter.
    Counter RHSFalseCnt = getRegionCounter(E->getRHS());

    // Extract the Parent Region Counter.
    Counter ParentCnt = getRegion().getCounter();

    // Create Branch Region around LHS condition.
    createBranchRegion(E->getLHS(), subtractCounters(ParentCnt, RHSExecCnt),
                       RHSExecCnt);

    // Create Branch Region around RHS condition.
    createBranchRegion(E->getRHS(), subtractCounters(RHSExecCnt, RHSFalseCnt),
                       RHSFalseCnt);
  }

  void VisitLambdaExpr(const LambdaExpr *LE) {
    // Lambdas are treated as their own functions for now, so we shouldn't
    // propagate counts into them.
  }
};

} // end anonymous namespace

static void dump(llvm::raw_ostream &OS, StringRef FunctionName,
                 ArrayRef<CounterExpression> Expressions,
                 ArrayRef<CounterMappingRegion> Regions) {
  OS << FunctionName << ":\n";
  CounterMappingContext Ctx(Expressions);
  for (const auto &R : Regions) {
    OS.indent(2);
    switch (R.Kind) {
    case CounterMappingRegion::CodeRegion:
      break;
    case CounterMappingRegion::ExpansionRegion:
      OS << "Expansion,";
      break;
    case CounterMappingRegion::SkippedRegion:
      OS << "Skipped,";
      break;
    case CounterMappingRegion::GapRegion:
      OS << "Gap,";
      break;
    case CounterMappingRegion::BranchRegion:
      OS << "Branch,";
      break;
    }

    OS << "File " << R.FileID << ", " << R.LineStart << ":" << R.ColumnStart
       << " -> " << R.LineEnd << ":" << R.ColumnEnd << " = ";
    Ctx.dump(R.Count, OS);

    if (R.Kind == CounterMappingRegion::BranchRegion) {
      OS << ", ";
      Ctx.dump(R.FalseCount, OS);
    }

    if (R.Kind == CounterMappingRegion::ExpansionRegion)
      OS << " (Expanded file = " << R.ExpandedFileID << ")";
    OS << "\n";
  }
}

CoverageMappingModuleGen::CoverageMappingModuleGen(
    CodeGenModule &CGM, CoverageSourceInfo &SourceInfo)
    : CGM(CGM), SourceInfo(SourceInfo) {
  CoveragePrefixMap = CGM.getCodeGenOpts().CoveragePrefixMap;
}

std::string CoverageMappingModuleGen::getCurrentDirname() {
  if (!CGM.getCodeGenOpts().CoverageCompilationDir.empty())
    return CGM.getCodeGenOpts().CoverageCompilationDir;

  SmallString<256> CWD;
  llvm::sys::fs::current_path(CWD);
  return CWD.str().str();
}

std::string CoverageMappingModuleGen::normalizeFilename(StringRef Filename) {
  llvm::SmallString<256> Path(Filename);
  llvm::sys::path::remove_dots(Path, /*remove_dot_dot=*/true);
  for (const auto &Entry : CoveragePrefixMap) {
    if (llvm::sys::path::replace_path_prefix(Path, Entry.first, Entry.second))
      break;
  }
  return Path.str().str();
}

static std::string getInstrProfSection(const CodeGenModule &CGM,
                                       llvm::InstrProfSectKind SK) {
  return llvm::getInstrProfSectionName(
      SK, CGM.getContext().getTargetInfo().getTriple().getObjectFormat());
}

void CoverageMappingModuleGen::emitFunctionMappingRecord(
    const FunctionInfo &Info, uint64_t FilenamesRef) {
  llvm::LLVMContext &Ctx = CGM.getLLVMContext();

  // Assign a name to the function record. This is used to merge duplicates.
  std::string FuncRecordName = "__covrec_" + llvm::utohexstr(Info.NameHash);

  // A dummy description for a function included-but-not-used in a TU can be
  // replaced by full description provided by a different TU. The two kinds of
  // descriptions play distinct roles: therefore, assign them different names
  // to prevent `linkonce_odr` merging.
  if (Info.IsUsed)
    FuncRecordName += "u";

  // Create the function record type.
  const uint64_t NameHash = Info.NameHash;
  const uint64_t FuncHash = Info.FuncHash;
  const std::string &CoverageMapping = Info.CoverageMapping;
#define COVMAP_FUNC_RECORD(Type, LLVMType, Name, Init) LLVMType,
  llvm::Type *FunctionRecordTypes[] = {
#include "llvm/ProfileData/InstrProfData.inc"
  };
  auto *FunctionRecordTy =
      llvm::StructType::get(Ctx, makeArrayRef(FunctionRecordTypes),
                            /*isPacked=*/true);

  // Create the function record constant.
#define COVMAP_FUNC_RECORD(Type, LLVMType, Name, Init) Init,
  llvm::Constant *FunctionRecordVals[] = {
      #include "llvm/ProfileData/InstrProfData.inc"
  };
  auto *FuncRecordConstant = llvm::ConstantStruct::get(
      FunctionRecordTy, makeArrayRef(FunctionRecordVals));

  // Create the function record global.
  auto *FuncRecord = new llvm::GlobalVariable(
      CGM.getModule(), FunctionRecordTy, /*isConstant=*/true,
      llvm::GlobalValue::LinkOnceODRLinkage, FuncRecordConstant,
      FuncRecordName);
  FuncRecord->setVisibility(llvm::GlobalValue::HiddenVisibility);
  FuncRecord->setSection(getInstrProfSection(CGM, llvm::IPSK_covfun));
  FuncRecord->setAlignment(llvm::Align(8));
  if (CGM.supportsCOMDAT())
    FuncRecord->setComdat(CGM.getModule().getOrInsertComdat(FuncRecordName));

  // Make sure the data doesn't get deleted.
  CGM.addUsedGlobal(FuncRecord);
}

void CoverageMappingModuleGen::addFunctionMappingRecord(
    llvm::GlobalVariable *NamePtr, StringRef NameValue, uint64_t FuncHash,
    const std::string &CoverageMapping, bool IsUsed) {
  llvm::LLVMContext &Ctx = CGM.getLLVMContext();
  const uint64_t NameHash = llvm::IndexedInstrProf::ComputeHash(NameValue);
  FunctionRecords.push_back({NameHash, FuncHash, CoverageMapping, IsUsed});

  if (!IsUsed)
    FunctionNames.push_back(
        llvm::ConstantExpr::getBitCast(NamePtr, llvm::Type::getInt8PtrTy(Ctx)));

  if (CGM.getCodeGenOpts().DumpCoverageMapping) {
    // Dump the coverage mapping data for this function by decoding the
    // encoded data. This allows us to dump the mapping regions which were
    // also processed by the CoverageMappingWriter which performs
    // additional minimization operations such as reducing the number of
    // expressions.
    llvm::SmallVector<std::string, 16> FilenameStrs;
    std::vector<StringRef> Filenames;
    std::vector<CounterExpression> Expressions;
    std::vector<CounterMappingRegion> Regions;
    FilenameStrs.resize(FileEntries.size() + 1);
    FilenameStrs[0] = normalizeFilename(getCurrentDirname());
    for (const auto &Entry : FileEntries) {
      auto I = Entry.second;
      FilenameStrs[I] = normalizeFilename(Entry.first->getName());
    }
    ArrayRef<std::string> FilenameRefs = llvm::makeArrayRef(FilenameStrs);
    RawCoverageMappingReader Reader(CoverageMapping, FilenameRefs, Filenames,
                                    Expressions, Regions);
    if (Reader.read())
      return;
    dump(llvm::outs(), NameValue, Expressions, Regions);
  }
}

void CoverageMappingModuleGen::emit() {
  if (FunctionRecords.empty())
    return;
  llvm::LLVMContext &Ctx = CGM.getLLVMContext();
  auto *Int32Ty = llvm::Type::getInt32Ty(Ctx);

  // Create the filenames and merge them with coverage mappings
  llvm::SmallVector<std::string, 16> FilenameStrs;
  FilenameStrs.resize(FileEntries.size() + 1);
  // The first filename is the current working directory.
  FilenameStrs[0] = normalizeFilename(getCurrentDirname());
  for (const auto &Entry : FileEntries) {
    auto I = Entry.second;
    FilenameStrs[I] = normalizeFilename(Entry.first->getName());
  }

  std::string Filenames;
  {
    llvm::raw_string_ostream OS(Filenames);
    CoverageFilenamesSectionWriter(FilenameStrs).write(OS);
  }
  auto *FilenamesVal =
      llvm::ConstantDataArray::getString(Ctx, Filenames, false);
  const int64_t FilenamesRef = llvm::IndexedInstrProf::ComputeHash(Filenames);

  // Emit the function records.
  for (const FunctionInfo &Info : FunctionRecords)
    emitFunctionMappingRecord(Info, FilenamesRef);

  const unsigned NRecords = 0;
  const size_t FilenamesSize = Filenames.size();
  const unsigned CoverageMappingSize = 0;
  llvm::Type *CovDataHeaderTypes[] = {
#define COVMAP_HEADER(Type, LLVMType, Name, Init) LLVMType,
#include "llvm/ProfileData/InstrProfData.inc"
  };
  auto CovDataHeaderTy =
      llvm::StructType::get(Ctx, makeArrayRef(CovDataHeaderTypes));
  llvm::Constant *CovDataHeaderVals[] = {
#define COVMAP_HEADER(Type, LLVMType, Name, Init) Init,
#include "llvm/ProfileData/InstrProfData.inc"
  };
  auto CovDataHeaderVal = llvm::ConstantStruct::get(
      CovDataHeaderTy, makeArrayRef(CovDataHeaderVals));

  // Create the coverage data record
  llvm::Type *CovDataTypes[] = {CovDataHeaderTy, FilenamesVal->getType()};
  auto CovDataTy = llvm::StructType::get(Ctx, makeArrayRef(CovDataTypes));
  llvm::Constant *TUDataVals[] = {CovDataHeaderVal, FilenamesVal};
  auto CovDataVal =
      llvm::ConstantStruct::get(CovDataTy, makeArrayRef(TUDataVals));
  auto CovData = new llvm::GlobalVariable(
      CGM.getModule(), CovDataTy, true, llvm::GlobalValue::PrivateLinkage,
      CovDataVal, llvm::getCoverageMappingVarName());

  CovData->setSection(getInstrProfSection(CGM, llvm::IPSK_covmap));
  CovData->setAlignment(llvm::Align(8));

  // Make sure the data doesn't get deleted.
  CGM.addUsedGlobal(CovData);
  // Create the deferred function records array
  if (!FunctionNames.empty()) {
    auto NamesArrTy = llvm::ArrayType::get(llvm::Type::getInt8PtrTy(Ctx),
                                           FunctionNames.size());
    auto NamesArrVal = llvm::ConstantArray::get(NamesArrTy, FunctionNames);
    // This variable will *NOT* be emitted to the object file. It is used
    // to pass the list of names referenced to codegen.
    new llvm::GlobalVariable(CGM.getModule(), NamesArrTy, true,
                             llvm::GlobalValue::InternalLinkage, NamesArrVal,
                             llvm::getCoverageUnusedNamesVarName());
  }
}

unsigned CoverageMappingModuleGen::getFileID(const FileEntry *File) {
  auto It = FileEntries.find(File);
  if (It != FileEntries.end())
    return It->second;
  unsigned FileID = FileEntries.size() + 1;
  FileEntries.insert(std::make_pair(File, FileID));
  return FileID;
}

void CoverageMappingGen::emitCounterMapping(const Decl *D,
                                            llvm::raw_ostream &OS) {
  assert(CounterMap);
  CounterCoverageMappingBuilder Walker(CVM, *CounterMap, SM, LangOpts);
  Walker.VisitDecl(D);
  Walker.write(OS);
}

void CoverageMappingGen::emitEmptyMapping(const Decl *D,
                                          llvm::raw_ostream &OS) {
  EmptyCoverageMappingBuilder Walker(CVM, SM, LangOpts);
  Walker.VisitDecl(D);
  Walker.write(OS);
}
