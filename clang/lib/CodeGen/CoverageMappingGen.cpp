//===--- CoverageMappingGen.cpp - Coverage mapping generation ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Instrumentation-based code coverage mapping generator
//
//===----------------------------------------------------------------------===//

#include "CoverageMappingGen.h"
#include "CodeGenFunction.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ProfileData/Coverage/CoverageMapping.h"
#include "llvm/ProfileData/Coverage/CoverageMappingReader.h"
#include "llvm/ProfileData/Coverage/CoverageMappingWriter.h"
#include "llvm/ProfileData/InstrProfReader.h"
#include "llvm/Support/FileSystem.h"

using namespace clang;
using namespace CodeGen;
using namespace llvm::coverage;

void CoverageSourceInfo::SourceRangeSkipped(SourceRange Range) {
  SkippedRanges.push_back(Range);
}

namespace {

/// \brief A region of source code that can be mapped to a counter.
class SourceMappingRegion {
  Counter Count;

  /// \brief The region's starting location.
  Optional<SourceLocation> LocStart;

  /// \brief The region's ending location.
  Optional<SourceLocation> LocEnd;

public:
  SourceMappingRegion(Counter Count, Optional<SourceLocation> LocStart,
                      Optional<SourceLocation> LocEnd)
      : Count(Count), LocStart(LocStart), LocEnd(LocEnd) {}

  const Counter &getCounter() const { return Count; }

  void setCounter(Counter C) { Count = C; }

  bool hasStartLoc() const { return LocStart.hasValue(); }

  void setStartLoc(SourceLocation Loc) { LocStart = Loc; }

  SourceLocation getStartLoc() const {
    assert(LocStart && "Region has no start location");
    return *LocStart;
  }

  bool hasEndLoc() const { return LocEnd.hasValue(); }

  void setEndLoc(SourceLocation Loc) { LocEnd = Loc; }

  SourceLocation getEndLoc() const {
    assert(LocEnd && "Region has no end location");
    return *LocEnd;
  }
};

/// \brief Provides the common functionality for the different
/// coverage mapping region builders.
class CoverageMappingBuilder {
public:
  CoverageMappingModuleGen &CVM;
  SourceManager &SM;
  const LangOptions &LangOpts;

private:
  /// \brief Map of clang's FileIDs to IDs used for coverage mapping.
  llvm::SmallDenseMap<FileID, std::pair<unsigned, SourceLocation>, 8>
      FileIDMapping;

public:
  /// \brief The coverage mapping regions for this function
  llvm::SmallVector<CounterMappingRegion, 32> MappingRegions;
  /// \brief The source mapping regions for this function.
  std::vector<SourceMappingRegion> SourceRegions;

  CoverageMappingBuilder(CoverageMappingModuleGen &CVM, SourceManager &SM,
                         const LangOptions &LangOpts)
      : CVM(CVM), SM(SM), LangOpts(LangOpts) {}

  /// \brief Return the precise end location for the given token.
  SourceLocation getPreciseTokenLocEnd(SourceLocation Loc) {
    // We avoid getLocForEndOfToken here, because it doesn't do what we want for
    // macro locations, which we just treat as expanded files.
    unsigned TokLen =
        Lexer::MeasureTokenLength(SM.getSpellingLoc(Loc), SM, LangOpts);
    return Loc.getLocWithOffset(TokLen);
  }

  /// \brief Return the start location of an included file or expanded macro.
  SourceLocation getStartOfFileOrMacro(SourceLocation Loc) {
    if (Loc.isMacroID())
      return Loc.getLocWithOffset(-SM.getFileOffset(Loc));
    return SM.getLocForStartOfFile(SM.getFileID(Loc));
  }

  /// \brief Return the end location of an included file or expanded macro.
  SourceLocation getEndOfFileOrMacro(SourceLocation Loc) {
    if (Loc.isMacroID())
      return Loc.getLocWithOffset(SM.getFileIDSize(SM.getFileID(Loc)) -
                                  SM.getFileOffset(Loc));
    return SM.getLocForEndOfFile(SM.getFileID(Loc));
  }

  /// \brief Find out where the current file is included or macro is expanded.
  SourceLocation getIncludeOrExpansionLoc(SourceLocation Loc) {
    return Loc.isMacroID() ? SM.getImmediateExpansionRange(Loc).first
                           : SM.getIncludeLoc(SM.getFileID(Loc));
  }

  /// \brief Return true if \c Loc is a location in a built-in macro.
  bool isInBuiltin(SourceLocation Loc) {
    return strcmp(SM.getBufferName(SM.getSpellingLoc(Loc)), "<built-in>") == 0;
  }

  /// \brief Check whether \c Loc is included or expanded from \c Parent.
  bool isNestedIn(SourceLocation Loc, FileID Parent) {
    do {
      Loc = getIncludeOrExpansionLoc(Loc);
      if (Loc.isInvalid())
        return false;
    } while (!SM.isInFileID(Loc, Parent));
    return true;
  }

  /// \brief Get the start of \c S ignoring macro arguments and builtin macros.
  SourceLocation getStart(const Stmt *S) {
    SourceLocation Loc = S->getLocStart();
    while (SM.isMacroArgExpansion(Loc) || isInBuiltin(Loc))
      Loc = SM.getImmediateExpansionRange(Loc).first;
    return Loc;
  }

  /// \brief Get the end of \c S ignoring macro arguments and builtin macros.
  SourceLocation getEnd(const Stmt *S) {
    SourceLocation Loc = S->getLocEnd();
    while (SM.isMacroArgExpansion(Loc) || isInBuiltin(Loc))
      Loc = SM.getImmediateExpansionRange(Loc).first;
    return getPreciseTokenLocEnd(Loc);
  }

  /// \brief Find the set of files we have regions for and assign IDs
  ///
  /// Fills \c Mapping with the virtual file mapping needed to write out
  /// coverage and collects the necessary file information to emit source and
  /// expansion regions.
  void gatherFileIDs(SmallVectorImpl<unsigned> &Mapping) {
    FileIDMapping.clear();

    llvm::SmallSet<FileID, 8> Visited;
    SmallVector<std::pair<SourceLocation, unsigned>, 8> FileLocs;
    for (const auto &Region : SourceRegions) {
      SourceLocation Loc = Region.getStartLoc();
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
    std::stable_sort(FileLocs.begin(), FileLocs.end(), llvm::less_second());

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

  /// \brief Get the coverage mapping file ID for \c Loc.
  ///
  /// If such file id doesn't exist, return None.
  Optional<unsigned> getCoverageFileID(SourceLocation Loc) {
    auto Mapping = FileIDMapping.find(SM.getFileID(Loc));
    if (Mapping != FileIDMapping.end())
      return Mapping->second.first;
    return None;
  }

  /// \brief Gather all the regions that were skipped by the preprocessor
  /// using the constructs like #if.
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
    for (const auto &I : SkippedRanges) {
      auto LocStart = I.getBegin();
      auto LocEnd = I.getEnd();
      assert(SM.isWrittenInSameFile(LocStart, LocEnd) &&
             "region spans multiple files");

      auto CovFileID = getCoverageFileID(LocStart);
      if (!CovFileID)
        continue;
      unsigned LineStart = SM.getSpellingLineNumber(LocStart);
      unsigned ColumnStart = SM.getSpellingColumnNumber(LocStart);
      unsigned LineEnd = SM.getSpellingLineNumber(LocEnd);
      unsigned ColumnEnd = SM.getSpellingColumnNumber(LocEnd);
      auto Region = CounterMappingRegion::makeSkipped(
          *CovFileID, LineStart, ColumnStart, LineEnd, ColumnEnd);
      // Make sure that we only collect the regions that are inside
      // the souce code of this function.
      if (Region.LineStart >= FileLineRanges[*CovFileID].first &&
          Region.LineEnd <= FileLineRanges[*CovFileID].second)
        MappingRegions.push_back(Region);
    }
  }

  /// \brief Generate the coverage counter mapping regions from collected
  /// source regions.
  void emitSourceRegions() {
    for (const auto &Region : SourceRegions) {
      assert(Region.hasEndLoc() && "incomplete region");

      SourceLocation LocStart = Region.getStartLoc();
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

      // Find the spilling locations for the mapping region.
      unsigned LineStart = SM.getSpellingLineNumber(LocStart);
      unsigned ColumnStart = SM.getSpellingColumnNumber(LocStart);
      unsigned LineEnd = SM.getSpellingLineNumber(LocEnd);
      unsigned ColumnEnd = SM.getSpellingColumnNumber(LocEnd);

      assert(LineStart <= LineEnd && "region start and end out of order");
      MappingRegions.push_back(CounterMappingRegion::makeRegion(
          Region.getCounter(), *CovFileID, LineStart, ColumnStart, LineEnd,
          ColumnEnd));
    }
  }

  /// \brief Generate expansion regions for each virtual file we've seen.
  void emitExpansionRegions() {
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

      unsigned LineStart = SM.getSpellingLineNumber(ParentLoc);
      unsigned ColumnStart = SM.getSpellingColumnNumber(ParentLoc);
      unsigned LineEnd = SM.getSpellingLineNumber(LocEnd);
      unsigned ColumnEnd = SM.getSpellingColumnNumber(LocEnd);

      MappingRegions.push_back(CounterMappingRegion::makeExpansion(
          *ParentFileID, *ExpandedFileID, LineStart, ColumnStart, LineEnd,
          ColumnEnd));
    }
  }
};

/// \brief Creates unreachable coverage regions for the functions that
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

  /// \brief Write the mapping data to the output stream
  void write(llvm::raw_ostream &OS) {
    SmallVector<unsigned, 16> FileIDMapping;
    gatherFileIDs(FileIDMapping);
    emitSourceRegions();

    CoverageMappingWriter Writer(FileIDMapping, None, MappingRegions);
    Writer.write(OS);
  }
};

/// \brief A StmtVisitor that creates coverage mapping regions which map
/// from the source code locations to the PGO counters.
struct CounterCoverageMappingBuilder
    : public CoverageMappingBuilder,
      public ConstStmtVisitor<CounterCoverageMappingBuilder> {
  /// \brief The map of statements to count values.
  llvm::DenseMap<const Stmt *, unsigned> &CounterMap;

  /// \brief A stack of currently live regions.
  std::vector<SourceMappingRegion> RegionStack;

  CounterExpressionBuilder Builder;

  /// \brief A location in the most recently visited file or macro.
  ///
  /// This is used to adjust the active source regions appropriately when
  /// expressions cross file or macro boundaries.
  SourceLocation MostRecentLocation;

  /// \brief Return a counter for the subtraction of \c RHS from \c LHS
  Counter subtractCounters(Counter LHS, Counter RHS) {
    return Builder.subtract(LHS, RHS);
  }

  /// \brief Return a counter for the sum of \c LHS and \c RHS.
  Counter addCounters(Counter LHS, Counter RHS) {
    return Builder.add(LHS, RHS);
  }

  Counter addCounters(Counter C1, Counter C2, Counter C3) {
    return addCounters(addCounters(C1, C2), C3);
  }

  /// \brief Return the region counter for the given statement.
  ///
  /// This should only be called on statements that have a dedicated counter.
  Counter getRegionCounter(const Stmt *S) {
    return Counter::getCounter(CounterMap[S]);
  }

  /// \brief Push a region onto the stack.
  ///
  /// Returns the index on the stack where the region was pushed. This can be
  /// used with popRegions to exit a "scope", ending the region that was pushed.
  size_t pushRegion(Counter Count, Optional<SourceLocation> StartLoc = None,
                    Optional<SourceLocation> EndLoc = None) {
    if (StartLoc)
      MostRecentLocation = *StartLoc;
    RegionStack.emplace_back(Count, StartLoc, EndLoc);

    return RegionStack.size() - 1;
  }

  /// \brief Pop regions from the stack into the function's list of regions.
  ///
  /// Adds all regions from \c ParentIndex to the top of the stack to the
  /// function's \c SourceRegions.
  void popRegions(size_t ParentIndex) {
    assert(RegionStack.size() >= ParentIndex && "parent not in stack");
    while (RegionStack.size() > ParentIndex) {
      SourceMappingRegion &Region = RegionStack.back();
      if (Region.hasStartLoc()) {
        SourceLocation StartLoc = Region.getStartLoc();
        SourceLocation EndLoc = Region.hasEndLoc()
                                    ? Region.getEndLoc()
                                    : RegionStack[ParentIndex].getEndLoc();
        while (!SM.isWrittenInSameFile(StartLoc, EndLoc)) {
          // The region ends in a nested file or macro expansion. Create a
          // separate region for each expansion.
          SourceLocation NestedLoc = getStartOfFileOrMacro(EndLoc);
          assert(SM.isWrittenInSameFile(NestedLoc, EndLoc));

          SourceRegions.emplace_back(Region.getCounter(), NestedLoc, EndLoc);

          EndLoc = getPreciseTokenLocEnd(getIncludeOrExpansionLoc(EndLoc));
          if (EndLoc.isInvalid())
            llvm::report_fatal_error("File exit not handled before popRegions");
        }
        Region.setEndLoc(EndLoc);

        MostRecentLocation = EndLoc;
        // If this region happens to span an entire expansion, we need to make
        // sure we don't overlap the parent region with it.
        if (StartLoc == getStartOfFileOrMacro(StartLoc) &&
            EndLoc == getEndOfFileOrMacro(EndLoc))
          MostRecentLocation = getIncludeOrExpansionLoc(EndLoc);

        assert(SM.isWrittenInSameFile(Region.getStartLoc(), EndLoc));
        SourceRegions.push_back(Region);
      }
      RegionStack.pop_back();
    }
  }

  /// \brief Return the currently active region.
  SourceMappingRegion &getRegion() {
    assert(!RegionStack.empty() && "statement has no region");
    return RegionStack.back();
  }

  /// \brief Propagate counts through the children of \c S.
  Counter propagateCounts(Counter TopCount, const Stmt *S) {
    size_t Index = pushRegion(TopCount, getStart(S), getEnd(S));
    Visit(S);
    Counter ExitCount = getRegion().getCounter();
    popRegions(Index);

    // The statement may be spanned by an expansion. Make sure we handle a file
    // exit out of this expansion before moving to the next statement.
    if (SM.isBeforeInTranslationUnit(getStart(S), S->getLocStart()))
      MostRecentLocation = getEnd(S);

    return ExitCount;
  }

  /// \brief Check whether a region with bounds \c StartLoc and \c EndLoc
  /// is already added to \c SourceRegions.
  bool isRegionAlreadyAdded(SourceLocation StartLoc, SourceLocation EndLoc) {
    return SourceRegions.rend() !=
           std::find_if(SourceRegions.rbegin(), SourceRegions.rend(),
                        [&](const SourceMappingRegion &Region) {
                          return Region.getStartLoc() == StartLoc &&
                                 Region.getEndLoc() == EndLoc;
                        });
  }

  /// \brief Adjust the most recently visited location to \c EndLoc.
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
                             MostRecentLocation))
      MostRecentLocation = getIncludeOrExpansionLoc(MostRecentLocation);
  }

  /// \brief Adjust regions and state when \c NewLoc exits a file.
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
      SourceLocation Loc = I.getStartLoc();
      if (!isNestedIn(Loc, ParentFile)) {
        ParentCounter = I.getCounter();
        break;
      }

      while (!SM.isInFileID(Loc, ParentFile)) {
        // The most nested region for each start location is the one with the
        // correct count. We avoid creating redundant regions by stopping once
        // we've seen this region.
        if (StartLocs.insert(Loc).second)
          SourceRegions.emplace_back(I.getCounter(), Loc,
                                     getEndOfFileOrMacro(Loc));
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
        if (StartLocs.insert(FileStart).second)
          SourceRegions.emplace_back(*ParentCounter, FileStart,
                                     getEndOfFileOrMacro(Loc));
        Loc = getIncludeOrExpansionLoc(Loc);
      }
    }

    MostRecentLocation = NewLoc;
  }

  /// \brief Ensure that \c S is included in the current region.
  void extendRegion(const Stmt *S) {
    SourceMappingRegion &Region = getRegion();
    SourceLocation StartLoc = getStart(S);

    handleFileExit(StartLoc);
    if (!Region.hasStartLoc())
      Region.setStartLoc(StartLoc);
  }

  /// \brief Mark \c S as a terminator, starting a zero region.
  void terminateRegion(const Stmt *S) {
    extendRegion(S);
    SourceMappingRegion &Region = getRegion();
    if (!Region.hasEndLoc())
      Region.setEndLoc(getEnd(S));
    pushRegion(Counter::getZero());
  }

  /// \brief Keep counts of breaks and continues inside loops.
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

  /// \brief Write the mapping data to the output stream
  void write(llvm::raw_ostream &OS) {
    llvm::SmallVector<unsigned, 8> VirtualFileMapping;
    gatherFileIDs(VirtualFileMapping);
    emitSourceRegions();
    emitExpansionRegions();
    gatherSkippedRegions();

    CoverageMappingWriter Writer(VirtualFileMapping, Builder.getExpressions(),
                                 MappingRegions);
    Writer.write(OS);
  }

  void VisitStmt(const Stmt *S) {
    if (S->getLocStart().isValid())
      extendRegion(S);
    for (const Stmt *Child : S->children())
      if (Child)
        this->Visit(Child);
    handleFileExit(getEnd(S));
  }

  void VisitDecl(const Decl *D) {
    Stmt *Body = D->getBody();
    propagateCounts(getRegionCounter(Body), Body);
  }

  void VisitReturnStmt(const ReturnStmt *S) {
    extendRegion(S);
    if (S->getRetValue())
      Visit(S->getRetValue());
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
    SourceLocation Start = getStart(S);
    // We can't extendRegion here or we risk overlapping with our new region.
    handleFileExit(Start);
    pushRegion(getRegionCounter(S), Start);
    Visit(S->getSubStmt());
  }

  void VisitBreakStmt(const BreakStmt *S) {
    assert(!BreakContinueStack.empty() && "break not in a loop or switch!");
    BreakContinueStack.back().BreakCount = addCounters(
        BreakContinueStack.back().BreakCount, getRegion().getCounter());
    terminateRegion(S);
  }

  void VisitContinueStmt(const ContinueStmt *S) {
    assert(!BreakContinueStack.empty() && "continue stmt not in a loop!");
    BreakContinueStack.back().ContinueCount = addCounters(
        BreakContinueStack.back().ContinueCount, getRegion().getCounter());
    terminateRegion(S);
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

    // Go back to handle the condition.
    Counter CondCount =
        addCounters(ParentCount, BackedgeCount, BC.ContinueCount);
    propagateCounts(CondCount, S->getCond());
    adjustForOutOfOrderTraversal(getEnd(S));

    Counter OutCount =
        addCounters(BC.BreakCount, subtractCounters(CondCount, BodyCount));
    if (OutCount != ParentCount)
      pushRegion(OutCount);
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

    Counter CondCount = addCounters(BackedgeCount, BC.ContinueCount);
    propagateCounts(CondCount, S->getCond());

    Counter OutCount =
        addCounters(BC.BreakCount, subtractCounters(CondCount, BodyCount));
    if (OutCount != ParentCount)
      pushRegion(OutCount);
  }

  void VisitForStmt(const ForStmt *S) {
    extendRegion(S);
    if (S->getInit())
      Visit(S->getInit());

    Counter ParentCount = getRegion().getCounter();
    Counter BodyCount = getRegionCounter(S);

    // Handle the body first so that we can get the backedge count.
    BreakContinueStack.push_back(BreakContinue());
    extendRegion(S->getBody());
    Counter BackedgeCount = propagateCounts(BodyCount, S->getBody());
    BreakContinue BC = BreakContinueStack.pop_back_val();

    // The increment is essentially part of the body but it needs to include
    // the count for all the continue statements.
    if (const Stmt *Inc = S->getInc())
      propagateCounts(addCounters(BackedgeCount, BC.ContinueCount), Inc);

    // Go back to handle the condition.
    Counter CondCount =
        addCounters(ParentCount, BackedgeCount, BC.ContinueCount);
    if (const Expr *Cond = S->getCond()) {
      propagateCounts(CondCount, Cond);
      adjustForOutOfOrderTraversal(getEnd(S));
    }

    Counter OutCount =
        addCounters(BC.BreakCount, subtractCounters(CondCount, BodyCount));
    if (OutCount != ParentCount)
      pushRegion(OutCount);
  }

  void VisitCXXForRangeStmt(const CXXForRangeStmt *S) {
    extendRegion(S);
    Visit(S->getLoopVarStmt());
    Visit(S->getRangeStmt());

    Counter ParentCount = getRegion().getCounter();
    Counter BodyCount = getRegionCounter(S);

    BreakContinueStack.push_back(BreakContinue());
    extendRegion(S->getBody());
    Counter BackedgeCount = propagateCounts(BodyCount, S->getBody());
    BreakContinue BC = BreakContinueStack.pop_back_val();

    Counter LoopCount =
        addCounters(ParentCount, BackedgeCount, BC.ContinueCount);
    Counter OutCount =
        addCounters(BC.BreakCount, subtractCounters(LoopCount, BodyCount));
    if (OutCount != ParentCount)
      pushRegion(OutCount);
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

    Counter LoopCount =
        addCounters(ParentCount, BackedgeCount, BC.ContinueCount);
    Counter OutCount =
        addCounters(BC.BreakCount, subtractCounters(LoopCount, BodyCount));
    if (OutCount != ParentCount)
      pushRegion(OutCount);
  }

  void VisitSwitchStmt(const SwitchStmt *S) {
    extendRegion(S);
    Visit(S->getCond());

    BreakContinueStack.push_back(BreakContinue());

    const Stmt *Body = S->getBody();
    extendRegion(Body);
    if (const auto *CS = dyn_cast<CompoundStmt>(Body)) {
      if (!CS->body_empty()) {
        // The body of the switch needs a zero region so that fallthrough counts
        // behave correctly, but it would be misleading to include the braces of
        // the compound statement in the zeroed area, so we need to handle this
        // specially.
        size_t Index =
            pushRegion(Counter::getZero(), getStart(CS->body_front()),
                       getEnd(CS->body_back()));
        for (const auto *Child : CS->children())
          Visit(Child);
        popRegions(Index);
      }
    } else
      propagateCounts(Counter::getZero(), Body);
    BreakContinue BC = BreakContinueStack.pop_back_val();

    if (!BreakContinueStack.empty())
      BreakContinueStack.back().ContinueCount = addCounters(
          BreakContinueStack.back().ContinueCount, BC.ContinueCount);

    Counter ExitCount = getRegionCounter(S);
    SourceLocation ExitLoc = getEnd(S);
    pushRegion(ExitCount, getStart(S), ExitLoc);
    handleFileExit(ExitLoc);
  }

  void VisitSwitchCase(const SwitchCase *S) {
    extendRegion(S);

    SourceMappingRegion &Parent = getRegion();

    Counter Count = addCounters(Parent.getCounter(), getRegionCounter(S));
    // Reuse the existing region if it starts at our label. This is typical of
    // the first case in a switch.
    if (Parent.hasStartLoc() && Parent.getStartLoc() == getStart(S))
      Parent.setCounter(Count);
    else
      pushRegion(Count, getStart(S));

    if (const auto *CS = dyn_cast<CaseStmt>(S)) {
      Visit(CS->getLHS());
      if (const Expr *RHS = CS->getRHS())
        Visit(RHS);
    }
    Visit(S->getSubStmt());
  }

  void VisitIfStmt(const IfStmt *S) {
    extendRegion(S);
    // Extend into the condition before we propagate through it below - this is
    // needed to handle macros that generate the "if" but not the condition.
    extendRegion(S->getCond());

    Counter ParentCount = getRegion().getCounter();
    Counter ThenCount = getRegionCounter(S);

    // Emitting a counter for the condition makes it easier to interpret the
    // counter for the body when looking at the coverage.
    propagateCounts(ParentCount, S->getCond());

    extendRegion(S->getThen());
    Counter OutCount = propagateCounts(ThenCount, S->getThen());

    Counter ElseCount = subtractCounters(ParentCount, ThenCount);
    if (const Stmt *Else = S->getElse()) {
      extendRegion(S->getElse());
      OutCount = addCounters(OutCount, propagateCounts(ElseCount, Else));
    } else
      OutCount = addCounters(OutCount, ElseCount);

    if (OutCount != ParentCount)
      pushRegion(OutCount);
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

    Visit(E->getCond());

    if (!isa<BinaryConditionalOperator>(E)) {
      extendRegion(E->getTrueExpr());
      propagateCounts(TrueCount, E->getTrueExpr());
    }
    extendRegion(E->getFalseExpr());
    propagateCounts(subtractCounters(ParentCount, TrueCount),
                    E->getFalseExpr());
  }

  void VisitBinLAnd(const BinaryOperator *E) {
    extendRegion(E);
    Visit(E->getLHS());

    extendRegion(E->getRHS());
    propagateCounts(getRegionCounter(E), E->getRHS());
  }

  void VisitBinLOr(const BinaryOperator *E) {
    extendRegion(E);
    Visit(E->getLHS());

    extendRegion(E->getRHS());
    propagateCounts(getRegionCounter(E), E->getRHS());
  }

  void VisitLambdaExpr(const LambdaExpr *LE) {
    // Lambdas are treated as their own functions for now, so we shouldn't
    // propagate counts into them.
  }
};
}

static bool isMachO(const CodeGenModule &CGM) {
  return CGM.getTarget().getTriple().isOSBinFormatMachO();
}

static StringRef getCoverageSection(const CodeGenModule &CGM) {
  return llvm::getInstrProfCoverageSectionName(isMachO(CGM));
}

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
    }

    OS << "File " << R.FileID << ", " << R.LineStart << ":" << R.ColumnStart
       << " -> " << R.LineEnd << ":" << R.ColumnEnd << " = ";
    Ctx.dump(R.Count, OS);
    if (R.Kind == CounterMappingRegion::ExpansionRegion)
      OS << " (Expanded file = " << R.ExpandedFileID << ")";
    OS << "\n";
  }
}

void CoverageMappingModuleGen::addFunctionMappingRecord(
    llvm::GlobalVariable *NamePtr, StringRef NameValue, uint64_t FuncHash,
    const std::string &CoverageMapping, bool IsUsed) {
  llvm::LLVMContext &Ctx = CGM.getLLVMContext();
  if (!FunctionRecordTy) {
#define COVMAP_FUNC_RECORD(Type, LLVMType, Name, Init) LLVMType,
    llvm::Type *FunctionRecordTypes[] = {
      #include "llvm/ProfileData/InstrProfData.inc"
    };
    FunctionRecordTy =
        llvm::StructType::get(Ctx, makeArrayRef(FunctionRecordTypes),
                              /*isPacked=*/true);
  }

  #define COVMAP_FUNC_RECORD(Type, LLVMType, Name, Init) Init,
  llvm::Constant *FunctionRecordVals[] = {
      #include "llvm/ProfileData/InstrProfData.inc"
  };
  FunctionRecords.push_back(llvm::ConstantStruct::get(
      FunctionRecordTy, makeArrayRef(FunctionRecordVals)));
  if (!IsUsed)
    FunctionNames.push_back(
        llvm::ConstantExpr::getBitCast(NamePtr, llvm::Type::getInt8PtrTy(Ctx)));
  CoverageMappings.push_back(CoverageMapping);

  if (CGM.getCodeGenOpts().DumpCoverageMapping) {
    // Dump the coverage mapping data for this function by decoding the
    // encoded data. This allows us to dump the mapping regions which were
    // also processed by the CoverageMappingWriter which performs
    // additional minimization operations such as reducing the number of
    // expressions.
    std::vector<StringRef> Filenames;
    std::vector<CounterExpression> Expressions;
    std::vector<CounterMappingRegion> Regions;
    llvm::SmallVector<StringRef, 16> FilenameRefs;
    FilenameRefs.resize(FileEntries.size());
    for (const auto &Entry : FileEntries)
      FilenameRefs[Entry.second] = Entry.first->getName();
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
  llvm::SmallVector<StringRef, 16> FilenameRefs;
  FilenameStrs.resize(FileEntries.size());
  FilenameRefs.resize(FileEntries.size());
  for (const auto &Entry : FileEntries) {
    llvm::SmallString<256> Path(Entry.first->getName());
    llvm::sys::fs::make_absolute(Path);

    auto I = Entry.second;
    FilenameStrs[I] = std::string(Path.begin(), Path.end());
    FilenameRefs[I] = FilenameStrs[I];
  }

  std::string FilenamesAndCoverageMappings;
  llvm::raw_string_ostream OS(FilenamesAndCoverageMappings);
  CoverageFilenamesSectionWriter(FilenameRefs).write(OS);
  std::string RawCoverageMappings =
      llvm::join(CoverageMappings.begin(), CoverageMappings.end(), "");
  OS << RawCoverageMappings;
  size_t CoverageMappingSize = RawCoverageMappings.size();
  size_t FilenamesSize = OS.str().size() - CoverageMappingSize;
  // Append extra zeroes if necessary to ensure that the size of the filenames
  // and coverage mappings is a multiple of 8.
  if (size_t Rem = OS.str().size() % 8) {
    CoverageMappingSize += 8 - Rem;
    for (size_t I = 0, S = 8 - Rem; I < S; ++I)
      OS << '\0';
  }
  auto *FilenamesAndMappingsVal =
      llvm::ConstantDataArray::getString(Ctx, OS.str(), false);

  // Create the deferred function records array
  auto RecordsTy =
      llvm::ArrayType::get(FunctionRecordTy, FunctionRecords.size());
  auto RecordsVal = llvm::ConstantArray::get(RecordsTy, FunctionRecords);

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
  llvm::Type *CovDataTypes[] = {CovDataHeaderTy, RecordsTy,
                                FilenamesAndMappingsVal->getType()};
  auto CovDataTy = llvm::StructType::get(Ctx, makeArrayRef(CovDataTypes));
  llvm::Constant *TUDataVals[] = {CovDataHeaderVal, RecordsVal,
                                  FilenamesAndMappingsVal};
  auto CovDataVal =
      llvm::ConstantStruct::get(CovDataTy, makeArrayRef(TUDataVals));
  auto CovData = new llvm::GlobalVariable(
      CGM.getModule(), CovDataTy, true, llvm::GlobalValue::InternalLinkage,
      CovDataVal, llvm::getCoverageMappingVarName());

  CovData->setSection(getCoverageSection(CGM));
  CovData->setAlignment(8);

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
  unsigned FileID = FileEntries.size();
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
