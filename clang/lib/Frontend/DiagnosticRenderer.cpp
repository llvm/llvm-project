//===--- DiagnosticRenderer.cpp - Diagnostic Pretty-Printing --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/DiagnosticRenderer.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Edit/Commit.h"
#include "clang/Edit/EditedSource.h"
#include "clang/Edit/EditsReceiver.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
using namespace clang;

DiagnosticRenderer::DiagnosticRenderer(const LangOptions &LangOpts,
                                       DiagnosticOptions *DiagOpts)
  : LangOpts(LangOpts), DiagOpts(DiagOpts), LastLevel() {}

DiagnosticRenderer::~DiagnosticRenderer() {}

namespace {

class FixitReceiver : public edit::EditsReceiver {
  SmallVectorImpl<FixItHint> &MergedFixits;

public:
  FixitReceiver(SmallVectorImpl<FixItHint> &MergedFixits)
    : MergedFixits(MergedFixits) { }
  void insert(SourceLocation loc, StringRef text) override {
    MergedFixits.push_back(FixItHint::CreateInsertion(loc, text));
  }
  void replace(CharSourceRange range, StringRef text) override {
    MergedFixits.push_back(FixItHint::CreateReplacement(range, text));
  }
};

}

static void mergeFixits(ArrayRef<FixItHint> FixItHints,
                        const SourceManager &SM, const LangOptions &LangOpts,
                        SmallVectorImpl<FixItHint> &MergedFixits) {
  edit::Commit commit(SM, LangOpts);
  for (ArrayRef<FixItHint>::const_iterator
         I = FixItHints.begin(), E = FixItHints.end(); I != E; ++I) {
    const FixItHint &Hint = *I;
    if (Hint.CodeToInsert.empty()) {
      if (Hint.InsertFromRange.isValid())
        commit.insertFromRange(Hint.RemoveRange.getBegin(),
                           Hint.InsertFromRange, /*afterToken=*/false,
                           Hint.BeforePreviousInsertions);
      else
        commit.remove(Hint.RemoveRange);
    } else {
      if (Hint.RemoveRange.isTokenRange() ||
          Hint.RemoveRange.getBegin() != Hint.RemoveRange.getEnd())
        commit.replace(Hint.RemoveRange, Hint.CodeToInsert);
      else
        commit.insert(Hint.RemoveRange.getBegin(), Hint.CodeToInsert,
                    /*afterToken=*/false, Hint.BeforePreviousInsertions);
    }
  }

  edit::EditedSource Editor(SM, LangOpts);
  if (Editor.commit(commit)) {
    FixitReceiver Rec(MergedFixits);
    Editor.applyRewrites(Rec);
  }
}

void DiagnosticRenderer::emitDiagnostic(SourceLocation Loc,
                                        DiagnosticsEngine::Level Level,
                                        StringRef Message,
                                        ArrayRef<CharSourceRange> Ranges,
                                        ArrayRef<FixItHint> FixItHints,
                                        const SourceManager *SM,
                                        DiagOrStoredDiag D) {
  assert(SM || Loc.isInvalid());

  beginDiagnostic(D, Level);

  if (!Loc.isValid())
    // If we have no source location, just emit the diagnostic message.
    emitDiagnosticMessage(Loc, PresumedLoc(), Level, Message, Ranges, SM, D);
  else {
    // Get the ranges into a local array we can hack on.
    SmallVector<CharSourceRange, 20> MutableRanges(Ranges.begin(),
                                                   Ranges.end());

    SmallVector<FixItHint, 8> MergedFixits;
    if (!FixItHints.empty()) {
      mergeFixits(FixItHints, *SM, LangOpts, MergedFixits);
      FixItHints = MergedFixits;
    }

    for (ArrayRef<FixItHint>::const_iterator I = FixItHints.begin(),
         E = FixItHints.end();
         I != E; ++I)
      if (I->RemoveRange.isValid())
        MutableRanges.push_back(I->RemoveRange);

    SourceLocation UnexpandedLoc = Loc;

    // Find the ultimate expansion location for the diagnostic.
    Loc = SM->getFileLoc(Loc);

    PresumedLoc PLoc = SM->getPresumedLoc(Loc, DiagOpts->ShowPresumedLoc);

    // First, if this diagnostic is not in the main file, print out the
    // "included from" lines.
    emitIncludeStack(Loc, PLoc, Level, *SM);

    // Next, emit the actual diagnostic message and caret.
    emitDiagnosticMessage(Loc, PLoc, Level, Message, Ranges, SM, D);
    emitCaret(Loc, Level, MutableRanges, FixItHints, *SM);

    // If this location is within a macro, walk from UnexpandedLoc up to Loc
    // and produce a macro backtrace.
    if (UnexpandedLoc.isValid() && UnexpandedLoc.isMacroID()) {
      emitMacroExpansions(UnexpandedLoc, Level, MutableRanges, FixItHints, *SM);
    }
  }

  LastLoc = Loc;
  LastLevel = Level;

  endDiagnostic(D, Level);
}


void DiagnosticRenderer::emitStoredDiagnostic(StoredDiagnostic &Diag) {
  emitDiagnostic(Diag.getLocation(), Diag.getLevel(), Diag.getMessage(),
                 Diag.getRanges(), Diag.getFixIts(),
                 Diag.getLocation().isValid() ? &Diag.getLocation().getManager()
                                              : nullptr,
                 &Diag);
}

void DiagnosticRenderer::emitBasicNote(StringRef Message) {
  emitDiagnosticMessage(
      SourceLocation(), PresumedLoc(), DiagnosticsEngine::Note, Message,
      None, nullptr, DiagOrStoredDiag());
}

/// \brief Prints an include stack when appropriate for a particular
/// diagnostic level and location.
///
/// This routine handles all the logic of suppressing particular include
/// stacks (such as those for notes) and duplicate include stacks when
/// repeated warnings occur within the same file. It also handles the logic
/// of customizing the formatting and display of the include stack.
///
/// \param Loc   The diagnostic location.
/// \param PLoc  The presumed location of the diagnostic location.
/// \param Level The diagnostic level of the message this stack pertains to.
void DiagnosticRenderer::emitIncludeStack(SourceLocation Loc,
                                          PresumedLoc PLoc,
                                          DiagnosticsEngine::Level Level,
                                          const SourceManager &SM) {
  SourceLocation IncludeLoc = PLoc.getIncludeLoc();

  // Skip redundant include stacks altogether.
  if (LastIncludeLoc == IncludeLoc)
    return;
  
  LastIncludeLoc = IncludeLoc;
  
  if (!DiagOpts->ShowNoteIncludeStack && Level == DiagnosticsEngine::Note)
    return;

  if (IncludeLoc.isValid())
    emitIncludeStackRecursively(IncludeLoc, SM);
  else {
    emitModuleBuildStack(SM);
    emitImportStack(Loc, SM);
  }
}

/// \brief Helper to recursivly walk up the include stack and print each layer
/// on the way back down.
void DiagnosticRenderer::emitIncludeStackRecursively(SourceLocation Loc,
                                                     const SourceManager &SM) {
  if (Loc.isInvalid()) {
    emitModuleBuildStack(SM);
    return;
  }
  
  PresumedLoc PLoc = SM.getPresumedLoc(Loc, DiagOpts->ShowPresumedLoc);
  if (PLoc.isInvalid())
    return;

  // If this source location was imported from a module, print the module
  // import stack rather than the 
  // FIXME: We want submodule granularity here.
  std::pair<SourceLocation, StringRef> Imported = SM.getModuleImportLoc(Loc);
  if (!Imported.second.empty()) {
    // This location was imported by a module. Emit the module import stack.
    emitImportStackRecursively(Imported.first, Imported.second, SM);
    return;
  }

  // Emit the other include frames first.
  emitIncludeStackRecursively(PLoc.getIncludeLoc(), SM);
  
  // Emit the inclusion text/note.
  emitIncludeLocation(Loc, PLoc, SM);
}

/// \brief Emit the module import stack associated with the current location.
void DiagnosticRenderer::emitImportStack(SourceLocation Loc,
                                         const SourceManager &SM) {
  if (Loc.isInvalid()) {
    emitModuleBuildStack(SM);
    return;
  }

  std::pair<SourceLocation, StringRef> NextImportLoc
    = SM.getModuleImportLoc(Loc);
  emitImportStackRecursively(NextImportLoc.first, NextImportLoc.second, SM);
}

/// \brief Helper to recursivly walk up the import stack and print each layer
/// on the way back down.
void DiagnosticRenderer::emitImportStackRecursively(SourceLocation Loc,
                                                    StringRef ModuleName,
                                                    const SourceManager &SM) {
  if (ModuleName.empty()) {
    return;
  }

  PresumedLoc PLoc = SM.getPresumedLoc(Loc, DiagOpts->ShowPresumedLoc);

  // Emit the other import frames first.
  std::pair<SourceLocation, StringRef> NextImportLoc
    = SM.getModuleImportLoc(Loc);
  emitImportStackRecursively(NextImportLoc.first, NextImportLoc.second, SM);

  // Emit the inclusion text/note.
  emitImportLocation(Loc, PLoc, ModuleName, SM);
}

/// \brief Emit the module build stack, for cases where a module is (re-)built
/// on demand.
void DiagnosticRenderer::emitModuleBuildStack(const SourceManager &SM) {
  ModuleBuildStack Stack = SM.getModuleBuildStack();
  for (unsigned I = 0, N = Stack.size(); I != N; ++I) {
    const SourceManager &CurSM = Stack[I].second.getManager();
    SourceLocation CurLoc = Stack[I].second;
    emitBuildingModuleLocation(CurLoc,
                               CurSM.getPresumedLoc(CurLoc,
                                                    DiagOpts->ShowPresumedLoc),
                               Stack[I].first,
                               CurSM);
  }
}

/// A recursive function to trace all possible backtrace locations
/// to match the \p CaretLocFileID.
static SourceLocation
retrieveMacroLocation(SourceLocation Loc, FileID MacroFileID,
                      FileID CaretFileID,
                      const SmallVectorImpl<FileID> &CommonArgExpansions,
                      bool IsBegin, const SourceManager *SM) {
  assert(SM->getFileID(Loc) == MacroFileID);
  if (MacroFileID == CaretFileID)
    return Loc;
  if (!Loc.isMacroID())
    return SourceLocation();

  SourceLocation MacroLocation, MacroArgLocation;

  if (SM->isMacroArgExpansion(Loc)) {
    // Only look at the immediate spelling location of this macro argument if
    // the other location in the source range is also present in that expansion.
    if (std::binary_search(CommonArgExpansions.begin(),
                           CommonArgExpansions.end(), MacroFileID))
      MacroLocation = SM->getImmediateSpellingLoc(Loc);
    MacroArgLocation = IsBegin ? SM->getImmediateExpansionRange(Loc).first
                               : SM->getImmediateExpansionRange(Loc).second;
  } else {
    MacroLocation = IsBegin ? SM->getImmediateExpansionRange(Loc).first
                            : SM->getImmediateExpansionRange(Loc).second;
    MacroArgLocation = SM->getImmediateSpellingLoc(Loc);
  }

  if (MacroLocation.isValid()) {
    MacroFileID = SM->getFileID(MacroLocation);
    MacroLocation =
        retrieveMacroLocation(MacroLocation, MacroFileID, CaretFileID,
                              CommonArgExpansions, IsBegin, SM);
    if (MacroLocation.isValid())
      return MacroLocation;
  }

  MacroFileID = SM->getFileID(MacroArgLocation);
  return retrieveMacroLocation(MacroArgLocation, MacroFileID, CaretFileID,
                               CommonArgExpansions, IsBegin, SM);
}

/// Walk up the chain of macro expansions and collect the FileIDs identifying the
/// expansions.
static void getMacroArgExpansionFileIDs(SourceLocation Loc,
                                        SmallVectorImpl<FileID> &IDs,
                                        bool IsBegin, const SourceManager *SM) {
  while (Loc.isMacroID()) {
    if (SM->isMacroArgExpansion(Loc)) {
      IDs.push_back(SM->getFileID(Loc));
      Loc = SM->getImmediateSpellingLoc(Loc);
    } else {
      auto ExpRange = SM->getImmediateExpansionRange(Loc);
      Loc = IsBegin ? ExpRange.first : ExpRange.second;
    }
  }
}

/// Collect the expansions of the begin and end locations and compute the set
/// intersection. Produces a sorted vector of FileIDs in CommonArgExpansions.
static void computeCommonMacroArgExpansionFileIDs(
    SourceLocation Begin, SourceLocation End, const SourceManager *SM,
    SmallVectorImpl<FileID> &CommonArgExpansions) {
  SmallVector<FileID, 4> BeginArgExpansions;
  SmallVector<FileID, 4> EndArgExpansions;
  getMacroArgExpansionFileIDs(Begin, BeginArgExpansions, /*IsBegin=*/true, SM);
  getMacroArgExpansionFileIDs(End, EndArgExpansions, /*IsBegin=*/false, SM);
  std::sort(BeginArgExpansions.begin(), BeginArgExpansions.end());
  std::sort(EndArgExpansions.begin(), EndArgExpansions.end());
  std::set_intersection(BeginArgExpansions.begin(), BeginArgExpansions.end(),
                        EndArgExpansions.begin(), EndArgExpansions.end(),
                        std::back_inserter(CommonArgExpansions));
}

// Helper function to fix up source ranges.  It takes in an array of ranges,
// and outputs an array of ranges where we want to draw the range highlighting
// around the location specified by CaretLoc.
//
// To find locations which correspond to the caret, we crawl the macro caller
// chain for the beginning and end of each range.  If the caret location
// is in a macro expansion, we search each chain for a location
// in the same expansion as the caret; otherwise, we crawl to the top of
// each chain. Two locations are part of the same macro expansion
// iff the FileID is the same.
static void mapDiagnosticRanges(
    SourceLocation CaretLoc,
    ArrayRef<CharSourceRange> Ranges,
    SmallVectorImpl<CharSourceRange> &SpellingRanges,
    const SourceManager *SM) {
  FileID CaretLocFileID = SM->getFileID(CaretLoc);

  for (auto I = Ranges.begin(), E = Ranges.end(); I != E; ++I) {
    if (I->isInvalid()) continue;

    SourceLocation Begin = I->getBegin(), End = I->getEnd();
    bool IsTokenRange = I->isTokenRange();

    FileID BeginFileID = SM->getFileID(Begin);
    FileID EndFileID = SM->getFileID(End);

    // Find the common parent for the beginning and end of the range.

    // First, crawl the expansion chain for the beginning of the range.
    llvm::SmallDenseMap<FileID, SourceLocation> BeginLocsMap;
    while (Begin.isMacroID() && BeginFileID != EndFileID) {
      BeginLocsMap[BeginFileID] = Begin;
      Begin = SM->getImmediateExpansionRange(Begin).first;
      BeginFileID = SM->getFileID(Begin);
    }

    // Then, crawl the expansion chain for the end of the range.
    if (BeginFileID != EndFileID) {
      while (End.isMacroID() && !BeginLocsMap.count(EndFileID)) {
        End = SM->getImmediateExpansionRange(End).second;
        EndFileID = SM->getFileID(End);
      }
      if (End.isMacroID()) {
        Begin = BeginLocsMap[EndFileID];
        BeginFileID = EndFileID;
      }
    }

    // Do the backtracking.
    SmallVector<FileID, 4> CommonArgExpansions;
    computeCommonMacroArgExpansionFileIDs(Begin, End, SM, CommonArgExpansions);
    Begin = retrieveMacroLocation(Begin, BeginFileID, CaretLocFileID,
                                  CommonArgExpansions, /*IsBegin=*/true, SM);
    End = retrieveMacroLocation(End, BeginFileID, CaretLocFileID,
                                CommonArgExpansions, /*IsBegin=*/false, SM);
    if (Begin.isInvalid() || End.isInvalid()) continue;

    // Return the spelling location of the beginning and end of the range.
    Begin = SM->getSpellingLoc(Begin);
    End = SM->getSpellingLoc(End);

    SpellingRanges.push_back(CharSourceRange(SourceRange(Begin, End),
                                             IsTokenRange));
  }
}

void DiagnosticRenderer::emitCaret(SourceLocation Loc,
                                   DiagnosticsEngine::Level Level,
                                   ArrayRef<CharSourceRange> Ranges,
                                   ArrayRef<FixItHint> Hints,
                                   const SourceManager &SM) {
  SmallVector<CharSourceRange, 4> SpellingRanges;
  mapDiagnosticRanges(Loc, Ranges, SpellingRanges, &SM);
  emitCodeContext(Loc, Level, SpellingRanges, Hints, SM);
}

/// \brief A helper function for emitMacroExpansion to print the
/// macro expansion message
void DiagnosticRenderer::emitSingleMacroExpansion(
    SourceLocation Loc,
    DiagnosticsEngine::Level Level,
    ArrayRef<CharSourceRange> Ranges,
    const SourceManager &SM) {
  // Find the spelling location for the macro definition. We must use the
  // spelling location here to avoid emitting a macro backtrace for the note.
  SourceLocation SpellingLoc = SM.getSpellingLoc(Loc);

  // Map the ranges into the FileID of the diagnostic location.
  SmallVector<CharSourceRange, 4> SpellingRanges;
  mapDiagnosticRanges(Loc, Ranges, SpellingRanges, &SM);

  SmallString<100> MessageStorage;
  llvm::raw_svector_ostream Message(MessageStorage);
  StringRef MacroName =
      Lexer::getImmediateMacroNameForDiagnostics(Loc, SM, LangOpts);
  if (MacroName.empty())
    Message << "expanded from here";
  else
    Message << "expanded from macro '" << MacroName << "'";

  emitDiagnostic(SpellingLoc, DiagnosticsEngine::Note, Message.str(),
                 SpellingRanges, None, &SM);
}

/// Check that the macro argument location of Loc starts with ArgumentLoc.
/// The starting location of the macro expansions is used to differeniate
/// different macro expansions.
static bool checkLocForMacroArgExpansion(SourceLocation Loc,
                                         const SourceManager &SM,
                                         SourceLocation ArgumentLoc) {
  SourceLocation MacroLoc;
  if (SM.isMacroArgExpansion(Loc, &MacroLoc)) {
    if (ArgumentLoc == MacroLoc) return true;
  }

  return false;
}

/// Check if all the locations in the range have the same macro argument
/// expansion, and that that expansion starts with ArgumentLoc.
static bool checkRangeForMacroArgExpansion(CharSourceRange Range,
                                           const SourceManager &SM,
                                           SourceLocation ArgumentLoc) {
  SourceLocation BegLoc = Range.getBegin(), EndLoc = Range.getEnd();
  while (BegLoc != EndLoc) {
    if (!checkLocForMacroArgExpansion(BegLoc, SM, ArgumentLoc))
      return false;
    BegLoc.getLocWithOffset(1);
  }

  return checkLocForMacroArgExpansion(BegLoc, SM, ArgumentLoc);
}

/// A helper function to check if the current ranges are all inside the same
/// macro argument expansion as Loc.
static bool checkRangesForMacroArgExpansion(SourceLocation Loc,
                                            ArrayRef<CharSourceRange> Ranges,
                                            const SourceManager &SM) {
  assert(Loc.isMacroID() && "Must be a macro expansion!");

  SmallVector<CharSourceRange, 4> SpellingRanges;
  mapDiagnosticRanges(Loc, Ranges, SpellingRanges, &SM);

  /// Count all valid ranges.
  unsigned ValidCount = 0;
  for (auto I : Ranges)
    if (I.isValid()) ValidCount++;

  if (ValidCount > SpellingRanges.size())
    return false;

  /// To store the source location of the argument location.
  SourceLocation ArgumentLoc;

  /// Set the ArgumentLoc to the beginning location of the expansion of Loc
  /// so to check if the ranges expands to the same beginning location.
  if (!SM.isMacroArgExpansion(Loc,&ArgumentLoc))
    return false;

  for (auto I = SpellingRanges.begin(), E = SpellingRanges.end(); I != E; ++I) {
    if (!checkRangeForMacroArgExpansion(*I, SM, ArgumentLoc))
      return false;
  }

  return true;
}

/// \brief Recursively emit notes for each macro expansion and caret
/// diagnostics where appropriate.
///
/// Walks up the macro expansion stack printing expansion notes, the code
/// snippet, caret, underlines and FixItHint display as appropriate at each
/// level.
///
/// \param Loc The location for this caret.
/// \param Level The diagnostic level currently being emitted.
/// \param Ranges The underlined ranges for this code snippet.
/// \param Hints The FixIt hints active for this diagnostic.
void DiagnosticRenderer::emitMacroExpansions(SourceLocation Loc,
                                             DiagnosticsEngine::Level Level,
                                             ArrayRef<CharSourceRange> Ranges,
                                             ArrayRef<FixItHint> Hints,
                                             const SourceManager &SM) {
  assert(Loc.isValid() && "must have a valid source location here");

  // Produce a stack of macro backtraces.
  SmallVector<SourceLocation, 8> LocationStack;
  unsigned IgnoredEnd = 0;
  while (Loc.isMacroID()) {
    // If this is the expansion of a macro argument, point the caret at the
    // use of the argument in the definition of the macro, not the expansion.
    if (SM.isMacroArgExpansion(Loc))
      LocationStack.push_back(SM.getImmediateExpansionRange(Loc).first);
    else
      LocationStack.push_back(Loc);

    if (checkRangesForMacroArgExpansion(Loc, Ranges, SM))
      IgnoredEnd = LocationStack.size();

    Loc = SM.getImmediateMacroCallerLoc(Loc);

    // Once the location no longer points into a macro, try stepping through
    // the last found location.  This sometimes produces additional useful
    // backtraces.
    if (Loc.isFileID())
      Loc = SM.getImmediateMacroCallerLoc(LocationStack.back());
    assert(Loc.isValid() && "must have a valid source location here");
  }

  LocationStack.erase(LocationStack.begin(),
                      LocationStack.begin() + IgnoredEnd);

  unsigned MacroDepth = LocationStack.size();
  unsigned MacroLimit = DiagOpts->MacroBacktraceLimit;
  if (MacroDepth <= MacroLimit || MacroLimit == 0) {
    for (auto I = LocationStack.rbegin(), E = LocationStack.rend();
         I != E; ++I)
      emitSingleMacroExpansion(*I, Level, Ranges, SM);
    return;
  }

  unsigned MacroStartMessages = MacroLimit / 2;
  unsigned MacroEndMessages = MacroLimit / 2 + MacroLimit % 2;

  for (auto I = LocationStack.rbegin(),
            E = LocationStack.rbegin() + MacroStartMessages;
       I != E; ++I)
    emitSingleMacroExpansion(*I, Level, Ranges, SM);

  SmallString<200> MessageStorage;
  llvm::raw_svector_ostream Message(MessageStorage);
  Message << "(skipping " << (MacroDepth - MacroLimit)
          << " expansions in backtrace; use -fmacro-backtrace-limit=0 to "
             "see all)";
  emitBasicNote(Message.str());

  for (auto I = LocationStack.rend() - MacroEndMessages,
            E = LocationStack.rend();
       I != E; ++I)
    emitSingleMacroExpansion(*I, Level, Ranges, SM);
}

DiagnosticNoteRenderer::~DiagnosticNoteRenderer() {}

void DiagnosticNoteRenderer::emitIncludeLocation(SourceLocation Loc,
                                                 PresumedLoc PLoc,
                                                 const SourceManager &SM) {
  // Generate a note indicating the include location.
  SmallString<200> MessageStorage;
  llvm::raw_svector_ostream Message(MessageStorage);
  Message << "in file included from " << PLoc.getFilename() << ':'
          << PLoc.getLine() << ":";
  emitNote(Loc, Message.str(), &SM);
}

void DiagnosticNoteRenderer::emitImportLocation(SourceLocation Loc,
                                                PresumedLoc PLoc,
                                                StringRef ModuleName,
                                                const SourceManager &SM) {
  // Generate a note indicating the include location.
  SmallString<200> MessageStorage;
  llvm::raw_svector_ostream Message(MessageStorage);
  Message << "in module '" << ModuleName;
  if (PLoc.isValid())
    Message << "' imported from " << PLoc.getFilename() << ':'
            << PLoc.getLine();
  Message << ":";
  emitNote(Loc, Message.str(), &SM);
}

void
DiagnosticNoteRenderer::emitBuildingModuleLocation(SourceLocation Loc,
                                                   PresumedLoc PLoc,
                                                   StringRef ModuleName,
                                                   const SourceManager &SM) {
  // Generate a note indicating the include location.
  SmallString<200> MessageStorage;
  llvm::raw_svector_ostream Message(MessageStorage);
  if (PLoc.getFilename())
    Message << "while building module '" << ModuleName << "' imported from "
            << PLoc.getFilename() << ':' << PLoc.getLine() << ":";
  else
    Message << "while building module '" << ModuleName << "':";
  emitNote(Loc, Message.str(), &SM);
}
