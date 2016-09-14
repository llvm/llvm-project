//===--- Replacement.cpp - Framework for clang refactoring tools ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Implements classes to support/store refactorings.
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Core/Replacement.h"

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_os_ostream.h"

namespace clang {
namespace tooling {

static const char * const InvalidLocation = "";

Replacement::Replacement()
  : FilePath(InvalidLocation) {}

Replacement::Replacement(StringRef FilePath, unsigned Offset, unsigned Length,
                         StringRef ReplacementText)
    : FilePath(FilePath), ReplacementRange(Offset, Length),
      ReplacementText(ReplacementText) {}

Replacement::Replacement(const SourceManager &Sources, SourceLocation Start,
                         unsigned Length, StringRef ReplacementText) {
  setFromSourceLocation(Sources, Start, Length, ReplacementText);
}

Replacement::Replacement(const SourceManager &Sources,
                         const CharSourceRange &Range,
                         StringRef ReplacementText,
                         const LangOptions &LangOpts) {
  setFromSourceRange(Sources, Range, ReplacementText, LangOpts);
}

bool Replacement::isApplicable() const {
  return FilePath != InvalidLocation;
}

bool Replacement::apply(Rewriter &Rewrite) const {
  SourceManager &SM = Rewrite.getSourceMgr();
  const FileEntry *Entry = SM.getFileManager().getFile(FilePath);
  if (!Entry)
    return false;

  FileID ID = SM.getOrCreateFileID(Entry, SrcMgr::C_User);
  const SourceLocation Start =
    SM.getLocForStartOfFile(ID).
    getLocWithOffset(ReplacementRange.getOffset());
  // ReplaceText returns false on success.
  // ReplaceText only fails if the source location is not a file location, in
  // which case we already returned false earlier.
  bool RewriteSucceeded = !Rewrite.ReplaceText(
      Start, ReplacementRange.getLength(), ReplacementText);
  assert(RewriteSucceeded);
  return RewriteSucceeded;
}

std::string Replacement::toString() const {
  std::string Result;
  llvm::raw_string_ostream Stream(Result);
  Stream << FilePath << ": " << ReplacementRange.getOffset() << ":+"
         << ReplacementRange.getLength() << ":\"" << ReplacementText << "\"";
  return Stream.str();
}

bool operator<(const Replacement &LHS, const Replacement &RHS) {
  if (LHS.getOffset() != RHS.getOffset())
    return LHS.getOffset() < RHS.getOffset();

  // Apply longer replacements first, specifically so that deletions are
  // executed before insertions. It is (hopefully) never the intention to
  // delete parts of newly inserted code.
  if (LHS.getLength() != RHS.getLength())
    return LHS.getLength() > RHS.getLength();

  if (LHS.getFilePath() != RHS.getFilePath())
    return LHS.getFilePath() < RHS.getFilePath();
  return LHS.getReplacementText() < RHS.getReplacementText();
}

bool operator==(const Replacement &LHS, const Replacement &RHS) {
  return LHS.getOffset() == RHS.getOffset() &&
         LHS.getLength() == RHS.getLength() &&
         LHS.getFilePath() == RHS.getFilePath() &&
         LHS.getReplacementText() == RHS.getReplacementText();
}

void Replacement::setFromSourceLocation(const SourceManager &Sources,
                                        SourceLocation Start, unsigned Length,
                                        StringRef ReplacementText) {
  const std::pair<FileID, unsigned> DecomposedLocation =
      Sources.getDecomposedLoc(Start);
  const FileEntry *Entry = Sources.getFileEntryForID(DecomposedLocation.first);
  this->FilePath = Entry ? Entry->getName() : InvalidLocation;
  this->ReplacementRange = Range(DecomposedLocation.second, Length);
  this->ReplacementText = ReplacementText;
}

// FIXME: This should go into the Lexer, but we need to figure out how
// to handle ranges for refactoring in general first - there is no obvious
// good way how to integrate this into the Lexer yet.
static int getRangeSize(const SourceManager &Sources,
                        const CharSourceRange &Range,
                        const LangOptions &LangOpts) {
  SourceLocation SpellingBegin = Sources.getSpellingLoc(Range.getBegin());
  SourceLocation SpellingEnd = Sources.getSpellingLoc(Range.getEnd());
  std::pair<FileID, unsigned> Start = Sources.getDecomposedLoc(SpellingBegin);
  std::pair<FileID, unsigned> End = Sources.getDecomposedLoc(SpellingEnd);
  if (Start.first != End.first) return -1;
  if (Range.isTokenRange())
    End.second += Lexer::MeasureTokenLength(SpellingEnd, Sources, LangOpts);
  return End.second - Start.second;
}

void Replacement::setFromSourceRange(const SourceManager &Sources,
                                     const CharSourceRange &Range,
                                     StringRef ReplacementText,
                                     const LangOptions &LangOpts) {
  setFromSourceLocation(Sources, Sources.getSpellingLoc(Range.getBegin()),
                        getRangeSize(Sources, Range, LangOpts),
                        ReplacementText);
}

llvm::Error makeConflictReplacementsError(const Replacement &New,
                                          const Replacement &Existing) {
  return llvm::make_error<llvm::StringError>(
      "New replacement:\n" + New.toString() +
          "\nconflicts with existing replacement:\n" + Existing.toString(),
      llvm::inconvertibleErrorCode());
}

llvm::Error Replacements::add(const Replacement &R) {
  // Check the file path.
  if (!Replaces.empty() && R.getFilePath() != Replaces.begin()->getFilePath())
    return llvm::make_error<llvm::StringError>(
        "All replacements must have the same file path. New replacement: " +
            R.getFilePath() + ", existing replacements: " +
            Replaces.begin()->getFilePath() + "\n",
        llvm::inconvertibleErrorCode());

  // Special-case header insertions.
  if (R.getOffset() == UINT_MAX) {
    Replaces.insert(R);
    return llvm::Error::success();
  }

  // This replacement cannot conflict with replacements that end before
  // this replacement starts or start after this replacement ends.
  // We also know that there currently are no overlapping replacements.
  // Thus, we know that all replacements that start after the end of the current
  // replacement cannot overlap.
  Replacement AtEnd(R.getFilePath(), R.getOffset() + R.getLength(), 0, "");

  // Find the first entry that starts after or at the end of R. Note that
  // entries that start at the end can still be conflicting if R is an
  // insertion.
  auto I = Replaces.lower_bound(AtEnd);
  // If `I` starts at the same offset as `R`, `R` must be an insertion.
  if (I != Replaces.end() && R.getOffset() == I->getOffset()) {
    assert(R.getLength() == 0);
    // `I` is also an insertion, `R` and `I` conflict.
    if (I->getLength() == 0)
      return makeConflictReplacementsError(R, *I);
    // Insertion `R` is adjacent to a non-insertion replacement `I`, so they
    // are order-independent. It is safe to assume that `R` will not conflict
    // with any replacement before `I` since all replacements before `I` must
    // either end before `R` or end at `R` but has length > 0 (if the
    // replacement before `I` is an insertion at `R`, it would have been `I`
    // since it is a lower bound of `AtEnd` and ordered before the current `I`
    // in the set).
    Replaces.insert(R);
    return llvm::Error::success();
  }

  // I is the smallest iterator whose entry cannot overlap.
  // If that is begin(), there are no overlaps.
  if (I == Replaces.begin()) {
    Replaces.insert(R);
    return llvm::Error::success();
  }
  --I;
  // If the previous entry does not overlap, we know that entries before it
  // can also not overlap.
  if (!Range(R.getOffset(), R.getLength())
           .overlapsWith(Range(I->getOffset(), I->getLength()))) {
    // If `R` and `I` do not have the same offset, it is safe to add `R` since
    // it must come after `I`. Otherwise:
    //   - If `R` is an insertion, `I` must not be an insertion since it would
    //   have come after `AtEnd` if it has length 0.
    //   - If `R` is not an insertion, `I` must be an insertion; otherwise, `R`
    //   and `I` would have overlapped.
    // In either case, we can safely insert `R`.
    Replaces.insert(R);
    return llvm::Error::success();
  }
  return makeConflictReplacementsError(R, *I);
}

namespace {

// Represents a merged replacement, i.e. a replacement consisting of multiple
// overlapping replacements from 'First' and 'Second' in mergeReplacements.
//
// Position projection:
// Offsets and lengths of the replacements can generally refer to two different
// coordinate spaces. Replacements from 'First' refer to the original text
// whereas replacements from 'Second' refer to the text after applying 'First'.
//
// MergedReplacement always operates in the coordinate space of the original
// text, i.e. transforms elements from 'Second' to take into account what was
// changed based on the elements from 'First'.
//
// We can correctly calculate this projection as we look at the replacements in
// order of strictly increasing offsets.
//
// Invariants:
// * We always merge elements from 'First' into elements from 'Second' and vice
//   versa. Within each set, the replacements are non-overlapping.
// * We only extend to the right, i.e. merge elements with strictly increasing
//   offsets.
class MergedReplacement {
public:
  MergedReplacement(const Replacement &R, bool MergeSecond, int D)
      : MergeSecond(MergeSecond), Delta(D), FilePath(R.getFilePath()),
        Offset(R.getOffset() + (MergeSecond ? 0 : Delta)), Length(R.getLength()),
        Text(R.getReplacementText()) {
    Delta += MergeSecond ? 0 : Text.size() - Length;
    DeltaFirst = MergeSecond ? Text.size() - Length : 0;
  }

  // Merges the next element 'R' into this merged element. As we always merge
  // from 'First' into 'Second' or vice versa, the MergedReplacement knows what
  // set the next element is coming from.
  void merge(const Replacement &R) {
    if (MergeSecond) {
      unsigned REnd = R.getOffset() + Delta + R.getLength();
      unsigned End = Offset + Text.size();
      if (REnd > End) {
        Length += REnd - End;
        MergeSecond = false;
      }
      StringRef TextRef = Text;
      StringRef Head = TextRef.substr(0, R.getOffset() + Delta - Offset);
      StringRef Tail = TextRef.substr(REnd - Offset);
      Text = (Head + R.getReplacementText() + Tail).str();
      Delta += R.getReplacementText().size() - R.getLength();
    } else {
      unsigned End = Offset + Length;
      StringRef RText = R.getReplacementText();
      StringRef Tail = RText.substr(End - R.getOffset());
      Text = (Text + Tail).str();
      if (R.getOffset() + RText.size() > End) {
        Length = R.getOffset() + R.getLength() - Offset;
        MergeSecond = true;
      } else {
        Length += R.getLength() - RText.size();
      }
      DeltaFirst += RText.size() - R.getLength();
    }
  }

  // Returns 'true' if 'R' starts strictly after the MergedReplacement and thus
  // doesn't need to be merged.
  bool endsBefore(const Replacement &R) const {
    if (MergeSecond)
      return Offset + Text.size() < R.getOffset() + Delta;
    return Offset + Length < R.getOffset();
  }

  // Returns 'true' if an element from the second set should be merged next.
  bool mergeSecond() const { return MergeSecond; }
  int deltaFirst() const { return DeltaFirst; }
  Replacement asReplacement() const { return {FilePath, Offset, Length, Text}; }

private:
  bool MergeSecond;

  // Amount of characters that elements from 'Second' need to be shifted by in
  // order to refer to the original text.
  int Delta;

  // Sum of all deltas (text-length - length) of elements from 'First' merged
  // into this element. This is used to update 'Delta' once the
  // MergedReplacement is completed.
  int DeltaFirst;

  // Data of the actually merged replacement. FilePath and Offset aren't changed
  // as the element is only extended to the right.
  const StringRef FilePath;
  const unsigned Offset;
  unsigned Length;
  std::string Text;
};

} // namespace

Replacements Replacements::merge(const Replacements &ReplacesToMerge) const {
  if (empty() || ReplacesToMerge.empty())
    return empty() ? ReplacesToMerge : *this;

  auto &First = Replaces;
  auto &Second = ReplacesToMerge.Replaces;
  // Delta is the amount of characters that replacements from 'Second' need to
  // be shifted so that their offsets refer to the original text.
  int Delta = 0;
  ReplacementsImpl Result;

  // Iterate over both sets and always add the next element (smallest total
  // Offset) from either 'First' or 'Second'. Merge that element with
  // subsequent replacements as long as they overlap. See more details in the
  // comment on MergedReplacement.
  for (auto FirstI = First.begin(), SecondI = Second.begin();
       FirstI != First.end() || SecondI != Second.end();) {
    bool NextIsFirst = SecondI == Second.end() ||
                       (FirstI != First.end() &&
                        FirstI->getOffset() < SecondI->getOffset() + Delta);
    MergedReplacement Merged(NextIsFirst ? *FirstI : *SecondI, NextIsFirst,
                             Delta);
    ++(NextIsFirst ? FirstI : SecondI);

    while ((Merged.mergeSecond() && SecondI != Second.end()) ||
           (!Merged.mergeSecond() && FirstI != First.end())) {
      auto &I = Merged.mergeSecond() ? SecondI : FirstI;
      if (Merged.endsBefore(*I))
        break;
      Merged.merge(*I);
      ++I;
    }
    Delta -= Merged.deltaFirst();
    Result.insert(Merged.asReplacement());
  }
  return Replacements(Result.begin(), Result.end());
}

// Combines overlapping ranges in \p Ranges and sorts the combined ranges.
// Returns a set of non-overlapping and sorted ranges that is equivalent to
// \p Ranges.
static std::vector<Range> combineAndSortRanges(std::vector<Range> Ranges) {
  std::sort(Ranges.begin(), Ranges.end(),
            [](const Range &LHS, const Range &RHS) {
              if (LHS.getOffset() != RHS.getOffset())
                return LHS.getOffset() < RHS.getOffset();
              return LHS.getLength() < RHS.getLength();
            });
  std::vector<Range> Result;
  for (const auto &R : Ranges) {
    if (Result.empty() ||
        Result.back().getOffset() + Result.back().getLength() < R.getOffset()) {
      Result.push_back(R);
    } else {
      unsigned NewEnd =
          std::max(Result.back().getOffset() + Result.back().getLength(),
                   R.getOffset() + R.getLength());
      Result[Result.size() - 1] =
          Range(Result.back().getOffset(), NewEnd - Result.back().getOffset());
    }
  }
  return Result;
}

std::vector<Range>
calculateRangesAfterReplacements(const Replacements &Replaces,
                                 const std::vector<Range> &Ranges) {
  // To calculate the new ranges,
  //   - Turn \p Ranges into Replacements at (offset, length) with an empty
  //     (unimportant) replacement text of length "length".
  //   - Merge with \p Replaces.
  //   - The new ranges will be the affected ranges of the merged replacements.
  auto MergedRanges = combineAndSortRanges(Ranges);
  if (Replaces.empty())
    return MergedRanges;
  tooling::Replacements FakeReplaces;
  for (const auto &R : MergedRanges) {
    auto Err = FakeReplaces.add(Replacement(Replaces.begin()->getFilePath(),
                                            R.getOffset(), R.getLength(),
                                            std::string(R.getLength(), ' ')));
    assert(!Err &&
           "Replacements must not conflict since ranges have been merged.");
    (void)Err;
  }
  return FakeReplaces.merge(Replaces).getAffectedRanges();
}

std::vector<Range> Replacements::getAffectedRanges() const {
  std::vector<Range> ChangedRanges;
  int Shift = 0;
  for (const Replacement &R : Replaces) {
    unsigned Offset = R.getOffset() + Shift;
    unsigned Length = R.getReplacementText().size();
    Shift += Length - R.getLength();
    ChangedRanges.push_back(Range(Offset, Length));
  }
  return combineAndSortRanges(ChangedRanges);
}

unsigned Replacements::getShiftedCodePosition(unsigned Position) const {
  unsigned Offset = 0;
  for (const auto& R : Replaces) {
    if (R.getOffset() + R.getLength() <= Position) {
      Offset += R.getReplacementText().size() - R.getLength();
      continue;
    }
    if (R.getOffset() < Position &&
        R.getOffset() + R.getReplacementText().size() <= Position) {
      Position = R.getOffset() + R.getReplacementText().size();
      if (R.getReplacementText().size() > 0)
        Position--;
    }
    break;
  }
  return Position + Offset;
}

bool applyAllReplacements(const Replacements &Replaces, Rewriter &Rewrite) {
  bool Result = true;
  for (Replacements::const_iterator I = Replaces.begin(),
                                    E = Replaces.end();
       I != E; ++I) {
    if (I->isApplicable()) {
      Result = I->apply(Rewrite) && Result;
    } else {
      Result = false;
    }
  }
  return Result;
}

llvm::Expected<std::string> applyAllReplacements(StringRef Code,
                                                const Replacements &Replaces) {
  if (Replaces.empty())
    return Code.str();

  IntrusiveRefCntPtr<vfs::InMemoryFileSystem> InMemoryFileSystem(
      new vfs::InMemoryFileSystem);
  FileManager Files(FileSystemOptions(), InMemoryFileSystem);
  DiagnosticsEngine Diagnostics(
      IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs),
      new DiagnosticOptions);
  SourceManager SourceMgr(Diagnostics, Files);
  Rewriter Rewrite(SourceMgr, LangOptions());
  InMemoryFileSystem->addFile(
      "<stdin>", 0, llvm::MemoryBuffer::getMemBuffer(Code, "<stdin>"));
  FileID ID = SourceMgr.createFileID(Files.getFile("<stdin>"), SourceLocation(),
                                     clang::SrcMgr::C_User);
  for (Replacements::const_iterator I = Replaces.begin(), E = Replaces.end();
       I != E; ++I) {
    Replacement Replace("<stdin>", I->getOffset(), I->getLength(),
                        I->getReplacementText());
    if (!Replace.apply(Rewrite))
      return llvm::make_error<llvm::StringError>(
          "Failed to apply replacement: " + Replace.toString(),
          llvm::inconvertibleErrorCode());
  }
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  Rewrite.getEditBuffer(ID).write(OS);
  OS.flush();
  return Result;
}

std::map<std::string, Replacements>
groupReplacementsByFile(const Replacements &Replaces) {
  std::map<std::string, Replacements> FileToReplaces;
  for (const auto &Replace : Replaces)
    // We can ignore the Error here since \p Replaces is already conflict-free.
    FileToReplaces[Replace.getFilePath()].add(Replace);
  return FileToReplaces;
}

} // end namespace tooling
} // end namespace clang
