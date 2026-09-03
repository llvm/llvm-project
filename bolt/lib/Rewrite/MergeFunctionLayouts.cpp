//===- bolt/Rewrite/MergeFunctionLayouts.cpp - Merge two function layouts -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements mergeFunctionLayouts() for llvm-bolt-align.
//
//===----------------------------------------------------------------------===//

#include "bolt/Rewrite/MergeFunctionLayouts.h"
#include "bolt/Utils/Utils.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <limits>
#include <vector>

using namespace llvm;
using namespace bolt;

namespace {

struct Entry {
  std::string Name;
  uint64_t Offset;
};

using Match = std::pair<const Entry *, const Entry *>;

/// Keep some spare space between functions. This absorbs small layout changes
/// in the final rewrite.
constexpr uint64_t LayoutSlack = 64;
constexpr uint64_t LayoutAlignment = 64;

static Expected<std::vector<Entry>> parseFile(StringRef Path) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> MB = MemoryBuffer::getFile(Path);
  if (std::error_code EC = MB.getError())
    return createStringError(EC, Twine("cannot open layout file '") + Path +
                                     "': " + EC.message());

  std::vector<Entry> Entries;
  for (line_iterator LI(*MB.get(), /*SkipBlanks=*/true, /*CommentMarker=*/'#');
       !LI.is_at_eof(); ++LI) {
    StringRef Line = LI->trim();
    if (Line.empty())
      continue;

    StringRef Name, OffsetStr;
    std::tie(Name, OffsetStr) = Line.split(' ');
    Name = Name.trim();
    OffsetStr = OffsetStr.trim();

    uint64_t Offset;
    if (Name.empty() || OffsetStr.empty() ||
        OffsetStr.getAsInteger(/*Radix=*/0, Offset))
      return createStringError(inconvertibleErrorCode(),
                               Twine("malformed entry at ") + Path + ":" +
                                   Twine(LI.line_number()));

    Entries.push_back({Name.str(), Offset});
  }
  return Entries;
}

/// Return all entries in both \p EntriesA and \p EntriesB in A's order.
static std::vector<Match> findMatches(ArrayRef<Entry> EntriesA,
                                      ArrayRef<Entry> EntriesB) {
  StringMap<const Entry *> ExactB;
  StringMap<const Entry *> CommonB;
  StringMap<unsigned> CommonCountA;
  StringMap<unsigned> CommonCountB;

  for (const Entry &E : EntriesA)
    if (std::optional<StringRef> Common = getLTOCommonName(E.Name))
      ++CommonCountA[*Common];

  for (const Entry &E : EntriesB) {
    ExactB[E.Name] = &E;
    if (std::optional<StringRef> Common = getLTOCommonName(E.Name)) {
      ++CommonCountB[*Common];
      CommonB[*Common] = &E;
    }
  }

  std::vector<Match> Matches;
  for (const Entry &E : EntriesA) {
    const Entry *Match = ExactB.lookup(E.Name);
    if (!Match) {
      std::optional<StringRef> Common = getLTOCommonName(E.Name);
      if (Common && CommonCountA.lookup(*Common) == 1 &&
          CommonCountB.lookup(*Common) == 1)
        Match = CommonB.lookup(*Common);
    }
    if (Match)
      Matches.emplace_back(&E, Match);
  }
  return Matches;
}

static std::vector<Match>
findLongestIncreasingSubsequence(ArrayRef<Match> Matches) {
  if (Matches.empty())
    return {};

  const size_t NoIndex = std::numeric_limits<size_t>::max();
  std::vector<size_t> Tails;
  std::vector<size_t> Previous(Matches.size(), NoIndex);

  for (size_t I = 0; I != Matches.size(); ++I) {
    const uint64_t Offset = Matches[I].second->Offset;
    auto It = std::lower_bound(Tails.begin(), Tails.end(), Offset,
                               [&](size_t Index, uint64_t Key) {
                                 return Matches[Index].second->Offset < Key;
                               });
    if (It != Tails.begin())
      Previous[I] = It[-1];

    if (It == Tails.end())
      Tails.push_back(I);
    else
      *It = I;
  }

  std::vector<Match> Result;
  for (size_t I = Tails.back(); I != NoIndex; I = Previous[I])
    Result.push_back(Matches[I]);
  std::reverse(Result.begin(), Result.end());
  return Result;
}

} // namespace

Error bolt::mergeFunctionLayouts(StringRef PathA, StringRef PathB,
                                 StringRef OutputPath, raw_ostream &Log) {
  Expected<std::vector<Entry>> EntriesA = parseFile(PathA);
  if (!EntriesA)
    return EntriesA.takeError();

  Expected<std::vector<Entry>> EntriesB = parseFile(PathB);
  if (!EntriesB)
    return EntriesB.takeError();

  const std::vector<Match> Matches =
      findLongestIncreasingSubsequence(findMatches(*EntriesA, *EntriesB));

  std::error_code EC;
  raw_fd_ostream OS(OutputPath, EC, sys::fs::OpenFlags::OF_None);
  if (EC)
    return createStringError(EC, Twine("cannot open output layout file '") +
                                     OutputPath + "': " + EC.message());

  uint64_t Matched = 0;
  bool First = true;
  uint64_t PrevMergedOff = 0, PrevAOff = 0, PrevBOff = 0;
  for (const Match &MatchPair : Matches) {
    const Entry &A = *MatchPair.first;
    const Entry &B = *MatchPair.second;
    const uint64_t AOff = A.Offset;
    const uint64_t BOff = B.Offset;

    uint64_t MergedOff;
    if (First) {
      MergedOff = alignTo(std::max(AOff, BOff), LayoutAlignment);
    } else {
      const uint64_t RequiredGap = std::max(AOff - PrevAOff, BOff - PrevBOff);
      MergedOff =
          alignTo(PrevMergedOff + RequiredGap + LayoutSlack, LayoutAlignment);
    }

    OS << A.Name << " 0x" << Twine::utohexstr(MergedOff) << "\n";
    if (B.Name != A.Name)
      OS << B.Name << " 0x" << Twine::utohexstr(MergedOff) << "\n";

    ++Matched;
    PrevMergedOff = MergedOff;
    PrevAOff = AOff;
    PrevBOff = BOff;
    First = false;
  }

  const uint64_t PossibleMatches = std::min(EntriesA->size(), EntriesB->size());
  const uint64_t MatchedPercent =
      PossibleMatches ? Matched * 100 / PossibleMatches : 0;
  Log << "BOLT-ALIGN: pinned " << Matched << " functions (" << MatchedPercent
      << "%)\n";

  return Error::success();
}
