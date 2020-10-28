//===-- release_test.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "tests/scudo_unit_test.h"

#include "list.h"
#include "release.h"
#include "size_class_map.h"

#include <string.h>

#include <algorithm>
#include <random>
#include <set>

TEST(ScudoReleaseTest, PackedCounterArray) {
  for (scudo::uptr I = 0; I < SCUDO_WORDSIZE; I++) {
    // Various valid counter's max values packed into one word.
    scudo::PackedCounterArray Counters2N(1U, 1U, 1UL << I);
    EXPECT_EQ(sizeof(scudo::uptr), Counters2N.getBufferSize());
    // Check the "all bit set" values too.
    scudo::PackedCounterArray Counters2N1_1(1U, 1U, ~0UL >> I);
    EXPECT_EQ(sizeof(scudo::uptr), Counters2N1_1.getBufferSize());
    // Verify the packing ratio, the counter is Expected to be packed into the
    // closest power of 2 bits.
    scudo::PackedCounterArray Counters(1U, SCUDO_WORDSIZE, 1UL << I);
    EXPECT_EQ(sizeof(scudo::uptr) * scudo::roundUpToPowerOfTwo(I + 1),
              Counters.getBufferSize());
  }

  // Go through 1, 2, 4, 8, .. {32,64} bits per counter.
  for (scudo::uptr I = 0; (SCUDO_WORDSIZE >> I) != 0; I++) {
    // Make sure counters request one memory page for the buffer.
    const scudo::uptr NumCounters =
        (scudo::getPageSizeCached() / 8) * (SCUDO_WORDSIZE >> I);
    scudo::PackedCounterArray Counters(1U, NumCounters,
                                       1UL << ((1UL << I) - 1));
    Counters.inc(0U, 0U);
    for (scudo::uptr C = 1; C < NumCounters - 1; C++) {
      EXPECT_EQ(0UL, Counters.get(0U, C));
      Counters.inc(0U, C);
      EXPECT_EQ(1UL, Counters.get(0U, C - 1));
    }
    EXPECT_EQ(0UL, Counters.get(0U, NumCounters - 1));
    Counters.inc(0U, NumCounters - 1);
    if (I > 0) {
      Counters.incRange(0u, 0U, NumCounters - 1);
      for (scudo::uptr C = 0; C < NumCounters; C++)
        EXPECT_EQ(2UL, Counters.get(0U, C));
    }
  }
}

class StringRangeRecorder {
public:
  std::string ReportedPages;

  StringRangeRecorder()
      : PageSizeScaledLog(scudo::getLog2(scudo::getPageSizeCached())) {}

  void releasePageRangeToOS(scudo::uptr From, scudo::uptr To) {
    From >>= PageSizeScaledLog;
    To >>= PageSizeScaledLog;
    EXPECT_LT(From, To);
    if (!ReportedPages.empty())
      EXPECT_LT(LastPageReported, From);
    ReportedPages.append(From - LastPageReported, '.');
    ReportedPages.append(To - From, 'x');
    LastPageReported = To;
  }

private:
  const scudo::uptr PageSizeScaledLog;
  scudo::uptr LastPageReported = 0;
};

TEST(ScudoReleaseTest, FreePagesRangeTracker) {
  // 'x' denotes a page to be released, '.' denotes a page to be kept around.
  const char *TestCases[] = {
      "",
      ".",
      "x",
      "........",
      "xxxxxxxxxxx",
      "..............xxxxx",
      "xxxxxxxxxxxxxxxxxx.....",
      "......xxxxxxxx........",
      "xxx..........xxxxxxxxxxxxxxx",
      "......xxxx....xxxx........",
      "xxx..........xxxxxxxx....xxxxxxx",
      "x.x.x.x.x.x.x.x.x.x.x.x.",
      ".x.x.x.x.x.x.x.x.x.x.x.x",
      ".x.x.x.x.x.x.x.x.x.x.x.x.",
      "x.x.x.x.x.x.x.x.x.x.x.x.x",
  };
  typedef scudo::FreePagesRangeTracker<StringRangeRecorder> RangeTracker;

  for (auto TestCase : TestCases) {
    StringRangeRecorder Recorder;
    RangeTracker Tracker(&Recorder);
    for (scudo::uptr I = 0; TestCase[I] != 0; I++)
      Tracker.processNextPage(TestCase[I] == 'x');
    Tracker.finish();
    // Strip trailing '.'-pages before comparing the results as they are not
    // going to be reported to range_recorder anyway.
    const char *LastX = strrchr(TestCase, 'x');
    std::string Expected(TestCase,
                         LastX == nullptr ? 0 : (LastX - TestCase + 1));
    EXPECT_STREQ(Expected.c_str(), Recorder.ReportedPages.c_str());
  }
}

class ReleasedPagesRecorder {
public:
  std::set<scudo::uptr> ReportedPages;

  void releasePageRangeToOS(scudo::uptr From, scudo::uptr To) {
    const scudo::uptr PageSize = scudo::getPageSizeCached();
    for (scudo::uptr I = From; I < To; I += PageSize)
      ReportedPages.insert(I);
  }
};

// Simplified version of a TransferBatch.
template <class SizeClassMap> struct FreeBatch {
  static const scudo::u32 MaxCount = SizeClassMap::MaxNumCachedHint;
  void clear() { Count = 0; }
  void add(scudo::uptr P) {
    DCHECK_LT(Count, MaxCount);
    Batch[Count++] = P;
  }
  scudo::u32 getCount() const { return Count; }
  scudo::uptr get(scudo::u32 I) const {
    DCHECK_LE(I, Count);
    return Batch[I];
  }
  FreeBatch *Next;

private:
  scudo::u32 Count;
  scudo::uptr Batch[MaxCount];
};

template <class SizeClassMap> void testReleaseFreeMemoryToOS() {
  typedef FreeBatch<SizeClassMap> Batch;
  const scudo::uptr PagesCount = 1024;
  const scudo::uptr PageSize = scudo::getPageSizeCached();
  std::mt19937 R;
  scudo::u32 RandState = 42;

  for (scudo::uptr I = 1; I <= SizeClassMap::LargestClassId; I++) {
    const scudo::uptr BlockSize = SizeClassMap::getSizeByClassId(I);
    const scudo::uptr MaxBlocks = PagesCount * PageSize / BlockSize;

    // Generate the random free list.
    std::vector<scudo::uptr> FreeArray;
    bool InFreeRange = false;
    scudo::uptr CurrentRangeEnd = 0;
    for (scudo::uptr I = 0; I < MaxBlocks; I++) {
      if (I == CurrentRangeEnd) {
        InFreeRange = (scudo::getRandomU32(&RandState) & 1U) == 1;
        CurrentRangeEnd += (scudo::getRandomU32(&RandState) & 0x7f) + 1;
      }
      if (InFreeRange)
        FreeArray.push_back(I * BlockSize);
    }
    if (FreeArray.empty())
      continue;
    // Shuffle the array to ensure that the order is irrelevant.
    std::shuffle(FreeArray.begin(), FreeArray.end(), R);

    // Build the FreeList from the FreeArray.
    scudo::SinglyLinkedList<Batch> FreeList;
    FreeList.clear();
    Batch *CurrentBatch = nullptr;
    for (auto const &Block : FreeArray) {
      if (!CurrentBatch) {
        CurrentBatch = new Batch;
        CurrentBatch->clear();
        FreeList.push_back(CurrentBatch);
      }
      CurrentBatch->add(Block);
      if (CurrentBatch->getCount() == Batch::MaxCount)
        CurrentBatch = nullptr;
    }

    // Release the memory.
    auto SkipRegion = [](UNUSED scudo::uptr RegionIndex) { return false; };
    ReleasedPagesRecorder Recorder;
    releaseFreeMemoryToOS(FreeList, 0, MaxBlocks * BlockSize, 1U, BlockSize,
                          &Recorder, SkipRegion);

    // Verify that there are no released pages touched by used chunks and all
    // ranges of free chunks big enough to contain the entire memory pages had
    // these pages released.
    scudo::uptr VerifiedReleasedPages = 0;
    std::set<scudo::uptr> FreeBlocks(FreeArray.begin(), FreeArray.end());

    scudo::uptr CurrentBlock = 0;
    InFreeRange = false;
    scudo::uptr CurrentFreeRangeStart = 0;
    for (scudo::uptr I = 0; I < MaxBlocks; I++) {
      const bool IsFreeBlock =
          FreeBlocks.find(CurrentBlock) != FreeBlocks.end();
      if (IsFreeBlock) {
        if (!InFreeRange) {
          InFreeRange = true;
          CurrentFreeRangeStart = CurrentBlock;
        }
      } else {
        // Verify that this used chunk does not touch any released page.
        const scudo::uptr StartPage = CurrentBlock / PageSize;
        const scudo::uptr EndPage = (CurrentBlock + BlockSize - 1) / PageSize;
        for (scudo::uptr J = StartPage; J <= EndPage; J++) {
          const bool PageReleased = Recorder.ReportedPages.find(J * PageSize) !=
                                    Recorder.ReportedPages.end();
          EXPECT_EQ(false, PageReleased);
        }

        if (InFreeRange) {
          InFreeRange = false;
          // Verify that all entire memory pages covered by this range of free
          // chunks were released.
          scudo::uptr P = scudo::roundUpTo(CurrentFreeRangeStart, PageSize);
          while (P + PageSize <= CurrentBlock) {
            const bool PageReleased =
                Recorder.ReportedPages.find(P) != Recorder.ReportedPages.end();
            EXPECT_EQ(true, PageReleased);
            VerifiedReleasedPages++;
            P += PageSize;
          }
        }
      }

      CurrentBlock += BlockSize;
    }

    if (InFreeRange) {
      scudo::uptr P = scudo::roundUpTo(CurrentFreeRangeStart, PageSize);
      const scudo::uptr EndPage =
          scudo::roundUpTo(MaxBlocks * BlockSize, PageSize);
      while (P + PageSize <= EndPage) {
        const bool PageReleased =
            Recorder.ReportedPages.find(P) != Recorder.ReportedPages.end();
        EXPECT_EQ(true, PageReleased);
        VerifiedReleasedPages++;
        P += PageSize;
      }
    }

    EXPECT_EQ(Recorder.ReportedPages.size(), VerifiedReleasedPages);

    while (!FreeList.empty()) {
      CurrentBatch = FreeList.front();
      FreeList.pop_front();
      delete CurrentBatch;
    }
  }
}

TEST(ScudoReleaseTest, ReleaseFreeMemoryToOSDefault) {
  testReleaseFreeMemoryToOS<scudo::DefaultSizeClassMap>();
}

TEST(ScudoReleaseTest, ReleaseFreeMemoryToOSAndroid) {
  testReleaseFreeMemoryToOS<scudo::AndroidSizeClassMap>();
}

TEST(ScudoReleaseTest, ReleaseFreeMemoryToOSSvelte) {
  testReleaseFreeMemoryToOS<scudo::SvelteSizeClassMap>();
}
