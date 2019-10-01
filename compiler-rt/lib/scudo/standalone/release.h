//===-- release.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SCUDO_RELEASE_H_
#define SCUDO_RELEASE_H_

#include "common.h"
#include "list.h"

namespace scudo {

class ReleaseRecorder {
public:
  ReleaseRecorder(uptr BaseAddress, MapPlatformData *Data = nullptr)
      : BaseAddress(BaseAddress), Data(Data) {}

  uptr getReleasedRangesCount() const { return ReleasedRangesCount; }

  uptr getReleasedBytes() const { return ReleasedBytes; }

  // Releases [From, To) range of pages back to OS.
  void releasePageRangeToOS(uptr From, uptr To) {
    const uptr Size = To - From;
    releasePagesToOS(BaseAddress, From, Size, Data);
    ReleasedRangesCount++;
    ReleasedBytes += Size;
  }

private:
  uptr ReleasedRangesCount = 0;
  uptr ReleasedBytes = 0;
  uptr BaseAddress = 0;
  MapPlatformData *Data = nullptr;
};

// A packed array of Counters. Each counter occupies 2^N bits, enough to store
// counter's MaxValue. Ctor will try to allocate the required Buffer via map()
// and the caller is expected to check whether the initialization was successful
// by checking isAllocated() result. For the performance sake, none of the
// accessors check the validity of the arguments, It is assumed that Index is
// always in [0, N) range and the value is not incremented past MaxValue.
class PackedCounterArray {
public:
  PackedCounterArray(uptr NumCounters, uptr MaxValue) : N(NumCounters) {
    CHECK_GT(NumCounters, 0);
    CHECK_GT(MaxValue, 0);
    constexpr uptr MaxCounterBits = sizeof(*Buffer) * 8UL;
    // Rounding counter storage size up to the power of two allows for using
    // bit shifts calculating particular counter's Index and offset.
    const uptr CounterSizeBits =
        roundUpToPowerOfTwo(getMostSignificantSetBitIndex(MaxValue) + 1);
    CHECK_LE(CounterSizeBits, MaxCounterBits);
    CounterSizeBitsLog = getLog2(CounterSizeBits);
    CounterMask = ~(static_cast<uptr>(0)) >> (MaxCounterBits - CounterSizeBits);

    const uptr PackingRatio = MaxCounterBits >> CounterSizeBitsLog;
    CHECK_GT(PackingRatio, 0);
    PackingRatioLog = getLog2(PackingRatio);
    BitOffsetMask = PackingRatio - 1;

    BufferSize = (roundUpTo(N, static_cast<uptr>(1U) << PackingRatioLog) >>
                  PackingRatioLog) *
                 sizeof(*Buffer);
    Buffer = reinterpret_cast<uptr *>(
        map(nullptr, BufferSize, "scudo:counters", MAP_ALLOWNOMEM));
  }
  ~PackedCounterArray() {
    if (isAllocated())
      unmap(reinterpret_cast<void *>(Buffer), BufferSize);
  }

  bool isAllocated() const { return !!Buffer; }

  uptr getCount() const { return N; }

  uptr get(uptr I) const {
    DCHECK_LT(I, N);
    const uptr Index = I >> PackingRatioLog;
    const uptr BitOffset = (I & BitOffsetMask) << CounterSizeBitsLog;
    return (Buffer[Index] >> BitOffset) & CounterMask;
  }

  void inc(uptr I) const {
    DCHECK_LT(get(I), CounterMask);
    const uptr Index = I >> PackingRatioLog;
    const uptr BitOffset = (I & BitOffsetMask) << CounterSizeBitsLog;
    DCHECK_LT(BitOffset, SCUDO_WORDSIZE);
    Buffer[Index] += static_cast<uptr>(1U) << BitOffset;
  }

  void incRange(uptr From, uptr To) const {
    DCHECK_LE(From, To);
    for (uptr I = From; I <= To; I++)
      inc(I);
  }

  uptr getBufferSize() const { return BufferSize; }

private:
  const uptr N;
  uptr CounterSizeBitsLog;
  uptr CounterMask;
  uptr PackingRatioLog;
  uptr BitOffsetMask;

  uptr BufferSize;
  uptr *Buffer;
};

template <class ReleaseRecorderT> class FreePagesRangeTracker {
public:
  explicit FreePagesRangeTracker(ReleaseRecorderT *Recorder)
      : Recorder(Recorder), PageSizeLog(getLog2(getPageSizeCached())) {}

  void processNextPage(bool Freed) {
    if (Freed) {
      if (!InRange) {
        CurrentRangeStatePage = CurrentPage;
        InRange = true;
      }
    } else {
      closeOpenedRange();
    }
    CurrentPage++;
  }

  void finish() { closeOpenedRange(); }

private:
  void closeOpenedRange() {
    if (InRange) {
      Recorder->releasePageRangeToOS((CurrentRangeStatePage << PageSizeLog),
                                     (CurrentPage << PageSizeLog));
      InRange = false;
    }
  }

  ReleaseRecorderT *const Recorder;
  const uptr PageSizeLog;
  bool InRange = false;
  uptr CurrentPage = 0;
  uptr CurrentRangeStatePage = 0;
};

template <class TransferBatchT, class ReleaseRecorderT>
NOINLINE void
releaseFreeMemoryToOS(const IntrusiveList<TransferBatchT> *FreeList, uptr Base,
                      uptr AllocatedPagesCount, uptr BlockSize,
                      ReleaseRecorderT *Recorder) {
  const uptr PageSize = getPageSizeCached();

  // Figure out the number of chunks per page and whether we can take a fast
  // path (the number of chunks per page is the same for all pages).
  uptr FullPagesBlockCountMax;
  bool SameBlockCountPerPage;
  if (BlockSize <= PageSize) {
    if (PageSize % BlockSize == 0) {
      // Same number of chunks per page, no cross overs.
      FullPagesBlockCountMax = PageSize / BlockSize;
      SameBlockCountPerPage = true;
    } else if (BlockSize % (PageSize % BlockSize) == 0) {
      // Some chunks are crossing page boundaries, which means that the page
      // contains one or two partial chunks, but all pages contain the same
      // number of chunks.
      FullPagesBlockCountMax = PageSize / BlockSize + 1;
      SameBlockCountPerPage = true;
    } else {
      // Some chunks are crossing page boundaries, which means that the page
      // contains one or two partial chunks.
      FullPagesBlockCountMax = PageSize / BlockSize + 2;
      SameBlockCountPerPage = false;
    }
  } else {
    if (BlockSize % PageSize == 0) {
      // One chunk covers multiple pages, no cross overs.
      FullPagesBlockCountMax = 1;
      SameBlockCountPerPage = true;
    } else {
      // One chunk covers multiple pages, Some chunks are crossing page
      // boundaries. Some pages contain one chunk, some contain two.
      FullPagesBlockCountMax = 2;
      SameBlockCountPerPage = false;
    }
  }

  PackedCounterArray Counters(AllocatedPagesCount, FullPagesBlockCountMax);
  if (!Counters.isAllocated())
    return;

  const uptr PageSizeLog = getLog2(PageSize);
  const uptr End = Base + AllocatedPagesCount * PageSize;

  // Iterate over free chunks and count how many free chunks affect each
  // allocated page.
  if (BlockSize <= PageSize && PageSize % BlockSize == 0) {
    // Each chunk affects one page only.
    for (auto It = FreeList->begin(); It != FreeList->end(); ++It) {
      for (u32 I = 0; I < (*It).getCount(); I++) {
        const uptr P = reinterpret_cast<uptr>((*It).get(I));
        if (P >= Base && P < End)
          Counters.inc((P - Base) >> PageSizeLog);
      }
    }
  } else {
    // In all other cases chunks might affect more than one page.
    for (auto It = FreeList->begin(); It != FreeList->end(); ++It) {
      for (u32 I = 0; I < (*It).getCount(); I++) {
        const uptr P = reinterpret_cast<uptr>((*It).get(I));
        if (P >= Base && P < End)
          Counters.incRange((P - Base) >> PageSizeLog,
                            (P - Base + BlockSize - 1) >> PageSizeLog);
      }
    }
  }

  // Iterate over pages detecting ranges of pages with chunk Counters equal
  // to the expected number of chunks for the particular page.
  FreePagesRangeTracker<ReleaseRecorderT> RangeTracker(Recorder);
  if (SameBlockCountPerPage) {
    // Fast path, every page has the same number of chunks affecting it.
    for (uptr I = 0; I < Counters.getCount(); I++)
      RangeTracker.processNextPage(Counters.get(I) == FullPagesBlockCountMax);
  } else {
    // Slow path, go through the pages keeping count how many chunks affect
    // each page.
    const uptr Pn = BlockSize < PageSize ? PageSize / BlockSize : 1;
    const uptr Pnc = Pn * BlockSize;
    // The idea is to increment the current page pointer by the first chunk
    // size, middle portion size (the portion of the page covered by chunks
    // except the first and the last one) and then the last chunk size, adding
    // up the number of chunks on the current page and checking on every step
    // whether the page boundary was crossed.
    uptr PrevPageBoundary = 0;
    uptr CurrentBoundary = 0;
    for (uptr I = 0; I < Counters.getCount(); I++) {
      const uptr PageBoundary = PrevPageBoundary + PageSize;
      uptr BlocksPerPage = Pn;
      if (CurrentBoundary < PageBoundary) {
        if (CurrentBoundary > PrevPageBoundary)
          BlocksPerPage++;
        CurrentBoundary += Pnc;
        if (CurrentBoundary < PageBoundary) {
          BlocksPerPage++;
          CurrentBoundary += BlockSize;
        }
      }
      PrevPageBoundary = PageBoundary;

      RangeTracker.processNextPage(Counters.get(I) == BlocksPerPage);
    }
  }
  RangeTracker.finish();
}

} // namespace scudo

#endif // SCUDO_RELEASE_H_
