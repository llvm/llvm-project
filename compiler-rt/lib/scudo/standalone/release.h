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
#include "mem_map.h"
#include "mutex.h"
#include "thread_annotations.h"

namespace scudo {

template <typename MemMapT> class RegionReleaseRecorder {
public:
  RegionReleaseRecorder(MemMapT *RegionMemMap, uptr Base, uptr Offset = 0)
      : RegionMemMap(RegionMemMap), Base(Base), Offset(Offset) {}

  uptr getReleasedRangesCount() const { return ReleasedRangesCount; }

  uptr getReleasedBytes() const { return ReleasedBytes; }

  uptr getBase() const { return Base; }

  // Releases [From, To) range of pages back to OS. Note that `From` and `To`
  // are offseted from `Base` + Offset.
  void releasePageRangeToOS(uptr From, uptr To) {
    const uptr Size = To - From;
    RegionMemMap->releasePagesToOS(getBase() + Offset + From, Size);
    ReleasedRangesCount++;
    ReleasedBytes += Size;
  }

private:
  uptr ReleasedRangesCount = 0;
  uptr ReleasedBytes = 0;
  MemMapT *RegionMemMap = nullptr;
  uptr Base = 0;
  // The release offset from Base. This is used when we know a given range after
  // Base will not be released.
  uptr Offset = 0;
};

class ReleaseRecorder {
public:
  ReleaseRecorder(uptr Base, uptr Offset = 0, MapPlatformData *Data = nullptr)
      : Base(Base), Offset(Offset), Data(Data) {}

  uptr getReleasedRangesCount() const { return ReleasedRangesCount; }

  uptr getReleasedBytes() const { return ReleasedBytes; }

  uptr getBase() const { return Base; }

  // Releases [From, To) range of pages back to OS.
  void releasePageRangeToOS(uptr From, uptr To) {
    const uptr Size = To - From;
    releasePagesToOS(Base, From + Offset, Size, Data);
    ReleasedRangesCount++;
    ReleasedBytes += Size;
  }

private:
  uptr ReleasedRangesCount = 0;
  uptr ReleasedBytes = 0;
  // The starting address to release. Note that we may want to combine (Base +
  // Offset) as a new Base. However, the Base is retrieved from
  // `MapPlatformData` on Fuchsia, which means the offset won't be aware.
  // Therefore, store them separately to make it work on all the platforms.
  uptr Base = 0;
  // The release offset from Base. This is used when we know a given range after
  // Base will not be released.
  uptr Offset = 0;
  MapPlatformData *Data = nullptr;
};

// A buffer pool which holds a fixed number of static buffers for fast buffer
// allocation. If the request size is greater than `StaticBufferSize`, it'll
// delegate the allocation to map().
template <uptr StaticBufferCount, uptr StaticBufferSize> class BufferPool {
public:
  // Preserve 1 bit in the `Mask` so that we don't need to do zero-check while
  // extracting the least significant bit from the `Mask`.
  static_assert(StaticBufferCount < SCUDO_WORDSIZE, "");
  static_assert(isAligned(StaticBufferSize, SCUDO_CACHE_LINE_SIZE), "");

  // Return a buffer which is at least `BufferSize`.
  uptr *getBuffer(const uptr BufferSize) {
    if (UNLIKELY(BufferSize > StaticBufferSize))
      return getDynamicBuffer(BufferSize);

    uptr index;
    {
      // TODO: In general, we expect this operation should be fast so the
      // waiting thread won't be put into sleep. The HybridMutex does implement
      // the busy-waiting but we may want to review the performance and see if
      // we need an explict spin lock here.
      ScopedLock L(Mutex);
      index = getLeastSignificantSetBitIndex(Mask);
      if (index < StaticBufferCount)
        Mask ^= static_cast<uptr>(1) << index;
    }

    if (index >= StaticBufferCount)
      return getDynamicBuffer(BufferSize);

    const uptr Offset = index * StaticBufferSize;
    memset(&RawBuffer[Offset], 0, StaticBufferSize);
    return &RawBuffer[Offset];
  }

  void releaseBuffer(uptr *Buffer, const uptr BufferSize) {
    const uptr index = getStaticBufferIndex(Buffer, BufferSize);
    if (index < StaticBufferCount) {
      ScopedLock L(Mutex);
      DCHECK_EQ((Mask & (static_cast<uptr>(1) << index)), 0U);
      Mask |= static_cast<uptr>(1) << index;
    } else {
      unmap(reinterpret_cast<void *>(Buffer),
            roundUp(BufferSize, getPageSizeCached()));
    }
  }

  bool isStaticBufferTestOnly(uptr *Buffer, uptr BufferSize) {
    return getStaticBufferIndex(Buffer, BufferSize) < StaticBufferCount;
  }

private:
  uptr getStaticBufferIndex(uptr *Buffer, uptr BufferSize) {
    if (UNLIKELY(BufferSize > StaticBufferSize))
      return StaticBufferCount;

    const uptr BufferBase = reinterpret_cast<uptr>(Buffer);
    const uptr RawBufferBase = reinterpret_cast<uptr>(RawBuffer);

    if (BufferBase < RawBufferBase ||
        BufferBase >= RawBufferBase + sizeof(RawBuffer)) {
      return StaticBufferCount;
    }

    DCHECK_LE(BufferSize, StaticBufferSize);
    DCHECK_LE(BufferBase + BufferSize, RawBufferBase + sizeof(RawBuffer));
    DCHECK_EQ((BufferBase - RawBufferBase) % StaticBufferSize, 0U);

    const uptr index =
        (BufferBase - RawBufferBase) / (StaticBufferSize * sizeof(uptr));
    DCHECK_LT(index, StaticBufferCount);
    return index;
  }

  uptr *getDynamicBuffer(const uptr BufferSize) {
    // When using a heap-based buffer, precommit the pages backing the
    // Vmar by passing |MAP_PRECOMMIT| flag. This allows an optimization
    // where page fault exceptions are skipped as the allocated memory
    // is accessed. So far, this is only enabled on Fuchsia. It hasn't proven a
    // performance benefit on other platforms.
    const uptr MmapFlags = MAP_ALLOWNOMEM | (SCUDO_FUCHSIA ? MAP_PRECOMMIT : 0);
    return reinterpret_cast<uptr *>(
        map(nullptr, roundUp(BufferSize, getPageSizeCached()), "scudo:counters",
            MmapFlags, &MapData));
  }

  HybridMutex Mutex;
  // '1' means that buffer index is not used. '0' means the buffer is in use.
  uptr Mask GUARDED_BY(Mutex) = ~static_cast<uptr>(0);
  uptr RawBuffer[StaticBufferCount * StaticBufferSize] GUARDED_BY(Mutex);
  [[no_unique_address]] MapPlatformData MapData = {};
};

// A Region page map is used to record the usage of pages in the regions. It
// implements a packed array of Counters. Each counter occupies 2^N bits, enough
// to store counter's MaxValue. Ctor will try to use a static buffer first, and
// if that fails (the buffer is too small or already locked), will allocate the
// required Buffer via map(). The caller is expected to check whether the
// initialization was successful by checking isAllocated() result. For
// performance sake, none of the accessors check the validity of the arguments,
// It is assumed that Index is always in [0, N) range and the value is not
// incremented past MaxValue.
class RegionPageMap {
public:
  RegionPageMap()
      : Regions(0),
        NumCounters(0),
        CounterSizeBitsLog(0),
        CounterMask(0),
        PackingRatioLog(0),
        BitOffsetMask(0),
        SizePerRegion(0),
        BufferSize(0),
        Buffer(nullptr) {}
  RegionPageMap(uptr NumberOfRegions, uptr CountersPerRegion, uptr MaxValue) {
    reset(NumberOfRegions, CountersPerRegion, MaxValue);
  }
  ~RegionPageMap() {
    if (!isAllocated())
      return;
    Buffers.releaseBuffer(Buffer, BufferSize);
    Buffer = nullptr;
  }

  // Lock of `StaticBuffer` is acquired conditionally and there's no easy way to
  // specify the thread-safety attribute properly in current code structure.
  // Besides, it's the only place we may want to check thread safety. Therefore,
  // it's fine to bypass the thread-safety analysis now.
  void reset(uptr NumberOfRegion, uptr CountersPerRegion, uptr MaxValue) {
    DCHECK_GT(NumberOfRegion, 0);
    DCHECK_GT(CountersPerRegion, 0);
    DCHECK_GT(MaxValue, 0);

    Regions = NumberOfRegion;
    NumCounters = CountersPerRegion;

    constexpr uptr MaxCounterBits = sizeof(*Buffer) * 8UL;
    // Rounding counter storage size up to the power of two allows for using
    // bit shifts calculating particular counter's Index and offset.
    const uptr CounterSizeBits =
        roundUpPowerOfTwo(getMostSignificantSetBitIndex(MaxValue) + 1);
    DCHECK_LE(CounterSizeBits, MaxCounterBits);
    CounterSizeBitsLog = getLog2(CounterSizeBits);
    CounterMask = ~(static_cast<uptr>(0)) >> (MaxCounterBits - CounterSizeBits);

    const uptr PackingRatio = MaxCounterBits >> CounterSizeBitsLog;
    DCHECK_GT(PackingRatio, 0);
    PackingRatioLog = getLog2(PackingRatio);
    BitOffsetMask = PackingRatio - 1;

    SizePerRegion =
        roundUp(NumCounters, static_cast<uptr>(1U) << PackingRatioLog) >>
        PackingRatioLog;
    BufferSize = SizePerRegion * sizeof(*Buffer) * Regions;
    Buffer = Buffers.getBuffer(BufferSize);
    DCHECK_NE(Buffer, nullptr);
  }

  bool isAllocated() const { return !!Buffer; }

  uptr getCount() const { return NumCounters; }

  uptr get(uptr Region, uptr I) const {
    DCHECK_LT(Region, Regions);
    DCHECK_LT(I, NumCounters);
    const uptr Index = I >> PackingRatioLog;
    const uptr BitOffset = (I & BitOffsetMask) << CounterSizeBitsLog;
    return (Buffer[Region * SizePerRegion + Index] >> BitOffset) & CounterMask;
  }

  void inc(uptr Region, uptr I) const {
    DCHECK_LT(get(Region, I), CounterMask);
    const uptr Index = I >> PackingRatioLog;
    const uptr BitOffset = (I & BitOffsetMask) << CounterSizeBitsLog;
    DCHECK_LT(BitOffset, SCUDO_WORDSIZE);
    DCHECK_EQ(isAllCounted(Region, I), false);
    Buffer[Region * SizePerRegion + Index] += static_cast<uptr>(1U)
                                              << BitOffset;
  }

  void incN(uptr Region, uptr I, uptr N) const {
    DCHECK_GT(N, 0U);
    DCHECK_LE(N, CounterMask);
    DCHECK_LE(get(Region, I), CounterMask - N);
    const uptr Index = I >> PackingRatioLog;
    const uptr BitOffset = (I & BitOffsetMask) << CounterSizeBitsLog;
    DCHECK_LT(BitOffset, SCUDO_WORDSIZE);
    DCHECK_EQ(isAllCounted(Region, I), false);
    Buffer[Region * SizePerRegion + Index] += N << BitOffset;
  }

  void incRange(uptr Region, uptr From, uptr To) const {
    DCHECK_LE(From, To);
    const uptr Top = Min(To + 1, NumCounters);
    for (uptr I = From; I < Top; I++)
      inc(Region, I);
  }

  // Set the counter to the max value. Note that the max number of blocks in a
  // page may vary. To provide an easier way to tell if all the blocks are
  // counted for different pages, set to the same max value to denote the
  // all-counted status.
  void setAsAllCounted(uptr Region, uptr I) const {
    DCHECK_LE(get(Region, I), CounterMask);
    const uptr Index = I >> PackingRatioLog;
    const uptr BitOffset = (I & BitOffsetMask) << CounterSizeBitsLog;
    DCHECK_LT(BitOffset, SCUDO_WORDSIZE);
    Buffer[Region * SizePerRegion + Index] |= CounterMask << BitOffset;
  }
  void setAsAllCountedRange(uptr Region, uptr From, uptr To) const {
    DCHECK_LE(From, To);
    const uptr Top = Min(To + 1, NumCounters);
    for (uptr I = From; I < Top; I++)
      setAsAllCounted(Region, I);
  }

  bool updateAsAllCountedIf(uptr Region, uptr I, uptr MaxCount) {
    const uptr Count = get(Region, I);
    if (Count == CounterMask)
      return true;
    if (Count == MaxCount) {
      setAsAllCounted(Region, I);
      return true;
    }
    return false;
  }
  bool isAllCounted(uptr Region, uptr I) const {
    return get(Region, I) == CounterMask;
  }

  uptr getBufferSize() const { return BufferSize; }

private:
  uptr Regions;
  uptr NumCounters;
  uptr CounterSizeBitsLog;
  uptr CounterMask;
  uptr PackingRatioLog;
  uptr BitOffsetMask;

  uptr SizePerRegion;
  uptr BufferSize;
  uptr *Buffer;

  // We may consider making this configurable if there are cases which may
  // benefit from this.
  static const uptr StaticBufferCount = 2U;
  static const uptr StaticBufferSize = 512U;
  static BufferPool<StaticBufferCount, StaticBufferSize> Buffers;
};

template <class ReleaseRecorderT> class FreePagesRangeTracker {
public:
  explicit FreePagesRangeTracker(ReleaseRecorderT &Recorder)
      : Recorder(Recorder), PageSizeLog(getLog2(getPageSizeCached())) {}

  void processNextPage(bool Released) {
    if (Released) {
      if (!InRange) {
        CurrentRangeStatePage = CurrentPage;
        InRange = true;
      }
    } else {
      closeOpenedRange();
    }
    CurrentPage++;
  }

  void skipPages(uptr N) {
    closeOpenedRange();
    CurrentPage += N;
  }

  void finish() { closeOpenedRange(); }

private:
  void closeOpenedRange() {
    if (InRange) {
      Recorder.releasePageRangeToOS((CurrentRangeStatePage << PageSizeLog),
                                    (CurrentPage << PageSizeLog));
      InRange = false;
    }
  }

  ReleaseRecorderT &Recorder;
  const uptr PageSizeLog;
  bool InRange = false;
  uptr CurrentPage = 0;
  uptr CurrentRangeStatePage = 0;
};

struct PageReleaseContext {
  PageReleaseContext(uptr BlockSize, uptr NumberOfRegions, uptr ReleaseSize,
                     uptr ReleaseOffset = 0)
      : BlockSize(BlockSize), NumberOfRegions(NumberOfRegions) {
    PageSize = getPageSizeCached();
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

    // TODO: For multiple regions, it's more complicated to support partial
    // region marking (which includes the complexity of how to handle the last
    // block in a region). We may consider this after markFreeBlocks() accepts
    // only free blocks from the same region.
    if (NumberOfRegions != 1)
      DCHECK_EQ(ReleaseOffset, 0U);

    PagesCount = roundUp(ReleaseSize, PageSize) / PageSize;
    PageSizeLog = getLog2(PageSize);
    ReleasePageOffset = ReleaseOffset >> PageSizeLog;
  }

  // PageMap is lazily allocated when markFreeBlocks() is invoked.
  bool hasBlockMarked() const {
    return PageMap.isAllocated();
  }

  void ensurePageMapAllocated() {
    if (PageMap.isAllocated())
      return;
    PageMap.reset(NumberOfRegions, PagesCount, FullPagesBlockCountMax);
    DCHECK(PageMap.isAllocated());
  }

  // Mark all the blocks in the given range [From, to). Instead of visiting all
  // the blocks, we will just mark the page as all counted. Note the `From` and
  // `To` has to be page aligned but with one exception, if `To` is equal to the
  // RegionSize, it's not necessary to be aligned with page size.
  void markRangeAsAllCounted(uptr From, uptr To, uptr Base,
                             const uptr RegionIndex, const uptr RegionSize) {
    DCHECK_LT(From, To);
    DCHECK_LE(To, Base + RegionSize);
    DCHECK_EQ(From % PageSize, 0U);
    DCHECK_LE(To - From, RegionSize);

    ensurePageMapAllocated();

    uptr FromInRegion = From - Base;
    uptr ToInRegion = To - Base;
    uptr FirstBlockInRange = roundUpSlow(FromInRegion, BlockSize);

    // The straddling block sits across entire range.
    if (FirstBlockInRange >= ToInRegion)
      return;

    // First block may not sit at the first pape in the range, move
    // `FromInRegion` to the first block page.
    FromInRegion = roundDown(FirstBlockInRange, PageSize);

    // When The first block is not aligned to the range boundary, which means
    // there is a block sitting acorss `From`, that looks like,
    //
    //   From                                             To
    //     V                                               V
    //     +-----------------------------------------------+
    //  +-----+-----+-----+-----+
    //  |     |     |     |     | ...
    //  +-----+-----+-----+-----+
    //     |-    first page     -||-    second page    -||- ...
    //
    // Therefore, we can't just mark the first page as all counted. Instead, we
    // increment the number of blocks in the first page in the page map and
    // then round up the `From` to the next page.
    if (FirstBlockInRange != FromInRegion) {
      DCHECK_GT(FromInRegion + PageSize, FirstBlockInRange);
      uptr NumBlocksInFirstPage =
          (FromInRegion + PageSize - FirstBlockInRange + BlockSize - 1) /
          BlockSize;
      PageMap.incN(RegionIndex, getPageIndex(FromInRegion),
                   NumBlocksInFirstPage);
      FromInRegion = roundUp(FromInRegion + 1, PageSize);
    }

    uptr LastBlockInRange = roundDownSlow(ToInRegion - 1, BlockSize);
    if (LastBlockInRange < FromInRegion)
      return;

    // When the last block sits across `To`, we can't just mark the pages
    // occupied by the last block as all counted. Instead, we increment the
    // counters of those pages by 1. The exception is that if it's the last
    // block in the region, it's fine to mark those pages as all counted.
    if (LastBlockInRange + BlockSize != RegionSize) {
      DCHECK_EQ(ToInRegion % PageSize, 0U);
      // The case below is like,
      //
      //   From                                      To
      //     V                                        V
      //     +----------------------------------------+
      //                          +-----+-----+-----+-----+
      //                          |     |     |     |     | ...
      //                          +-----+-----+-----+-----+
      //                    ... -||-    last page    -||-    next page    -|
      //
      // The last block is not aligned to `To`, we need to increment the
      // counter of `next page` by 1.
      if (LastBlockInRange + BlockSize != ToInRegion) {
        PageMap.incRange(RegionIndex, getPageIndex(ToInRegion),
                         getPageIndex(LastBlockInRange + BlockSize - 1));
      }
    } else {
      ToInRegion = RegionSize;
    }

    // After handling the first page and the last block, it's safe to mark any
    // page in between the range [From, To).
    if (FromInRegion < ToInRegion) {
      PageMap.setAsAllCountedRange(RegionIndex, getPageIndex(FromInRegion),
                                   getPageIndex(ToInRegion - 1));
    }
  }

  template <class TransferBatchT, typename DecompactPtrT>
  void markFreeBlocksInRegion(const IntrusiveList<TransferBatchT> &FreeList,
                              DecompactPtrT DecompactPtr, const uptr Base,
                              const uptr RegionIndex, const uptr RegionSize,
                              bool MayContainLastBlockInRegion) {
    ensurePageMapAllocated();

    if (MayContainLastBlockInRegion) {
      const uptr LastBlockInRegion =
          ((RegionSize / BlockSize) - 1U) * BlockSize;
      // The last block in a region may not use the entire page, we mark the
      // following "pretend" memory block(s) as free in advance.
      //
      //     Region Boundary
      //         v
      //  -----+-----------------------+
      //       |      Last Page        | <- Rounded Region Boundary
      //  -----+-----------------------+
      //   |-----||- trailing blocks  -|
      //      ^
      //   last block
      const uptr RoundedRegionSize = roundUp(RegionSize, PageSize);
      const uptr TrailingBlockBase = LastBlockInRegion + BlockSize;
      // Only the last page touched by the last block needs to mark the trailing
      // blocks. If the difference between `RoundedRegionSize` and
      // `TrailingBlockBase` is larger than a page, that implies the reported
      // `RegionSize` may not be accurate.
      DCHECK_LT(RoundedRegionSize - TrailingBlockBase, PageSize);
      uptr NumTrailingBlocks =
          roundUpSlow(RoundedRegionSize - TrailingBlockBase, BlockSize) /
          BlockSize;
      if (NumTrailingBlocks > 0) {
        PageMap.incN(RegionIndex, getPageIndex(TrailingBlockBase),
                     NumTrailingBlocks);
      }
    }

    // Iterate over free chunks and count how many free chunks affect each
    // allocated page.
    if (BlockSize <= PageSize && PageSize % BlockSize == 0) {
      // Each chunk affects one page only.
      for (const auto &It : FreeList) {
        for (u16 I = 0; I < It.getCount(); I++) {
          const uptr PInRegion = DecompactPtr(It.get(I)) - Base;
          DCHECK_LT(PInRegion, RegionSize);
          PageMap.inc(RegionIndex, getPageIndex(PInRegion));
        }
      }
    } else {
      // In all other cases chunks might affect more than one page.
      DCHECK_GE(RegionSize, BlockSize);
      for (const auto &It : FreeList) {
        for (u16 I = 0; I < It.getCount(); I++) {
          const uptr PInRegion = DecompactPtr(It.get(I)) - Base;
          PageMap.incRange(RegionIndex, getPageIndex(PInRegion),
                           getPageIndex(PInRegion + BlockSize - 1));
        }
      }
    }
  }

  uptr getPageIndex(uptr P) { return (P >> PageSizeLog) - ReleasePageOffset; }

  uptr BlockSize;
  uptr NumberOfRegions;
  // For partial region marking, some pages in front are not needed to be
  // counted.
  uptr ReleasePageOffset;
  uptr PageSize;
  uptr PagesCount;
  uptr PageSizeLog;
  uptr FullPagesBlockCountMax;
  bool SameBlockCountPerPage;
  RegionPageMap PageMap;
};

// Try to release the page which doesn't have any in-used block, i.e., they are
// all free blocks. The `PageMap` will record the number of free blocks in each
// page.
template <class ReleaseRecorderT, typename SkipRegionT>
NOINLINE void
releaseFreeMemoryToOS(PageReleaseContext &Context,
                      ReleaseRecorderT &Recorder, SkipRegionT SkipRegion) {
  const uptr PageSize = Context.PageSize;
  const uptr BlockSize = Context.BlockSize;
  const uptr PagesCount = Context.PagesCount;
  const uptr NumberOfRegions = Context.NumberOfRegions;
  const uptr ReleasePageOffset = Context.ReleasePageOffset;
  const uptr FullPagesBlockCountMax = Context.FullPagesBlockCountMax;
  const bool SameBlockCountPerPage = Context.SameBlockCountPerPage;
  RegionPageMap &PageMap = Context.PageMap;

  // Iterate over pages detecting ranges of pages with chunk Counters equal
  // to the expected number of chunks for the particular page.
  FreePagesRangeTracker<ReleaseRecorderT> RangeTracker(Recorder);
  if (SameBlockCountPerPage) {
    // Fast path, every page has the same number of chunks affecting it.
    for (uptr I = 0; I < NumberOfRegions; I++) {
      if (SkipRegion(I)) {
        RangeTracker.skipPages(PagesCount);
        continue;
      }
      for (uptr J = 0; J < PagesCount; J++) {
        const bool CanRelease =
            PageMap.updateAsAllCountedIf(I, J, FullPagesBlockCountMax);
        RangeTracker.processNextPage(CanRelease);
      }
    }
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
    for (uptr I = 0; I < NumberOfRegions; I++) {
      if (SkipRegion(I)) {
        RangeTracker.skipPages(PagesCount);
        continue;
      }
      uptr PrevPageBoundary = 0;
      uptr CurrentBoundary = 0;
      if (ReleasePageOffset > 0) {
        PrevPageBoundary = ReleasePageOffset * PageSize;
        CurrentBoundary = roundUpSlow(PrevPageBoundary, BlockSize);
      }
      for (uptr J = 0; J < PagesCount; J++) {
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
        const bool CanRelease =
            PageMap.updateAsAllCountedIf(I, J, BlocksPerPage);
        RangeTracker.processNextPage(CanRelease);
      }
    }
  }
  RangeTracker.finish();
}

} // namespace scudo

#endif // SCUDO_RELEASE_H_
