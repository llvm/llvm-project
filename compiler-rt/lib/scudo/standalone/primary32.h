//===-- primary32.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SCUDO_PRIMARY32_H_
#define SCUDO_PRIMARY32_H_

#include "bytemap.h"
#include "common.h"
#include "list.h"
#include "local_cache.h"
#include "release.h"
#include "report.h"
#include "stats.h"
#include "string_utils.h"

namespace scudo {

// SizeClassAllocator32 is an allocator for 32 or 64-bit address space.
//
// It maps Regions of 2^RegionSizeLog bytes aligned on a 2^RegionSizeLog bytes
// boundary, and keeps a bytemap of the mappable address space to track the size
// class they are associated with.
//
// Mapped regions are split into equally sized Blocks according to the size
// class they belong to, and the associated pointers are shuffled to prevent any
// predictable address pattern (the predictability increases with the block
// size).
//
// Regions for size class 0 are special and used to hold TransferBatches, which
// allow to transfer arrays of pointers from the global size class freelist to
// the thread specific freelist for said class, and back.
//
// Memory used by this allocator is never unmapped but can be partially
// reclaimed if the platform allows for it.

template <class SizeClassMapT, uptr RegionSizeLog> class SizeClassAllocator32 {
public:
  typedef SizeClassMapT SizeClassMap;
  // Regions should be large enough to hold the largest Block.
  COMPILER_CHECK((1UL << RegionSizeLog) >= SizeClassMap::MaxSize);
  typedef SizeClassAllocator32<SizeClassMapT, RegionSizeLog> ThisT;
  typedef SizeClassAllocatorLocalCache<ThisT> CacheT;
  typedef typename CacheT::TransferBatch TransferBatch;

  static uptr getSizeByClassId(uptr ClassId) {
    return (ClassId == SizeClassMap::BatchClassId)
               ? sizeof(TransferBatch)
               : SizeClassMap::getSizeByClassId(ClassId);
  }

  static bool canAllocate(uptr Size) { return Size <= SizeClassMap::MaxSize; }

  void initLinkerInitialized(s32 ReleaseToOsInterval) {
    if (SCUDO_FUCHSIA)
      reportError("SizeClassAllocator32 is not supported on Fuchsia");

    PossibleRegions.initLinkerInitialized();
    MinRegionIndex = NumRegions; // MaxRegionIndex is already initialized to 0.

    u32 Seed;
    if (UNLIKELY(!getRandom(reinterpret_cast<void *>(&Seed), sizeof(Seed))))
      Seed =
          static_cast<u32>(getMonotonicTime() ^
                           (reinterpret_cast<uptr>(SizeClassInfoArray) >> 6));
    const uptr PageSize = getPageSizeCached();
    for (uptr I = 0; I < NumClasses; I++) {
      SizeClassInfo *Sci = getSizeClassInfo(I);
      Sci->RandState = getRandomU32(&Seed);
      // See comment in the 64-bit primary about releasing smaller size classes.
      Sci->CanRelease = (ReleaseToOsInterval >= 0) &&
                        (I != SizeClassMap::BatchClassId) &&
                        (getSizeByClassId(I) >= (PageSize / 32));
    }
    ReleaseToOsIntervalMs = ReleaseToOsInterval;
  }
  void init(s32 ReleaseToOsInterval) {
    memset(this, 0, sizeof(*this));
    initLinkerInitialized(ReleaseToOsInterval);
  }

  void unmapTestOnly() {
    while (NumberOfStashedRegions > 0)
      unmap(reinterpret_cast<void *>(RegionsStash[--NumberOfStashedRegions]),
            RegionSize);
    // TODO(kostyak): unmap the TransferBatch regions as well.
    for (uptr I = 0; I < NumRegions; I++)
      if (PossibleRegions[I])
        unmap(reinterpret_cast<void *>(I * RegionSize), RegionSize);
    PossibleRegions.unmapTestOnly();
  }

  TransferBatch *popBatch(CacheT *C, uptr ClassId) {
    DCHECK_LT(ClassId, NumClasses);
    SizeClassInfo *Sci = getSizeClassInfo(ClassId);
    ScopedLock L(Sci->Mutex);
    TransferBatch *B = Sci->FreeList.front();
    if (B) {
      Sci->FreeList.pop_front();
    } else {
      B = populateFreeList(C, ClassId, Sci);
      if (UNLIKELY(!B))
        return nullptr;
    }
    DCHECK_GT(B->getCount(), 0);
    Sci->Stats.PoppedBlocks += B->getCount();
    return B;
  }

  void pushBatch(uptr ClassId, TransferBatch *B) {
    DCHECK_LT(ClassId, NumClasses);
    DCHECK_GT(B->getCount(), 0);
    SizeClassInfo *Sci = getSizeClassInfo(ClassId);
    ScopedLock L(Sci->Mutex);
    Sci->FreeList.push_front(B);
    Sci->Stats.PushedBlocks += B->getCount();
    if (Sci->CanRelease)
      releaseToOSMaybe(Sci, ClassId);
  }

  void disable() {
    for (uptr I = 0; I < NumClasses; I++)
      getSizeClassInfo(I)->Mutex.lock();
  }

  void enable() {
    for (sptr I = static_cast<sptr>(NumClasses) - 1; I >= 0; I--)
      getSizeClassInfo(static_cast<uptr>(I))->Mutex.unlock();
  }

  template <typename F> void iterateOverBlocks(F Callback) {
    for (uptr I = MinRegionIndex; I <= MaxRegionIndex; I++)
      if (PossibleRegions[I]) {
        const uptr BlockSize = getSizeByClassId(PossibleRegions[I]);
        const uptr From = I * RegionSize;
        const uptr To = From + (RegionSize / BlockSize) * BlockSize;
        for (uptr Block = From; Block < To; Block += BlockSize)
          Callback(Block);
      }
  }

  void getStats(ScopedString *Str) {
    // TODO(kostyak): get the RSS per region.
    uptr TotalMapped = 0;
    uptr PoppedBlocks = 0;
    uptr PushedBlocks = 0;
    for (uptr I = 0; I < NumClasses; I++) {
      SizeClassInfo *Sci = getSizeClassInfo(I);
      TotalMapped += Sci->AllocatedUser;
      PoppedBlocks += Sci->Stats.PoppedBlocks;
      PushedBlocks += Sci->Stats.PushedBlocks;
    }
    Str->append("Stats: SizeClassAllocator32: %zuM mapped in %zu allocations; "
                "remains %zu\n",
                TotalMapped >> 20, PoppedBlocks, PoppedBlocks - PushedBlocks);
    for (uptr I = 0; I < NumClasses; I++)
      getStats(Str, I, 0);
  }

  uptr releaseToOS() {
    uptr TotalReleasedBytes = 0;
    for (uptr I = 0; I < NumClasses; I++) {
      if (I == SizeClassMap::BatchClassId)
        continue;
      SizeClassInfo *Sci = getSizeClassInfo(I);
      ScopedLock L(Sci->Mutex);
      TotalReleasedBytes += releaseToOSMaybe(Sci, I, /*Force=*/true);
    }
    return TotalReleasedBytes;
  }

private:
  static const uptr NumClasses = SizeClassMap::NumClasses;
  static const uptr RegionSize = 1UL << RegionSizeLog;
  static const uptr NumRegions = SCUDO_MMAP_RANGE_SIZE >> RegionSizeLog;
#if SCUDO_WORDSIZE == 32U
  typedef FlatByteMap<NumRegions> ByteMap;
#else
  typedef TwoLevelByteMap<(NumRegions >> 12), 1UL << 12> ByteMap;
#endif

  struct SizeClassStats {
    uptr PoppedBlocks;
    uptr PushedBlocks;
  };

  struct ReleaseToOsInfo {
    uptr PushedBlocksAtLastRelease;
    uptr RangesReleased;
    uptr LastReleasedBytes;
    u64 LastReleaseAtNs;
  };

  struct ALIGNED(SCUDO_CACHE_LINE_SIZE) SizeClassInfo {
    HybridMutex Mutex;
    SinglyLinkedList<TransferBatch> FreeList;
    SizeClassStats Stats;
    bool CanRelease;
    u32 RandState;
    uptr AllocatedUser;
    ReleaseToOsInfo ReleaseInfo;
  };
  COMPILER_CHECK(sizeof(SizeClassInfo) % SCUDO_CACHE_LINE_SIZE == 0);

  uptr computeRegionId(uptr Mem) {
    const uptr Id = Mem >> RegionSizeLog;
    CHECK_LT(Id, NumRegions);
    return Id;
  }

  uptr allocateRegionSlow() {
    uptr MapSize = 2 * RegionSize;
    const uptr MapBase = reinterpret_cast<uptr>(
        map(nullptr, MapSize, "scudo:primary", MAP_ALLOWNOMEM));
    if (UNLIKELY(!MapBase))
      return 0;
    const uptr MapEnd = MapBase + MapSize;
    uptr Region = MapBase;
    if (isAligned(Region, RegionSize)) {
      ScopedLock L(RegionsStashMutex);
      if (NumberOfStashedRegions < MaxStashedRegions)
        RegionsStash[NumberOfStashedRegions++] = MapBase + RegionSize;
      else
        MapSize = RegionSize;
    } else {
      Region = roundUpTo(MapBase, RegionSize);
      unmap(reinterpret_cast<void *>(MapBase), Region - MapBase);
      MapSize = RegionSize;
    }
    const uptr End = Region + MapSize;
    if (End != MapEnd)
      unmap(reinterpret_cast<void *>(End), MapEnd - End);
    return Region;
  }

  uptr allocateRegion(uptr ClassId) {
    DCHECK_LT(ClassId, NumClasses);
    uptr Region = 0;
    {
      ScopedLock L(RegionsStashMutex);
      if (NumberOfStashedRegions > 0)
        Region = RegionsStash[--NumberOfStashedRegions];
    }
    if (!Region)
      Region = allocateRegionSlow();
    if (LIKELY(Region)) {
      if (ClassId) {
        const uptr RegionIndex = computeRegionId(Region);
        if (RegionIndex < MinRegionIndex)
          MinRegionIndex = RegionIndex;
        if (RegionIndex > MaxRegionIndex)
          MaxRegionIndex = RegionIndex;
        PossibleRegions.set(RegionIndex, static_cast<u8>(ClassId));
      }
    }
    return Region;
  }

  SizeClassInfo *getSizeClassInfo(uptr ClassId) {
    DCHECK_LT(ClassId, NumClasses);
    return &SizeClassInfoArray[ClassId];
  }

  bool populateBatches(CacheT *C, SizeClassInfo *Sci, uptr ClassId,
                       TransferBatch **CurrentBatch, u32 MaxCount,
                       void **PointersArray, u32 Count) {
    if (ClassId != SizeClassMap::BatchClassId)
      shuffle(PointersArray, Count, &Sci->RandState);
    TransferBatch *B = *CurrentBatch;
    for (uptr I = 0; I < Count; I++) {
      if (B && B->getCount() == MaxCount) {
        Sci->FreeList.push_back(B);
        B = nullptr;
      }
      if (!B) {
        B = C->createBatch(ClassId, PointersArray[I]);
        if (UNLIKELY(!B))
          return false;
        B->clear();
      }
      B->add(PointersArray[I]);
    }
    *CurrentBatch = B;
    return true;
  }

  NOINLINE TransferBatch *populateFreeList(CacheT *C, uptr ClassId,
                                           SizeClassInfo *Sci) {
    const uptr Region = allocateRegion(ClassId);
    if (UNLIKELY(!Region))
      return nullptr;
    C->getStats().add(StatMapped, RegionSize);
    const uptr Size = getSizeByClassId(ClassId);
    const u32 MaxCount = TransferBatch::getMaxCached(Size);
    DCHECK_GT(MaxCount, 0);
    const uptr NumberOfBlocks = RegionSize / Size;
    DCHECK_GT(NumberOfBlocks, 0);
    TransferBatch *B = nullptr;
    constexpr uptr ShuffleArraySize = 48;
    void *ShuffleArray[ShuffleArraySize];
    u32 Count = 0;
    const uptr AllocatedUser = NumberOfBlocks * Size;
    for (uptr I = Region; I < Region + AllocatedUser; I += Size) {
      ShuffleArray[Count++] = reinterpret_cast<void *>(I);
      if (Count == ShuffleArraySize) {
        if (UNLIKELY(!populateBatches(C, Sci, ClassId, &B, MaxCount,
                                      ShuffleArray, Count)))
          return nullptr;
        Count = 0;
      }
    }
    if (Count) {
      if (UNLIKELY(!populateBatches(C, Sci, ClassId, &B, MaxCount, ShuffleArray,
                                    Count)))
        return nullptr;
    }
    DCHECK(B);
    DCHECK_GT(B->getCount(), 0);

    C->getStats().add(StatFree, AllocatedUser);
    Sci->AllocatedUser += AllocatedUser;
    if (Sci->CanRelease)
      Sci->ReleaseInfo.LastReleaseAtNs = getMonotonicTime();
    return B;
  }

  void getStats(ScopedString *Str, uptr ClassId, uptr Rss) {
    SizeClassInfo *Sci = getSizeClassInfo(ClassId);
    if (Sci->AllocatedUser == 0)
      return;
    const uptr InUse = Sci->Stats.PoppedBlocks - Sci->Stats.PushedBlocks;
    const uptr AvailableChunks = Sci->AllocatedUser / getSizeByClassId(ClassId);
    Str->append("  %02zu (%6zu): mapped: %6zuK popped: %7zu pushed: %7zu "
                "inuse: %6zu avail: %6zu rss: %6zuK\n",
                ClassId, getSizeByClassId(ClassId), Sci->AllocatedUser >> 10,
                Sci->Stats.PoppedBlocks, Sci->Stats.PushedBlocks, InUse,
                AvailableChunks, Rss >> 10);
  }

  NOINLINE uptr releaseToOSMaybe(SizeClassInfo *Sci, uptr ClassId,
                                 bool Force = false) {
    const uptr BlockSize = getSizeByClassId(ClassId);
    const uptr PageSize = getPageSizeCached();

    CHECK_GE(Sci->Stats.PoppedBlocks, Sci->Stats.PushedBlocks);
    const uptr BytesInFreeList =
        Sci->AllocatedUser -
        (Sci->Stats.PoppedBlocks - Sci->Stats.PushedBlocks) * BlockSize;
    if (BytesInFreeList < PageSize)
      return 0; // No chance to release anything.
    if ((Sci->Stats.PushedBlocks - Sci->ReleaseInfo.PushedBlocksAtLastRelease) *
            BlockSize <
        PageSize) {
      return 0; // Nothing new to release.
    }

    if (!Force) {
      const s32 IntervalMs = ReleaseToOsIntervalMs;
      if (IntervalMs < 0)
        return 0;
      if (Sci->ReleaseInfo.LastReleaseAtNs +
              static_cast<uptr>(IntervalMs) * 1000000ULL >
          getMonotonicTime()) {
        return 0; // Memory was returned recently.
      }
    }

    // TODO(kostyak): currently not ideal as we loop over all regions and
    // iterate multiple times over the same freelist if a ClassId spans multiple
    // regions. But it will have to do for now.
    uptr TotalReleasedBytes = 0;
    for (uptr I = MinRegionIndex; I <= MaxRegionIndex; I++) {
      if (PossibleRegions[I] == ClassId) {
        ReleaseRecorder Recorder(I * RegionSize);
        releaseFreeMemoryToOS(Sci->FreeList, I * RegionSize,
                              RegionSize / PageSize, BlockSize, &Recorder);
        if (Recorder.getReleasedRangesCount() > 0) {
          Sci->ReleaseInfo.PushedBlocksAtLastRelease = Sci->Stats.PushedBlocks;
          Sci->ReleaseInfo.RangesReleased += Recorder.getReleasedRangesCount();
          Sci->ReleaseInfo.LastReleasedBytes = Recorder.getReleasedBytes();
          TotalReleasedBytes += Sci->ReleaseInfo.LastReleasedBytes;
        }
      }
    }
    Sci->ReleaseInfo.LastReleaseAtNs = getMonotonicTime();
    return TotalReleasedBytes;
  }

  SizeClassInfo SizeClassInfoArray[NumClasses];

  ByteMap PossibleRegions;
  // Keep track of the lowest & highest regions allocated to avoid looping
  // through the whole NumRegions.
  uptr MinRegionIndex;
  uptr MaxRegionIndex;
  s32 ReleaseToOsIntervalMs;
  // Unless several threads request regions simultaneously from different size
  // classes, the stash rarely contains more than 1 entry.
  static constexpr uptr MaxStashedRegions = 4;
  HybridMutex RegionsStashMutex;
  uptr NumberOfStashedRegions;
  uptr RegionsStash[MaxStashedRegions];
};

} // namespace scudo

#endif // SCUDO_PRIMARY32_H_
