//===-- primary64.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SCUDO_PRIMARY64_H_
#define SCUDO_PRIMARY64_H_

#include "bytemap.h"
#include "common.h"
#include "list.h"
#include "local_cache.h"
#include "release.h"
#include "stats.h"
#include "string_utils.h"

namespace scudo {

// SizeClassAllocator64 is an allocator tuned for 64-bit address space.
//
// It starts by reserving NumClasses * 2^RegionSizeLog bytes, equally divided in
// Regions, specific to each size class. Note that the base of that mapping is
// random (based to the platform specific map() capabilities), and that each
// Region actually starts at a random offset from its base.
//
// Regions are mapped incrementally on demand to fulfill allocation requests,
// those mappings being split into equally sized Blocks based on the size class
// they belong to. The Blocks created are shuffled to prevent predictable
// address patterns (the predictability increases with the size of the Blocks).
//
// The 1st Region (for size class 0) holds the TransferBatches. This is a
// structure used to transfer arrays of available pointers from the class size
// freelist to the thread specific freelist, and back.
//
// The memory used by this allocator is never unmapped, but can be partially
// released it the platform allows for it.

template <class SizeClassMapT, uptr RegionSizeLog> class SizeClassAllocator64 {
public:
  typedef SizeClassMapT SizeClassMap;
  typedef SizeClassAllocator64<SizeClassMap, RegionSizeLog> ThisT;
  typedef SizeClassAllocatorLocalCache<ThisT> CacheT;
  typedef typename CacheT::TransferBatch TransferBatch;

  static uptr getSizeByClassId(uptr ClassId) {
    return (ClassId == SizeClassMap::BatchClassId)
               ? sizeof(TransferBatch)
               : SizeClassMap::getSizeByClassId(ClassId);
  }

  static bool canAllocate(uptr Size) { return Size <= SizeClassMap::MaxSize; }

  void initLinkerInitialized(s32 ReleaseToOsInterval) {
    // Reserve the space required for the Primary.
    PrimaryBase = reinterpret_cast<uptr>(
        map(nullptr, PrimarySize, "scudo:primary", MAP_NOACCESS, &Data));

    RegionInfoArray = reinterpret_cast<RegionInfo *>(
        map(nullptr, sizeof(RegionInfo) * NumClasses, "scudo:regioninfo"));
    DCHECK_EQ(reinterpret_cast<uptr>(RegionInfoArray) % SCUDO_CACHE_LINE_SIZE,
              0);

    u32 Seed;
    if (UNLIKELY(!getRandom(reinterpret_cast<void *>(&Seed), sizeof(Seed))))
      Seed = static_cast<u32>(getMonotonicTime() ^ (PrimaryBase >> 12));
    const uptr PageSize = getPageSizeCached();
    for (uptr I = 0; I < NumClasses; I++) {
      RegionInfo *Region = getRegionInfo(I);
      // The actual start of a region is offseted by a random number of pages.
      Region->RegionBeg =
          getRegionBaseByClassId(I) + (getRandomModN(&Seed, 16) + 1) * PageSize;
      // Releasing smaller size classes doesn't necessarily yield to a
      // meaningful RSS impact: there are more blocks per page, they are
      // randomized around, and thus pages are less likely to be entirely empty.
      // On top of this, attempting to release those require more iterations and
      // memory accesses which ends up being fairly costly. The current lower
      // limit is mostly arbitrary and based on empirical observations.
      // TODO(kostyak): make the lower limit a runtime option
      Region->CanRelease = (ReleaseToOsInterval > 0) &&
                           (I != SizeClassMap::BatchClassId) &&
                           (getSizeByClassId(I) >= (PageSize / 32));
      Region->RandState = getRandomU32(&Seed);
    }
    ReleaseToOsIntervalMs = ReleaseToOsInterval;
  }
  void init(s32 ReleaseToOsInterval) {
    memset(this, 0, sizeof(*this));
    initLinkerInitialized(ReleaseToOsInterval);
  }

  void unmapTestOnly() {
    unmap(reinterpret_cast<void *>(PrimaryBase), PrimarySize, UNMAP_ALL, &Data);
    unmap(reinterpret_cast<void *>(RegionInfoArray),
          sizeof(RegionInfo) * NumClasses);
  }

  TransferBatch *popBatch(CacheT *C, uptr ClassId) {
    DCHECK_LT(ClassId, NumClasses);
    RegionInfo *Region = getRegionInfo(ClassId);
    BlockingMutexLock L(&Region->Mutex);
    TransferBatch *B = Region->FreeList.front();
    if (B)
      Region->FreeList.pop_front();
    else {
      B = populateFreeList(C, ClassId, Region);
      if (UNLIKELY(!B))
        return nullptr;
    }
    DCHECK_GT(B->getCount(), 0);
    Region->Stats.PoppedBlocks += B->getCount();
    return B;
  }

  void pushBatch(uptr ClassId, TransferBatch *B) {
    DCHECK_GT(B->getCount(), 0);
    RegionInfo *Region = getRegionInfo(ClassId);
    BlockingMutexLock L(&Region->Mutex);
    Region->FreeList.push_front(B);
    Region->Stats.PushedBlocks += B->getCount();
    if (Region->CanRelease)
      releaseToOSMaybe(Region, ClassId);
  }

  void disable() {
    for (uptr I = 0; I < NumClasses; I++)
      getRegionInfo(I)->Mutex.lock();
  }

  void enable() {
    for (sptr I = static_cast<sptr>(NumClasses) - 1; I >= 0; I--)
      getRegionInfo(I)->Mutex.unlock();
  }

  template <typename F> void iterateOverBlocks(F Callback) const {
    for (uptr I = 1; I < NumClasses; I++) {
      const RegionInfo *Region = getRegionInfo(I);
      const uptr BlockSize = getSizeByClassId(I);
      const uptr From = Region->RegionBeg;
      const uptr To = From + Region->AllocatedUser;
      for (uptr Block = From; Block < To; Block += BlockSize)
        Callback(Block);
    }
  }

  void printStats() const {
    // TODO(kostyak): get the RSS per region.
    uptr TotalMapped = 0;
    uptr PoppedBlocks = 0;
    uptr PushedBlocks = 0;
    for (uptr I = 0; I < NumClasses; I++) {
      RegionInfo *Region = getRegionInfo(I);
      if (Region->MappedUser)
        TotalMapped += Region->MappedUser;
      PoppedBlocks += Region->Stats.PoppedBlocks;
      PushedBlocks += Region->Stats.PushedBlocks;
    }
    Printf("Stats: Primary64: %zuM mapped (%zuM rss) in %zu allocations; "
           "remains %zu\n",
           TotalMapped >> 20, 0, PoppedBlocks, PoppedBlocks - PushedBlocks);

    for (uptr I = 0; I < NumClasses; I++)
      printStats(I, 0);
  }

  void releaseToOS() {
    for (uptr I = 1; I < NumClasses; I++) {
      RegionInfo *Region = getRegionInfo(I);
      BlockingMutexLock L(&Region->Mutex);
      releaseToOSMaybe(Region, I, /*Force=*/true);
    }
  }

private:
  static const uptr RegionSize = 1UL << RegionSizeLog;
  static const uptr NumClasses = SizeClassMap::NumClasses;
  static const uptr PrimarySize = RegionSize * NumClasses;

  // Call map for user memory with at least this size.
  static const uptr MapSizeIncrement = 1UL << 16;

  struct RegionStats {
    uptr PoppedBlocks;
    uptr PushedBlocks;
  };

  struct ReleaseToOsInfo {
    uptr PushedBlocksAtLastRelease;
    uptr RangesReleased;
    uptr LastReleasedBytes;
    u64 LastReleaseAtNs;
  };

  struct ALIGNED(SCUDO_CACHE_LINE_SIZE) RegionInfo {
    BlockingMutex Mutex;
    IntrusiveList<TransferBatch> FreeList;
    RegionStats Stats;
    bool CanRelease;
    bool Exhausted;
    u32 RandState;
    uptr RegionBeg;
    uptr MappedUser;    // Bytes mapped for user memory.
    uptr AllocatedUser; // Bytes allocated for user memory.
    MapPlatformData Data;
    ReleaseToOsInfo ReleaseInfo;
  };
  COMPILER_CHECK(sizeof(RegionInfo) % SCUDO_CACHE_LINE_SIZE == 0);

  uptr PrimaryBase;
  RegionInfo *RegionInfoArray;
  MapPlatformData Data;
  s32 ReleaseToOsIntervalMs;

  RegionInfo *getRegionInfo(uptr ClassId) const {
    DCHECK_LT(ClassId, NumClasses);
    return &RegionInfoArray[ClassId];
  }

  uptr getRegionBaseByClassId(uptr ClassId) const {
    return PrimaryBase + (ClassId << RegionSizeLog);
  }

  bool populateBatches(CacheT *C, RegionInfo *Region, uptr ClassId,
                       TransferBatch **CurrentBatch, u32 MaxCount,
                       void **PointersArray, u32 Count) {
    // No need to shuffle the batches size class.
    if (ClassId != SizeClassMap::BatchClassId)
      shuffle(PointersArray, Count, &Region->RandState);
    TransferBatch *B = *CurrentBatch;
    for (uptr I = 0; I < Count; I++) {
      if (B && B->getCount() == MaxCount) {
        Region->FreeList.push_back(B);
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
                                           RegionInfo *Region) {
    const uptr Size = getSizeByClassId(ClassId);
    const u32 MaxCount = TransferBatch::MaxCached(Size);

    const uptr RegionBeg = Region->RegionBeg;
    const uptr MappedUser = Region->MappedUser;
    const uptr TotalUserBytes = Region->AllocatedUser + MaxCount * Size;
    // Map more space for blocks, if necessary.
    if (LIKELY(TotalUserBytes > MappedUser)) {
      // Do the mmap for the user memory.
      const uptr UserMapSize =
          roundUpTo(TotalUserBytes - MappedUser, MapSizeIncrement);
      const uptr RegionBase = RegionBeg - getRegionBaseByClassId(ClassId);
      if (UNLIKELY(RegionBase + MappedUser + UserMapSize > RegionSize)) {
        if (!Region->Exhausted) {
          Region->Exhausted = true;
          printStats();
          Printf(
              "Scudo OOM: The process has Exhausted %zuM for size class %zu.\n",
              RegionSize >> 20, Size);
        }
        return nullptr;
      }
      if (MappedUser == 0)
        Region->Data = Data;
      if (UNLIKELY(!map(reinterpret_cast<void *>(RegionBeg + MappedUser),
                        UserMapSize, "scudo:primary",
                        MAP_ALLOWNOMEM | MAP_RESIZABLE, &Region->Data)))
        return nullptr;
      Region->MappedUser += UserMapSize;
      C->getStats().add(StatMapped, UserMapSize);
    }

    const uptr NumberOfBlocks = Min(
        8UL * MaxCount, (Region->MappedUser - Region->AllocatedUser) / Size);
    DCHECK_GT(NumberOfBlocks, 0);

    TransferBatch *B = nullptr;
    constexpr uptr ShuffleArraySize = 48;
    void *ShuffleArray[ShuffleArraySize];
    u32 Count = 0;
    const uptr P = RegionBeg + Region->AllocatedUser;
    const uptr AllocatedUser = NumberOfBlocks * Size;
    for (uptr I = P; I < P + AllocatedUser; I += Size) {
      ShuffleArray[Count++] = reinterpret_cast<void *>(I);
      if (Count == ShuffleArraySize) {
        if (UNLIKELY(!populateBatches(C, Region, ClassId, &B, MaxCount,
                                      ShuffleArray, Count)))
          return nullptr;
        Count = 0;
      }
    }
    if (Count) {
      if (UNLIKELY(!populateBatches(C, Region, ClassId, &B, MaxCount,
                                    ShuffleArray, Count)))
        return nullptr;
    }
    DCHECK(B);
    CHECK_GT(B->getCount(), 0);

    Region->AllocatedUser += AllocatedUser;
    Region->Exhausted = false;
    if (Region->CanRelease)
      Region->ReleaseInfo.LastReleaseAtNs = getMonotonicTime();

    return B;
  }

  void printStats(uptr ClassId, uptr Rss) const {
    RegionInfo *Region = getRegionInfo(ClassId);
    if (Region->MappedUser == 0)
      return;
    const uptr InUse = Region->Stats.PoppedBlocks - Region->Stats.PushedBlocks;
    const uptr AvailableChunks =
        Region->AllocatedUser / getSizeByClassId(ClassId);
    Printf("%s %02zu (%6zu): mapped: %6zuK popped: %7zu pushed: %7zu inuse: "
           "%6zu avail: %6zu rss: %6zuK releases: %6zu last released: %6zuK "
           "region: 0x%zx (0x%zx)\n",
           Region->Exhausted ? "F" : " ", ClassId, getSizeByClassId(ClassId),
           Region->MappedUser >> 10, Region->Stats.PoppedBlocks,
           Region->Stats.PushedBlocks, InUse, AvailableChunks, Rss >> 10,
           Region->ReleaseInfo.RangesReleased,
           Region->ReleaseInfo.LastReleasedBytes >> 10, Region->RegionBeg,
           getRegionBaseByClassId(ClassId));
  }

  NOINLINE void releaseToOSMaybe(RegionInfo *Region, uptr ClassId,
                                 bool Force = false) {
    const uptr BlockSize = getSizeByClassId(ClassId);
    const uptr PageSize = getPageSizeCached();

    CHECK_GE(Region->Stats.PoppedBlocks, Region->Stats.PushedBlocks);
    const uptr N = Region->Stats.PoppedBlocks - Region->Stats.PushedBlocks;
    if (N * BlockSize < PageSize)
      return; // No chance to release anything.
    if ((Region->Stats.PushedBlocks -
         Region->ReleaseInfo.PushedBlocksAtLastRelease) *
            BlockSize <
        PageSize) {
      return; // Nothing new to release.
    }

    if (!Force) {
      const s32 IntervalMs = ReleaseToOsIntervalMs;
      if (IntervalMs < 0)
        return;
      if (Region->ReleaseInfo.LastReleaseAtNs + IntervalMs * 1000000ULL >
          getMonotonicTime()) {
        return; // Memory was returned recently.
      }
    }

    ReleaseRecorder Recorder(Region->RegionBeg, &Region->Data);
    releaseFreeMemoryToOS(&Region->FreeList, Region->RegionBeg,
                          roundUpTo(Region->AllocatedUser, PageSize) / PageSize,
                          BlockSize, &Recorder);

    if (Recorder.getReleasedRangesCount() > 0) {
      Region->ReleaseInfo.PushedBlocksAtLastRelease =
          Region->Stats.PushedBlocks;
      Region->ReleaseInfo.RangesReleased += Recorder.getReleasedRangesCount();
      Region->ReleaseInfo.LastReleasedBytes = Recorder.getReleasedBytes();
    }
    Region->ReleaseInfo.LastReleaseAtNs = getMonotonicTime();
  }
};

} // namespace scudo

#endif // SCUDO_PRIMARY64_H_
