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
#include "memtag.h"
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
// released if the platform allows for it.

template <class SizeClassMapT, uptr RegionSizeLog,
          bool MaySupportMemoryTagging = false>
class SizeClassAllocator64 {
public:
  typedef SizeClassMapT SizeClassMap;
  typedef SizeClassAllocator64<SizeClassMap, RegionSizeLog,
                               MaySupportMemoryTagging>
      ThisT;
  typedef SizeClassAllocatorLocalCache<ThisT> CacheT;
  typedef typename CacheT::TransferBatch TransferBatch;
  static const bool SupportsMemoryTagging =
      MaySupportMemoryTagging && archSupportsMemoryTagging();

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
      Region->CanRelease = (ReleaseToOsInterval >= 0) &&
                           (getSizeByClassId(I) >= (PageSize / 64));
      Region->RandState = getRandomU32(&Seed);
    }
    ReleaseToOsIntervalMs = ReleaseToOsInterval;

    if (SupportsMemoryTagging)
      UseMemoryTagging = systemSupportsMemoryTagging();
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
    ScopedLock L(Region->Mutex);
    TransferBatch *B = Region->FreeList.front();
    if (B) {
      Region->FreeList.pop_front();
    } else {
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
    ScopedLock L(Region->Mutex);
    Region->FreeList.push_front(B);
    Region->Stats.PushedBlocks += B->getCount();
    if (Region->CanRelease)
      releaseToOSMaybe(Region, ClassId);
  }

  void disable() {
    // The BatchClassId must be locked last since other classes can use it.
    for (sptr I = static_cast<sptr>(NumClasses) - 1; I >= 0; I--) {
      if (static_cast<uptr>(I) == SizeClassMap::BatchClassId)
        continue;
      getRegionInfo(static_cast<uptr>(I))->Mutex.lock();
    }
    getRegionInfo(SizeClassMap::BatchClassId)->Mutex.lock();
  }

  void enable() {
    getRegionInfo(SizeClassMap::BatchClassId)->Mutex.unlock();
    for (uptr I = 0; I < NumClasses; I++) {
      if (I == SizeClassMap::BatchClassId)
        continue;
      getRegionInfo(I)->Mutex.unlock();
    }
  }

  template <typename F> void iterateOverBlocks(F Callback) const {
    for (uptr I = 0; I < NumClasses; I++) {
      if (I == SizeClassMap::BatchClassId)
        continue;
      const RegionInfo *Region = getRegionInfo(I);
      const uptr BlockSize = getSizeByClassId(I);
      const uptr From = Region->RegionBeg;
      const uptr To = From + Region->AllocatedUser;
      for (uptr Block = From; Block < To; Block += BlockSize)
        Callback(Block);
    }
  }

  void getStats(ScopedString *Str) const {
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
    Str->append("Stats: SizeClassAllocator64: %zuM mapped (%zuM rss) in %zu "
                "allocations; remains %zu\n",
                TotalMapped >> 20, 0, PoppedBlocks,
                PoppedBlocks - PushedBlocks);

    for (uptr I = 0; I < NumClasses; I++)
      getStats(Str, I, 0);
  }

  uptr releaseToOS() {
    uptr TotalReleasedBytes = 0;
    for (uptr I = 0; I < NumClasses; I++) {
      if (I == SizeClassMap::BatchClassId)
        continue;
      RegionInfo *Region = getRegionInfo(I);
      ScopedLock L(Region->Mutex);
      TotalReleasedBytes += releaseToOSMaybe(Region, I, /*Force=*/true);
    }
    return TotalReleasedBytes;
  }

  bool useMemoryTagging() const {
    return SupportsMemoryTagging && UseMemoryTagging;
  }
  void disableMemoryTagging() { UseMemoryTagging = false; }

private:
  static const uptr RegionSize = 1UL << RegionSizeLog;
  static const uptr NumClasses = SizeClassMap::NumClasses;
  static const uptr PrimarySize = RegionSize * NumClasses;

  // Call map for user memory with at least this size.
  static const uptr MapSizeIncrement = 1UL << 17;
  // Fill at most this number of batches from the newly map'd memory.
  static const u32 MaxNumBatches = 8U;

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
    HybridMutex Mutex;
    SinglyLinkedList<TransferBatch> FreeList;
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
  static_assert(sizeof(RegionInfo) % SCUDO_CACHE_LINE_SIZE == 0, "");

  uptr PrimaryBase;
  RegionInfo *RegionInfoArray;
  MapPlatformData Data;
  s32 ReleaseToOsIntervalMs;
  bool UseMemoryTagging;

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
    const u32 MaxCount = TransferBatch::getMaxCached(Size);

    const uptr RegionBeg = Region->RegionBeg;
    const uptr MappedUser = Region->MappedUser;
    const uptr TotalUserBytes = Region->AllocatedUser + MaxCount * Size;
    // Map more space for blocks, if necessary.
    if (TotalUserBytes > MappedUser) {
      // Do the mmap for the user memory.
      const uptr UserMapSize =
          roundUpTo(TotalUserBytes - MappedUser, MapSizeIncrement);
      const uptr RegionBase = RegionBeg - getRegionBaseByClassId(ClassId);
      if (UNLIKELY(RegionBase + MappedUser + UserMapSize > RegionSize)) {
        if (!Region->Exhausted) {
          Region->Exhausted = true;
          ScopedString Str(1024);
          getStats(&Str);
          Str.append(
              "Scudo OOM: The process has Exhausted %zuM for size class %zu.\n",
              RegionSize >> 20, Size);
          Str.output();
        }
        return nullptr;
      }
      if (UNLIKELY(MappedUser == 0))
        Region->Data = Data;
      if (UNLIKELY(!map(reinterpret_cast<void *>(RegionBeg + MappedUser),
                        UserMapSize, "scudo:primary",
                        MAP_ALLOWNOMEM | MAP_RESIZABLE |
                            (useMemoryTagging() ? MAP_MEMTAG : 0),
                        &Region->Data)))
        return nullptr;
      Region->MappedUser += UserMapSize;
      C->getStats().add(StatMapped, UserMapSize);
    }

    const u32 NumberOfBlocks = Min(
        MaxNumBatches * MaxCount,
        static_cast<u32>((Region->MappedUser - Region->AllocatedUser) / Size));
    DCHECK_GT(NumberOfBlocks, 0);

    TransferBatch *B = nullptr;
    constexpr u32 ShuffleArraySize =
        MaxNumBatches * TransferBatch::MaxNumCached;
    void *ShuffleArray[ShuffleArraySize];
    u32 Count = 0;
    const uptr P = RegionBeg + Region->AllocatedUser;
    const uptr AllocatedUser = Size * NumberOfBlocks;
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
    if (!Region->FreeList.empty()) {
      Region->FreeList.push_back(B);
      B = Region->FreeList.front();
      Region->FreeList.pop_front();
    }
    DCHECK_GT(B->getCount(), 0);

    C->getStats().add(StatFree, AllocatedUser);
    Region->AllocatedUser += AllocatedUser;
    Region->Exhausted = false;
    if (Region->CanRelease)
      Region->ReleaseInfo.LastReleaseAtNs = getMonotonicTime();

    return B;
  }

  void getStats(ScopedString *Str, uptr ClassId, uptr Rss) const {
    RegionInfo *Region = getRegionInfo(ClassId);
    if (Region->MappedUser == 0)
      return;
    const uptr InUse = Region->Stats.PoppedBlocks - Region->Stats.PushedBlocks;
    const uptr TotalChunks = Region->AllocatedUser / getSizeByClassId(ClassId);
    Str->append("%s %02zu (%6zu): mapped: %6zuK popped: %7zu pushed: %7zu "
                "inuse: %6zu total: %6zu rss: %6zuK releases: %6zu last "
                "released: %6zuK region: 0x%zx (0x%zx)\n",
                Region->Exhausted ? "F" : " ", ClassId,
                getSizeByClassId(ClassId), Region->MappedUser >> 10,
                Region->Stats.PoppedBlocks, Region->Stats.PushedBlocks, InUse,
                TotalChunks, Rss >> 10, Region->ReleaseInfo.RangesReleased,
                Region->ReleaseInfo.LastReleasedBytes >> 10, Region->RegionBeg,
                getRegionBaseByClassId(ClassId));
  }

  NOINLINE uptr releaseToOSMaybe(RegionInfo *Region, uptr ClassId,
                                 bool Force = false) {
    const uptr BlockSize = getSizeByClassId(ClassId);
    const uptr PageSize = getPageSizeCached();

    CHECK_GE(Region->Stats.PoppedBlocks, Region->Stats.PushedBlocks);
    const uptr BytesInFreeList =
        Region->AllocatedUser -
        (Region->Stats.PoppedBlocks - Region->Stats.PushedBlocks) * BlockSize;
    if (BytesInFreeList < PageSize)
      return 0; // No chance to release anything.
    if ((Region->Stats.PushedBlocks -
         Region->ReleaseInfo.PushedBlocksAtLastRelease) *
            BlockSize <
        PageSize) {
      return 0; // Nothing new to release.
    }

    if (!Force) {
      const s32 IntervalMs = ReleaseToOsIntervalMs;
      if (IntervalMs < 0)
        return 0;
      if (Region->ReleaseInfo.LastReleaseAtNs +
              static_cast<u64>(IntervalMs) * 1000000 >
          getMonotonicTime()) {
        return 0; // Memory was returned recently.
      }
    }

    ReleaseRecorder Recorder(Region->RegionBeg, &Region->Data);
    releaseFreeMemoryToOS(Region->FreeList, Region->RegionBeg,
                          roundUpTo(Region->AllocatedUser, PageSize) / PageSize,
                          BlockSize, &Recorder);

    if (Recorder.getReleasedRangesCount() > 0) {
      Region->ReleaseInfo.PushedBlocksAtLastRelease =
          Region->Stats.PushedBlocks;
      Region->ReleaseInfo.RangesReleased += Recorder.getReleasedRangesCount();
      Region->ReleaseInfo.LastReleasedBytes = Recorder.getReleasedBytes();
    }
    Region->ReleaseInfo.LastReleaseAtNs = getMonotonicTime();
    return Recorder.getReleasedBytes();
  }
};

} // namespace scudo

#endif // SCUDO_PRIMARY64_H_
