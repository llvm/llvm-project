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
#include "options.h"
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

template <typename Config> class SizeClassAllocator64 {
public:
  typedef typename Config::SizeClassMap SizeClassMap;
  typedef SizeClassAllocator64<Config> ThisT;
  typedef SizeClassAllocatorLocalCache<ThisT> CacheT;
  typedef typename CacheT::TransferBatch TransferBatch;
  static const bool SupportsMemoryTagging =
      allocatorSupportsMemoryTagging<Config>();

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

    u32 Seed;
    const u64 Time = getMonotonicTime();
    if (UNLIKELY(!getRandom(reinterpret_cast<void *>(&Seed), sizeof(Seed))))
      Seed = static_cast<u32>(Time ^ (PrimaryBase >> 12));
    const uptr PageSize = getPageSizeCached();
    for (uptr I = 0; I < NumClasses; I++) {
      RegionInfo *Region = getRegionInfo(I);
      // The actual start of a region is offseted by a random number of pages.
      Region->RegionBeg =
          getRegionBaseByClassId(I) + (getRandomModN(&Seed, 16) + 1) * PageSize;
      Region->RandState = getRandomU32(&Seed);
      Region->ReleaseInfo.LastReleaseAtNs = Time;
    }
    setOption(Option::ReleaseInterval, static_cast<sptr>(ReleaseToOsInterval));

    if (SupportsMemoryTagging && systemSupportsMemoryTagging())
      Options.set(OptionBit::UseMemoryTagging);
  }
  void init(s32 ReleaseToOsInterval) {
    memset(this, 0, sizeof(*this));
    initLinkerInitialized(ReleaseToOsInterval);
  }

  void unmapTestOnly() {
    unmap(reinterpret_cast<void *>(PrimaryBase), PrimarySize, UNMAP_ALL, &Data);
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
    if (ClassId != SizeClassMap::BatchClassId)
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

  template <typename F> void iterateOverBlocks(F Callback) {
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

  void getStats(ScopedString *Str) {
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

  bool setOption(Option O, sptr Value) {
    if (O == Option::ReleaseInterval) {
      const s32 Interval = Max(
          Min(static_cast<s32>(Value), Config::PrimaryMaxReleaseToOsIntervalMs),
          Config::PrimaryMinReleaseToOsIntervalMs);
      atomic_store_relaxed(&ReleaseToOsIntervalMs, Interval);
      return true;
    }
    // Not supported by the Primary, but not an error either.
    return true;
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

  static bool useMemoryTagging(Options Options) {
    return SupportsMemoryTagging && Options.get(OptionBit::UseMemoryTagging);
  }
  void disableMemoryTagging() { Options.clear(OptionBit::UseMemoryTagging); }

  const char *getRegionInfoArrayAddress() const {
    return reinterpret_cast<const char *>(RegionInfoArray);
  }

  static uptr getRegionInfoArraySize() { return sizeof(RegionInfoArray); }

  static BlockInfo findNearestBlock(const char *RegionInfoData, uptr Ptr) {
    const RegionInfo *RegionInfoArray =
        reinterpret_cast<const RegionInfo *>(RegionInfoData);
    uptr ClassId;
    uptr MinDistance = -1UL;
    for (uptr I = 0; I != NumClasses; ++I) {
      if (I == SizeClassMap::BatchClassId)
        continue;
      uptr Begin = RegionInfoArray[I].RegionBeg;
      uptr End = Begin + RegionInfoArray[I].AllocatedUser;
      if (Begin > End || End - Begin < SizeClassMap::getSizeByClassId(I))
        continue;
      uptr RegionDistance;
      if (Begin <= Ptr) {
        if (Ptr < End)
          RegionDistance = 0;
        else
          RegionDistance = Ptr - End;
      } else {
        RegionDistance = Begin - Ptr;
      }

      if (RegionDistance < MinDistance) {
        MinDistance = RegionDistance;
        ClassId = I;
      }
    }

    BlockInfo B = {};
    if (MinDistance <= 8192) {
      B.RegionBegin = RegionInfoArray[ClassId].RegionBeg;
      B.RegionEnd = B.RegionBegin + RegionInfoArray[ClassId].AllocatedUser;
      B.BlockSize = SizeClassMap::getSizeByClassId(ClassId);
      B.BlockBegin =
          B.RegionBegin + uptr(sptr(Ptr - B.RegionBegin) / sptr(B.BlockSize) *
                               sptr(B.BlockSize));
      while (B.BlockBegin < B.RegionBegin)
        B.BlockBegin += B.BlockSize;
      while (B.RegionEnd < B.BlockBegin + B.BlockSize)
        B.BlockBegin -= B.BlockSize;
    }
    return B;
  }

  AtomicOptions Options;

private:
  static const uptr RegionSize = 1UL << Config::PrimaryRegionSizeLog;
  static const uptr NumClasses = SizeClassMap::NumClasses;
  static const uptr PrimarySize = RegionSize * NumClasses;

  // Call map for user memory with at least this size.
  static const uptr MapSizeIncrement = 1UL << 18;
  // Fill at most this number of batches from the newly map'd memory.
  static const u32 MaxNumBatches = SCUDO_ANDROID ? 4U : 8U;

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

  struct UnpaddedRegionInfo {
    HybridMutex Mutex;
    SinglyLinkedList<TransferBatch> FreeList;
    RegionStats Stats;
    bool Exhausted;
    u32 RandState;
    uptr RegionBeg;
    uptr MappedUser;    // Bytes mapped for user memory.
    uptr AllocatedUser; // Bytes allocated for user memory.
    MapPlatformData Data;
    ReleaseToOsInfo ReleaseInfo;
  };
  struct RegionInfo : UnpaddedRegionInfo {
    char Padding[SCUDO_CACHE_LINE_SIZE -
                 (sizeof(UnpaddedRegionInfo) % SCUDO_CACHE_LINE_SIZE)];
  };
  static_assert(sizeof(RegionInfo) % SCUDO_CACHE_LINE_SIZE == 0, "");

  uptr PrimaryBase;
  MapPlatformData Data;
  atomic_s32 ReleaseToOsIntervalMs;
  alignas(SCUDO_CACHE_LINE_SIZE) RegionInfo RegionInfoArray[NumClasses];

  RegionInfo *getRegionInfo(uptr ClassId) {
    DCHECK_LT(ClassId, NumClasses);
    return &RegionInfoArray[ClassId];
  }

  uptr getRegionBaseByClassId(uptr ClassId) const {
    return PrimaryBase + (ClassId << Config::PrimaryRegionSizeLog);
  }

  NOINLINE TransferBatch *populateFreeList(CacheT *C, uptr ClassId,
                                           RegionInfo *Region) {
    const uptr Size = getSizeByClassId(ClassId);
    const u32 MaxCount = TransferBatch::getMaxCached(Size);

    const uptr RegionBeg = Region->RegionBeg;
    const uptr MappedUser = Region->MappedUser;
    const uptr TotalUserBytes = Region->AllocatedUser + MaxCount * Size;
    // Map more space for blocks, if necessary.
    if (UNLIKELY(TotalUserBytes > MappedUser)) {
      // Do the mmap for the user memory.
      const uptr UserMapSize =
          roundUpTo(TotalUserBytes - MappedUser, MapSizeIncrement);
      const uptr RegionBase = RegionBeg - getRegionBaseByClassId(ClassId);
      if (RegionBase + MappedUser + UserMapSize > RegionSize) {
        if (!Region->Exhausted) {
          Region->Exhausted = true;
          ScopedString Str(1024);
          getStats(&Str);
          Str.append(
              "Scudo OOM: The process has exhausted %zuM for size class %zu.\n",
              RegionSize >> 20, Size);
          Str.output();
        }
        return nullptr;
      }
      if (MappedUser == 0)
        Region->Data = Data;
      if (!map(reinterpret_cast<void *>(RegionBeg + MappedUser), UserMapSize,
               "scudo:primary",
               MAP_ALLOWNOMEM | MAP_RESIZABLE |
                   (useMemoryTagging(Options.load()) ? MAP_MEMTAG : 0),
               &Region->Data))
        return nullptr;
      Region->MappedUser += UserMapSize;
      C->getStats().add(StatMapped, UserMapSize);
    }

    const u32 NumberOfBlocks = Min(
        MaxNumBatches * MaxCount,
        static_cast<u32>((Region->MappedUser - Region->AllocatedUser) / Size));
    DCHECK_GT(NumberOfBlocks, 0);

    constexpr u32 ShuffleArraySize =
        MaxNumBatches * TransferBatch::MaxNumCached;
    void *ShuffleArray[ShuffleArraySize];
    DCHECK_LE(NumberOfBlocks, ShuffleArraySize);

    uptr P = RegionBeg + Region->AllocatedUser;
    for (u32 I = 0; I < NumberOfBlocks; I++, P += Size)
      ShuffleArray[I] = reinterpret_cast<void *>(P);
    // No need to shuffle the batches size class.
    if (ClassId != SizeClassMap::BatchClassId)
      shuffle(ShuffleArray, NumberOfBlocks, &Region->RandState);
    for (u32 I = 0; I < NumberOfBlocks;) {
      TransferBatch *B = C->createBatch(ClassId, ShuffleArray[I]);
      if (UNLIKELY(!B))
        return nullptr;
      const u32 N = Min(MaxCount, NumberOfBlocks - I);
      B->setFromArray(&ShuffleArray[I], N);
      Region->FreeList.push_back(B);
      I += N;
    }
    TransferBatch *B = Region->FreeList.front();
    Region->FreeList.pop_front();
    DCHECK(B);
    DCHECK_GT(B->getCount(), 0);

    const uptr AllocatedUser = Size * NumberOfBlocks;
    C->getStats().add(StatFree, AllocatedUser);
    Region->AllocatedUser += AllocatedUser;

    return B;
  }

  void getStats(ScopedString *Str, uptr ClassId, uptr Rss) {
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

    DCHECK_GE(Region->Stats.PoppedBlocks, Region->Stats.PushedBlocks);
    const uptr BytesInFreeList =
        Region->AllocatedUser -
        (Region->Stats.PoppedBlocks - Region->Stats.PushedBlocks) * BlockSize;
    if (BytesInFreeList < PageSize)
      return 0; // No chance to release anything.
    const uptr BytesPushed = (Region->Stats.PushedBlocks -
                              Region->ReleaseInfo.PushedBlocksAtLastRelease) *
                             BlockSize;
    if (BytesPushed < PageSize)
      return 0; // Nothing new to release.

    // Releasing smaller blocks is expensive, so we want to make sure that a
    // significant amount of bytes are free, and that there has been a good
    // amount of batches pushed to the freelist before attempting to release.
    if (BlockSize < PageSize / 16U) {
      if (!Force && BytesPushed < Region->AllocatedUser / 16U)
        return 0;
      // We want 8x% to 9x% free bytes (the larger the bock, the lower the %).
      if ((BytesInFreeList * 100U) / Region->AllocatedUser <
          (100U - 1U - BlockSize / 16U))
        return 0;
    }

    if (!Force) {
      const s32 IntervalMs = atomic_load_relaxed(&ReleaseToOsIntervalMs);
      if (IntervalMs < 0)
        return 0;
      if (Region->ReleaseInfo.LastReleaseAtNs +
              static_cast<u64>(IntervalMs) * 1000000 >
          getMonotonicTime()) {
        return 0; // Memory was returned recently.
      }
    }

    auto SkipRegion = [](UNUSED uptr RegionIndex) { return false; };
    ReleaseRecorder Recorder(Region->RegionBeg, &Region->Data);
    releaseFreeMemoryToOS(Region->FreeList, Region->RegionBeg,
                          Region->AllocatedUser, 1U, BlockSize, &Recorder,
                          SkipRegion);

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
