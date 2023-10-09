//===-- primary64.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SCUDO_PRIMARY64_H_
#define SCUDO_PRIMARY64_H_

#include "allocator_common.h"
#include "bytemap.h"
#include "common.h"
#include "list.h"
#include "local_cache.h"
#include "mem_map.h"
#include "memtag.h"
#include "options.h"
#include "release.h"
#include "stats.h"
#include "string_utils.h"
#include "thread_annotations.h"

namespace scudo {

// SizeClassAllocator64 is an allocator tuned for 64-bit address space.
//
// It starts by reserving NumClasses * 2^RegionSizeLog bytes, equally divided in
// Regions, specific to each size class. Note that the base of that mapping is
// random (based to the platform specific map() capabilities). If
// PrimaryEnableRandomOffset is set, each Region actually starts at a random
// offset from its base.
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
  typedef typename Config::Primary::CompactPtrT CompactPtrT;
  typedef typename Config::Primary::SizeClassMap SizeClassMap;
  static const uptr CompactPtrScale = Config::Primary::CompactPtrScale;
  static const uptr RegionSizeLog = Config::Primary::RegionSizeLog;
  static const uptr GroupSizeLog = Config::Primary::GroupSizeLog;
  static_assert(RegionSizeLog >= GroupSizeLog,
                "Group size shouldn't be greater than the region size");
  static const uptr GroupScale = GroupSizeLog - CompactPtrScale;
  typedef SizeClassAllocator64<Config> ThisT;
  typedef SizeClassAllocatorLocalCache<ThisT> CacheT;
  typedef TransferBatch<ThisT> TransferBatchT;
  typedef BatchGroup<ThisT> BatchGroupT;

  static_assert(sizeof(BatchGroupT) <= sizeof(TransferBatchT),
                "BatchGroupT uses the same class size as TransferBatchT");

  static uptr getSizeByClassId(uptr ClassId) {
    return (ClassId == SizeClassMap::BatchClassId)
               ? roundUp(sizeof(TransferBatchT), 1U << CompactPtrScale)
               : SizeClassMap::getSizeByClassId(ClassId);
  }

  static bool canAllocate(uptr Size) { return Size <= SizeClassMap::MaxSize; }

  void init(s32 ReleaseToOsInterval) NO_THREAD_SAFETY_ANALYSIS {
    DCHECK(isAligned(reinterpret_cast<uptr>(this), alignof(ThisT)));

    const uptr PageSize = getPageSizeCached();
    const uptr GroupSize = (1UL << GroupSizeLog);
    const uptr PagesInGroup = GroupSize / PageSize;
    const uptr MinSizeClass = getSizeByClassId(1);
    // When trying to release pages back to memory, visiting smaller size
    // classes is expensive. Therefore, we only try to release smaller size
    // classes when the amount of free blocks goes over a certain threshold (See
    // the comment in releaseToOSMaybe() for more details). For example, for
    // size class 32, we only do the release when the size of free blocks is
    // greater than 97% of pages in a group. However, this may introduce another
    // issue that if the number of free blocks is bouncing between 97% ~ 100%.
    // Which means we may try many page releases but only release very few of
    // them (less than 3% in a group). Even though we have
    // `&ReleaseToOsIntervalMs` which slightly reduce the frequency of these
    // calls but it will be better to have another guard to mitigate this issue.
    //
    // Here we add another constraint on the minimum size requirement. The
    // constraint is determined by the size of in-use blocks in the minimal size
    // class. Take size class 32 as an example,
    //
    //   +-     one memory group      -+
    //   +----------------------+------+
    //   |  97% of free blocks  |      |
    //   +----------------------+------+
    //                           \    /
    //                      3% in-use blocks
    //
    //   * The release size threshold is 97%.
    //
    // The 3% size in a group is about 7 pages. For two consecutive
    // releaseToOSMaybe(), we require the difference between `PushedBlocks`
    // should be greater than 7 pages. This mitigates the page releasing
    // thrashing which is caused by memory usage bouncing around the threshold.
    // The smallest size class takes longest time to do the page release so we
    // use its size of in-use blocks as a heuristic.
    SmallerBlockReleasePageDelta =
        PagesInGroup * (1 + MinSizeClass / 16U) / 100;

    // Reserve the space required for the Primary.
    CHECK(ReservedMemory.create(/*Addr=*/0U, PrimarySize,
                                "scudo:primary_reserve"));
    PrimaryBase = ReservedMemory.getBase();
    DCHECK_NE(PrimaryBase, 0U);

    u32 Seed;
    const u64 Time = getMonotonicTimeFast();
    if (!getRandom(reinterpret_cast<void *>(&Seed), sizeof(Seed)))
      Seed = static_cast<u32>(Time ^ (PrimaryBase >> 12));

    for (uptr I = 0; I < NumClasses; I++) {
      RegionInfo *Region = getRegionInfo(I);
      // The actual start of a region is offset by a random number of pages
      // when PrimaryEnableRandomOffset is set.
      Region->RegionBeg = (PrimaryBase + (I << RegionSizeLog)) +
                          (Config::Primary::EnableRandomOffset
                               ? ((getRandomModN(&Seed, 16) + 1) * PageSize)
                               : 0);
      Region->RandState = getRandomU32(&Seed);
      // Releasing small blocks is expensive, set a higher threshold to avoid
      // frequent page releases.
      if (isSmallBlock(getSizeByClassId(I)))
        Region->TryReleaseThreshold = PageSize * SmallerBlockReleasePageDelta;
      else
        Region->TryReleaseThreshold = PageSize;
      Region->ReleaseInfo.LastReleaseAtNs = Time;

      Region->MemMapInfo.MemMap = ReservedMemory.dispatch(
          PrimaryBase + (I << RegionSizeLog), RegionSize);
      CHECK(Region->MemMapInfo.MemMap.isAllocated());
    }
    shuffle(RegionInfoArray, NumClasses, &Seed);

    setOption(Option::ReleaseInterval, static_cast<sptr>(ReleaseToOsInterval));
  }

  void unmapTestOnly() NO_THREAD_SAFETY_ANALYSIS {
    for (uptr I = 0; I < NumClasses; I++) {
      RegionInfo *Region = getRegionInfo(I);
      *Region = {};
    }
    if (PrimaryBase)
      ReservedMemory.release();
    PrimaryBase = 0U;
  }

  // When all blocks are freed, it has to be the same size as `AllocatedUser`.
  void verifyAllBlocksAreReleasedTestOnly() {
    // `BatchGroup` and `TransferBatch` also use the blocks from BatchClass.
    uptr BatchClassUsedInFreeLists = 0;
    for (uptr I = 0; I < NumClasses; I++) {
      // We have to count BatchClassUsedInFreeLists in other regions first.
      if (I == SizeClassMap::BatchClassId)
        continue;
      RegionInfo *Region = getRegionInfo(I);
      ScopedLock ML(Region->MMLock);
      ScopedLock FL(Region->FLLock);
      const uptr BlockSize = getSizeByClassId(I);
      uptr TotalBlocks = 0;
      for (BatchGroupT &BG : Region->FreeListInfo.BlockList) {
        // `BG::Batches` are `TransferBatches`. +1 for `BatchGroup`.
        BatchClassUsedInFreeLists += BG.Batches.size() + 1;
        for (const auto &It : BG.Batches)
          TotalBlocks += It.getCount();
      }

      DCHECK_EQ(TotalBlocks, Region->MemMapInfo.AllocatedUser / BlockSize);
      DCHECK_EQ(Region->FreeListInfo.PushedBlocks,
                Region->FreeListInfo.PoppedBlocks);
    }

    RegionInfo *Region = getRegionInfo(SizeClassMap::BatchClassId);
    ScopedLock ML(Region->MMLock);
    ScopedLock FL(Region->FLLock);
    const uptr BlockSize = getSizeByClassId(SizeClassMap::BatchClassId);
    uptr TotalBlocks = 0;
    for (BatchGroupT &BG : Region->FreeListInfo.BlockList) {
      if (LIKELY(!BG.Batches.empty())) {
        for (const auto &It : BG.Batches)
          TotalBlocks += It.getCount();
      } else {
        // `BatchGroup` with empty freelist doesn't have `TransferBatch` record
        // itself.
        ++TotalBlocks;
      }
    }
    DCHECK_EQ(TotalBlocks + BatchClassUsedInFreeLists,
              Region->MemMapInfo.AllocatedUser / BlockSize);
    DCHECK_GE(Region->FreeListInfo.PoppedBlocks,
              Region->FreeListInfo.PushedBlocks);
    const uptr BlocksInUse =
        Region->FreeListInfo.PoppedBlocks - Region->FreeListInfo.PushedBlocks;
    DCHECK_EQ(BlocksInUse, BatchClassUsedInFreeLists);
  }

  u16 popBlocks(CacheT *C, uptr ClassId, CompactPtrT *ToArray) {
    TransferBatchT *B = popBatch(C, ClassId);
    if (!B)
      return 0;

    const u16 Count = B->getCount();
    DCHECK_GT(Count, 0U);
    B->moveToArray(ToArray);

    if (ClassId != SizeClassMap::BatchClassId)
      C->deallocate(SizeClassMap::BatchClassId, B);

    return Count;
  }

  TransferBatchT *popBatch(CacheT *C, uptr ClassId) {
    DCHECK_LT(ClassId, NumClasses);
    RegionInfo *Region = getRegionInfo(ClassId);

    {
      ScopedLock L(Region->FLLock);
      TransferBatchT *B = popBatchImpl(C, ClassId, Region);
      if (LIKELY(B))
        return B;
    }

    bool ReportRegionExhausted = false;
    TransferBatchT *B = nullptr;

    while (true) {
      // When two threads compete for `Region->MMLock`, we only want one of them
      // does the populateFreeListAndPopBatch(). To avoid both of them doing
      // that, always check the freelist before mapping new pages.
      //
      // TODO(chiahungduan): Use a condition variable so that we don't need to
      // hold `Region->MMLock` here.
      ScopedLock ML(Region->MMLock);
      {
        ScopedLock FL(Region->FLLock);
        B = popBatchImpl(C, ClassId, Region);
        if (LIKELY(B))
          return B;
      }

      const bool RegionIsExhausted = Region->Exhausted;
      if (!RegionIsExhausted)
        B = populateFreeListAndPopBatch(C, ClassId, Region);
      ReportRegionExhausted = !RegionIsExhausted && Region->Exhausted;
      break;
    }

    if (UNLIKELY(ReportRegionExhausted)) {
      Printf("Can't populate more pages for size class %zu.\n",
             getSizeByClassId(ClassId));

      // Theoretically, BatchClass shouldn't be used up. Abort immediately  when
      // it happens.
      if (ClassId == SizeClassMap::BatchClassId)
        reportOutOfBatchClass();
    }

    return B;
  }

  // Push the array of free blocks to the designated batch group.
  void pushBlocks(CacheT *C, uptr ClassId, CompactPtrT *Array, u32 Size) {
    DCHECK_LT(ClassId, NumClasses);
    DCHECK_GT(Size, 0);

    RegionInfo *Region = getRegionInfo(ClassId);
    if (ClassId == SizeClassMap::BatchClassId) {
      ScopedLock L(Region->FLLock);
      pushBatchClassBlocks(Region, Array, Size);
      return;
    }

    // TODO(chiahungduan): Consider not doing grouping if the group size is not
    // greater than the block size with a certain scale.

    bool SameGroup = true;
    if (GroupSizeLog < RegionSizeLog) {
      // Sort the blocks so that blocks belonging to the same group can be
      // pushed together.
      for (u32 I = 1; I < Size; ++I) {
        if (compactPtrGroup(Array[I - 1]) != compactPtrGroup(Array[I]))
          SameGroup = false;
        CompactPtrT Cur = Array[I];
        u32 J = I;
        while (J > 0 && compactPtrGroup(Cur) < compactPtrGroup(Array[J - 1])) {
          Array[J] = Array[J - 1];
          --J;
        }
        Array[J] = Cur;
      }
    }

    {
      ScopedLock L(Region->FLLock);
      pushBlocksImpl(C, ClassId, Region, Array, Size, SameGroup);
    }
  }

  void disable() NO_THREAD_SAFETY_ANALYSIS {
    // The BatchClassId must be locked last since other classes can use it.
    for (sptr I = static_cast<sptr>(NumClasses) - 1; I >= 0; I--) {
      if (static_cast<uptr>(I) == SizeClassMap::BatchClassId)
        continue;
      getRegionInfo(static_cast<uptr>(I))->MMLock.lock();
      getRegionInfo(static_cast<uptr>(I))->FLLock.lock();
    }
    getRegionInfo(SizeClassMap::BatchClassId)->MMLock.lock();
    getRegionInfo(SizeClassMap::BatchClassId)->FLLock.lock();
  }

  void enable() NO_THREAD_SAFETY_ANALYSIS {
    getRegionInfo(SizeClassMap::BatchClassId)->FLLock.unlock();
    getRegionInfo(SizeClassMap::BatchClassId)->MMLock.unlock();
    for (uptr I = 0; I < NumClasses; I++) {
      if (I == SizeClassMap::BatchClassId)
        continue;
      getRegionInfo(I)->FLLock.unlock();
      getRegionInfo(I)->MMLock.unlock();
    }
  }

  template <typename F> void iterateOverBlocks(F Callback) {
    for (uptr I = 0; I < NumClasses; I++) {
      if (I == SizeClassMap::BatchClassId)
        continue;
      RegionInfo *Region = getRegionInfo(I);
      // TODO: The call of `iterateOverBlocks` requires disabling
      // SizeClassAllocator64. We may consider locking each region on demand
      // only.
      Region->FLLock.assertHeld();
      Region->MMLock.assertHeld();
      const uptr BlockSize = getSizeByClassId(I);
      const uptr From = Region->RegionBeg;
      const uptr To = From + Region->MemMapInfo.AllocatedUser;
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
      {
        ScopedLock L(Region->MMLock);
        TotalMapped += Region->MemMapInfo.MappedUser;
      }
      {
        ScopedLock L(Region->FLLock);
        PoppedBlocks += Region->FreeListInfo.PoppedBlocks;
        PushedBlocks += Region->FreeListInfo.PushedBlocks;
      }
    }
    Str->append("Stats: SizeClassAllocator64: %zuM mapped (%uM rss) in %zu "
                "allocations; remains %zu\n",
                TotalMapped >> 20, 0U, PoppedBlocks,
                PoppedBlocks - PushedBlocks);

    for (uptr I = 0; I < NumClasses; I++) {
      RegionInfo *Region = getRegionInfo(I);
      ScopedLock L1(Region->MMLock);
      ScopedLock L2(Region->FLLock);
      getStats(Str, I, Region);
    }
  }

  void getFragmentationInfo(ScopedString *Str) {
    Str->append(
        "Fragmentation Stats: SizeClassAllocator64: page size = %zu bytes\n",
        getPageSizeCached());

    for (uptr I = 1; I < NumClasses; I++) {
      RegionInfo *Region = getRegionInfo(I);
      ScopedLock L(Region->MMLock);
      getRegionFragmentationInfo(Region, I, Str);
    }
  }

  bool setOption(Option O, sptr Value) {
    if (O == Option::ReleaseInterval) {
      const s32 Interval = Max(Min(static_cast<s32>(Value),
                                   Config::Primary::MaxReleaseToOsIntervalMs),
                               Config::Primary::MinReleaseToOsIntervalMs);
      atomic_store_relaxed(&ReleaseToOsIntervalMs, Interval);
      return true;
    }
    // Not supported by the Primary, but not an error either.
    return true;
  }

  uptr tryReleaseToOS(uptr ClassId, ReleaseToOS ReleaseType) {
    RegionInfo *Region = getRegionInfo(ClassId);
    // Note that the tryLock() may fail spuriously, given that it should rarely
    // happen and page releasing is fine to skip, we don't take certain
    // approaches to ensure one page release is done.
    if (Region->MMLock.tryLock()) {
      uptr BytesReleased = releaseToOSMaybe(Region, ClassId, ReleaseType);
      Region->MMLock.unlock();
      return BytesReleased;
    }
    return 0;
  }

  uptr releaseToOS(ReleaseToOS ReleaseType) {
    uptr TotalReleasedBytes = 0;
    for (uptr I = 0; I < NumClasses; I++) {
      if (I == SizeClassMap::BatchClassId)
        continue;
      RegionInfo *Region = getRegionInfo(I);
      ScopedLock L(Region->MMLock);
      TotalReleasedBytes += releaseToOSMaybe(Region, I, ReleaseType);
    }
    return TotalReleasedBytes;
  }

  const char *getRegionInfoArrayAddress() const {
    return reinterpret_cast<const char *>(RegionInfoArray);
  }

  static uptr getRegionInfoArraySize() { return sizeof(RegionInfoArray); }

  uptr getCompactPtrBaseByClassId(uptr ClassId) {
    return getRegionInfo(ClassId)->RegionBeg;
  }

  CompactPtrT compactPtr(uptr ClassId, uptr Ptr) {
    DCHECK_LE(ClassId, SizeClassMap::LargestClassId);
    return compactPtrInternal(getCompactPtrBaseByClassId(ClassId), Ptr);
  }

  void *decompactPtr(uptr ClassId, CompactPtrT CompactPtr) {
    DCHECK_LE(ClassId, SizeClassMap::LargestClassId);
    return reinterpret_cast<void *>(
        decompactPtrInternal(getCompactPtrBaseByClassId(ClassId), CompactPtr));
  }

  static BlockInfo findNearestBlock(const char *RegionInfoData,
                                    uptr Ptr) NO_THREAD_SAFETY_ANALYSIS {
    const RegionInfo *RegionInfoArray =
        reinterpret_cast<const RegionInfo *>(RegionInfoData);

    uptr ClassId;
    uptr MinDistance = -1UL;
    for (uptr I = 0; I != NumClasses; ++I) {
      if (I == SizeClassMap::BatchClassId)
        continue;
      uptr Begin = RegionInfoArray[I].RegionBeg;
      // TODO(chiahungduan): In fact, We need to lock the RegionInfo::MMLock.
      // However, the RegionInfoData is passed with const qualifier and lock the
      // mutex requires modifying RegionInfoData, which means we need to remove
      // the const qualifier. This may lead to another undefined behavior (The
      // first one is accessing `AllocatedUser` without locking. It's better to
      // pass `RegionInfoData` as `void *` then we can lock the mutex properly.
      uptr End = Begin + RegionInfoArray[I].MemMapInfo.AllocatedUser;
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
      B.RegionEnd =
          B.RegionBegin + RegionInfoArray[ClassId].MemMapInfo.AllocatedUser;
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
  static const uptr RegionSize = 1UL << RegionSizeLog;
  static const uptr NumClasses = SizeClassMap::NumClasses;
  static const uptr PrimarySize = RegionSize * NumClasses;

  static const uptr MapSizeIncrement = Config::Primary::MapSizeIncrement;
  // Fill at most this number of batches from the newly map'd memory.
  static const u32 MaxNumBatches = SCUDO_ANDROID ? 4U : 8U;

  struct ReleaseToOsInfo {
    uptr BytesInFreeListAtLastCheckpoint;
    uptr RangesReleased;
    uptr LastReleasedBytes;
    u64 LastReleaseAtNs;
  };

  struct BlocksInfo {
    SinglyLinkedList<BatchGroupT> BlockList = {};
    uptr PoppedBlocks = 0;
    uptr PushedBlocks = 0;
  };

  struct PagesInfo {
    MemMapT MemMap = {};
    // Bytes mapped for user memory.
    uptr MappedUser = 0;
    // Bytes allocated for user memory.
    uptr AllocatedUser = 0;
  };

  struct UnpaddedRegionInfo {
    // Mutex for operations on freelist
    HybridMutex FLLock;
    // Mutex for memmap operations
    HybridMutex MMLock ACQUIRED_BEFORE(FLLock);
    // `RegionBeg` is initialized before thread creation and won't be changed.
    uptr RegionBeg = 0;
    u32 RandState GUARDED_BY(MMLock) = 0;
    BlocksInfo FreeListInfo GUARDED_BY(FLLock);
    PagesInfo MemMapInfo GUARDED_BY(MMLock);
    // The minimum size of pushed blocks to trigger page release.
    uptr TryReleaseThreshold GUARDED_BY(MMLock) = 0;
    ReleaseToOsInfo ReleaseInfo GUARDED_BY(MMLock) = {};
    bool Exhausted GUARDED_BY(MMLock) = false;
  };
  struct RegionInfo : UnpaddedRegionInfo {
    char Padding[SCUDO_CACHE_LINE_SIZE -
                 (sizeof(UnpaddedRegionInfo) % SCUDO_CACHE_LINE_SIZE)] = {};
  };
  static_assert(sizeof(RegionInfo) % SCUDO_CACHE_LINE_SIZE == 0, "");

  RegionInfo *getRegionInfo(uptr ClassId) {
    DCHECK_LT(ClassId, NumClasses);
    return &RegionInfoArray[ClassId];
  }

  uptr getRegionBaseByClassId(uptr ClassId) {
    return roundDown(getRegionInfo(ClassId)->RegionBeg - PrimaryBase,
                     RegionSize) +
           PrimaryBase;
  }

  static CompactPtrT compactPtrInternal(uptr Base, uptr Ptr) {
    return static_cast<CompactPtrT>((Ptr - Base) >> CompactPtrScale);
  }

  static uptr decompactPtrInternal(uptr Base, CompactPtrT CompactPtr) {
    return Base + (static_cast<uptr>(CompactPtr) << CompactPtrScale);
  }

  static uptr compactPtrGroup(CompactPtrT CompactPtr) {
    const uptr Mask = (static_cast<uptr>(1) << GroupScale) - 1;
    return static_cast<uptr>(CompactPtr) & ~Mask;
  }
  static uptr decompactGroupBase(uptr Base, uptr CompactPtrGroupBase) {
    DCHECK_EQ(CompactPtrGroupBase % (static_cast<uptr>(1) << (GroupScale)), 0U);
    return Base + (CompactPtrGroupBase << CompactPtrScale);
  }

  ALWAYS_INLINE static bool isSmallBlock(uptr BlockSize) {
    const uptr PageSize = getPageSizeCached();
    return BlockSize < PageSize / 16U;
  }

  ALWAYS_INLINE static bool isLargeBlock(uptr BlockSize) {
    const uptr PageSize = getPageSizeCached();
    return BlockSize > PageSize;
  }

  void pushBatchClassBlocks(RegionInfo *Region, CompactPtrT *Array, u32 Size)
      REQUIRES(Region->FLLock) {
    DCHECK_EQ(Region, getRegionInfo(SizeClassMap::BatchClassId));

    // Free blocks are recorded by TransferBatch in freelist for all
    // size-classes. In addition, TransferBatch is allocated from BatchClassId.
    // In order not to use additional block to record the free blocks in
    // BatchClassId, they are self-contained. I.e., A TransferBatch records the
    // block address of itself. See the figure below:
    //
    // TransferBatch at 0xABCD
    // +----------------------------+
    // | Free blocks' addr          |
    // | +------+------+------+     |
    // | |0xABCD|...   |...   |     |
    // | +------+------+------+     |
    // +----------------------------+
    //
    // When we allocate all the free blocks in the TransferBatch, the block used
    // by TransferBatch is also free for use. We don't need to recycle the
    // TransferBatch. Note that the correctness is maintained by the invariant,
    //
    //   The unit of each popBatch() request is entire TransferBatch. Return
    //   part of the blocks in a TransferBatch is invalid.
    //
    // This ensures that TransferBatch won't leak the address itself while it's
    // still holding other valid data.
    //
    // Besides, BatchGroup is also allocated from BatchClassId and has its
    // address recorded in the TransferBatch too. To maintain the correctness,
    //
    //   The address of BatchGroup is always recorded in the last TransferBatch
    //   in the freelist (also imply that the freelist should only be
    //   updated with push_front). Once the last TransferBatch is popped,
    //   the block used by BatchGroup is also free for use.
    //
    // With this approach, the blocks used by BatchGroup and TransferBatch are
    // reusable and don't need additional space for them.

    Region->FreeListInfo.PushedBlocks += Size;
    BatchGroupT *BG = Region->FreeListInfo.BlockList.front();

    if (BG == nullptr) {
      // Construct `BatchGroup` on the last element.
      BG = reinterpret_cast<BatchGroupT *>(
          decompactPtr(SizeClassMap::BatchClassId, Array[Size - 1]));
      --Size;
      BG->Batches.clear();
      // BatchClass hasn't enabled memory group. Use `0` to indicate there's no
      // memory group here.
      BG->CompactPtrGroupBase = 0;
      // `BG` is also the block of BatchClassId. Note that this is different
      // from `CreateGroup` in `pushBlocksImpl`
      BG->PushedBlocks = 1;
      BG->BytesInBGAtLastCheckpoint = 0;
      BG->MaxCachedPerBatch =
          CacheT::getMaxCached(getSizeByClassId(SizeClassMap::BatchClassId));

      Region->FreeListInfo.BlockList.push_front(BG);
    }

    if (UNLIKELY(Size == 0))
      return;

    // This happens under 2 cases.
    //   1. just allocated a new `BatchGroup`.
    //   2. Only 1 block is pushed when the freelist is empty.
    if (BG->Batches.empty()) {
      // Construct the `TransferBatch` on the last element.
      TransferBatchT *TB = reinterpret_cast<TransferBatchT *>(
          decompactPtr(SizeClassMap::BatchClassId, Array[Size - 1]));
      TB->clear();
      // As mentioned above, addresses of `TransferBatch` and `BatchGroup` are
      // recorded in the TransferBatch.
      TB->add(Array[Size - 1]);
      TB->add(
          compactPtr(SizeClassMap::BatchClassId, reinterpret_cast<uptr>(BG)));
      --Size;
      DCHECK_EQ(BG->PushedBlocks, 1U);
      // `TB` is also the block of BatchClassId.
      BG->PushedBlocks += 1;
      BG->Batches.push_front(TB);
    }

    TransferBatchT *CurBatch = BG->Batches.front();
    DCHECK_NE(CurBatch, nullptr);

    for (u32 I = 0; I < Size;) {
      u16 UnusedSlots =
          static_cast<u16>(BG->MaxCachedPerBatch - CurBatch->getCount());
      if (UnusedSlots == 0) {
        CurBatch = reinterpret_cast<TransferBatchT *>(
            decompactPtr(SizeClassMap::BatchClassId, Array[I]));
        CurBatch->clear();
        // Self-contained
        CurBatch->add(Array[I]);
        ++I;
        // TODO(chiahungduan): Avoid the use of push_back() in `Batches` of
        // BatchClassId.
        BG->Batches.push_front(CurBatch);
        UnusedSlots = static_cast<u16>(BG->MaxCachedPerBatch - 1);
      }
      // `UnusedSlots` is u16 so the result will be also fit in u16.
      const u16 AppendSize = static_cast<u16>(Min<u32>(UnusedSlots, Size - I));
      CurBatch->appendFromArray(&Array[I], AppendSize);
      I += AppendSize;
    }

    BG->PushedBlocks += Size;
  }

  // Push the blocks to their batch group. The layout will be like,
  //
  // FreeListInfo.BlockList - > BG -> BG -> BG
  //                            |     |     |
  //                            v     v     v
  //                            TB    TB    TB
  //                            |
  //                            v
  //                            TB
  //
  // Each BlockGroup(BG) will associate with unique group id and the free blocks
  // are managed by a list of TransferBatch(TB). To reduce the time of inserting
  // blocks, BGs are sorted and the input `Array` are supposed to be sorted so
  // that we can get better performance of maintaining sorted property.
  // Use `SameGroup=true` to indicate that all blocks in the array are from the
  // same group then we will skip checking the group id of each block.
  void pushBlocksImpl(CacheT *C, uptr ClassId, RegionInfo *Region,
                      CompactPtrT *Array, u32 Size, bool SameGroup = false)
      REQUIRES(Region->FLLock) {
    DCHECK_NE(ClassId, SizeClassMap::BatchClassId);
    DCHECK_GT(Size, 0U);

    auto CreateGroup = [&](uptr CompactPtrGroupBase) {
      BatchGroupT *BG =
          reinterpret_cast<BatchGroupT *>(C->getBatchClassBlock());
      BG->Batches.clear();
      TransferBatchT *TB =
          reinterpret_cast<TransferBatchT *>(C->getBatchClassBlock());
      TB->clear();

      BG->CompactPtrGroupBase = CompactPtrGroupBase;
      BG->Batches.push_front(TB);
      BG->PushedBlocks = 0;
      BG->BytesInBGAtLastCheckpoint = 0;
      BG->MaxCachedPerBatch = CacheT::getMaxCached(getSizeByClassId(ClassId));

      return BG;
    };

    auto InsertBlocks = [&](BatchGroupT *BG, CompactPtrT *Array, u32 Size) {
      SinglyLinkedList<TransferBatchT> &Batches = BG->Batches;
      TransferBatchT *CurBatch = Batches.front();
      DCHECK_NE(CurBatch, nullptr);

      for (u32 I = 0; I < Size;) {
        DCHECK_GE(BG->MaxCachedPerBatch, CurBatch->getCount());
        u16 UnusedSlots =
            static_cast<u16>(BG->MaxCachedPerBatch - CurBatch->getCount());
        if (UnusedSlots == 0) {
          CurBatch =
              reinterpret_cast<TransferBatchT *>(C->getBatchClassBlock());
          CurBatch->clear();
          Batches.push_front(CurBatch);
          UnusedSlots = BG->MaxCachedPerBatch;
        }
        // `UnusedSlots` is u16 so the result will be also fit in u16.
        u16 AppendSize = static_cast<u16>(Min<u32>(UnusedSlots, Size - I));
        CurBatch->appendFromArray(&Array[I], AppendSize);
        I += AppendSize;
      }

      BG->PushedBlocks += Size;
    };

    Region->FreeListInfo.PushedBlocks += Size;
    BatchGroupT *Cur = Region->FreeListInfo.BlockList.front();

    // In the following, `Cur` always points to the BatchGroup for blocks that
    // will be pushed next. `Prev` is the element right before `Cur`.
    BatchGroupT *Prev = nullptr;

    while (Cur != nullptr &&
           compactPtrGroup(Array[0]) > Cur->CompactPtrGroupBase) {
      Prev = Cur;
      Cur = Cur->Next;
    }

    if (Cur == nullptr ||
        compactPtrGroup(Array[0]) != Cur->CompactPtrGroupBase) {
      Cur = CreateGroup(compactPtrGroup(Array[0]));
      if (Prev == nullptr)
        Region->FreeListInfo.BlockList.push_front(Cur);
      else
        Region->FreeListInfo.BlockList.insert(Prev, Cur);
    }

    // All the blocks are from the same group, just push without checking group
    // id.
    if (SameGroup) {
      for (u32 I = 0; I < Size; ++I)
        DCHECK_EQ(compactPtrGroup(Array[I]), Cur->CompactPtrGroupBase);

      InsertBlocks(Cur, Array, Size);
      return;
    }

    // The blocks are sorted by group id. Determine the segment of group and
    // push them to their group together.
    u32 Count = 1;
    for (u32 I = 1; I < Size; ++I) {
      if (compactPtrGroup(Array[I - 1]) != compactPtrGroup(Array[I])) {
        DCHECK_EQ(compactPtrGroup(Array[I - 1]), Cur->CompactPtrGroupBase);
        InsertBlocks(Cur, Array + I - Count, Count);

        while (Cur != nullptr &&
               compactPtrGroup(Array[I]) > Cur->CompactPtrGroupBase) {
          Prev = Cur;
          Cur = Cur->Next;
        }

        if (Cur == nullptr ||
            compactPtrGroup(Array[I]) != Cur->CompactPtrGroupBase) {
          Cur = CreateGroup(compactPtrGroup(Array[I]));
          DCHECK_NE(Prev, nullptr);
          Region->FreeListInfo.BlockList.insert(Prev, Cur);
        }

        Count = 1;
      } else {
        ++Count;
      }
    }

    InsertBlocks(Cur, Array + Size - Count, Count);
  }

  // Pop one TransferBatch from a BatchGroup. The BatchGroup with the smallest
  // group id will be considered first.
  //
  // The region mutex needs to be held while calling this method.
  TransferBatchT *popBatchImpl(CacheT *C, uptr ClassId, RegionInfo *Region)
      REQUIRES(Region->FLLock) {
    if (Region->FreeListInfo.BlockList.empty())
      return nullptr;

    SinglyLinkedList<TransferBatchT> &Batches =
        Region->FreeListInfo.BlockList.front()->Batches;

    if (Batches.empty()) {
      DCHECK_EQ(ClassId, SizeClassMap::BatchClassId);
      BatchGroupT *BG = Region->FreeListInfo.BlockList.front();
      Region->FreeListInfo.BlockList.pop_front();

      // Block used by `BatchGroup` is from BatchClassId. Turn the block into
      // `TransferBatch` with single block.
      TransferBatchT *TB = reinterpret_cast<TransferBatchT *>(BG);
      TB->clear();
      TB->add(
          compactPtr(SizeClassMap::BatchClassId, reinterpret_cast<uptr>(TB)));
      Region->FreeListInfo.PoppedBlocks += 1;
      return TB;
    }

    TransferBatchT *B = Batches.front();
    Batches.pop_front();
    DCHECK_NE(B, nullptr);
    DCHECK_GT(B->getCount(), 0U);

    if (Batches.empty()) {
      BatchGroupT *BG = Region->FreeListInfo.BlockList.front();
      Region->FreeListInfo.BlockList.pop_front();

      // We don't keep BatchGroup with zero blocks to avoid empty-checking while
      // allocating. Note that block used by constructing BatchGroup is recorded
      // as free blocks in the last element of BatchGroup::Batches. Which means,
      // once we pop the last TransferBatch, the block is implicitly
      // deallocated.
      if (ClassId != SizeClassMap::BatchClassId)
        C->deallocate(SizeClassMap::BatchClassId, BG);
    }

    Region->FreeListInfo.PoppedBlocks += B->getCount();

    return B;
  }

  // Refill the freelist and return one batch.
  NOINLINE TransferBatchT *populateFreeListAndPopBatch(CacheT *C, uptr ClassId,
                                                       RegionInfo *Region)
      REQUIRES(Region->MMLock) EXCLUDES(Region->FLLock) {
    const uptr Size = getSizeByClassId(ClassId);
    const u16 MaxCount = CacheT::getMaxCached(Size);

    const uptr RegionBeg = Region->RegionBeg;
    const uptr MappedUser = Region->MemMapInfo.MappedUser;
    const uptr TotalUserBytes =
        Region->MemMapInfo.AllocatedUser + MaxCount * Size;
    // Map more space for blocks, if necessary.
    if (TotalUserBytes > MappedUser) {
      // Do the mmap for the user memory.
      const uptr MapSize =
          roundUp(TotalUserBytes - MappedUser, MapSizeIncrement);
      const uptr RegionBase = RegionBeg - getRegionBaseByClassId(ClassId);
      if (UNLIKELY(RegionBase + MappedUser + MapSize > RegionSize)) {
        Region->Exhausted = true;
        return nullptr;
      }

      if (UNLIKELY(!Region->MemMapInfo.MemMap.remap(
              RegionBeg + MappedUser, MapSize, "scudo:primary",
              MAP_ALLOWNOMEM | MAP_RESIZABLE |
                  (useMemoryTagging<Config>(Options.load()) ? MAP_MEMTAG
                                                            : 0)))) {
        return nullptr;
      }
      Region->MemMapInfo.MappedUser += MapSize;
      C->getStats().add(StatMapped, MapSize);
    }

    const u32 NumberOfBlocks =
        Min(MaxNumBatches * MaxCount,
            static_cast<u32>((Region->MemMapInfo.MappedUser -
                              Region->MemMapInfo.AllocatedUser) /
                             Size));
    DCHECK_GT(NumberOfBlocks, 0);

    constexpr u32 ShuffleArraySize =
        MaxNumBatches * TransferBatchT::MaxNumCached;
    CompactPtrT ShuffleArray[ShuffleArraySize];
    DCHECK_LE(NumberOfBlocks, ShuffleArraySize);

    const uptr CompactPtrBase = getCompactPtrBaseByClassId(ClassId);
    uptr P = RegionBeg + Region->MemMapInfo.AllocatedUser;
    for (u32 I = 0; I < NumberOfBlocks; I++, P += Size)
      ShuffleArray[I] = compactPtrInternal(CompactPtrBase, P);

    ScopedLock L(Region->FLLock);

    if (ClassId != SizeClassMap::BatchClassId) {
      u32 N = 1;
      uptr CurGroup = compactPtrGroup(ShuffleArray[0]);
      for (u32 I = 1; I < NumberOfBlocks; I++) {
        if (UNLIKELY(compactPtrGroup(ShuffleArray[I]) != CurGroup)) {
          shuffle(ShuffleArray + I - N, N, &Region->RandState);
          pushBlocksImpl(C, ClassId, Region, ShuffleArray + I - N, N,
                         /*SameGroup=*/true);
          N = 1;
          CurGroup = compactPtrGroup(ShuffleArray[I]);
        } else {
          ++N;
        }
      }

      shuffle(ShuffleArray + NumberOfBlocks - N, N, &Region->RandState);
      pushBlocksImpl(C, ClassId, Region, &ShuffleArray[NumberOfBlocks - N], N,
                     /*SameGroup=*/true);
    } else {
      pushBatchClassBlocks(Region, ShuffleArray, NumberOfBlocks);
    }

    TransferBatchT *B = popBatchImpl(C, ClassId, Region);
    DCHECK_NE(B, nullptr);

    // Note that `PushedBlocks` and `PoppedBlocks` are supposed to only record
    // the requests from `PushBlocks` and `PopBatch` which are external
    // interfaces. `populateFreeListAndPopBatch` is the internal interface so we
    // should set the values back to avoid incorrectly setting the stats.
    Region->FreeListInfo.PushedBlocks -= NumberOfBlocks;

    const uptr AllocatedUser = Size * NumberOfBlocks;
    C->getStats().add(StatFree, AllocatedUser);
    Region->MemMapInfo.AllocatedUser += AllocatedUser;

    return B;
  }

  void getStats(ScopedString *Str, uptr ClassId, RegionInfo *Region)
      REQUIRES(Region->MMLock, Region->FLLock) {
    if (Region->MemMapInfo.MappedUser == 0)
      return;
    const uptr BlockSize = getSizeByClassId(ClassId);
    const uptr InUseBlocks =
        Region->FreeListInfo.PoppedBlocks - Region->FreeListInfo.PushedBlocks;
    const uptr BytesInFreeList =
        Region->MemMapInfo.AllocatedUser - InUseBlocks * BlockSize;
    uptr RegionPushedBytesDelta = 0;
    if (BytesInFreeList >=
        Region->ReleaseInfo.BytesInFreeListAtLastCheckpoint) {
      RegionPushedBytesDelta =
          BytesInFreeList - Region->ReleaseInfo.BytesInFreeListAtLastCheckpoint;
    }
    const uptr TotalChunks = Region->MemMapInfo.AllocatedUser / BlockSize;
    Str->append(
        "%s %02zu (%6zu): mapped: %6zuK popped: %7zu pushed: %7zu "
        "inuse: %6zu total: %6zu releases: %6zu last "
        "released: %6zuK latest pushed bytes: %6zuK region: 0x%zx (0x%zx)\n",
        Region->Exhausted ? "E" : " ", ClassId, getSizeByClassId(ClassId),
        Region->MemMapInfo.MappedUser >> 10, Region->FreeListInfo.PoppedBlocks,
        Region->FreeListInfo.PushedBlocks, InUseBlocks, TotalChunks,
        Region->ReleaseInfo.RangesReleased,
        Region->ReleaseInfo.LastReleasedBytes >> 10,
        RegionPushedBytesDelta >> 10, Region->RegionBeg,
        getRegionBaseByClassId(ClassId));
  }

  void getRegionFragmentationInfo(RegionInfo *Region, uptr ClassId,
                                  ScopedString *Str) REQUIRES(Region->MMLock) {
    const uptr BlockSize = getSizeByClassId(ClassId);
    const uptr AllocatedUserEnd =
        Region->MemMapInfo.AllocatedUser + Region->RegionBeg;

    SinglyLinkedList<BatchGroupT> GroupsToRelease;
    {
      ScopedLock L(Region->FLLock);
      GroupsToRelease = Region->FreeListInfo.BlockList;
      Region->FreeListInfo.BlockList.clear();
    }

    FragmentationRecorder Recorder;
    if (!GroupsToRelease.empty()) {
      PageReleaseContext Context =
          markFreeBlocks(Region, BlockSize, AllocatedUserEnd,
                         getCompactPtrBaseByClassId(ClassId), GroupsToRelease);
      auto SkipRegion = [](UNUSED uptr RegionIndex) { return false; };
      releaseFreeMemoryToOS(Context, Recorder, SkipRegion);

      mergeGroupsToReleaseBack(Region, GroupsToRelease);
    }

    ScopedLock L(Region->FLLock);
    const uptr PageSize = getPageSizeCached();
    const uptr TotalBlocks = Region->MemMapInfo.AllocatedUser / BlockSize;
    const uptr InUseBlocks =
        Region->FreeListInfo.PoppedBlocks - Region->FreeListInfo.PushedBlocks;
    const uptr AllocatedPagesCount =
        roundUp(Region->MemMapInfo.AllocatedUser, PageSize) / PageSize;
    DCHECK_GE(AllocatedPagesCount, Recorder.getReleasedPagesCount());
    const uptr InUsePages =
        AllocatedPagesCount - Recorder.getReleasedPagesCount();
    const uptr InUseBytes = InUsePages * PageSize;

    Str->append("  %02zu (%6zu): inuse/total blocks: %6zu/%6zu inuse/total "
                "pages: %6zu/%6zu inuse bytes: %6zuK\n",
                ClassId, BlockSize, InUseBlocks, TotalBlocks, InUsePages,
                AllocatedPagesCount, InUseBytes >> 10);
  }

  NOINLINE uptr releaseToOSMaybe(RegionInfo *Region, uptr ClassId,
                                 ReleaseToOS ReleaseType = ReleaseToOS::Normal)
      REQUIRES(Region->MMLock) EXCLUDES(Region->FLLock) {
    const uptr BlockSize = getSizeByClassId(ClassId);
    uptr BytesInFreeList;
    const uptr AllocatedUserEnd =
        Region->MemMapInfo.AllocatedUser + Region->RegionBeg;
    SinglyLinkedList<BatchGroupT> GroupsToRelease;

    {
      ScopedLock L(Region->FLLock);

      BytesInFreeList = Region->MemMapInfo.AllocatedUser -
                        (Region->FreeListInfo.PoppedBlocks -
                         Region->FreeListInfo.PushedBlocks) *
                            BlockSize;
      if (UNLIKELY(BytesInFreeList == 0))
        return false;

      // ==================================================================== //
      // 1. Check if we have enough free blocks and if it's worth doing a page
      //    release.
      // ==================================================================== //
      if (ReleaseType != ReleaseToOS::ForceAll &&
          !hasChanceToReleasePages(Region, BlockSize, BytesInFreeList,
                                   ReleaseType)) {
        return 0;
      }

      // ==================================================================== //
      // 2. Determine which groups can release the pages. Use a heuristic to
      //    gather groups that are candidates for doing a release.
      // ==================================================================== //
      if (ReleaseType == ReleaseToOS::ForceAll) {
        GroupsToRelease = Region->FreeListInfo.BlockList;
        Region->FreeListInfo.BlockList.clear();
      } else {
        GroupsToRelease =
            collectGroupsToRelease(Region, BlockSize, AllocatedUserEnd,
                                   getCompactPtrBaseByClassId(ClassId));
      }
      if (GroupsToRelease.empty())
        return 0;
    }

    // Note that we have extracted the `GroupsToRelease` from region freelist.
    // It's safe to let pushBlocks()/popBatches() access the remaining region
    // freelist. In the steps 3 and 4, we will temporarily release the FLLock
    // and lock it again before step 5.

    // ==================================================================== //
    // 3. Mark the free blocks in `GroupsToRelease` in the `PageReleaseContext`.
    //    Then we can tell which pages are in-use by querying
    //    `PageReleaseContext`.
    // ==================================================================== //
    PageReleaseContext Context =
        markFreeBlocks(Region, BlockSize, AllocatedUserEnd,
                       getCompactPtrBaseByClassId(ClassId), GroupsToRelease);
    if (UNLIKELY(!Context.hasBlockMarked())) {
      mergeGroupsToReleaseBack(Region, GroupsToRelease);
      return 0;
    }

    // ==================================================================== //
    // 4. Release the unused physical pages back to the OS.
    // ==================================================================== //
    RegionReleaseRecorder<MemMapT> Recorder(&Region->MemMapInfo.MemMap,
                                            Region->RegionBeg,
                                            Context.getReleaseOffset());
    auto SkipRegion = [](UNUSED uptr RegionIndex) { return false; };
    releaseFreeMemoryToOS(Context, Recorder, SkipRegion);
    if (Recorder.getReleasedRangesCount() > 0) {
      Region->ReleaseInfo.BytesInFreeListAtLastCheckpoint = BytesInFreeList;
      Region->ReleaseInfo.RangesReleased += Recorder.getReleasedRangesCount();
      Region->ReleaseInfo.LastReleasedBytes = Recorder.getReleasedBytes();
    }
    Region->ReleaseInfo.LastReleaseAtNs = getMonotonicTimeFast();

    // ====================================================================== //
    // 5. Merge the `GroupsToRelease` back to the freelist.
    // ====================================================================== //
    mergeGroupsToReleaseBack(Region, GroupsToRelease);

    return Recorder.getReleasedBytes();
  }

  bool hasChanceToReleasePages(RegionInfo *Region, uptr BlockSize,
                               uptr BytesInFreeList, ReleaseToOS ReleaseType)
      REQUIRES(Region->MMLock, Region->FLLock) {
    DCHECK_GE(Region->FreeListInfo.PoppedBlocks,
              Region->FreeListInfo.PushedBlocks);
    const uptr PageSize = getPageSizeCached();

    // Always update `BytesInFreeListAtLastCheckpoint` with the smallest value
    // so that we won't underestimate the releasable pages. For example, the
    // following is the region usage,
    //
    //  BytesInFreeListAtLastCheckpoint   AllocatedUser
    //                v                         v
    //  |--------------------------------------->
    //         ^                   ^
    //  BytesInFreeList     ReleaseThreshold
    //
    // In general, if we have collected enough bytes and the amount of free
    // bytes meets the ReleaseThreshold, we will try to do page release. If we
    // don't update `BytesInFreeListAtLastCheckpoint` when the current
    // `BytesInFreeList` is smaller, we may take longer time to wait for enough
    // freed blocks because we miss the bytes between
    // (BytesInFreeListAtLastCheckpoint - BytesInFreeList).
    if (BytesInFreeList <=
        Region->ReleaseInfo.BytesInFreeListAtLastCheckpoint) {
      Region->ReleaseInfo.BytesInFreeListAtLastCheckpoint = BytesInFreeList;
    }

    const uptr RegionPushedBytesDelta =
        BytesInFreeList - Region->ReleaseInfo.BytesInFreeListAtLastCheckpoint;
    if (RegionPushedBytesDelta < PageSize)
      return false;

    // Releasing smaller blocks is expensive, so we want to make sure that a
    // significant amount of bytes are free, and that there has been a good
    // amount of batches pushed to the freelist before attempting to release.
    if (isSmallBlock(BlockSize) && ReleaseType == ReleaseToOS::Normal)
      if (RegionPushedBytesDelta < Region->TryReleaseThreshold)
        return false;

    if (ReleaseType == ReleaseToOS::Normal) {
      const s32 IntervalMs = atomic_load_relaxed(&ReleaseToOsIntervalMs);
      if (IntervalMs < 0)
        return false;

      // The constant 8 here is selected from profiling some apps and the number
      // of unreleased pages in the large size classes is around 16 pages or
      // more. Choose half of it as a heuristic and which also avoids page
      // release every time for every pushBlocks() attempt by large blocks.
      const bool ByPassReleaseInterval =
          isLargeBlock(BlockSize) && RegionPushedBytesDelta > 8 * PageSize;
      if (!ByPassReleaseInterval) {
        if (Region->ReleaseInfo.LastReleaseAtNs +
                static_cast<u64>(IntervalMs) * 1000000 >
            getMonotonicTimeFast()) {
          // Memory was returned recently.
          return false;
        }
      }
    } // if (ReleaseType == ReleaseToOS::Normal)

    return true;
  }

  SinglyLinkedList<BatchGroupT>
  collectGroupsToRelease(RegionInfo *Region, const uptr BlockSize,
                         const uptr AllocatedUserEnd, const uptr CompactPtrBase)
      REQUIRES(Region->MMLock, Region->FLLock) {
    const uptr GroupSize = (1UL << GroupSizeLog);
    const uptr PageSize = getPageSizeCached();
    SinglyLinkedList<BatchGroupT> GroupsToRelease;

    // We are examining each group and will take the minimum distance to the
    // release threshold as the next Region::TryReleaseThreshold(). Note that if
    // the size of free blocks has reached the release threshold, the distance
    // to the next release will be PageSize * SmallerBlockReleasePageDelta. See
    // the comment on `SmallerBlockReleasePageDelta` for more details.
    uptr MinDistToThreshold = GroupSize;

    for (BatchGroupT *BG = Region->FreeListInfo.BlockList.front(),
                     *Prev = nullptr;
         BG != nullptr;) {
      // Group boundary is always GroupSize-aligned from CompactPtr base. The
      // layout of memory groups is like,
      //
      //     (CompactPtrBase)
      // #1 CompactPtrGroupBase   #2 CompactPtrGroupBase            ...
      //           |                       |                       |
      //           v                       v                       v
      //           +-----------------------+-----------------------+
      //            \                     / \                     /
      //             ---   GroupSize   ---   ---   GroupSize   ---
      //
      // After decompacting the CompactPtrGroupBase, we expect the alignment
      // property is held as well.
      const uptr BatchGroupBase =
          decompactGroupBase(CompactPtrBase, BG->CompactPtrGroupBase);
      DCHECK_LE(Region->RegionBeg, BatchGroupBase);
      DCHECK_GE(AllocatedUserEnd, BatchGroupBase);
      DCHECK_EQ((Region->RegionBeg - BatchGroupBase) % GroupSize, 0U);
      // TransferBatches are pushed in front of BG.Batches. The first one may
      // not have all caches used.
      const uptr NumBlocks = (BG->Batches.size() - 1) * BG->MaxCachedPerBatch +
                             BG->Batches.front()->getCount();
      const uptr BytesInBG = NumBlocks * BlockSize;

      if (BytesInBG <= BG->BytesInBGAtLastCheckpoint) {
        BG->BytesInBGAtLastCheckpoint = BytesInBG;
        Prev = BG;
        BG = BG->Next;
        continue;
      }

      const uptr PushedBytesDelta = BG->BytesInBGAtLastCheckpoint - BytesInBG;

      // Given the randomness property, we try to release the pages only if the
      // bytes used by free blocks exceed certain proportion of group size. Note
      // that this heuristic only applies when all the spaces in a BatchGroup
      // are allocated.
      if (isSmallBlock(BlockSize)) {
        const uptr BatchGroupEnd = BatchGroupBase + GroupSize;
        const uptr AllocatedGroupSize = AllocatedUserEnd >= BatchGroupEnd
                                            ? GroupSize
                                            : AllocatedUserEnd - BatchGroupBase;
        const uptr ReleaseThreshold =
            (AllocatedGroupSize * (100 - 1U - BlockSize / 16U)) / 100U;
        const bool HighDensity = BytesInBG >= ReleaseThreshold;
        const bool MayHaveReleasedAll = NumBlocks >= (GroupSize / BlockSize);
        // If all blocks in the group are released, we will do range marking
        // which is fast. Otherwise, we will wait until we have accumulated
        // a certain amount of free memory.
        const bool ReachReleaseDelta =
            MayHaveReleasedAll
                ? true
                : PushedBytesDelta >= PageSize * SmallerBlockReleasePageDelta;

        if (!HighDensity) {
          DCHECK_LE(BytesInBG, ReleaseThreshold);
          // The following is the usage of a memroy group,
          //
          //     BytesInBG             ReleaseThreshold
          //  /             \                 v
          //  +---+---------------------------+-----+
          //  |   |         |                 |     |
          //  +---+---------------------------+-----+
          //       \        /                       ^
          //    PushedBytesDelta                 GroupEnd
          MinDistToThreshold =
              Min(MinDistToThreshold,
                  ReleaseThreshold - BytesInBG + PushedBytesDelta);
        } else {
          // If it reaches high density at this round, the next time we will try
          // to release is based on SmallerBlockReleasePageDelta
          MinDistToThreshold =
              Min(MinDistToThreshold, PageSize * SmallerBlockReleasePageDelta);
        }

        if (!HighDensity || !ReachReleaseDelta) {
          Prev = BG;
          BG = BG->Next;
          continue;
        }
      }

      // If `BG` is the first BatchGroupT in the list, we only need to advance
      // `BG` and call FreeListInfo.BlockList::pop_front(). No update is needed
      // for `Prev`.
      //
      //         (BG)   (BG->Next)
      // Prev     Cur      BG
      //   |       |       |
      //   v       v       v
      //  nil     +--+    +--+
      //          |X | -> |  | -> ...
      //          +--+    +--+
      //
      // Otherwise, `Prev` will be used to extract the `Cur` from the
      // `FreeListInfo.BlockList`.
      //
      //         (BG)   (BG->Next)
      // Prev     Cur      BG
      //   |       |       |
      //   v       v       v
      //  +--+    +--+    +--+
      //  |  | -> |X | -> |  | -> ...
      //  +--+    +--+    +--+
      //
      // After FreeListInfo.BlockList::extract(),
      //
      // Prev     Cur       BG
      //   |       |        |
      //   v       v        v
      //  +--+    +--+     +--+
      //  |  |-+  |X |  +->|  | -> ...
      //  +--+ |  +--+  |  +--+
      //       +--------+
      //
      // Note that we need to advance before pushing this BatchGroup to
      // GroupsToRelease because it's a destructive operation.

      BatchGroupT *Cur = BG;
      BG = BG->Next;

      // Ideally, we may want to update this only after successful release.
      // However, for smaller blocks, each block marking is a costly operation.
      // Therefore, we update it earlier.
      // TODO: Consider updating this after releasing pages if `ReleaseRecorder`
      // can tell the released bytes in each group.
      Cur->BytesInBGAtLastCheckpoint = BytesInBG;

      if (Prev != nullptr)
        Region->FreeListInfo.BlockList.extract(Prev, Cur);
      else
        Region->FreeListInfo.BlockList.pop_front();
      GroupsToRelease.push_back(Cur);
    }

    // Only small blocks have the adaptive `TryReleaseThreshold`.
    if (isSmallBlock(BlockSize)) {
      // If the MinDistToThreshold is not updated, that means each memory group
      // may have only pushed less than a page size. In that case, just set it
      // back to normal.
      if (MinDistToThreshold == GroupSize)
        MinDistToThreshold = PageSize * SmallerBlockReleasePageDelta;
      Region->TryReleaseThreshold = MinDistToThreshold;
    }

    return GroupsToRelease;
  }

  PageReleaseContext
  markFreeBlocks(RegionInfo *Region, const uptr BlockSize,
                 const uptr AllocatedUserEnd, const uptr CompactPtrBase,
                 SinglyLinkedList<BatchGroupT> &GroupsToRelease)
      REQUIRES(Region->MMLock) EXCLUDES(Region->FLLock) {
    const uptr GroupSize = (1UL << GroupSizeLog);
    auto DecompactPtr = [CompactPtrBase](CompactPtrT CompactPtr) {
      return decompactPtrInternal(CompactPtrBase, CompactPtr);
    };

    const uptr ReleaseBase = decompactGroupBase(
        CompactPtrBase, GroupsToRelease.front()->CompactPtrGroupBase);
    const uptr LastGroupEnd =
        Min(decompactGroupBase(CompactPtrBase,
                               GroupsToRelease.back()->CompactPtrGroupBase) +
                GroupSize,
            AllocatedUserEnd);
    // The last block may straddle the group boundary. Rounding up to BlockSize
    // to get the exact range.
    const uptr ReleaseEnd =
        roundUpSlow(LastGroupEnd - Region->RegionBeg, BlockSize) +
        Region->RegionBeg;
    const uptr ReleaseRangeSize = ReleaseEnd - ReleaseBase;
    const uptr ReleaseOffset = ReleaseBase - Region->RegionBeg;

    PageReleaseContext Context(BlockSize, /*NumberOfRegions=*/1U,
                               ReleaseRangeSize, ReleaseOffset);
    // We may not be able to do the page release in a rare case that we may
    // fail on PageMap allocation.
    if (UNLIKELY(!Context.ensurePageMapAllocated()))
      return Context;

    for (BatchGroupT &BG : GroupsToRelease) {
      const uptr BatchGroupBase =
          decompactGroupBase(CompactPtrBase, BG.CompactPtrGroupBase);
      const uptr BatchGroupEnd = BatchGroupBase + GroupSize;
      const uptr AllocatedGroupSize = AllocatedUserEnd >= BatchGroupEnd
                                          ? GroupSize
                                          : AllocatedUserEnd - BatchGroupBase;
      const uptr BatchGroupUsedEnd = BatchGroupBase + AllocatedGroupSize;
      const bool MayContainLastBlockInRegion =
          BatchGroupUsedEnd == AllocatedUserEnd;
      const bool BlockAlignedWithUsedEnd =
          (BatchGroupUsedEnd - Region->RegionBeg) % BlockSize == 0;

      uptr MaxContainedBlocks = AllocatedGroupSize / BlockSize;
      if (!BlockAlignedWithUsedEnd)
        ++MaxContainedBlocks;

      const uptr NumBlocks = (BG.Batches.size() - 1) * BG.MaxCachedPerBatch +
                             BG.Batches.front()->getCount();

      if (NumBlocks == MaxContainedBlocks) {
        for (const auto &It : BG.Batches) {
          if (&It != BG.Batches.front())
            DCHECK_EQ(It.getCount(), BG.MaxCachedPerBatch);
          for (u16 I = 0; I < It.getCount(); ++I)
            DCHECK_EQ(compactPtrGroup(It.get(I)), BG.CompactPtrGroupBase);
        }

        Context.markRangeAsAllCounted(BatchGroupBase, BatchGroupUsedEnd,
                                      Region->RegionBeg, /*RegionIndex=*/0,
                                      Region->MemMapInfo.AllocatedUser);
      } else {
        DCHECK_LT(NumBlocks, MaxContainedBlocks);
        // Note that we don't always visit blocks in each BatchGroup so that we
        // may miss the chance of releasing certain pages that cross
        // BatchGroups.
        Context.markFreeBlocksInRegion(
            BG.Batches, DecompactPtr, Region->RegionBeg, /*RegionIndex=*/0,
            Region->MemMapInfo.AllocatedUser, MayContainLastBlockInRegion);
      }
    }

    DCHECK(Context.hasBlockMarked());

    return Context;
  }

  void mergeGroupsToReleaseBack(RegionInfo *Region,
                                SinglyLinkedList<BatchGroupT> &GroupsToRelease)
      REQUIRES(Region->MMLock) EXCLUDES(Region->FLLock) {
    ScopedLock L(Region->FLLock);

    // After merging two freelists, we may have redundant `BatchGroup`s that
    // need to be recycled. The number of unused `BatchGroup`s is expected to be
    // small. Pick a constant which is inferred from real programs.
    constexpr uptr MaxUnusedSize = 8;
    CompactPtrT Blocks[MaxUnusedSize];
    u32 Idx = 0;
    RegionInfo *BatchClassRegion = getRegionInfo(SizeClassMap::BatchClassId);
    // We can't call pushBatchClassBlocks() to recycle the unused `BatchGroup`s
    // when we are manipulating the freelist of `BatchClassRegion`. Instead, we
    // should just push it back to the freelist when we merge two `BatchGroup`s.
    // This logic hasn't been implemented because we haven't supported releasing
    // pages in `BatchClassRegion`.
    DCHECK_NE(BatchClassRegion, Region);

    // Merge GroupsToRelease back to the Region::FreeListInfo.BlockList. Note
    // that both `Region->FreeListInfo.BlockList` and `GroupsToRelease` are
    // sorted.
    for (BatchGroupT *BG = Region->FreeListInfo.BlockList.front(),
                     *Prev = nullptr;
         ;) {
      if (BG == nullptr || GroupsToRelease.empty()) {
        if (!GroupsToRelease.empty())
          Region->FreeListInfo.BlockList.append_back(&GroupsToRelease);
        break;
      }

      DCHECK(!BG->Batches.empty());

      if (BG->CompactPtrGroupBase <
          GroupsToRelease.front()->CompactPtrGroupBase) {
        Prev = BG;
        BG = BG->Next;
        continue;
      }

      BatchGroupT *Cur = GroupsToRelease.front();
      TransferBatchT *UnusedTransferBatch = nullptr;
      GroupsToRelease.pop_front();

      if (BG->CompactPtrGroupBase == Cur->CompactPtrGroupBase) {
        BG->PushedBlocks += Cur->PushedBlocks;
        // We have updated `BatchGroup::BytesInBGAtLastCheckpoint` while
        // collecting the `GroupsToRelease`.
        BG->BytesInBGAtLastCheckpoint = Cur->BytesInBGAtLastCheckpoint;
        const uptr MaxCachedPerBatch = BG->MaxCachedPerBatch;

        // Note that the first TransferBatches in both `Batches` may not be
        // full and only the first TransferBatch can have non-full blocks. Thus
        // we have to merge them before appending one to another.
        if (Cur->Batches.front()->getCount() == MaxCachedPerBatch) {
          BG->Batches.append_back(&Cur->Batches);
        } else {
          TransferBatchT *NonFullBatch = Cur->Batches.front();
          Cur->Batches.pop_front();
          const u16 NonFullBatchCount = NonFullBatch->getCount();
          // The remaining Batches in `Cur` are full.
          BG->Batches.append_back(&Cur->Batches);

          if (BG->Batches.front()->getCount() == MaxCachedPerBatch) {
            // Only 1 non-full TransferBatch, push it to the front.
            BG->Batches.push_front(NonFullBatch);
          } else {
            const u16 NumBlocksToMove = static_cast<u16>(
                Min(static_cast<u16>(MaxCachedPerBatch -
                                     BG->Batches.front()->getCount()),
                    NonFullBatchCount));
            BG->Batches.front()->appendFromTransferBatch(NonFullBatch,
                                                         NumBlocksToMove);
            if (NonFullBatch->isEmpty())
              UnusedTransferBatch = NonFullBatch;
            else
              BG->Batches.push_front(NonFullBatch);
          }
        }

        const u32 NeededSlots = UnusedTransferBatch == nullptr ? 1U : 2U;
        if (UNLIKELY(Idx + NeededSlots > MaxUnusedSize)) {
          ScopedLock L(BatchClassRegion->FLLock);
          pushBatchClassBlocks(BatchClassRegion, Blocks, Idx);
          Idx = 0;
        }
        Blocks[Idx++] =
            compactPtr(SizeClassMap::BatchClassId, reinterpret_cast<uptr>(Cur));
        if (UnusedTransferBatch) {
          Blocks[Idx++] =
              compactPtr(SizeClassMap::BatchClassId,
                         reinterpret_cast<uptr>(UnusedTransferBatch));
        }
        Prev = BG;
        BG = BG->Next;
        continue;
      }

      // At here, the `BG` is the first BatchGroup with CompactPtrGroupBase
      // larger than the first element in `GroupsToRelease`. We need to insert
      // `GroupsToRelease::front()` (which is `Cur` below)  before `BG`.
      //
      //   1. If `Prev` is nullptr, we simply push `Cur` to the front of
      //      FreeListInfo.BlockList.
      //   2. Otherwise, use `insert()` which inserts an element next to `Prev`.
      //
      // Afterwards, we don't need to advance `BG` because the order between
      // `BG` and the new `GroupsToRelease::front()` hasn't been checked.
      if (Prev == nullptr)
        Region->FreeListInfo.BlockList.push_front(Cur);
      else
        Region->FreeListInfo.BlockList.insert(Prev, Cur);
      DCHECK_EQ(Cur->Next, BG);
      Prev = Cur;
    }

    if (Idx != 0) {
      ScopedLock L(BatchClassRegion->FLLock);
      pushBatchClassBlocks(BatchClassRegion, Blocks, Idx);
    }

    if (SCUDO_DEBUG) {
      BatchGroupT *Prev = Region->FreeListInfo.BlockList.front();
      for (BatchGroupT *Cur = Prev->Next; Cur != nullptr;
           Prev = Cur, Cur = Cur->Next) {
        CHECK_LT(Prev->CompactPtrGroupBase, Cur->CompactPtrGroupBase);
      }
    }
  }

  // TODO: `PrimaryBase` can be obtained from ReservedMemory. This needs to be
  // deprecated.
  uptr PrimaryBase = 0;
  ReservedMemoryT ReservedMemory = {};
  // The minimum size of pushed blocks that we will try to release the pages in
  // that size class.
  uptr SmallerBlockReleasePageDelta = 0;
  atomic_s32 ReleaseToOsIntervalMs = {};
  alignas(SCUDO_CACHE_LINE_SIZE) RegionInfo RegionInfoArray[NumClasses];
};

} // namespace scudo

#endif // SCUDO_PRIMARY64_H_
