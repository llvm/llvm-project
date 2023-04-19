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
  typedef typename Config::PrimaryCompactPtrT CompactPtrT;
  static const uptr CompactPtrScale = Config::PrimaryCompactPtrScale;
  static const uptr GroupSizeLog = Config::PrimaryGroupSizeLog;
  static const uptr GroupScale = GroupSizeLog - CompactPtrScale;
  typedef typename Config::SizeClassMap SizeClassMap;
  typedef SizeClassAllocator64<Config> ThisT;
  typedef SizeClassAllocatorLocalCache<ThisT> CacheT;
  typedef typename CacheT::TransferBatch TransferBatch;
  typedef typename CacheT::BatchGroup BatchGroup;

  static uptr getSizeByClassId(uptr ClassId) {
    return (ClassId == SizeClassMap::BatchClassId)
               ? roundUp(sizeof(TransferBatch), 1U << CompactPtrScale)
               : SizeClassMap::getSizeByClassId(ClassId);
  }

  static bool canAllocate(uptr Size) { return Size <= SizeClassMap::MaxSize; }

  void init(s32 ReleaseToOsInterval) NO_THREAD_SAFETY_ANALYSIS {
    DCHECK(isAligned(reinterpret_cast<uptr>(this), alignof(ThisT)));

    const uptr PageSize = getPageSizeCached();
    const uptr GroupSize = (1U << GroupSizeLog);
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
      Region->RegionBeg = (PrimaryBase + (I << Config::PrimaryRegionSizeLog)) +
                          (Config::PrimaryEnableRandomOffset
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

  TransferBatch *popBatch(CacheT *C, uptr ClassId) {
    DCHECK_LT(ClassId, NumClasses);
    RegionInfo *Region = getRegionInfo(ClassId);
    bool PrintStats = false;
    {
      ScopedLock L(Region->Mutex);
      TransferBatch *B = popBatchImpl(C, ClassId, Region);
      if (LIKELY(B)) {
        Region->Stats.PoppedBlocks += B->getCount();
        return B;
      }

      const bool RegionIsExhausted = Region->Exhausted;
      if (UNLIKELY(RegionIsExhausted ||
                   !populateFreeList(C, ClassId, Region))) {
        PrintStats = !RegionIsExhausted && Region->Exhausted;
      } else {
        B = popBatchImpl(C, ClassId, Region);
        // if `populateFreeList` succeeded, we are supposed to get free blocks.
        DCHECK_NE(B, nullptr);
        Region->Stats.PoppedBlocks += B->getCount();
        return B;
      }
    }

    // Note that `getStats()` requires locking each region so we can't call it
    // while locking the Region->Mutex in the above.
    if (UNLIKELY(PrintStats)) {
      ScopedString Str;
      getStats(&Str);
      Str.append(
          "Scudo OOM: The process has exhausted %zuM for size class %zu.\n",
          RegionSize >> 20, getSizeByClassId(ClassId));
      Str.output();
    }
    return nullptr;
  }

  // Push the array of free blocks to the designated batch group.
  void pushBlocks(CacheT *C, uptr ClassId, CompactPtrT *Array, u32 Size) {
    DCHECK_LT(ClassId, NumClasses);
    DCHECK_GT(Size, 0);

    RegionInfo *Region = getRegionInfo(ClassId);
    if (ClassId == SizeClassMap::BatchClassId) {
      bool PrintStats = false;
      {
        ScopedLock L(Region->Mutex);
        // Constructing a batch group in the free list will use two blocks in
        // BatchClassId. If we are pushing BatchClassId blocks, we will use the
        // blocks in the array directly (can't delegate local cache which will
        // cause a recursive allocation). However, The number of free blocks may
        // be less than two. Therefore, populate the free list before inserting
        // the blocks.
        if (Size >= 2U) {
          pushBlocksImpl(C, SizeClassMap::BatchClassId, Region, Array, Size);
          Region->Stats.PushedBlocks += Size;
        } else {
          const bool RegionIsExhausted = Region->Exhausted;
          if (UNLIKELY(
                  RegionIsExhausted ||
                  !populateFreeList(C, SizeClassMap::BatchClassId, Region))) {
            PrintStats = !RegionIsExhausted && Region->Exhausted;
          }
        }
      }

      // Note that `getStats()` requires the lock of each region so we can't
      // call it while locking the Region->Mutex in the above.
      if (UNLIKELY(PrintStats)) {
        ScopedString Str;
        getStats(&Str);
        Str.append(
            "Scudo OOM: The process has exhausted %zuM for size class %zu.\n",
            RegionSize >> 20, getSizeByClassId(ClassId));
        Str.output();
        // Theoretically, BatchClass shouldn't be used up. Abort immediately
        // when it happens.
        reportOutOfBatchClass();
      }
      return;
    }

    // TODO(chiahungduan): Consider not doing grouping if the group size is not
    // greater than the block size with a certain scale.

    // Sort the blocks so that blocks belonging to the same group can be pushed
    // together.
    bool SameGroup = true;
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

    ScopedLock L(Region->Mutex);
    pushBlocksImpl(C, ClassId, Region, Array, Size, SameGroup);

    Region->Stats.PushedBlocks += Size;
    if (ClassId != SizeClassMap::BatchClassId)
      releaseToOSMaybe(Region, ClassId);
  }

  void disable() NO_THREAD_SAFETY_ANALYSIS {
    // The BatchClassId must be locked last since other classes can use it.
    for (sptr I = static_cast<sptr>(NumClasses) - 1; I >= 0; I--) {
      if (static_cast<uptr>(I) == SizeClassMap::BatchClassId)
        continue;
      getRegionInfo(static_cast<uptr>(I))->Mutex.lock();
    }
    getRegionInfo(SizeClassMap::BatchClassId)->Mutex.lock();
  }

  void enable() NO_THREAD_SAFETY_ANALYSIS {
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
      RegionInfo *Region = getRegionInfo(I);
      // TODO: The call of `iterateOverBlocks` requires disabling
      // SizeClassAllocator64. We may consider locking each region on demand
      // only.
      Region->Mutex.assertHeld();
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
      ScopedLock L(Region->Mutex);
      if (Region->MappedUser)
        TotalMapped += Region->MappedUser;
      PoppedBlocks += Region->Stats.PoppedBlocks;
      PushedBlocks += Region->Stats.PushedBlocks;
    }
    Str->append("Stats: SizeClassAllocator64: %zuM mapped (%uM rss) in %zu "
                "allocations; remains %zu\n",
                TotalMapped >> 20, 0U, PoppedBlocks,
                PoppedBlocks - PushedBlocks);

    for (uptr I = 0; I < NumClasses; I++) {
      RegionInfo *Region = getRegionInfo(I);
      ScopedLock L(Region->Mutex);
      getStats(Str, I, Region, 0);
    }
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

  uptr releaseToOS(ReleaseToOS ReleaseType) {
    uptr TotalReleasedBytes = 0;
    for (uptr I = 0; I < NumClasses; I++) {
      if (I == SizeClassMap::BatchClassId)
        continue;
      RegionInfo *Region = getRegionInfo(I);
      ScopedLock L(Region->Mutex);
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
      // TODO(chiahungduan): In fact, We need to lock the RegionInfo::Mutex.
      // However, the RegionInfoData is passed with const qualifier and lock the
      // mutex requires modifying RegionInfoData, which means we need to remove
      // the const qualifier. This may lead to another undefined behavior (The
      // first one is accessing `AllocatedUser` without locking. It's better to
      // pass `RegionInfoData` as `void *` then we can lock the mutex properly.
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

  static const uptr MapSizeIncrement = Config::PrimaryMapSizeIncrement;
  // Fill at most this number of batches from the newly map'd memory.
  static const u32 MaxNumBatches = SCUDO_ANDROID ? 4U : 8U;

  struct RegionStats {
    uptr PoppedBlocks;
    uptr PushedBlocks;
  };

  struct ReleaseToOsInfo {
    uptr BytesInFreeListAtLastCheckpoint;
    uptr RangesReleased;
    uptr LastReleasedBytes;
    u64 LastReleaseAtNs;
  };

  struct UnpaddedRegionInfo {
    HybridMutex Mutex;
    SinglyLinkedList<BatchGroup> FreeList GUARDED_BY(Mutex);
    // This is initialized before thread creation.
    uptr RegionBeg = 0;
    RegionStats Stats GUARDED_BY(Mutex) = {};
    u32 RandState GUARDED_BY(Mutex) = 0;
    // Bytes mapped for user memory.
    uptr MappedUser GUARDED_BY(Mutex) = 0;
    // Bytes allocated for user memory.
    uptr AllocatedUser GUARDED_BY(Mutex) = 0;
    // The minimum size of pushed blocks to trigger page release.
    uptr TryReleaseThreshold GUARDED_BY(Mutex) = 0;
    MemMapT MemMap = {};
    ReleaseToOsInfo ReleaseInfo GUARDED_BY(Mutex) = {};
    bool Exhausted GUARDED_BY(Mutex) = false;
  };
  struct RegionInfo : UnpaddedRegionInfo {
    char Padding[SCUDO_CACHE_LINE_SIZE -
                 (sizeof(UnpaddedRegionInfo) % SCUDO_CACHE_LINE_SIZE)] = {};
  };
  static_assert(sizeof(RegionInfo) % SCUDO_CACHE_LINE_SIZE == 0, "");

  // TODO: `PrimaryBase` can be obtained from ReservedMemory. This needs to be
  // deprecated.
  uptr PrimaryBase = 0;
  ReservedMemoryT ReservedMemory = {};
  // The minimum size of pushed blocks that we will try to release the pages in
  // that size class.
  uptr SmallerBlockReleasePageDelta = 0;
  atomic_s32 ReleaseToOsIntervalMs = {};
  alignas(SCUDO_CACHE_LINE_SIZE) RegionInfo RegionInfoArray[NumClasses];

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

  // Push the blocks to their batch group. The layout will be like,
  //
  // FreeList - > BG -> BG -> BG
  //              |     |     |
  //              v     v     v
  //              TB    TB    TB
  //              |
  //              v
  //              TB
  //
  // Each BlockGroup(BG) will associate with unique group id and the free blocks
  // are managed by a list of TransferBatch(TB). To reduce the time of inserting
  // blocks, BGs are sorted and the input `Array` are supposed to be sorted so
  // that we can get better performance of maintaining sorted property.
  // Use `SameGroup=true` to indicate that all blocks in the array are from the
  // same group then we will skip checking the group id of each block.
  //
  // The region mutex needs to be held while calling this method.
  void pushBlocksImpl(CacheT *C, uptr ClassId, RegionInfo *Region,
                      CompactPtrT *Array, u32 Size, bool SameGroup = false)
      REQUIRES(Region->Mutex) {
    DCHECK_GT(Size, 0U);

    auto CreateGroup = [&](uptr CompactPtrGroupBase) {
      BatchGroup *BG = nullptr;
      TransferBatch *TB = nullptr;
      if (ClassId == SizeClassMap::BatchClassId) {
        DCHECK_GE(Size, 2U);

        // Free blocks are recorded by TransferBatch in freelist, blocks of
        // BatchClassId are included. In order not to use additional memory to
        // record blocks of BatchClassId, they are self-contained. I.e., A
        // TransferBatch may record the block address of itself. See the figure
        // below:
        //
        // TransferBatch at 0xABCD
        // +----------------------------+
        // | Free blocks' addr          |
        // | +------+------+------+     |
        // | |0xABCD|...   |...   |     |
        // | +------+------+------+     |
        // +----------------------------+
        //
        // The safeness of manipulating TransferBatch is kept by the invariant,
        //
        //   The unit of each pop-block request is a TransferBatch. Return
        //   part of the blocks in a TransferBatch is not allowed.
        //
        // This ensures that TransferBatch won't leak the address itself while
        // it's still holding other valid data.
        //
        // Besides, BatchGroup uses the same size-class as TransferBatch does
        // and its address is recorded in the TransferBatch too. To maintain the
        // safeness, the invariant to keep is,
        //
        //   The address of itself is always recorded in the last TransferBatch
        //   of the freelist (also imply that the freelist should only be
        //   updated with push_front). Once the last TransferBatch is popped,
        //   the BatchGroup becomes invalid.
        //
        // As a result, the blocks used by BatchGroup and TransferBatch are
        // reusable and don't need additional space for them.
        BG = reinterpret_cast<BatchGroup *>(
            decompactPtr(ClassId, Array[Size - 1]));
        BG->Batches.clear();

        TB = reinterpret_cast<TransferBatch *>(
            decompactPtr(ClassId, Array[Size - 2]));
        TB->clear();

        // Append the blocks used by BatchGroup and TransferBatch immediately so
        // that we ensure that they are in the last TransBatch.
        TB->appendFromArray(Array + Size - 2, 2);
        Size -= 2;
      } else {
        BG = C->createGroup();
        BG->Batches.clear();

        TB = C->createBatch(ClassId, nullptr);
        TB->clear();
      }

      BG->CompactPtrGroupBase = CompactPtrGroupBase;
      // TODO(chiahungduan): Avoid the use of push_back() in `Batches`.
      BG->Batches.push_front(TB);
      BG->PushedBlocks = 0;
      BG->BytesInBGAtLastCheckpoint = 0;
      BG->MaxCachedPerBatch =
          TransferBatch::getMaxCached(getSizeByClassId(ClassId));

      return BG;
    };

    auto InsertBlocks = [&](BatchGroup *BG, CompactPtrT *Array, u32 Size) {
      SinglyLinkedList<TransferBatch> &Batches = BG->Batches;
      TransferBatch *CurBatch = Batches.front();
      DCHECK_NE(CurBatch, nullptr);

      for (u32 I = 0; I < Size;) {
        DCHECK_GE(BG->MaxCachedPerBatch, CurBatch->getCount());
        u16 UnusedSlots =
            static_cast<u16>(BG->MaxCachedPerBatch - CurBatch->getCount());
        if (UnusedSlots == 0) {
          CurBatch = C->createBatch(
              ClassId,
              reinterpret_cast<void *>(decompactPtr(ClassId, Array[I])));
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

    BatchGroup *Cur = Region->FreeList.front();

    if (ClassId == SizeClassMap::BatchClassId) {
      if (Cur == nullptr) {
        // Don't need to classify BatchClassId.
        Cur = CreateGroup(/*CompactPtrGroupBase=*/0);
        Region->FreeList.push_front(Cur);
      }
      InsertBlocks(Cur, Array, Size);
      return;
    }

    // In the following, `Cur` always points to the BatchGroup for blocks that
    // will be pushed next. `Prev` is the element right before `Cur`.
    BatchGroup *Prev = nullptr;

    while (Cur != nullptr &&
           compactPtrGroup(Array[0]) > Cur->CompactPtrGroupBase) {
      Prev = Cur;
      Cur = Cur->Next;
    }

    if (Cur == nullptr ||
        compactPtrGroup(Array[0]) != Cur->CompactPtrGroupBase) {
      Cur = CreateGroup(compactPtrGroup(Array[0]));
      if (Prev == nullptr)
        Region->FreeList.push_front(Cur);
      else
        Region->FreeList.insert(Prev, Cur);
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
          Region->FreeList.insert(Prev, Cur);
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
  TransferBatch *popBatchImpl(CacheT *C, uptr ClassId, RegionInfo *Region)
      REQUIRES(Region->Mutex) {
    if (Region->FreeList.empty())
      return nullptr;

    SinglyLinkedList<TransferBatch> &Batches =
        Region->FreeList.front()->Batches;
    DCHECK(!Batches.empty());

    TransferBatch *B = Batches.front();
    Batches.pop_front();
    DCHECK_NE(B, nullptr);
    DCHECK_GT(B->getCount(), 0U);

    if (Batches.empty()) {
      BatchGroup *BG = Region->FreeList.front();
      Region->FreeList.pop_front();

      // We don't keep BatchGroup with zero blocks to avoid empty-checking while
      // allocating. Note that block used by constructing BatchGroup is recorded
      // as free blocks in the last element of BatchGroup::Batches. Which means,
      // once we pop the last TransferBatch, the block is implicitly
      // deallocated.
      if (ClassId != SizeClassMap::BatchClassId)
        C->deallocate(SizeClassMap::BatchClassId, BG);
    }

    return B;
  }

  NOINLINE bool populateFreeList(CacheT *C, uptr ClassId, RegionInfo *Region)
      REQUIRES(Region->Mutex) {
    const uptr Size = getSizeByClassId(ClassId);
    const u16 MaxCount = TransferBatch::getMaxCached(Size);

    const uptr RegionBeg = Region->RegionBeg;
    const uptr MappedUser = Region->MappedUser;
    const uptr TotalUserBytes = Region->AllocatedUser + MaxCount * Size;
    // Map more space for blocks, if necessary.
    if (TotalUserBytes > MappedUser) {
      // Do the mmap for the user memory.
      const uptr MapSize =
          roundUp(TotalUserBytes - MappedUser, MapSizeIncrement);
      const uptr RegionBase = RegionBeg - getRegionBaseByClassId(ClassId);
      if (UNLIKELY(RegionBase + MappedUser + MapSize > RegionSize)) {
        Region->Exhausted = true;
        return false;
      }
      // TODO: Consider allocating MemMap in init().
      if (!Region->MemMap.isAllocated()) {
        // TODO: Ideally, a region should reserve RegionSize because the memory
        // between `RegionBeg` and region base is still belong to a region and
        // it's just not used. In order to make it work on every platform (some
        // of them don't support `remap()` across the unused range), dispatch
        // from `RegionBeg` for now.
        const uptr ReserveSize =
            RegionSize - (RegionBeg - getRegionBaseByClassId(ClassId));
        Region->MemMap = ReservedMemory.dispatch(RegionBeg, ReserveSize);
      }
      DCHECK(Region->MemMap.isAllocated());

      if (UNLIKELY(!Region->MemMap.remap(
              RegionBeg + MappedUser, MapSize, "scudo:primary",
              MAP_ALLOWNOMEM | MAP_RESIZABLE |
                  (useMemoryTagging<Config>(Options.load()) ? MAP_MEMTAG
                                                            : 0)))) {
        return false;
      }
      Region->MappedUser += MapSize;
      C->getStats().add(StatMapped, MapSize);
    }

    const u32 NumberOfBlocks = Min(
        MaxNumBatches * MaxCount,
        static_cast<u32>((Region->MappedUser - Region->AllocatedUser) / Size));
    DCHECK_GT(NumberOfBlocks, 0);

    constexpr u32 ShuffleArraySize =
        MaxNumBatches * TransferBatch::MaxNumCached;
    CompactPtrT ShuffleArray[ShuffleArraySize];
    DCHECK_LE(NumberOfBlocks, ShuffleArraySize);

    const uptr CompactPtrBase = getCompactPtrBaseByClassId(ClassId);
    uptr P = RegionBeg + Region->AllocatedUser;
    for (u32 I = 0; I < NumberOfBlocks; I++, P += Size)
      ShuffleArray[I] = compactPtrInternal(CompactPtrBase, P);

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
      pushBlocksImpl(C, ClassId, Region, ShuffleArray, NumberOfBlocks,
                     /*SameGroup=*/true);
    }

    const uptr AllocatedUser = Size * NumberOfBlocks;
    C->getStats().add(StatFree, AllocatedUser);
    Region->AllocatedUser += AllocatedUser;

    return true;
  }

  void getStats(ScopedString *Str, uptr ClassId, RegionInfo *Region, uptr Rss)
      REQUIRES(Region->Mutex) {
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
                                 ReleaseToOS ReleaseType = ReleaseToOS::Normal)
      REQUIRES(Region->Mutex) {
    const uptr BlockSize = getSizeByClassId(ClassId);
    const uptr PageSize = getPageSizeCached();

    DCHECK_GE(Region->Stats.PoppedBlocks, Region->Stats.PushedBlocks);
    const uptr BytesInFreeList =
        Region->AllocatedUser -
        (Region->Stats.PoppedBlocks - Region->Stats.PushedBlocks) * BlockSize;

    bool MaySkip = false;

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
      MaySkip = true;
    }

    const uptr RegionPushedBytesDelta =
        BytesInFreeList - Region->ReleaseInfo.BytesInFreeListAtLastCheckpoint;
    if (RegionPushedBytesDelta < PageSize)
      MaySkip = true;

    const bool CheckDensity = isSmallBlock(BlockSize);
    // Releasing smaller blocks is expensive, so we want to make sure that a
    // significant amount of bytes are free, and that there has been a good
    // amount of batches pushed to the freelist before attempting to release.
    if (CheckDensity) {
      if (ReleaseType == ReleaseToOS::Normal &&
          RegionPushedBytesDelta < Region->TryReleaseThreshold) {
        MaySkip = true;
      }
    }

    if (MaySkip && ReleaseType != ReleaseToOS::ForceAll)
      return 0;

    if (ReleaseType == ReleaseToOS::Normal) {
      const s32 IntervalMs = atomic_load_relaxed(&ReleaseToOsIntervalMs);
      if (IntervalMs < 0)
        return 0;
      if (Region->ReleaseInfo.LastReleaseAtNs +
              static_cast<u64>(IntervalMs) * 1000000 >
          getMonotonicTimeFast()) {
        return 0; // Memory was returned recently.
      }
    }

    const uptr GroupSize = (1U << GroupSizeLog);
    const uptr AllocatedUserEnd = Region->AllocatedUser + Region->RegionBeg;
    const uptr CompactPtrBase = getCompactPtrBaseByClassId(ClassId);
    auto DecompactPtr = [CompactPtrBase](CompactPtrT CompactPtr) {
      return decompactPtrInternal(CompactPtrBase, CompactPtr);
    };

    // Instead of always preparing PageMap for the entire region, we only do it
    // for the range of releasing groups. To do that, the free-block marking
    // process includes visiting BlockGroups twice.

    // The first visit is to determine the range of BatchGroups we are going to
    // release. And we will extract those BatchGroups out and push into
    // `GroupToRelease`.
    SinglyLinkedList<BatchGroup> GroupToRelease;
    GroupToRelease.clear();

    // This is only used for debugging to ensure the consistency of the number
    // of groups.
    uptr NumberOfBatchGroups = Region->FreeList.size();

    // We are examining each group and will take the minimum distance to the
    // release threshold as the next Region::TryReleaseThreshold(). Note that if
    // the size of free blocks has reached the release threshold, the distance
    // to the next release will be PageSize * SmallerBlockReleasePageDelta. See
    // the comment on `SmallerBlockReleasePageDelta` for more details.
    uptr MinDistToThreshold = GroupSize;

    for (BatchGroup *BG = Region->FreeList.front(), *Prev = nullptr;
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
      const uptr BatchGroupEnd = BatchGroupBase + GroupSize;
      const uptr AllocatedGroupSize = AllocatedUserEnd >= BatchGroupEnd
                                          ? GroupSize
                                          : AllocatedUserEnd - BatchGroupBase;
      if (AllocatedGroupSize == 0) {
        Prev = BG;
        BG = BG->Next;
        continue;
      }

      // TransferBatches are pushed in front of BG.Batches. The first one may
      // not have all caches used.
      const uptr NumBlocks = (BG->Batches.size() - 1) * BG->MaxCachedPerBatch +
                             BG->Batches.front()->getCount();
      const uptr BytesInBG = NumBlocks * BlockSize;

      if (ReleaseType != ReleaseToOS::ForceAll &&
          BytesInBG <= BG->BytesInBGAtLastCheckpoint) {
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
      if (CheckDensity) {
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

      // If `BG` is the first BatchGroup in the list, we only need to advance
      // `BG` and call FreeList::pop_front(). No update is needed for `Prev`.
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
      // `FreeList`.
      //
      //         (BG)   (BG->Next)
      // Prev     Cur      BG
      //   |       |       |
      //   v       v       v
      //  +--+    +--+    +--+
      //  |  | -> |X | -> |  | -> ...
      //  +--+    +--+    +--+
      //
      // After FreeList::extract(),
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
      // GroupToRelease because it's a destructive operation.

      BatchGroup *Cur = BG;
      BG = BG->Next;

      // Ideally, we may want to update this only after successful release.
      // However, for smaller blocks, each block marking is a costly operation.
      // Therefore, we update it earlier.
      // TODO: Consider updating this after page release if `ReleaseRecorder`
      // can tell the releasd bytes in each group.
      Cur->BytesInBGAtLastCheckpoint = BytesInBG;

      if (Prev != nullptr)
        Region->FreeList.extract(Prev, Cur);
      else
        Region->FreeList.pop_front();
      GroupToRelease.push_back(Cur);
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

    if (GroupToRelease.empty())
      return 0;

    const uptr ReleaseBase = decompactGroupBase(
        CompactPtrBase, GroupToRelease.front()->CompactPtrGroupBase);
    const uptr LastGroupEnd =
        Min(decompactGroupBase(CompactPtrBase,
                               GroupToRelease.back()->CompactPtrGroupBase) +
                GroupSize,
            AllocatedUserEnd);
    // The last block may straddle the group boundary. Rounding up to BlockSize
    // to get the exact range.
    const uptr ReleaseEnd =
        roundUpSlow(LastGroupEnd - Region->RegionBeg, BlockSize) +
        Region->RegionBeg;
    const uptr ReleaseRangeSize = ReleaseEnd - ReleaseBase;
    const uptr ReleaseOffset = ReleaseBase - Region->RegionBeg;

    RegionReleaseRecorder<MemMapT> Recorder(&Region->MemMap, Region->RegionBeg,
                                            ReleaseOffset);
    PageReleaseContext Context(BlockSize, /*NumberOfRegions=*/1U,
                               ReleaseRangeSize, ReleaseOffset);

    for (BatchGroup &BG : GroupToRelease) {
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
        for (const auto &It : BG.Batches)
          for (u16 I = 0; I < It.getCount(); ++I)
            DCHECK_EQ(compactPtrGroup(It.get(I)), BG.CompactPtrGroupBase);

        Context.markRangeAsAllCounted(BatchGroupBase, BatchGroupUsedEnd,
                                      Region->RegionBeg, /*RegionIndex=*/0,
                                      Region->AllocatedUser);
      } else {
        DCHECK_LT(NumBlocks, MaxContainedBlocks);
        // Note that we don't always visit blocks in each BatchGroup so that we
        // may miss the chance of releasing certain pages that cross
        // BatchGroups.
        Context.markFreeBlocksInRegion(
            BG.Batches, DecompactPtr, Region->RegionBeg, /*RegionIndex=*/0,
            Region->AllocatedUser, MayContainLastBlockInRegion);
      }
    }

    DCHECK(Context.hasBlockMarked());

    auto SkipRegion = [](UNUSED uptr RegionIndex) { return false; };
    releaseFreeMemoryToOS(Context, Recorder, SkipRegion);

    if (Recorder.getReleasedRangesCount() > 0) {
      Region->ReleaseInfo.BytesInFreeListAtLastCheckpoint = BytesInFreeList;
      Region->ReleaseInfo.RangesReleased += Recorder.getReleasedRangesCount();
      Region->ReleaseInfo.LastReleasedBytes = Recorder.getReleasedBytes();
    }
    Region->ReleaseInfo.LastReleaseAtNs = getMonotonicTimeFast();

    // Merge GroupToRelease back to the Region::FreeList. Note that both
    // `Region->FreeList` and `GroupToRelease` are sorted.
    for (BatchGroup *BG = Region->FreeList.front(), *Prev = nullptr;;) {
      if (BG == nullptr || GroupToRelease.empty()) {
        if (!GroupToRelease.empty())
          Region->FreeList.append_back(&GroupToRelease);
        break;
      }

      DCHECK_NE(BG->CompactPtrGroupBase,
                GroupToRelease.front()->CompactPtrGroupBase);

      if (BG->CompactPtrGroupBase <
          GroupToRelease.front()->CompactPtrGroupBase) {
        Prev = BG;
        BG = BG->Next;
        continue;
      }

      // At here, the `BG` is the first BatchGroup with CompactPtrGroupBase
      // larger than the first element in `GroupToRelease`. We need to insert
      // `GroupToRelease::front()` (which is `Cur` below)  before `BG`.
      //
      //   1. If `Prev` is nullptr, we simply push `Cur` to the front of
      //      FreeList.
      //   2. Otherwise, use `insert()` which inserts an element next to `Prev`.
      //
      // Afterwards, we don't need to advance `BG` because the order between
      // `BG` and the new `GroupToRelease::front()` hasn't been checked.
      BatchGroup *Cur = GroupToRelease.front();
      GroupToRelease.pop_front();
      if (Prev == nullptr)
        Region->FreeList.push_front(Cur);
      else
        Region->FreeList.insert(Prev, Cur);
      DCHECK_EQ(Cur->Next, BG);
      Prev = Cur;
    }

    DCHECK_EQ(Region->FreeList.size(), NumberOfBatchGroups);
    (void)NumberOfBatchGroups;

    if (SCUDO_DEBUG) {
      BatchGroup *Prev = Region->FreeList.front();
      for (BatchGroup *Cur = Prev->Next; Cur != nullptr;
           Prev = Cur, Cur = Cur->Next) {
        CHECK_LT(Prev->CompactPtrGroupBase, Cur->CompactPtrGroupBase);
      }
    }

    return Recorder.getReleasedBytes();
  }
};

} // namespace scudo

#endif // SCUDO_PRIMARY64_H_
