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
#include "options.h"
#include "release.h"
#include "report.h"
#include "stats.h"
#include "string_utils.h"
#include "thread_annotations.h"

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

template <typename Config> class SizeClassAllocator32 {
public:
  typedef typename Config::PrimaryCompactPtrT CompactPtrT;
  typedef typename Config::SizeClassMap SizeClassMap;
  static const uptr GroupSizeLog = Config::PrimaryGroupSizeLog;
  // The bytemap can only track UINT8_MAX - 1 classes.
  static_assert(SizeClassMap::LargestClassId <= (UINT8_MAX - 1), "");
  // Regions should be large enough to hold the largest Block.
  static_assert((1UL << Config::PrimaryRegionSizeLog) >= SizeClassMap::MaxSize,
                "");
  typedef SizeClassAllocator32<Config> ThisT;
  typedef SizeClassAllocatorLocalCache<ThisT> CacheT;
  typedef typename CacheT::TransferBatch TransferBatch;
  typedef typename CacheT::BatchGroup BatchGroup;

  static uptr getSizeByClassId(uptr ClassId) {
    return (ClassId == SizeClassMap::BatchClassId)
               ? sizeof(TransferBatch)
               : SizeClassMap::getSizeByClassId(ClassId);
  }

  static bool canAllocate(uptr Size) { return Size <= SizeClassMap::MaxSize; }

  void init(s32 ReleaseToOsInterval) NO_THREAD_SAFETY_ANALYSIS {
    if (SCUDO_FUCHSIA)
      reportError("SizeClassAllocator32 is not supported on Fuchsia");

    if (SCUDO_TRUSTY)
      reportError("SizeClassAllocator32 is not supported on Trusty");

    DCHECK(isAligned(reinterpret_cast<uptr>(this), alignof(ThisT)));
    PossibleRegions.init();
    u32 Seed;
    const u64 Time = getMonotonicTimeFast();
    if (!getRandom(reinterpret_cast<void *>(&Seed), sizeof(Seed)))
      Seed = static_cast<u32>(
          Time ^ (reinterpret_cast<uptr>(SizeClassInfoArray) >> 6));
    for (uptr I = 0; I < NumClasses; I++) {
      SizeClassInfo *Sci = getSizeClassInfo(I);
      Sci->RandState = getRandomU32(&Seed);
      // Sci->MaxRegionIndex is already initialized to 0.
      Sci->MinRegionIndex = NumRegions;
      Sci->ReleaseInfo.LastReleaseAtNs = Time;
    }
    setOption(Option::ReleaseInterval, static_cast<sptr>(ReleaseToOsInterval));
  }

  void unmapTestOnly() {
    {
      ScopedLock L(RegionsStashMutex);
      while (NumberOfStashedRegions > 0) {
        unmap(reinterpret_cast<void *>(RegionsStash[--NumberOfStashedRegions]),
              RegionSize);
      }
    }

    uptr MinRegionIndex = NumRegions, MaxRegionIndex = 0;
    for (uptr I = 0; I < NumClasses; I++) {
      SizeClassInfo *Sci = getSizeClassInfo(I);
      ScopedLock L(Sci->Mutex);
      if (Sci->MinRegionIndex < MinRegionIndex)
        MinRegionIndex = Sci->MinRegionIndex;
      if (Sci->MaxRegionIndex > MaxRegionIndex)
        MaxRegionIndex = Sci->MaxRegionIndex;
      *Sci = {};
    }

    ScopedLock L(ByteMapMutex);
    for (uptr I = MinRegionIndex; I < MaxRegionIndex; I++)
      if (PossibleRegions[I])
        unmap(reinterpret_cast<void *>(I * RegionSize), RegionSize);
    PossibleRegions.unmapTestOnly();
  }

  CompactPtrT compactPtr(UNUSED uptr ClassId, uptr Ptr) const {
    return static_cast<CompactPtrT>(Ptr);
  }

  void *decompactPtr(UNUSED uptr ClassId, CompactPtrT CompactPtr) const {
    return reinterpret_cast<void *>(static_cast<uptr>(CompactPtr));
  }

  uptr compactPtrGroupBase(CompactPtrT CompactPtr) {
    const uptr Mask = (static_cast<uptr>(1) << GroupSizeLog) - 1;
    return CompactPtr & ~Mask;
  }

  uptr decompactGroupBase(uptr CompactPtrGroupBase) {
    return CompactPtrGroupBase;
  }

  TransferBatch *popBatch(CacheT *C, uptr ClassId) {
    DCHECK_LT(ClassId, NumClasses);
    SizeClassInfo *Sci = getSizeClassInfo(ClassId);
    ScopedLock L(Sci->Mutex);
    TransferBatch *B = popBatchImpl(C, ClassId, Sci);
    if (UNLIKELY(!B)) {
      if (UNLIKELY(!populateFreeList(C, ClassId, Sci)))
        return nullptr;
      B = popBatchImpl(C, ClassId, Sci);
      // if `populateFreeList` succeeded, we are supposed to get free blocks.
      DCHECK_NE(B, nullptr);
    }
    Sci->Stats.PoppedBlocks += B->getCount();
    return B;
  }

  // Push the array of free blocks to the designated batch group.
  void pushBlocks(CacheT *C, uptr ClassId, CompactPtrT *Array, u32 Size) {
    DCHECK_LT(ClassId, NumClasses);
    DCHECK_GT(Size, 0);

    SizeClassInfo *Sci = getSizeClassInfo(ClassId);
    if (ClassId == SizeClassMap::BatchClassId) {
      ScopedLock L(Sci->Mutex);
      // Constructing a batch group in the free list will use two blocks in
      // BatchClassId. If we are pushing BatchClassId blocks, we will use the
      // blocks in the array directly (can't delegate local cache which will
      // cause a recursive allocation). However, The number of free blocks may
      // be less than two. Therefore, populate the free list before inserting
      // the blocks.
      if (Size == 1 && !populateFreeList(C, ClassId, Sci))
        return;
      pushBlocksImpl(C, ClassId, Sci, Array, Size);
      Sci->Stats.PushedBlocks += Size;
      return;
    }

    // TODO(chiahungduan): Consider not doing grouping if the group size is not
    // greater than the block size with a certain scale.

    // Sort the blocks so that blocks belonging to the same group can be pushed
    // together.
    bool SameGroup = true;
    for (u32 I = 1; I < Size; ++I) {
      if (compactPtrGroupBase(Array[I - 1]) != compactPtrGroupBase(Array[I]))
        SameGroup = false;
      CompactPtrT Cur = Array[I];
      u32 J = I;
      while (J > 0 &&
             compactPtrGroupBase(Cur) < compactPtrGroupBase(Array[J - 1])) {
        Array[J] = Array[J - 1];
        --J;
      }
      Array[J] = Cur;
    }

    ScopedLock L(Sci->Mutex);
    pushBlocksImpl(C, ClassId, Sci, Array, Size, SameGroup);

    Sci->Stats.PushedBlocks += Size;
    if (ClassId != SizeClassMap::BatchClassId)
      releaseToOSMaybe(Sci, ClassId);
  }

  void disable() NO_THREAD_SAFETY_ANALYSIS {
    // The BatchClassId must be locked last since other classes can use it.
    for (sptr I = static_cast<sptr>(NumClasses) - 1; I >= 0; I--) {
      if (static_cast<uptr>(I) == SizeClassMap::BatchClassId)
        continue;
      getSizeClassInfo(static_cast<uptr>(I))->Mutex.lock();
    }
    getSizeClassInfo(SizeClassMap::BatchClassId)->Mutex.lock();
    RegionsStashMutex.lock();
    ByteMapMutex.lock();
  }

  void enable() NO_THREAD_SAFETY_ANALYSIS {
    ByteMapMutex.unlock();
    RegionsStashMutex.unlock();
    getSizeClassInfo(SizeClassMap::BatchClassId)->Mutex.unlock();
    for (uptr I = 0; I < NumClasses; I++) {
      if (I == SizeClassMap::BatchClassId)
        continue;
      getSizeClassInfo(I)->Mutex.unlock();
    }
  }

  template <typename F> void iterateOverBlocks(F Callback) {
    uptr MinRegionIndex = NumRegions, MaxRegionIndex = 0;
    for (uptr I = 0; I < NumClasses; I++) {
      SizeClassInfo *Sci = getSizeClassInfo(I);
      // TODO: The call of `iterateOverBlocks` requires disabling
      // SizeClassAllocator32. We may consider locking each region on demand
      // only.
      Sci->Mutex.assertHeld();
      if (Sci->MinRegionIndex < MinRegionIndex)
        MinRegionIndex = Sci->MinRegionIndex;
      if (Sci->MaxRegionIndex > MaxRegionIndex)
        MaxRegionIndex = Sci->MaxRegionIndex;
    }

    // SizeClassAllocator32 is disabled, i.e., ByteMapMutex is held.
    ByteMapMutex.assertHeld();

    for (uptr I = MinRegionIndex; I <= MaxRegionIndex; I++) {
      if (PossibleRegions[I] &&
          (PossibleRegions[I] - 1U) != SizeClassMap::BatchClassId) {
        const uptr BlockSize = getSizeByClassId(PossibleRegions[I] - 1U);
        const uptr From = I * RegionSize;
        const uptr To = From + (RegionSize / BlockSize) * BlockSize;
        for (uptr Block = From; Block < To; Block += BlockSize)
          Callback(Block);
      }
    }
  }

  void getStats(ScopedString *Str) {
    // TODO(kostyak): get the RSS per region.
    uptr TotalMapped = 0;
    uptr PoppedBlocks = 0;
    uptr PushedBlocks = 0;
    for (uptr I = 0; I < NumClasses; I++) {
      SizeClassInfo *Sci = getSizeClassInfo(I);
      ScopedLock L(Sci->Mutex);
      TotalMapped += Sci->AllocatedUser;
      PoppedBlocks += Sci->Stats.PoppedBlocks;
      PushedBlocks += Sci->Stats.PushedBlocks;
    }
    Str->append("Stats: SizeClassAllocator32: %zuM mapped in %zu allocations; "
                "remains %zu\n",
                TotalMapped >> 20, PoppedBlocks, PoppedBlocks - PushedBlocks);
    for (uptr I = 0; I < NumClasses; I++) {
      SizeClassInfo *Sci = getSizeClassInfo(I);
      ScopedLock L(Sci->Mutex);
      getStats(Str, I, Sci, 0);
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
      SizeClassInfo *Sci = getSizeClassInfo(I);
      ScopedLock L(Sci->Mutex);
      TotalReleasedBytes += releaseToOSMaybe(Sci, I, ReleaseType);
    }
    return TotalReleasedBytes;
  }

  const char *getRegionInfoArrayAddress() const { return nullptr; }
  static uptr getRegionInfoArraySize() { return 0; }

  static BlockInfo findNearestBlock(UNUSED const char *RegionInfoData,
                                    UNUSED uptr Ptr) {
    return {};
  }

  AtomicOptions Options;

private:
  static const uptr NumClasses = SizeClassMap::NumClasses;
  static const uptr RegionSize = 1UL << Config::PrimaryRegionSizeLog;
  static const uptr NumRegions =
      SCUDO_MMAP_RANGE_SIZE >> Config::PrimaryRegionSizeLog;
  static const u32 MaxNumBatches = SCUDO_ANDROID ? 4U : 8U;
  typedef FlatByteMap<NumRegions> ByteMap;

  struct SizeClassStats {
    uptr PoppedBlocks;
    uptr PushedBlocks;
  };

  struct ReleaseToOsInfo {
    uptr BytesInFreeListAtLastCheckpoint;
    uptr RangesReleased;
    uptr LastReleasedBytes;
    u64 LastReleaseAtNs;
  };

  struct alignas(SCUDO_CACHE_LINE_SIZE) SizeClassInfo {
    HybridMutex Mutex;
    SinglyLinkedList<BatchGroup> FreeList GUARDED_BY(Mutex);
    uptr CurrentRegion GUARDED_BY(Mutex);
    uptr CurrentRegionAllocated GUARDED_BY(Mutex);
    SizeClassStats Stats GUARDED_BY(Mutex);
    u32 RandState;
    uptr AllocatedUser GUARDED_BY(Mutex);
    // Lowest & highest region index allocated for this size class, to avoid
    // looping through the whole NumRegions.
    uptr MinRegionIndex GUARDED_BY(Mutex);
    uptr MaxRegionIndex GUARDED_BY(Mutex);
    ReleaseToOsInfo ReleaseInfo GUARDED_BY(Mutex);
  };
  static_assert(sizeof(SizeClassInfo) % SCUDO_CACHE_LINE_SIZE == 0, "");

  uptr computeRegionId(uptr Mem) {
    const uptr Id = Mem >> Config::PrimaryRegionSizeLog;
    CHECK_LT(Id, NumRegions);
    return Id;
  }

  uptr allocateRegionSlow() {
    uptr MapSize = 2 * RegionSize;
    const uptr MapBase = reinterpret_cast<uptr>(
        map(nullptr, MapSize, "scudo:primary", MAP_ALLOWNOMEM));
    if (!MapBase)
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
      Region = roundUp(MapBase, RegionSize);
      unmap(reinterpret_cast<void *>(MapBase), Region - MapBase);
      MapSize = RegionSize;
    }
    const uptr End = Region + MapSize;
    if (End != MapEnd)
      unmap(reinterpret_cast<void *>(End), MapEnd - End);

    DCHECK_EQ(Region % RegionSize, 0U);
    static_assert(Config::PrimaryRegionSizeLog == GroupSizeLog,
                  "Memory group should be the same size as Region");

    return Region;
  }

  uptr allocateRegion(SizeClassInfo *Sci, uptr ClassId) REQUIRES(Sci->Mutex) {
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
      // Sci->Mutex is held by the caller, updating the Min/Max is safe.
      const uptr RegionIndex = computeRegionId(Region);
      if (RegionIndex < Sci->MinRegionIndex)
        Sci->MinRegionIndex = RegionIndex;
      if (RegionIndex > Sci->MaxRegionIndex)
        Sci->MaxRegionIndex = RegionIndex;
      ScopedLock L(ByteMapMutex);
      PossibleRegions.set(RegionIndex, static_cast<u8>(ClassId + 1U));
    }
    return Region;
  }

  SizeClassInfo *getSizeClassInfo(uptr ClassId) {
    DCHECK_LT(ClassId, NumClasses);
    return &SizeClassInfoArray[ClassId];
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
  void pushBlocksImpl(CacheT *C, uptr ClassId, SizeClassInfo *Sci,
                      CompactPtrT *Array, u32 Size, bool SameGroup = false)
      REQUIRES(Sci->Mutex) {
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

    BatchGroup *Cur = Sci->FreeList.front();

    if (ClassId == SizeClassMap::BatchClassId) {
      if (Cur == nullptr) {
        // Don't need to classify BatchClassId.
        Cur = CreateGroup(/*CompactPtrGroupBase=*/0);
        Sci->FreeList.push_front(Cur);
      }
      InsertBlocks(Cur, Array, Size);
      return;
    }

    // In the following, `Cur` always points to the BatchGroup for blocks that
    // will be pushed next. `Prev` is the element right before `Cur`.
    BatchGroup *Prev = nullptr;

    while (Cur != nullptr &&
           compactPtrGroupBase(Array[0]) > Cur->CompactPtrGroupBase) {
      Prev = Cur;
      Cur = Cur->Next;
    }

    if (Cur == nullptr ||
        compactPtrGroupBase(Array[0]) != Cur->CompactPtrGroupBase) {
      Cur = CreateGroup(compactPtrGroupBase(Array[0]));
      if (Prev == nullptr)
        Sci->FreeList.push_front(Cur);
      else
        Sci->FreeList.insert(Prev, Cur);
    }

    // All the blocks are from the same group, just push without checking group
    // id.
    if (SameGroup) {
      for (u32 I = 0; I < Size; ++I)
        DCHECK_EQ(compactPtrGroupBase(Array[I]), Cur->CompactPtrGroupBase);

      InsertBlocks(Cur, Array, Size);
      return;
    }

    // The blocks are sorted by group id. Determine the segment of group and
    // push them to their group together.
    u32 Count = 1;
    for (u32 I = 1; I < Size; ++I) {
      if (compactPtrGroupBase(Array[I - 1]) != compactPtrGroupBase(Array[I])) {
        DCHECK_EQ(compactPtrGroupBase(Array[I - 1]), Cur->CompactPtrGroupBase);
        InsertBlocks(Cur, Array + I - Count, Count);

        while (Cur != nullptr &&
               compactPtrGroupBase(Array[I]) > Cur->CompactPtrGroupBase) {
          Prev = Cur;
          Cur = Cur->Next;
        }

        if (Cur == nullptr ||
            compactPtrGroupBase(Array[I]) != Cur->CompactPtrGroupBase) {
          Cur = CreateGroup(compactPtrGroupBase(Array[I]));
          DCHECK_NE(Prev, nullptr);
          Sci->FreeList.insert(Prev, Cur);
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
  TransferBatch *popBatchImpl(CacheT *C, uptr ClassId, SizeClassInfo *Sci)
      REQUIRES(Sci->Mutex) {
    if (Sci->FreeList.empty())
      return nullptr;

    SinglyLinkedList<TransferBatch> &Batches = Sci->FreeList.front()->Batches;
    DCHECK(!Batches.empty());

    TransferBatch *B = Batches.front();
    Batches.pop_front();
    DCHECK_NE(B, nullptr);
    DCHECK_GT(B->getCount(), 0U);

    if (Batches.empty()) {
      BatchGroup *BG = Sci->FreeList.front();
      Sci->FreeList.pop_front();

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

  NOINLINE bool populateFreeList(CacheT *C, uptr ClassId, SizeClassInfo *Sci)
      REQUIRES(Sci->Mutex) {
    uptr Region;
    uptr Offset;
    // If the size-class currently has a region associated to it, use it. The
    // newly created blocks will be located after the currently allocated memory
    // for that region (up to RegionSize). Otherwise, create a new region, where
    // the new blocks will be carved from the beginning.
    if (Sci->CurrentRegion) {
      Region = Sci->CurrentRegion;
      DCHECK_GT(Sci->CurrentRegionAllocated, 0U);
      Offset = Sci->CurrentRegionAllocated;
    } else {
      DCHECK_EQ(Sci->CurrentRegionAllocated, 0U);
      Region = allocateRegion(Sci, ClassId);
      if (UNLIKELY(!Region))
        return false;
      C->getStats().add(StatMapped, RegionSize);
      Sci->CurrentRegion = Region;
      Offset = 0;
    }

    const uptr Size = getSizeByClassId(ClassId);
    const u16 MaxCount = TransferBatch::getMaxCached(Size);
    DCHECK_GT(MaxCount, 0U);
    // The maximum number of blocks we should carve in the region is dictated
    // by the maximum number of batches we want to fill, and the amount of
    // memory left in the current region (we use the lowest of the two). This
    // will not be 0 as we ensure that a region can at least hold one block (via
    // static_assert and at the end of this function).
    const u32 NumberOfBlocks =
        Min(MaxNumBatches * MaxCount,
            static_cast<u32>((RegionSize - Offset) / Size));
    DCHECK_GT(NumberOfBlocks, 0U);

    constexpr u32 ShuffleArraySize =
        MaxNumBatches * TransferBatch::MaxNumCached;
    // Fill the transfer batches and put them in the size-class freelist. We
    // need to randomize the blocks for security purposes, so we first fill a
    // local array that we then shuffle before populating the batches.
    CompactPtrT ShuffleArray[ShuffleArraySize];
    DCHECK_LE(NumberOfBlocks, ShuffleArraySize);

    uptr P = Region + Offset;
    for (u32 I = 0; I < NumberOfBlocks; I++, P += Size)
      ShuffleArray[I] = reinterpret_cast<CompactPtrT>(P);

    if (ClassId != SizeClassMap::BatchClassId) {
      u32 N = 1;
      uptr CurGroup = compactPtrGroupBase(ShuffleArray[0]);
      for (u32 I = 1; I < NumberOfBlocks; I++) {
        if (UNLIKELY(compactPtrGroupBase(ShuffleArray[I]) != CurGroup)) {
          shuffle(ShuffleArray + I - N, N, &Sci->RandState);
          pushBlocksImpl(C, ClassId, Sci, ShuffleArray + I - N, N,
                         /*SameGroup=*/true);
          N = 1;
          CurGroup = compactPtrGroupBase(ShuffleArray[I]);
        } else {
          ++N;
        }
      }

      shuffle(ShuffleArray + NumberOfBlocks - N, N, &Sci->RandState);
      pushBlocksImpl(C, ClassId, Sci, &ShuffleArray[NumberOfBlocks - N], N,
                     /*SameGroup=*/true);
    } else {
      pushBlocksImpl(C, ClassId, Sci, ShuffleArray, NumberOfBlocks,
                     /*SameGroup=*/true);
    }

    const uptr AllocatedUser = Size * NumberOfBlocks;
    C->getStats().add(StatFree, AllocatedUser);
    DCHECK_LE(Sci->CurrentRegionAllocated + AllocatedUser, RegionSize);
    // If there is not enough room in the region currently associated to fit
    // more blocks, we deassociate the region by resetting CurrentRegion and
    // CurrentRegionAllocated. Otherwise, update the allocated amount.
    if (RegionSize - (Sci->CurrentRegionAllocated + AllocatedUser) < Size) {
      Sci->CurrentRegion = 0;
      Sci->CurrentRegionAllocated = 0;
    } else {
      Sci->CurrentRegionAllocated += AllocatedUser;
    }
    Sci->AllocatedUser += AllocatedUser;

    return true;
  }

  void getStats(ScopedString *Str, uptr ClassId, SizeClassInfo *Sci, uptr Rss)
      REQUIRES(Sci->Mutex) {
    if (Sci->AllocatedUser == 0)
      return;
    const uptr InUse = Sci->Stats.PoppedBlocks - Sci->Stats.PushedBlocks;
    const uptr AvailableChunks = Sci->AllocatedUser / getSizeByClassId(ClassId);
    Str->append("  %02zu (%6zu): mapped: %6zuK popped: %7zu pushed: %7zu "
                "inuse: %6zu avail: %6zu rss: %6zuK releases: %6zu\n",
                ClassId, getSizeByClassId(ClassId), Sci->AllocatedUser >> 10,
                Sci->Stats.PoppedBlocks, Sci->Stats.PushedBlocks, InUse,
                AvailableChunks, Rss >> 10, Sci->ReleaseInfo.RangesReleased);
  }

  NOINLINE uptr releaseToOSMaybe(SizeClassInfo *Sci, uptr ClassId,
                                 ReleaseToOS ReleaseType = ReleaseToOS::Normal)
      REQUIRES(Sci->Mutex) {
    const uptr BlockSize = getSizeByClassId(ClassId);
    const uptr PageSize = getPageSizeCached();

    DCHECK_GE(Sci->Stats.PoppedBlocks, Sci->Stats.PushedBlocks);
    const uptr BytesInFreeList =
        Sci->AllocatedUser -
        (Sci->Stats.PoppedBlocks - Sci->Stats.PushedBlocks) * BlockSize;

    if (UNLIKELY(BytesInFreeList == 0))
      return 0;

    bool MaySkip = false;

    if (BytesInFreeList <= Sci->ReleaseInfo.BytesInFreeListAtLastCheckpoint) {
      Sci->ReleaseInfo.BytesInFreeListAtLastCheckpoint = BytesInFreeList;
      MaySkip = true;
    }

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
    const uptr PushedBytesDelta =
        BytesInFreeList - Sci->ReleaseInfo.BytesInFreeListAtLastCheckpoint;
    if (PushedBytesDelta < PageSize)
      MaySkip = true;

    const bool CheckDensity =
        BlockSize < PageSize / 16U && ReleaseType != ReleaseToOS::ForceAll;
    // Releasing smaller blocks is expensive, so we want to make sure that a
    // significant amount of bytes are free, and that there has been a good
    // amount of batches pushed to the freelist before attempting to release.
    if (CheckDensity) {
      if (ReleaseType == ReleaseToOS::Normal &&
          PushedBytesDelta < Sci->AllocatedUser / 16U) {
        MaySkip = true;
      }
    }

    if (MaySkip && ReleaseType != ReleaseToOS::ForceAll)
      return 0;

    if (ReleaseType == ReleaseToOS::Normal) {
      const s32 IntervalMs = atomic_load_relaxed(&ReleaseToOsIntervalMs);
      if (IntervalMs < 0)
        return 0;
      if (Sci->ReleaseInfo.LastReleaseAtNs +
              static_cast<u64>(IntervalMs) * 1000000 >
          getMonotonicTimeFast()) {
        return 0; // Memory was returned recently.
      }
    }

    const uptr First = Sci->MinRegionIndex;
    const uptr Last = Sci->MaxRegionIndex;
    DCHECK_NE(Last, 0U);
    DCHECK_LE(First, Last);
    uptr TotalReleasedBytes = 0;
    const uptr Base = First * RegionSize;
    const uptr NumberOfRegions = Last - First + 1U;
    const uptr GroupSize = (1U << GroupSizeLog);
    const uptr CurGroupBase =
        compactPtrGroupBase(compactPtr(ClassId, Sci->CurrentRegion));

    ReleaseRecorder Recorder(Base);
    PageReleaseContext Context(BlockSize, NumberOfRegions,
                               /*ReleaseSize=*/RegionSize);

    auto DecompactPtr = [](CompactPtrT CompactPtr) {
      return reinterpret_cast<uptr>(CompactPtr);
    };
    for (BatchGroup &BG : Sci->FreeList) {
      const uptr GroupBase = decompactGroupBase(BG.CompactPtrGroupBase);
      // The `GroupSize` may not be divided by `BlockSize`, which means there is
      // an unused space at the end of Region. Exclude that space to avoid
      // unused page map entry.
      uptr AllocatedGroupSize = GroupBase == CurGroupBase
                                    ? Sci->CurrentRegionAllocated
                                    : roundDownSlow(GroupSize, BlockSize);
      if (AllocatedGroupSize == 0)
        continue;

      // TransferBatches are pushed in front of BG.Batches. The first one may
      // not have all caches used.
      const uptr NumBlocks = (BG.Batches.size() - 1) * BG.MaxCachedPerBatch +
                             BG.Batches.front()->getCount();
      const uptr BytesInBG = NumBlocks * BlockSize;

      if (ReleaseType != ReleaseToOS::ForceAll &&
          BytesInBG <= BG.BytesInBGAtLastCheckpoint) {
        BG.BytesInBGAtLastCheckpoint = BytesInBG;
        continue;
      }
      const uptr PushedBytesDelta = BytesInBG - BG.BytesInBGAtLastCheckpoint;
      if (PushedBytesDelta < PageSize)
        continue;

      // Given the randomness property, we try to release the pages only if the
      // bytes used by free blocks exceed certain proportion of allocated
      // spaces.
      if (CheckDensity && (BytesInBG * 100U) / AllocatedGroupSize <
                              (100U - 1U - BlockSize / 16U)) {
        continue;
      }

      // TODO: Consider updating this after page release if `ReleaseRecorder`
      // can tell the releasd bytes in each group.
      BG.BytesInBGAtLastCheckpoint = BytesInBG;

      const uptr MaxContainedBlocks = AllocatedGroupSize / BlockSize;
      const uptr RegionIndex = (GroupBase - Base) / RegionSize;

      if (NumBlocks == MaxContainedBlocks) {
        for (const auto &It : BG.Batches)
          for (u16 I = 0; I < It.getCount(); ++I)
            DCHECK_EQ(compactPtrGroupBase(It.get(I)), BG.CompactPtrGroupBase);

        const uptr To = GroupBase + AllocatedGroupSize;
        Context.markRangeAsAllCounted(GroupBase, To, GroupBase, RegionIndex,
                                      AllocatedGroupSize);
      } else {
        DCHECK_LT(NumBlocks, MaxContainedBlocks);

        // Note that we don't always visit blocks in each BatchGroup so that we
        // may miss the chance of releasing certain pages that cross
        // BatchGroups.
        Context.markFreeBlocksInRegion(BG.Batches, DecompactPtr, GroupBase,
                                       RegionIndex, AllocatedGroupSize,
                                       /*MayContainLastBlockInRegion=*/true);
      }
    }

    if (!Context.hasBlockMarked())
      return 0;

    auto SkipRegion = [this, First, ClassId](uptr RegionIndex) {
      ScopedLock L(ByteMapMutex);
      return (PossibleRegions[First + RegionIndex] - 1U) != ClassId;
    };
    releaseFreeMemoryToOS(Context, Recorder, SkipRegion);

    if (Recorder.getReleasedRangesCount() > 0) {
      Sci->ReleaseInfo.BytesInFreeListAtLastCheckpoint = BytesInFreeList;
      Sci->ReleaseInfo.RangesReleased += Recorder.getReleasedRangesCount();
      Sci->ReleaseInfo.LastReleasedBytes = Recorder.getReleasedBytes();
      TotalReleasedBytes += Sci->ReleaseInfo.LastReleasedBytes;
    }
    Sci->ReleaseInfo.LastReleaseAtNs = getMonotonicTimeFast();

    return TotalReleasedBytes;
  }

  SizeClassInfo SizeClassInfoArray[NumClasses] = {};

  HybridMutex ByteMapMutex;
  // Track the regions in use, 0 is unused, otherwise store ClassId + 1.
  ByteMap PossibleRegions GUARDED_BY(ByteMapMutex) = {};
  atomic_s32 ReleaseToOsIntervalMs = {};
  // Unless several threads request regions simultaneously from different size
  // classes, the stash rarely contains more than 1 entry.
  static constexpr uptr MaxStashedRegions = 4;
  HybridMutex RegionsStashMutex;
  uptr NumberOfStashedRegions GUARDED_BY(RegionsStashMutex) = 0;
  uptr RegionsStash[MaxStashedRegions] GUARDED_BY(RegionsStashMutex) = {};
};

} // namespace scudo

#endif // SCUDO_PRIMARY32_H_
