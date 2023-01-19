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

  void init(s32 ReleaseToOsInterval) {
    if (SCUDO_FUCHSIA)
      reportError("SizeClassAllocator32 is not supported on Fuchsia");

    if (SCUDO_TRUSTY)
      reportError("SizeClassAllocator32 is not supported on Trusty");

    DCHECK(isAligned(reinterpret_cast<uptr>(this), alignof(ThisT)));
    PossibleRegions.init();
    u32 Seed;
    const u64 Time = getMonotonicTime();
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
    while (NumberOfStashedRegions > 0)
      unmap(reinterpret_cast<void *>(RegionsStash[--NumberOfStashedRegions]),
            RegionSize);
    uptr MinRegionIndex = NumRegions, MaxRegionIndex = 0;
    for (uptr I = 0; I < NumClasses; I++) {
      SizeClassInfo *Sci = getSizeClassInfo(I);
      if (Sci->MinRegionIndex < MinRegionIndex)
        MinRegionIndex = Sci->MinRegionIndex;
      if (Sci->MaxRegionIndex > MaxRegionIndex)
        MaxRegionIndex = Sci->MaxRegionIndex;
      *Sci = {};
    }
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

  uptr compactPtrGroup(CompactPtrT CompactPtr) {
    return CompactPtr >> GroupSizeLog;
  }

  TransferBatch *popBatch(CacheT *C, uptr ClassId) {
    DCHECK_LT(ClassId, NumClasses);
    SizeClassInfo *Sci = getSizeClassInfo(ClassId);
    ScopedLock L(Sci->Mutex);
    TransferBatch *B = popBatchImpl(C, ClassId);
    if (UNLIKELY(!B)) {
      if (UNLIKELY(!populateFreeList(C, ClassId, Sci)))
        return nullptr;
      B = popBatchImpl(C, ClassId);
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
      pushBlocksImpl(C, ClassId, Array, Size);
      Sci->Stats.PushedBlocks += Size;
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

    ScopedLock L(Sci->Mutex);
    pushBlocksImpl(C, ClassId, Array, Size, SameGroup);

    Sci->Stats.PushedBlocks += Size;
    if (ClassId != SizeClassMap::BatchClassId)
      releaseToOSMaybe(Sci, ClassId);
  }

  void disable() {
    // The BatchClassId must be locked last since other classes can use it.
    for (sptr I = static_cast<sptr>(NumClasses) - 1; I >= 0; I--) {
      if (static_cast<uptr>(I) == SizeClassMap::BatchClassId)
        continue;
      getSizeClassInfo(static_cast<uptr>(I))->Mutex.lock();
    }
    getSizeClassInfo(SizeClassMap::BatchClassId)->Mutex.lock();
    RegionsStashMutex.lock();
    PossibleRegions.disable();
  }

  void enable() {
    PossibleRegions.enable();
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
      if (Sci->MinRegionIndex < MinRegionIndex)
        MinRegionIndex = Sci->MinRegionIndex;
      if (Sci->MaxRegionIndex > MaxRegionIndex)
        MaxRegionIndex = Sci->MaxRegionIndex;
    }
    for (uptr I = MinRegionIndex; I <= MaxRegionIndex; I++)
      if (PossibleRegions[I] &&
          (PossibleRegions[I] - 1U) != SizeClassMap::BatchClassId) {
        const uptr BlockSize = getSizeByClassId(PossibleRegions[I] - 1U);
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
      getStats(Str, I, 0);
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
    uptr PushedBlocksAtLastRelease;
    uptr RangesReleased;
    uptr LastReleasedBytes;
    u64 LastReleaseAtNs;
  };

  struct alignas(SCUDO_CACHE_LINE_SIZE) SizeClassInfo {
    HybridMutex Mutex;
    SinglyLinkedList<BatchGroup> FreeList;
    uptr CurrentRegion;
    uptr CurrentRegionAllocated;
    SizeClassStats Stats;
    u32 RandState;
    uptr AllocatedUser;
    // Lowest & highest region index allocated for this size class, to avoid
    // looping through the whole NumRegions.
    uptr MinRegionIndex;
    uptr MaxRegionIndex;
    ReleaseToOsInfo ReleaseInfo;
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
      Region = roundUpTo(MapBase, RegionSize);
      unmap(reinterpret_cast<void *>(MapBase), Region - MapBase);
      MapSize = RegionSize;
    }
    const uptr End = Region + MapSize;
    if (End != MapEnd)
      unmap(reinterpret_cast<void *>(End), MapEnd - End);
    return Region;
  }

  uptr allocateRegion(SizeClassInfo *Sci, uptr ClassId) {
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
  // Note that this aims to have a better management of dirty pages, i.e., the
  // RSS usage won't grow indefinitely. There's an exception that we may not put
  // a block to its associated group. While populating new blocks, we may have
  // blocks cross different groups. However, most cases will fall into same
  // group and they are supposed to be popped soon. In that case, it's not worth
  // sorting the array with the almost-sorted property. Therefore, we use
  // `SameGroup=true` instead.
  //
  // The region mutex needs to be held while calling this method.
  void pushBlocksImpl(CacheT *C, uptr ClassId, CompactPtrT *Array, u32 Size,
                      bool SameGroup = false) {
    DCHECK_GT(Size, 0U);
    SizeClassInfo *Sci = getSizeClassInfo(ClassId);

    auto CreateGroup = [&](uptr GroupId) {
      BatchGroup *BG = nullptr;
      TransferBatch *TB = nullptr;
      if (ClassId == SizeClassMap::BatchClassId) {
        DCHECK_GE(Size, 2U);
        BG = reinterpret_cast<BatchGroup *>(
            decompactPtr(ClassId, Array[Size - 1]));
        BG->Batches.clear();

        TB = reinterpret_cast<TransferBatch *>(
            decompactPtr(ClassId, Array[Size - 2]));
        TB->clear();
      } else {
        BG = C->createGroup();
        BG->Batches.clear();

        TB = C->createBatch(ClassId, nullptr);
        TB->clear();
      }

      BG->GroupId = GroupId;
      BG->Batches.push_front(TB);
      BG->PushedBlocks = 0;
      BG->PushedBlocksAtLastCheckpoint = 0;
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
        Cur = CreateGroup(/*GroupId=*/0);
        Sci->FreeList.push_front(Cur);
      }
      InsertBlocks(Cur, Array, Size);
      return;
    }

    // In the following, `Cur` always points to the BatchGroup for blocks that
    // will be pushed next. `Prev` is the element right before `Cur`.
    BatchGroup *Prev = nullptr;

    while (Cur != nullptr && compactPtrGroup(Array[0]) > Cur->GroupId) {
      Prev = Cur;
      Cur = Cur->Next;
    }

    if (Cur == nullptr || compactPtrGroup(Array[0]) != Cur->GroupId) {
      Cur = CreateGroup(compactPtrGroup(Array[0]));
      if (Prev == nullptr)
        Sci->FreeList.push_front(Cur);
      else
        Sci->FreeList.insert(Prev, Cur);
    }

    // All the blocks are from the same group, just push without checking group
    // id.
    if (SameGroup) {
      InsertBlocks(Cur, Array, Size);
      return;
    }

    // The blocks are sorted by group id. Determine the segment of group and
    // push them to their group together.
    u32 Count = 1;
    for (u32 I = 1; I < Size; ++I) {
      if (compactPtrGroup(Array[I - 1]) != compactPtrGroup(Array[I])) {
        DCHECK_EQ(compactPtrGroup(Array[I - 1]), Cur->GroupId);
        InsertBlocks(Cur, Array + I - Count, Count);

        while (Cur != nullptr && compactPtrGroup(Array[I]) > Cur->GroupId) {
          Prev = Cur;
          Cur = Cur->Next;
        }

        if (Cur == nullptr || compactPtrGroup(Array[I]) != Cur->GroupId) {
          Cur = CreateGroup(compactPtrGroup(Array[I]));
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
  TransferBatch *popBatchImpl(CacheT *C, uptr ClassId) {
    SizeClassInfo *Sci = getSizeClassInfo(ClassId);
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

  NOINLINE bool populateFreeList(CacheT *C, uptr ClassId, SizeClassInfo *Sci) {
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
    // No need to shuffle the batches size class.
    if (ClassId != SizeClassMap::BatchClassId)
      shuffle(ShuffleArray, NumberOfBlocks, &Sci->RandState);
    for (u32 I = 0; I < NumberOfBlocks;) {
      // `MaxCount` is u16 so the result will also fit in u16.
      const u16 N = static_cast<u16>(Min<u32>(MaxCount, NumberOfBlocks - I));
      // Note that the N blocks here may have different group ids. Given that
      // it only happens when it crosses the group size boundary. Instead of
      // sorting them, treat them as same group here to avoid sorting the
      // almost-sorted blocks.
      pushBlocksImpl(C, ClassId, &ShuffleArray[I], N, /*SameGroup=*/true);
      I += N;
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

  void getStats(ScopedString *Str, uptr ClassId, uptr Rss) {
    SizeClassInfo *Sci = getSizeClassInfo(ClassId);
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
                                 bool Force = false) {
    const uptr BlockSize = getSizeByClassId(ClassId);
    const uptr PageSize = getPageSizeCached();

    DCHECK_GE(Sci->Stats.PoppedBlocks, Sci->Stats.PushedBlocks);
    const uptr BytesInFreeList =
        Sci->AllocatedUser -
        (Sci->Stats.PoppedBlocks - Sci->Stats.PushedBlocks) * BlockSize;
    if (BytesInFreeList < PageSize)
      return 0; // No chance to release anything.
    const uptr BytesPushed =
        (Sci->Stats.PushedBlocks - Sci->ReleaseInfo.PushedBlocksAtLastRelease) *
        BlockSize;
    if (BytesPushed < PageSize)
      return 0; // Nothing new to release.

    const bool CheckDensity = BlockSize < PageSize / 16U;
    // Releasing smaller blocks is expensive, so we want to make sure that a
    // significant amount of bytes are free, and that there has been a good
    // amount of batches pushed to the freelist before attempting to release.
    if (CheckDensity) {
      if (!Force && BytesPushed < Sci->AllocatedUser / 16U)
        return 0;
    }

    if (!Force) {
      const s32 IntervalMs = atomic_load_relaxed(&ReleaseToOsIntervalMs);
      if (IntervalMs < 0)
        return 0;
      if (Sci->ReleaseInfo.LastReleaseAtNs +
              static_cast<u64>(IntervalMs) * 1000000 >
          getMonotonicTime()) {
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
    const uptr CurRegionGroupId =
        compactPtrGroup(compactPtr(ClassId, Sci->CurrentRegion));

    ReleaseRecorder Recorder(Base);
    PageReleaseContext Context(BlockSize, RegionSize, NumberOfRegions);

    auto DecompactPtr = [](CompactPtrT CompactPtr) {
      return reinterpret_cast<uptr>(CompactPtr);
    };
    for (BatchGroup &BG : Sci->FreeList) {
      const uptr PushedBytesDelta =
          BG.PushedBlocks - BG.PushedBlocksAtLastCheckpoint;
      if (PushedBytesDelta * BlockSize < PageSize)
        continue;

      uptr AllocatedGroupSize = BG.GroupId == CurRegionGroupId
                                    ? Sci->CurrentRegionAllocated
                                    : GroupSize;
      if (AllocatedGroupSize == 0)
        continue;

      // TransferBatches are pushed in front of BG.Batches. The first one may
      // not have all caches used.
      const uptr NumBlocks = (BG.Batches.size() - 1) * BG.MaxCachedPerBatch +
                             BG.Batches.front()->getCount();
      const uptr BytesInBG = NumBlocks * BlockSize;
      // Given the randomness property, we try to release the pages only if the
      // bytes used by free blocks exceed certain proportion of allocated
      // spaces.
      if (CheckDensity && (BytesInBG * 100U) / AllocatedGroupSize <
                              (100U - 1U - BlockSize / 16U)) {
        continue;
      }

      BG.PushedBlocksAtLastCheckpoint = BG.PushedBlocks;
      // Note that we don't always visit blocks in each BatchGroup so that we
      // may miss the chance of releasing certain pages that cross BatchGroups.
      Context.markFreeBlocks(BG.Batches, DecompactPtr, Base);
    }

    if (!Context.hasBlockMarked())
      return 0;

    auto SkipRegion = [this, First, ClassId](uptr RegionIndex) {
      return (PossibleRegions[First + RegionIndex] - 1U) != ClassId;
    };
    releaseFreeMemoryToOS(Context, Recorder, SkipRegion);

    if (Recorder.getReleasedRangesCount() > 0) {
      Sci->ReleaseInfo.PushedBlocksAtLastRelease = Sci->Stats.PushedBlocks;
      Sci->ReleaseInfo.RangesReleased += Recorder.getReleasedRangesCount();
      Sci->ReleaseInfo.LastReleasedBytes = Recorder.getReleasedBytes();
      TotalReleasedBytes += Sci->ReleaseInfo.LastReleasedBytes;
    }
    Sci->ReleaseInfo.LastReleaseAtNs = getMonotonicTime();

    return TotalReleasedBytes;
  }

  SizeClassInfo SizeClassInfoArray[NumClasses] = {};

  // Track the regions in use, 0 is unused, otherwise store ClassId + 1.
  ByteMap PossibleRegions = {};
  atomic_s32 ReleaseToOsIntervalMs = {};
  // Unless several threads request regions simultaneously from different size
  // classes, the stash rarely contains more than 1 entry.
  static constexpr uptr MaxStashedRegions = 4;
  HybridMutex RegionsStashMutex;
  uptr NumberOfStashedRegions = 0;
  uptr RegionsStash[MaxStashedRegions] = {};
};

} // namespace scudo

#endif // SCUDO_PRIMARY32_H_
