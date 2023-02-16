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
  typedef typename Config::SizeClassMap SizeClassMap;
  typedef SizeClassAllocator64<Config> ThisT;
  typedef SizeClassAllocatorLocalCache<ThisT> CacheT;
  typedef typename CacheT::TransferBatch TransferBatch;
  typedef typename CacheT::BatchGroup BatchGroup;

  static uptr getSizeByClassId(uptr ClassId) {
    return (ClassId == SizeClassMap::BatchClassId)
               ? roundUpTo(sizeof(TransferBatch), 1U << CompactPtrScale)
               : SizeClassMap::getSizeByClassId(ClassId);
  }

  static bool canAllocate(uptr Size) { return Size <= SizeClassMap::MaxSize; }

  void init(s32 ReleaseToOsInterval) {
    DCHECK(isAligned(reinterpret_cast<uptr>(this), alignof(ThisT)));
    DCHECK_EQ(PrimaryBase, 0U);
    // Reserve the space required for the Primary.
    PrimaryBase = reinterpret_cast<uptr>(
        map(nullptr, PrimarySize, nullptr, MAP_NOACCESS, &Data));

    u32 Seed;
    const u64 Time = getMonotonicTime();
    if (!getRandom(reinterpret_cast<void *>(&Seed), sizeof(Seed)))
      Seed = static_cast<u32>(Time ^ (PrimaryBase >> 12));
    const uptr PageSize = getPageSizeCached();
    for (uptr I = 0; I < NumClasses; I++) {
      RegionInfo *Region = getRegionInfo(I);
      // The actual start of a region is offset by a random number of pages
      // when PrimaryEnableRandomOffset is set.
      Region->RegionBeg = getRegionBaseByClassId(I) +
                          (Config::PrimaryEnableRandomOffset
                               ? ((getRandomModN(&Seed, 16) + 1) * PageSize)
                               : 0);
      Region->RandState = getRandomU32(&Seed);
      Region->ReleaseInfo.LastReleaseAtNs = Time;
    }
    setOption(Option::ReleaseInterval, static_cast<sptr>(ReleaseToOsInterval));
  }

  void unmapTestOnly() {
    for (uptr I = 0; I < NumClasses; I++) {
      RegionInfo *Region = getRegionInfo(I);
      *Region = {};
    }
    if (PrimaryBase)
      unmap(reinterpret_cast<void *>(PrimaryBase), PrimarySize, UNMAP_ALL,
            &Data);
    PrimaryBase = 0U;
  }

  TransferBatch *popBatch(CacheT *C, uptr ClassId) {
    DCHECK_LT(ClassId, NumClasses);
    RegionInfo *Region = getRegionInfo(ClassId);
    ScopedLock L(Region->Mutex);
    TransferBatch *B = popBatchImpl(C, ClassId);
    if (UNLIKELY(!B)) {
      if (UNLIKELY(!populateFreeList(C, ClassId, Region)))
        return nullptr;
      B = popBatchImpl(C, ClassId);
      // if `populateFreeList` succeeded, we are supposed to get free blocks.
      DCHECK_NE(B, nullptr);
    }
    Region->Stats.PoppedBlocks += B->getCount();
    return B;
  }

  // Push the array of free blocks to the designated batch group.
  void pushBlocks(CacheT *C, uptr ClassId, CompactPtrT *Array, u32 Size) {
    DCHECK_LT(ClassId, NumClasses);
    DCHECK_GT(Size, 0);

    RegionInfo *Region = getRegionInfo(ClassId);
    if (ClassId == SizeClassMap::BatchClassId) {
      ScopedLock L(Region->Mutex);
      // Constructing a batch group in the free list will use two blocks in
      // BatchClassId. If we are pushing BatchClassId blocks, we will use the
      // blocks in the array directly (can't delegate local cache which will
      // cause a recursive allocation). However, The number of free blocks may
      // be less than two. Therefore, populate the free list before inserting
      // the blocks.
      if (Size == 1 && UNLIKELY(!populateFreeList(C, ClassId, Region)))
        return;
      pushBlocksImpl(C, ClassId, Array, Size);
      Region->Stats.PushedBlocks += Size;
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
    pushBlocksImpl(C, ClassId, Array, Size, SameGroup);

    Region->Stats.PushedBlocks += Size;
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
      RegionInfo *Region = getRegionInfo(I);
      ScopedLock L(Region->Mutex);
      TotalReleasedBytes += releaseToOSMaybe(Region, I, /*Force=*/true);
    }
    return TotalReleasedBytes;
  }

  const char *getRegionInfoArrayAddress() const {
    return reinterpret_cast<const char *>(RegionInfoArray);
  }

  static uptr getRegionInfoArraySize() { return sizeof(RegionInfoArray); }

  uptr getCompactPtrBaseByClassId(uptr ClassId) {
    // If we are not compacting pointers, base everything off of 0.
    if (sizeof(CompactPtrT) == sizeof(uptr) && CompactPtrScale == 0)
      return 0;
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

  static const uptr MapSizeIncrement = Config::PrimaryMapSizeIncrement;
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
    SinglyLinkedList<BatchGroup> FreeList;
    uptr RegionBeg = 0;
    RegionStats Stats = {};
    u32 RandState = 0;
    uptr MappedUser = 0;    // Bytes mapped for user memory.
    uptr AllocatedUser = 0; // Bytes allocated for user memory.
    MapPlatformData Data = {};
    ReleaseToOsInfo ReleaseInfo = {};
    bool Exhausted = false;
  };
  struct RegionInfo : UnpaddedRegionInfo {
    char Padding[SCUDO_CACHE_LINE_SIZE -
                 (sizeof(UnpaddedRegionInfo) % SCUDO_CACHE_LINE_SIZE)] = {};
  };
  static_assert(sizeof(RegionInfo) % SCUDO_CACHE_LINE_SIZE == 0, "");

  uptr PrimaryBase = 0;
  MapPlatformData Data = {};
  atomic_s32 ReleaseToOsIntervalMs = {};
  alignas(SCUDO_CACHE_LINE_SIZE) RegionInfo RegionInfoArray[NumClasses];

  RegionInfo *getRegionInfo(uptr ClassId) {
    DCHECK_LT(ClassId, NumClasses);
    return &RegionInfoArray[ClassId];
  }

  uptr getRegionBaseByClassId(uptr ClassId) const {
    return PrimaryBase + (ClassId << Config::PrimaryRegionSizeLog);
  }

  static CompactPtrT compactPtrInternal(uptr Base, uptr Ptr) {
    return static_cast<CompactPtrT>((Ptr - Base) >> CompactPtrScale);
  }

  static uptr decompactPtrInternal(uptr Base, CompactPtrT CompactPtr) {
    return Base + (static_cast<uptr>(CompactPtr) << CompactPtrScale);
  }

  static uptr compactPtrGroup(CompactPtrT CompactPtr) {
    return static_cast<uptr>(CompactPtr) >> (GroupSizeLog - CompactPtrScale);
  }
  static uptr batchGroupBase(uptr Base, uptr GroupId) {
    return (GroupId << GroupSizeLog) + Base;
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
    RegionInfo *Region = getRegionInfo(ClassId);

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

    BatchGroup *Cur = Region->FreeList.front();

    if (ClassId == SizeClassMap::BatchClassId) {
      if (Cur == nullptr) {
        // Don't need to classify BatchClassId.
        Cur = CreateGroup(/*GroupId=*/0);
        Region->FreeList.push_front(Cur);
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
        Region->FreeList.push_front(Cur);
      else
        Region->FreeList.insert(Prev, Cur);
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
  TransferBatch *popBatchImpl(CacheT *C, uptr ClassId) {
    RegionInfo *Region = getRegionInfo(ClassId);
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

  NOINLINE bool populateFreeList(CacheT *C, uptr ClassId, RegionInfo *Region) {
    const uptr Size = getSizeByClassId(ClassId);
    const u16 MaxCount = TransferBatch::getMaxCached(Size);

    const uptr RegionBeg = Region->RegionBeg;
    const uptr MappedUser = Region->MappedUser;
    const uptr TotalUserBytes = Region->AllocatedUser + MaxCount * Size;
    // Map more space for blocks, if necessary.
    if (TotalUserBytes > MappedUser) {
      // Do the mmap for the user memory.
      const uptr MapSize =
          roundUpTo(TotalUserBytes - MappedUser, MapSizeIncrement);
      const uptr RegionBase = RegionBeg - getRegionBaseByClassId(ClassId);
      if (UNLIKELY(RegionBase + MappedUser + MapSize > RegionSize)) {
        if (!Region->Exhausted) {
          Region->Exhausted = true;
          ScopedString Str;
          // FIXME: getStats() needs to go over all the regions and
          // will take the locks of them. Which means we will try to recursively
          // acquire the `Region->Mutex` which is not supported. It will be
          // better to log this somewhere else.
          // getStats(&Str);
          Str.append(
              "Scudo OOM: The process has exhausted %zuM for size class %zu.\n",
              RegionSize >> 20, Size);
          Str.output();
        }
        return false;
      }
      if (MappedUser == 0)
        Region->Data = Data;
      if (UNLIKELY(!map(
              reinterpret_cast<void *>(RegionBeg + MappedUser), MapSize,
              "scudo:primary",
              MAP_ALLOWNOMEM | MAP_RESIZABLE |
                  (useMemoryTagging<Config>(Options.load()) ? MAP_MEMTAG : 0),
              &Region->Data))) {
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
    // No need to shuffle the batches size class.
    if (ClassId != SizeClassMap::BatchClassId)
      shuffle(ShuffleArray, NumberOfBlocks, &Region->RandState);
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
    Region->AllocatedUser += AllocatedUser;

    return true;
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

    bool CheckDensity = BlockSize < PageSize / 16U;
    // Releasing smaller blocks is expensive, so we want to make sure that a
    // significant amount of bytes are free, and that there has been a good
    // amount of batches pushed to the freelist before attempting to release.
    if (CheckDensity) {
      if (!Force && BytesPushed < Region->AllocatedUser / 16U)
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

    const uptr GroupSize = (1U << GroupSizeLog);
    const uptr AllocatedUserEnd = Region->AllocatedUser + Region->RegionBeg;
    ReleaseRecorder Recorder(Region->RegionBeg, &Region->Data);
    PageReleaseContext Context(BlockSize, Region->AllocatedUser,
                               /*NumberOfRegions=*/1U);

    const uptr CompactPtrBase = getCompactPtrBaseByClassId(ClassId);
    auto DecompactPtr = [CompactPtrBase](CompactPtrT CompactPtr) {
      return decompactPtrInternal(CompactPtrBase, CompactPtr);
    };
    for (BatchGroup &BG : Region->FreeList) {
      const uptr PushedBytesDelta =
          BG.PushedBlocks - BG.PushedBlocksAtLastCheckpoint;
      if (PushedBytesDelta * BlockSize < PageSize)
        continue;

      // Group boundary does not necessarily have the same alignment as Region.
      // It may sit across a Region boundary. Which means that we may have the
      // following two cases,
      //
      // 1. Group boundary sits before RegionBeg.
      //
      //                (BatchGroupBeg)
      // batchGroupBase  RegionBeg       BatchGroupEnd
      //        |            |                |
      //        v            v                v
      //        +------------+----------------+
      //         \                           /
      //          ------   GroupSize   ------
      //
      // 2. Group boundary sits after RegionBeg.
      //
      //               (BatchGroupBeg)
      //    RegionBeg  batchGroupBase               BatchGroupEnd
      //        |           |                             |
      //        v           v                             v
      //        +-----------+-----------------------------+
      //                     \                           /
      //                      ------   GroupSize   ------
      //
      // Note that in the first case, the group range before RegionBeg is never
      // used. Therefore, while calculating the used group size, we should
      // exclude that part to get the correct size.
      const uptr BatchGroupBeg =
          Max(batchGroupBase(CompactPtrBase, BG.GroupId), Region->RegionBeg);
      DCHECK_GE(AllocatedUserEnd, BatchGroupBeg);
      const uptr BatchGroupEnd =
          batchGroupBase(CompactPtrBase, BG.GroupId) + GroupSize;
      const uptr AllocatedGroupSize = AllocatedUserEnd >= BatchGroupEnd
                                          ? BatchGroupEnd - BatchGroupBeg
                                          : AllocatedUserEnd - BatchGroupBeg;
      if (AllocatedGroupSize == 0)
        continue;

      // TransferBatches are pushed in front of BG.Batches. The first one may
      // not have all caches used.
      const uptr NumBlocks = (BG.Batches.size() - 1) * BG.MaxCachedPerBatch +
                             BG.Batches.front()->getCount();
      const uptr BytesInBG = NumBlocks * BlockSize;
      // Given the randomness property, we try to release the pages only if the
      // bytes used by free blocks exceed certain proportion of group size. Note
      // that this heuristic only applies when all the spaces in a BatchGroup
      // are allocated.
      if (CheckDensity && (BytesInBG * 100U) / AllocatedGroupSize <
                              (100U - 1U - BlockSize / 16U)) {
        continue;
      }

      BG.PushedBlocksAtLastCheckpoint = BG.PushedBlocks;
      // Note that we don't always visit blocks in each BatchGroup so that we
      // may miss the chance of releasing certain pages that cross BatchGroups.
      Context.markFreeBlocks(BG.Batches, DecompactPtr, Region->RegionBeg);
    }

    if (!Context.hasBlockMarked())
      return 0;

    auto SkipRegion = [](UNUSED uptr RegionIndex) { return false; };
    releaseFreeMemoryToOS(Context, Recorder, SkipRegion);

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
