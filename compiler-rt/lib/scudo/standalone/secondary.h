//===-- secondary.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SCUDO_SECONDARY_H_
#define SCUDO_SECONDARY_H_

#include "chunk.h"
#include "common.h"
#include "list.h"
#include "memtag.h"
#include "mutex.h"
#include "options.h"
#include "stats.h"
#include "string_utils.h"

namespace scudo {

// This allocator wraps the platform allocation primitives, and as such is on
// the slower side and should preferably be used for larger sized allocations.
// Blocks allocated will be preceded and followed by a guard page, and hold
// their own header that is not checksummed: the guard pages and the Combined
// header should be enough for our purpose.

namespace LargeBlock {

struct alignas(Max<uptr>(archSupportsMemoryTagging()
                             ? archMemoryTagGranuleSize()
                             : 1,
                         1U << SCUDO_MIN_ALIGNMENT_LOG)) Header {
  LargeBlock::Header *Prev;
  LargeBlock::Header *Next;
  uptr CommitBase;
  uptr CommitSize;
  uptr MapBase;
  uptr MapSize;
  [[no_unique_address]] MapPlatformData Data;
};

static_assert(sizeof(Header) % (1U << SCUDO_MIN_ALIGNMENT_LOG) == 0, "");
static_assert(!archSupportsMemoryTagging() ||
                  sizeof(Header) % archMemoryTagGranuleSize() == 0,
              "");

constexpr uptr getHeaderSize() { return sizeof(Header); }

template <typename Config> static uptr addHeaderTag(uptr Ptr) {
  if (allocatorSupportsMemoryTagging<Config>())
    return addFixedTag(Ptr, 1);
  return Ptr;
}

template <typename Config> static Header *getHeader(uptr Ptr) {
  return reinterpret_cast<Header *>(addHeaderTag<Config>(Ptr)) - 1;
}

template <typename Config> static Header *getHeader(const void *Ptr) {
  return getHeader<Config>(reinterpret_cast<uptr>(Ptr));
}

} // namespace LargeBlock

static void unmap(LargeBlock::Header *H) {
  MapPlatformData Data = H->Data;
  unmap(reinterpret_cast<void *>(H->MapBase), H->MapSize, UNMAP_ALL, &Data);
}

class MapAllocatorNoCache {
public:
  void init(UNUSED s32 ReleaseToOsInterval) {}
  bool retrieve(UNUSED Options Options, UNUSED uptr Size, UNUSED uptr Alignment,
                UNUSED LargeBlock::Header **H, UNUSED bool *Zeroed) {
    return false;
  }
  void store(UNUSED Options Options, LargeBlock::Header *H) { unmap(H); }
  bool canCache(UNUSED uptr Size) { return false; }
  void disable() {}
  void enable() {}
  void releaseToOS() {}
  void disableMemoryTagging() {}
  void unmapTestOnly() {}
  bool setOption(Option O, UNUSED sptr Value) {
    if (O == Option::ReleaseInterval || O == Option::MaxCacheEntriesCount ||
        O == Option::MaxCacheEntrySize)
      return false;
    // Not supported by the Secondary Cache, but not an error either.
    return true;
  }
};

static const uptr MaxUnusedCachePages = 4U;

template <typename Config>
void mapSecondary(Options Options, uptr CommitBase, uptr CommitSize,
                  uptr AllocPos, uptr Flags, MapPlatformData *Data) {
  const uptr MaxUnusedCacheBytes = MaxUnusedCachePages * getPageSizeCached();
  if (useMemoryTagging<Config>(Options) && CommitSize > MaxUnusedCacheBytes) {
    const uptr UntaggedPos = Max(AllocPos, CommitBase + MaxUnusedCacheBytes);
    map(reinterpret_cast<void *>(CommitBase), UntaggedPos - CommitBase,
        "scudo:secondary", MAP_RESIZABLE | MAP_MEMTAG | Flags, Data);
    map(reinterpret_cast<void *>(UntaggedPos),
        CommitBase + CommitSize - UntaggedPos, "scudo:secondary",
        MAP_RESIZABLE | Flags, Data);
  } else {
    map(reinterpret_cast<void *>(CommitBase), CommitSize, "scudo:secondary",
        MAP_RESIZABLE | (useMemoryTagging<Config>(Options) ? MAP_MEMTAG : 0) |
            Flags,
        Data);
  }
}

template <typename Config> class MapAllocatorCache {
public:
  // Ensure the default maximum specified fits the array.
  static_assert(Config::SecondaryCacheDefaultMaxEntriesCount <=
                    Config::SecondaryCacheEntriesArraySize,
                "");

  void init(s32 ReleaseToOsInterval) {
    DCHECK_EQ(EntriesCount, 0U);
    setOption(Option::MaxCacheEntriesCount,
              static_cast<sptr>(Config::SecondaryCacheDefaultMaxEntriesCount));
    setOption(Option::MaxCacheEntrySize,
              static_cast<sptr>(Config::SecondaryCacheDefaultMaxEntrySize));
    setOption(Option::ReleaseInterval, static_cast<sptr>(ReleaseToOsInterval));
  }

  void store(Options Options, LargeBlock::Header *H) {
    if (!canCache(H->CommitSize))
      return unmap(H);

    bool EntryCached = false;
    bool EmptyCache = false;
    const s32 Interval = atomic_load_relaxed(&ReleaseToOsIntervalMs);
    const u64 Time = getMonotonicTime();
    const u32 MaxCount = atomic_load_relaxed(&MaxEntriesCount);
    CachedBlock Entry;
    Entry.CommitBase = H->CommitBase;
    Entry.CommitSize = H->CommitSize;
    Entry.MapBase = H->MapBase;
    Entry.MapSize = H->MapSize;
    Entry.BlockBegin = reinterpret_cast<uptr>(H + 1);
    Entry.Data = H->Data;
    Entry.Time = Time;
    if (useMemoryTagging<Config>(Options)) {
      if (Interval == 0 && !SCUDO_FUCHSIA) {
        // Release the memory and make it inaccessible at the same time by
        // creating a new MAP_NOACCESS mapping on top of the existing mapping.
        // Fuchsia does not support replacing mappings by creating a new mapping
        // on top so we just do the two syscalls there.
        Entry.Time = 0;
        mapSecondary<Config>(Options, Entry.CommitBase, Entry.CommitSize,
                             Entry.CommitBase, MAP_NOACCESS, &Entry.Data);
      } else {
        setMemoryPermission(Entry.CommitBase, Entry.CommitSize, MAP_NOACCESS,
                            &Entry.Data);
      }
    } else if (Interval == 0) {
      releasePagesToOS(Entry.CommitBase, 0, Entry.CommitSize, &Entry.Data);
      Entry.Time = 0;
    }
    do {
      ScopedLock L(Mutex);
      if (useMemoryTagging<Config>(Options) && QuarantinePos == -1U) {
        // If we get here then memory tagging was disabled in between when we
        // read Options and when we locked Mutex. We can't insert our entry into
        // the quarantine or the cache because the permissions would be wrong so
        // just unmap it.
        break;
      }
      if (Config::SecondaryCacheQuarantineSize &&
          useMemoryTagging<Config>(Options)) {
        QuarantinePos =
            (QuarantinePos + 1) % Max(Config::SecondaryCacheQuarantineSize, 1u);
        if (!Quarantine[QuarantinePos].CommitBase) {
          Quarantine[QuarantinePos] = Entry;
          return;
        }
        CachedBlock PrevEntry = Quarantine[QuarantinePos];
        Quarantine[QuarantinePos] = Entry;
        if (OldestTime == 0)
          OldestTime = Entry.Time;
        Entry = PrevEntry;
      }
      if (EntriesCount >= MaxCount) {
        if (IsFullEvents++ == 4U)
          EmptyCache = true;
      } else {
        for (u32 I = 0; I < MaxCount; I++) {
          if (Entries[I].CommitBase)
            continue;
          if (I != 0)
            Entries[I] = Entries[0];
          Entries[0] = Entry;
          EntriesCount++;
          if (OldestTime == 0)
            OldestTime = Entry.Time;
          EntryCached = true;
          break;
        }
      }
    } while (0);
    if (EmptyCache)
      empty();
    else if (Interval >= 0)
      releaseOlderThan(Time - static_cast<u64>(Interval) * 1000000);
    if (!EntryCached)
      unmap(reinterpret_cast<void *>(Entry.MapBase), Entry.MapSize, UNMAP_ALL,
            &Entry.Data);
  }

  bool retrieve(Options Options, uptr Size, uptr Alignment,
                LargeBlock::Header **H, bool *Zeroed) {
    const uptr PageSize = getPageSizeCached();
    const u32 MaxCount = atomic_load_relaxed(&MaxEntriesCount);
    bool Found = false;
    CachedBlock Entry;
    uptr HeaderPos;
    {
      ScopedLock L(Mutex);
      if (EntriesCount == 0)
        return false;
      for (u32 I = 0; I < MaxCount; I++) {
        const uptr CommitBase = Entries[I].CommitBase;
        if (!CommitBase)
          continue;
        const uptr CommitSize = Entries[I].CommitSize;
        const uptr AllocPos =
            roundDownTo(CommitBase + CommitSize - Size, Alignment);
        HeaderPos =
            AllocPos - Chunk::getHeaderSize() - LargeBlock::getHeaderSize();
        if (HeaderPos > CommitBase + CommitSize)
          continue;
        if (HeaderPos < CommitBase ||
            AllocPos > CommitBase + PageSize * MaxUnusedCachePages)
          continue;
        Found = true;
        Entry = Entries[I];
        Entries[I].CommitBase = 0;
        break;
      }
    }
    if (Found) {
      *H = reinterpret_cast<LargeBlock::Header *>(
          LargeBlock::addHeaderTag<Config>(HeaderPos));
      *Zeroed = Entry.Time == 0;
      if (useMemoryTagging<Config>(Options))
        setMemoryPermission(Entry.CommitBase, Entry.CommitSize, 0, &Entry.Data);
      uptr NewBlockBegin = reinterpret_cast<uptr>(*H + 1);
      if (useMemoryTagging<Config>(Options)) {
        if (*Zeroed)
          storeTags(LargeBlock::addHeaderTag<Config>(Entry.CommitBase),
                    NewBlockBegin);
        else if (Entry.BlockBegin < NewBlockBegin)
          storeTags(Entry.BlockBegin, NewBlockBegin);
        else
          storeTags(untagPointer(NewBlockBegin),
                    untagPointer(Entry.BlockBegin));
      }
      (*H)->CommitBase = Entry.CommitBase;
      (*H)->CommitSize = Entry.CommitSize;
      (*H)->MapBase = Entry.MapBase;
      (*H)->MapSize = Entry.MapSize;
      (*H)->Data = Entry.Data;
      EntriesCount--;
    }
    return Found;
  }

  bool canCache(uptr Size) {
    return atomic_load_relaxed(&MaxEntriesCount) != 0U &&
           Size <= atomic_load_relaxed(&MaxEntrySize);
  }

  bool setOption(Option O, sptr Value) {
    if (O == Option::ReleaseInterval) {
      const s32 Interval =
          Max(Min(static_cast<s32>(Value),
                  Config::SecondaryCacheMaxReleaseToOsIntervalMs),
              Config::SecondaryCacheMinReleaseToOsIntervalMs);
      atomic_store_relaxed(&ReleaseToOsIntervalMs, Interval);
      return true;
    }
    if (O == Option::MaxCacheEntriesCount) {
      const u32 MaxCount = static_cast<u32>(Value);
      if (MaxCount > Config::SecondaryCacheEntriesArraySize)
        return false;
      atomic_store_relaxed(&MaxEntriesCount, MaxCount);
      return true;
    }
    if (O == Option::MaxCacheEntrySize) {
      atomic_store_relaxed(&MaxEntrySize, static_cast<uptr>(Value));
      return true;
    }
    // Not supported by the Secondary Cache, but not an error either.
    return true;
  }

  void releaseToOS() { releaseOlderThan(UINT64_MAX); }

  void disableMemoryTagging() {
    ScopedLock L(Mutex);
    for (u32 I = 0; I != Config::SecondaryCacheQuarantineSize; ++I) {
      if (Quarantine[I].CommitBase) {
        unmap(reinterpret_cast<void *>(Quarantine[I].MapBase),
              Quarantine[I].MapSize, UNMAP_ALL, &Quarantine[I].Data);
        Quarantine[I].CommitBase = 0;
      }
    }
    const u32 MaxCount = atomic_load_relaxed(&MaxEntriesCount);
    for (u32 I = 0; I < MaxCount; I++)
      if (Entries[I].CommitBase)
        setMemoryPermission(Entries[I].CommitBase, Entries[I].CommitSize, 0,
                            &Entries[I].Data);
    QuarantinePos = -1U;
  }

  void disable() { Mutex.lock(); }

  void enable() { Mutex.unlock(); }

  void unmapTestOnly() { empty(); }

private:
  void empty() {
    struct {
      void *MapBase;
      uptr MapSize;
      MapPlatformData Data;
    } MapInfo[Config::SecondaryCacheEntriesArraySize];
    uptr N = 0;
    {
      ScopedLock L(Mutex);
      for (uptr I = 0; I < Config::SecondaryCacheEntriesArraySize; I++) {
        if (!Entries[I].CommitBase)
          continue;
        MapInfo[N].MapBase = reinterpret_cast<void *>(Entries[I].MapBase);
        MapInfo[N].MapSize = Entries[I].MapSize;
        MapInfo[N].Data = Entries[I].Data;
        Entries[I].CommitBase = 0;
        N++;
      }
      EntriesCount = 0;
      IsFullEvents = 0;
    }
    for (uptr I = 0; I < N; I++)
      unmap(MapInfo[I].MapBase, MapInfo[I].MapSize, UNMAP_ALL,
            &MapInfo[I].Data);
  }

  struct CachedBlock {
    uptr CommitBase;
    uptr CommitSize;
    uptr MapBase;
    uptr MapSize;
    uptr BlockBegin;
    [[no_unique_address]] MapPlatformData Data;
    u64 Time;
  };

  void releaseIfOlderThan(CachedBlock &Entry, u64 Time) {
    if (!Entry.CommitBase || !Entry.Time)
      return;
    if (Entry.Time > Time) {
      if (OldestTime == 0 || Entry.Time < OldestTime)
        OldestTime = Entry.Time;
      return;
    }
    releasePagesToOS(Entry.CommitBase, 0, Entry.CommitSize, &Entry.Data);
    Entry.Time = 0;
  }

  void releaseOlderThan(u64 Time) {
    ScopedLock L(Mutex);
    if (!EntriesCount || OldestTime == 0 || OldestTime > Time)
      return;
    OldestTime = 0;
    for (uptr I = 0; I < Config::SecondaryCacheQuarantineSize; I++)
      releaseIfOlderThan(Quarantine[I], Time);
    for (uptr I = 0; I < Config::SecondaryCacheEntriesArraySize; I++)
      releaseIfOlderThan(Entries[I], Time);
  }

  HybridMutex Mutex;
  u32 EntriesCount = 0;
  u32 QuarantinePos = 0;
  atomic_u32 MaxEntriesCount = {};
  atomic_uptr MaxEntrySize = {};
  u64 OldestTime = 0;
  u32 IsFullEvents = 0;
  atomic_s32 ReleaseToOsIntervalMs = {};

  CachedBlock Entries[Config::SecondaryCacheEntriesArraySize] = {};
  CachedBlock Quarantine[Config::SecondaryCacheQuarantineSize] = {};
};

template <typename Config> class MapAllocator {
public:
  void init(GlobalStats *S, s32 ReleaseToOsInterval = -1) {
    DCHECK_EQ(AllocatedBytes, 0U);
    DCHECK_EQ(FreedBytes, 0U);
    Cache.init(ReleaseToOsInterval);
    Stats.init();
    if (LIKELY(S))
      S->link(&Stats);
  }

  void *allocate(Options Options, uptr Size, uptr AlignmentHint = 0,
                 uptr *BlockEnd = nullptr,
                 FillContentsMode FillContents = NoFill);

  void deallocate(Options Options, void *Ptr);

  static uptr getBlockEnd(void *Ptr) {
    auto *B = LargeBlock::getHeader<Config>(Ptr);
    return B->CommitBase + B->CommitSize;
  }

  static uptr getBlockSize(void *Ptr) {
    return getBlockEnd(Ptr) - reinterpret_cast<uptr>(Ptr);
  }

  void getStats(ScopedString *Str) const;

  void disable() {
    Mutex.lock();
    Cache.disable();
  }

  void enable() {
    Cache.enable();
    Mutex.unlock();
  }

  template <typename F> void iterateOverBlocks(F Callback) const {
    for (const auto &H : InUseBlocks) {
      uptr Ptr = reinterpret_cast<uptr>(&H) + LargeBlock::getHeaderSize();
      if (allocatorSupportsMemoryTagging<Config>())
        Ptr = untagPointer(Ptr);
      Callback(Ptr);
    }
  }

  uptr canCache(uptr Size) { return Cache.canCache(Size); }

  bool setOption(Option O, sptr Value) { return Cache.setOption(O, Value); }

  void releaseToOS() { Cache.releaseToOS(); }

  void disableMemoryTagging() { Cache.disableMemoryTagging(); }

  void unmapTestOnly() { Cache.unmapTestOnly(); }

private:
  typename Config::SecondaryCache Cache;

  HybridMutex Mutex;
  DoublyLinkedList<LargeBlock::Header> InUseBlocks;
  uptr AllocatedBytes = 0;
  uptr FreedBytes = 0;
  uptr LargestSize = 0;
  u32 NumberOfAllocs = 0;
  u32 NumberOfFrees = 0;
  LocalStats Stats;
};

// As with the Primary, the size passed to this function includes any desired
// alignment, so that the frontend can align the user allocation. The hint
// parameter allows us to unmap spurious memory when dealing with larger
// (greater than a page) alignments on 32-bit platforms.
// Due to the sparsity of address space available on those platforms, requesting
// an allocation from the Secondary with a large alignment would end up wasting
// VA space (even though we are not committing the whole thing), hence the need
// to trim off some of the reserved space.
// For allocations requested with an alignment greater than or equal to a page,
// the committed memory will amount to something close to Size - AlignmentHint
// (pending rounding and headers).
template <typename Config>
void *MapAllocator<Config>::allocate(Options Options, uptr Size, uptr Alignment,
                                     uptr *BlockEndPtr,
                                     FillContentsMode FillContents) {
  if (Options.get(OptionBit::AddLargeAllocationSlack))
    Size += 1UL << SCUDO_MIN_ALIGNMENT_LOG;
  Alignment = Max(Alignment, 1UL << SCUDO_MIN_ALIGNMENT_LOG);
  const uptr PageSize = getPageSizeCached();
  uptr RoundedSize =
      roundUpTo(roundUpTo(Size, Alignment) + LargeBlock::getHeaderSize() +
                    Chunk::getHeaderSize(),
                PageSize);
  if (Alignment > PageSize)
    RoundedSize += Alignment - PageSize;

  if (Alignment < PageSize && Cache.canCache(RoundedSize)) {
    LargeBlock::Header *H;
    bool Zeroed;
    if (Cache.retrieve(Options, Size, Alignment, &H, &Zeroed)) {
      const uptr BlockEnd = H->CommitBase + H->CommitSize;
      if (BlockEndPtr)
        *BlockEndPtr = BlockEnd;
      uptr HInt = reinterpret_cast<uptr>(H);
      if (allocatorSupportsMemoryTagging<Config>())
        HInt = untagPointer(HInt);
      const uptr PtrInt = HInt + LargeBlock::getHeaderSize();
      void *Ptr = reinterpret_cast<void *>(PtrInt);
      if (FillContents && !Zeroed)
        memset(Ptr, FillContents == ZeroFill ? 0 : PatternFillByte,
               BlockEnd - PtrInt);
      const uptr BlockSize = BlockEnd - HInt;
      {
        ScopedLock L(Mutex);
        InUseBlocks.push_back(H);
        AllocatedBytes += BlockSize;
        NumberOfAllocs++;
        Stats.add(StatAllocated, BlockSize);
        Stats.add(StatMapped, H->MapSize);
      }
      return Ptr;
    }
  }

  MapPlatformData Data = {};
  const uptr MapSize = RoundedSize + 2 * PageSize;
  uptr MapBase = reinterpret_cast<uptr>(
      map(nullptr, MapSize, nullptr, MAP_NOACCESS | MAP_ALLOWNOMEM, &Data));
  if (UNLIKELY(!MapBase))
    return nullptr;
  uptr CommitBase = MapBase + PageSize;
  uptr MapEnd = MapBase + MapSize;

  // In the unlikely event of alignments larger than a page, adjust the amount
  // of memory we want to commit, and trim the extra memory.
  if (UNLIKELY(Alignment >= PageSize)) {
    // For alignments greater than or equal to a page, the user pointer (eg: the
    // pointer that is returned by the C or C++ allocation APIs) ends up on a
    // page boundary , and our headers will live in the preceding page.
    CommitBase = roundUpTo(MapBase + PageSize + 1, Alignment) - PageSize;
    const uptr NewMapBase = CommitBase - PageSize;
    DCHECK_GE(NewMapBase, MapBase);
    // We only trim the extra memory on 32-bit platforms: 64-bit platforms
    // are less constrained memory wise, and that saves us two syscalls.
    if (SCUDO_WORDSIZE == 32U && NewMapBase != MapBase) {
      unmap(reinterpret_cast<void *>(MapBase), NewMapBase - MapBase, 0, &Data);
      MapBase = NewMapBase;
    }
    const uptr NewMapEnd =
        CommitBase + PageSize + roundUpTo(Size, PageSize) + PageSize;
    DCHECK_LE(NewMapEnd, MapEnd);
    if (SCUDO_WORDSIZE == 32U && NewMapEnd != MapEnd) {
      unmap(reinterpret_cast<void *>(NewMapEnd), MapEnd - NewMapEnd, 0, &Data);
      MapEnd = NewMapEnd;
    }
  }

  const uptr CommitSize = MapEnd - PageSize - CommitBase;
  const uptr AllocPos = roundDownTo(CommitBase + CommitSize - Size, Alignment);
  mapSecondary<Config>(Options, CommitBase, CommitSize, AllocPos, 0, &Data);
  const uptr HeaderPos =
      AllocPos - Chunk::getHeaderSize() - LargeBlock::getHeaderSize();
  LargeBlock::Header *H = reinterpret_cast<LargeBlock::Header *>(
      LargeBlock::addHeaderTag<Config>(HeaderPos));
  if (useMemoryTagging<Config>(Options))
    storeTags(LargeBlock::addHeaderTag<Config>(CommitBase),
              reinterpret_cast<uptr>(H + 1));
  H->MapBase = MapBase;
  H->MapSize = MapEnd - MapBase;
  H->CommitBase = CommitBase;
  H->CommitSize = CommitSize;
  H->Data = Data;
  if (BlockEndPtr)
    *BlockEndPtr = CommitBase + CommitSize;
  {
    ScopedLock L(Mutex);
    InUseBlocks.push_back(H);
    AllocatedBytes += CommitSize;
    if (LargestSize < CommitSize)
      LargestSize = CommitSize;
    NumberOfAllocs++;
    Stats.add(StatAllocated, CommitSize);
    Stats.add(StatMapped, H->MapSize);
  }
  return reinterpret_cast<void *>(HeaderPos + LargeBlock::getHeaderSize());
}

template <typename Config>
void MapAllocator<Config>::deallocate(Options Options, void *Ptr) {
  LargeBlock::Header *H = LargeBlock::getHeader<Config>(Ptr);
  const uptr CommitSize = H->CommitSize;
  {
    ScopedLock L(Mutex);
    InUseBlocks.remove(H);
    FreedBytes += CommitSize;
    NumberOfFrees++;
    Stats.sub(StatAllocated, CommitSize);
    Stats.sub(StatMapped, H->MapSize);
  }
  Cache.store(Options, H);
}

template <typename Config>
void MapAllocator<Config>::getStats(ScopedString *Str) const {
  Str->append(
      "Stats: MapAllocator: allocated %zu times (%zuK), freed %zu times "
      "(%zuK), remains %zu (%zuK) max %zuM\n",
      NumberOfAllocs, AllocatedBytes >> 10, NumberOfFrees, FreedBytes >> 10,
      NumberOfAllocs - NumberOfFrees, (AllocatedBytes - FreedBytes) >> 10,
      LargestSize >> 20);
}

} // namespace scudo

#endif // SCUDO_SECONDARY_H_
