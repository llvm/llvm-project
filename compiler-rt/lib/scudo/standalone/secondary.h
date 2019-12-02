//===-- secondary.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SCUDO_SECONDARY_H_
#define SCUDO_SECONDARY_H_

#include "common.h"
#include "list.h"
#include "mutex.h"
#include "stats.h"
#include "string_utils.h"

namespace scudo {

// This allocator wraps the platform allocation primitives, and as such is on
// the slower side and should preferably be used for larger sized allocations.
// Blocks allocated will be preceded and followed by a guard page, and hold
// their own header that is not checksummed: the guard pages and the Combined
// header should be enough for our purpose.

namespace LargeBlock {

struct Header {
  LargeBlock::Header *Prev;
  LargeBlock::Header *Next;
  uptr BlockEnd;
  uptr MapBase;
  uptr MapSize;
  MapPlatformData Data;
};

constexpr uptr getHeaderSize() {
  return roundUpTo(sizeof(Header), 1U << SCUDO_MIN_ALIGNMENT_LOG);
}

static Header *getHeader(uptr Ptr) {
  return reinterpret_cast<Header *>(Ptr - getHeaderSize());
}

static Header *getHeader(const void *Ptr) {
  return getHeader(reinterpret_cast<uptr>(Ptr));
}

} // namespace LargeBlock

template <uptr MaxFreeListSize = 32U> class MapAllocator {
public:
  // Ensure the freelist is disabled on Fuchsia, since it doesn't support
  // releasing Secondary blocks yet.
  COMPILER_CHECK(!SCUDO_FUCHSIA || MaxFreeListSize == 0U);

  void initLinkerInitialized(GlobalStats *S) {
    Stats.initLinkerInitialized();
    if (LIKELY(S))
      S->link(&Stats);
  }
  void init(GlobalStats *S) {
    memset(this, 0, sizeof(*this));
    initLinkerInitialized(S);
  }

  void *allocate(uptr Size, uptr AlignmentHint = 0, uptr *BlockEnd = nullptr,
                 bool ZeroContents = false);

  void deallocate(void *Ptr);

  static uptr getBlockEnd(void *Ptr) {
    return LargeBlock::getHeader(Ptr)->BlockEnd;
  }

  static uptr getBlockSize(void *Ptr) {
    return getBlockEnd(Ptr) - reinterpret_cast<uptr>(Ptr);
  }

  void getStats(ScopedString *Str) const;

  void disable() { Mutex.lock(); }

  void enable() { Mutex.unlock(); }

  template <typename F> void iterateOverBlocks(F Callback) const {
    for (const auto &H : InUseBlocks)
      Callback(reinterpret_cast<uptr>(&H) + LargeBlock::getHeaderSize());
  }

  static uptr getMaxFreeListSize(void) { return MaxFreeListSize; }

private:
  HybridMutex Mutex;
  DoublyLinkedList<LargeBlock::Header> InUseBlocks;
  // The free list is sorted based on the committed size of blocks.
  DoublyLinkedList<LargeBlock::Header> FreeBlocks;
  uptr AllocatedBytes;
  uptr FreedBytes;
  uptr LargestSize;
  u32 NumberOfAllocs;
  u32 NumberOfFrees;
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
template <uptr MaxFreeListSize>
void *MapAllocator<MaxFreeListSize>::allocate(uptr Size, uptr AlignmentHint,
                                              uptr *BlockEnd,
                                              bool ZeroContents) {
  DCHECK_GE(Size, AlignmentHint);
  const uptr PageSize = getPageSizeCached();
  const uptr RoundedSize =
      roundUpTo(Size + LargeBlock::getHeaderSize(), PageSize);

  if (MaxFreeListSize && AlignmentHint < PageSize) {
    ScopedLock L(Mutex);
    for (auto &H : FreeBlocks) {
      const uptr FreeBlockSize = H.BlockEnd - reinterpret_cast<uptr>(&H);
      if (FreeBlockSize < RoundedSize)
        continue;
      // Candidate free block should only be at most 4 pages larger.
      if (FreeBlockSize > RoundedSize + 4 * PageSize)
        break;
      FreeBlocks.remove(&H);
      InUseBlocks.push_back(&H);
      AllocatedBytes += FreeBlockSize;
      NumberOfAllocs++;
      Stats.add(StatAllocated, FreeBlockSize);
      if (BlockEnd)
        *BlockEnd = H.BlockEnd;
      void *Ptr = reinterpret_cast<void *>(reinterpret_cast<uptr>(&H) +
                                           LargeBlock::getHeaderSize());
      if (ZeroContents)
        memset(Ptr, 0, H.BlockEnd - reinterpret_cast<uptr>(Ptr));
      return Ptr;
    }
  }

  MapPlatformData Data = {};
  const uptr MapSize = RoundedSize + 2 * PageSize;
  uptr MapBase =
      reinterpret_cast<uptr>(map(nullptr, MapSize, "scudo:secondary",
                                 MAP_NOACCESS | MAP_ALLOWNOMEM, &Data));
  if (UNLIKELY(!MapBase))
    return nullptr;
  uptr CommitBase = MapBase + PageSize;
  uptr MapEnd = MapBase + MapSize;

  // In the unlikely event of alignments larger than a page, adjust the amount
  // of memory we want to commit, and trim the extra memory.
  if (UNLIKELY(AlignmentHint >= PageSize)) {
    // For alignments greater than or equal to a page, the user pointer (eg: the
    // pointer that is returned by the C or C++ allocation APIs) ends up on a
    // page boundary , and our headers will live in the preceding page.
    CommitBase = roundUpTo(MapBase + PageSize + 1, AlignmentHint) - PageSize;
    const uptr NewMapBase = CommitBase - PageSize;
    DCHECK_GE(NewMapBase, MapBase);
    // We only trim the extra memory on 32-bit platforms: 64-bit platforms
    // are less constrained memory wise, and that saves us two syscalls.
    if (SCUDO_WORDSIZE == 32U && NewMapBase != MapBase) {
      unmap(reinterpret_cast<void *>(MapBase), NewMapBase - MapBase, 0, &Data);
      MapBase = NewMapBase;
    }
    const uptr NewMapEnd = CommitBase + PageSize +
                           roundUpTo((Size - AlignmentHint), PageSize) +
                           PageSize;
    DCHECK_LE(NewMapEnd, MapEnd);
    if (SCUDO_WORDSIZE == 32U && NewMapEnd != MapEnd) {
      unmap(reinterpret_cast<void *>(NewMapEnd), MapEnd - NewMapEnd, 0, &Data);
      MapEnd = NewMapEnd;
    }
  }

  const uptr CommitSize = MapEnd - PageSize - CommitBase;
  const uptr Ptr =
      reinterpret_cast<uptr>(map(reinterpret_cast<void *>(CommitBase),
                                 CommitSize, "scudo:secondary", 0, &Data));
  LargeBlock::Header *H = reinterpret_cast<LargeBlock::Header *>(Ptr);
  H->MapBase = MapBase;
  H->MapSize = MapEnd - MapBase;
  H->BlockEnd = CommitBase + CommitSize;
  H->Data = Data;
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
  if (BlockEnd)
    *BlockEnd = CommitBase + CommitSize;
  return reinterpret_cast<void *>(Ptr + LargeBlock::getHeaderSize());
}

template <uptr MaxFreeListSize>
void MapAllocator<MaxFreeListSize>::deallocate(void *Ptr) {
  LargeBlock::Header *H = LargeBlock::getHeader(Ptr);
  const uptr Block = reinterpret_cast<uptr>(H);
  {
    ScopedLock L(Mutex);
    InUseBlocks.remove(H);
    const uptr CommitSize = H->BlockEnd - Block;
    FreedBytes += CommitSize;
    NumberOfFrees++;
    Stats.sub(StatAllocated, CommitSize);
    if (MaxFreeListSize && FreeBlocks.size() < MaxFreeListSize) {
      bool Inserted = false;
      for (auto &F : FreeBlocks) {
        const uptr FreeBlockSize = F.BlockEnd - reinterpret_cast<uptr>(&F);
        if (FreeBlockSize >= CommitSize) {
          FreeBlocks.insert(H, &F);
          Inserted = true;
          break;
        }
      }
      if (!Inserted)
        FreeBlocks.push_back(H);
      const uptr RoundedAllocationStart =
          roundUpTo(Block + LargeBlock::getHeaderSize(), getPageSizeCached());
      MapPlatformData Data = H->Data;
      // TODO(kostyak): use release_to_os_interval_ms
      releasePagesToOS(Block, RoundedAllocationStart - Block,
                       H->BlockEnd - RoundedAllocationStart, &Data);
      return;
    }
    Stats.sub(StatMapped, H->MapSize);
  }
  void *Addr = reinterpret_cast<void *>(H->MapBase);
  const uptr Size = H->MapSize;
  MapPlatformData Data = H->Data;
  unmap(Addr, Size, UNMAP_ALL, &Data);
}

template <uptr MaxFreeListSize>
void MapAllocator<MaxFreeListSize>::getStats(ScopedString *Str) const {
  Str->append(
      "Stats: MapAllocator: allocated %zu times (%zuK), freed %zu times "
      "(%zuK), remains %zu (%zuK) max %zuM\n",
      NumberOfAllocs, AllocatedBytes >> 10, NumberOfFrees, FreedBytes >> 10,
      NumberOfAllocs - NumberOfFrees, (AllocatedBytes - FreedBytes) >> 10,
      LargestSize >> 20);
}

} // namespace scudo

#endif // SCUDO_SECONDARY_H_
