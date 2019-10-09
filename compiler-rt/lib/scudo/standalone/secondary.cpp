//===-- secondary.cpp -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "secondary.h"

#include "string_utils.h"

namespace scudo {

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
void *MapAllocator::allocate(uptr Size, uptr AlignmentHint, uptr *BlockEnd) {
  DCHECK_GT(Size, AlignmentHint);
  const uptr PageSize = getPageSizeCached();
  const uptr MapSize =
      roundUpTo(Size + LargeBlock::getHeaderSize(), PageSize) + 2 * PageSize;
  MapPlatformData Data = {};
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
    if (LIKELY(Tail)) {
      Tail->Next = H;
      H->Prev = Tail;
    }
    Tail = H;
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

void MapAllocator::deallocate(void *Ptr) {
  LargeBlock::Header *H = LargeBlock::getHeader(Ptr);
  {
    ScopedLock L(Mutex);
    LargeBlock::Header *Prev = H->Prev;
    LargeBlock::Header *Next = H->Next;
    if (Prev) {
      CHECK_EQ(Prev->Next, H);
      Prev->Next = Next;
    }
    if (Next) {
      CHECK_EQ(Next->Prev, H);
      Next->Prev = Prev;
    }
    if (UNLIKELY(Tail == H)) {
      CHECK(!Next);
      Tail = Prev;
    } else {
      CHECK(Next);
    }
    const uptr CommitSize = H->BlockEnd - reinterpret_cast<uptr>(H);
    FreedBytes += CommitSize;
    NumberOfFrees++;
    Stats.sub(StatAllocated, CommitSize);
    Stats.sub(StatMapped, H->MapSize);
  }
  void *Addr = reinterpret_cast<void *>(H->MapBase);
  const uptr Size = H->MapSize;
  MapPlatformData Data;
  Data = H->Data;
  unmap(Addr, Size, UNMAP_ALL, &Data);
}

void MapAllocator::getStats(ScopedString *Str) const {
  Str->append(
      "Stats: MapAllocator: allocated %zu times (%zuK), freed %zu times "
      "(%zuK), remains %zu (%zuK) max %zuM\n",
      NumberOfAllocs, AllocatedBytes >> 10, NumberOfFrees, FreedBytes >> 10,
      NumberOfAllocs - NumberOfFrees, (AllocatedBytes - FreedBytes) >> 10,
      LargestSize >> 20);
}

} // namespace scudo
