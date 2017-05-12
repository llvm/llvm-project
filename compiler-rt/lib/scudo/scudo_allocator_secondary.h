//===-- scudo_allocator_secondary.h -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// Scudo Secondary Allocator.
/// This services allocation that are too large to be serviced by the Primary
/// Allocator. It is directly backed by the memory mapping functions of the
/// operating system.
///
//===----------------------------------------------------------------------===//

#ifndef SCUDO_ALLOCATOR_SECONDARY_H_
#define SCUDO_ALLOCATOR_SECONDARY_H_

#ifndef SCUDO_ALLOCATOR_H_
# error "This file must be included inside scudo_allocator.h."
#endif

class ScudoLargeMmapAllocator {
 public:

  void Init(bool AllocatorMayReturnNull) {
    PageSize = GetPageSizeCached();
    atomic_store_relaxed(&MayReturnNull, AllocatorMayReturnNull);
  }

  void *Allocate(AllocatorStats *Stats, uptr Size, uptr Alignment) {
    uptr UserSize = Size - AlignedChunkHeaderSize;
    // The Scudo frontend prevents us from allocating more than
    // MaxAllowedMallocSize, so integer overflow checks would be superfluous.
    uptr MapSize = Size + SecondaryHeaderSize;
    if (Alignment > MinAlignment)
      MapSize += Alignment;
    MapSize = RoundUpTo(MapSize, PageSize);
    // Account for 2 guard pages, one before and one after the chunk.
    MapSize += 2 * PageSize;

    uptr MapBeg = reinterpret_cast<uptr>(MmapNoAccess(MapSize));
    if (MapBeg == ~static_cast<uptr>(0))
      return ReturnNullOrDieOnOOM();
    // A page-aligned pointer is assumed after that, so check it now.
    CHECK(IsAligned(MapBeg, PageSize));
    uptr MapEnd = MapBeg + MapSize;
    // The beginning of the user area for that allocation comes after the
    // initial guard page, and both headers. This is the pointer that has to
    // abide by alignment requirements.
    uptr UserBeg = MapBeg + PageSize + HeadersSize;
    uptr UserEnd = UserBeg + UserSize;

    // In the rare event of larger alignments, we will attempt to fit the mmap
    // area better and unmap extraneous memory. This will also ensure that the
    // offset and unused bytes field of the header stay small.
    if (Alignment > MinAlignment) {
      if (!IsAligned(UserBeg, Alignment)) {
        UserBeg = RoundUpTo(UserBeg, Alignment);
        CHECK_GE(UserBeg, MapBeg);
        uptr NewMapBeg = RoundDownTo(UserBeg - HeadersSize, PageSize) -
            PageSize;
        CHECK_GE(NewMapBeg, MapBeg);
        if (NewMapBeg != MapBeg) {
          UnmapOrDie(reinterpret_cast<void *>(MapBeg), NewMapBeg - MapBeg);
          MapBeg = NewMapBeg;
        }
        UserEnd = UserBeg + UserSize;
      }
      uptr NewMapEnd = RoundUpTo(UserEnd, PageSize) + PageSize;
      if (NewMapEnd != MapEnd) {
        UnmapOrDie(reinterpret_cast<void *>(NewMapEnd), MapEnd - NewMapEnd);
        MapEnd = NewMapEnd;
      }
      MapSize = MapEnd - MapBeg;
    }

    CHECK_LE(UserEnd, MapEnd - PageSize);
    // Actually mmap the memory, preserving the guard pages on either side.
    CHECK_EQ(MapBeg + PageSize, reinterpret_cast<uptr>(
        MmapFixedOrDie(MapBeg + PageSize, MapSize - 2 * PageSize)));
    uptr Ptr = UserBeg - AlignedChunkHeaderSize;
    SecondaryHeader *Header = getHeader(Ptr);
    Header->MapBeg = MapBeg;
    Header->MapSize = MapSize;
    // The primary adds the whole class size to the stats when allocating a
    // chunk, so we will do something similar here. But we will not account for
    // the guard pages.
    {
      SpinMutexLock l(&StatsMutex);
      Stats->Add(AllocatorStatAllocated, MapSize - 2 * PageSize);
      Stats->Add(AllocatorStatMapped, MapSize - 2 * PageSize);
    }

    return reinterpret_cast<void *>(Ptr);
  }

  void *ReturnNullOrDieOnOOM() {
    if (atomic_load_relaxed(&MayReturnNull))
      return nullptr;
    ReportAllocatorCannotReturnNull(true);
  }

  void Deallocate(AllocatorStats *Stats, void *Ptr) {
    SecondaryHeader *Header = getHeader(Ptr);
    {
      SpinMutexLock l(&StatsMutex);
      Stats->Sub(AllocatorStatAllocated, Header->MapSize - 2 * PageSize);
      Stats->Sub(AllocatorStatMapped, Header->MapSize - 2 * PageSize);
    }
    UnmapOrDie(reinterpret_cast<void *>(Header->MapBeg), Header->MapSize);
  }

  uptr GetActuallyAllocatedSize(void *Ptr) {
    SecondaryHeader *Header = getHeader(Ptr);
    // Deduct PageSize as MapSize includes the trailing guard page.
    uptr MapEnd = Header->MapBeg + Header->MapSize - PageSize;
    return MapEnd - reinterpret_cast<uptr>(Ptr);
  }

 private:
  // A Secondary allocated chunk header contains the base of the mapping and
  // its size, which comprises the guard pages.
  struct SecondaryHeader {
    uptr MapBeg;
    uptr MapSize;
  };
  // Check that sizeof(SecondaryHeader) is a multiple of MinAlignment.
  COMPILER_CHECK((sizeof(SecondaryHeader) & (MinAlignment - 1)) == 0);

  SecondaryHeader *getHeader(uptr Ptr) {
    return reinterpret_cast<SecondaryHeader*>(Ptr - sizeof(SecondaryHeader));
  }
  SecondaryHeader *getHeader(const void *Ptr) {
    return getHeader(reinterpret_cast<uptr>(Ptr));
  }

  const uptr SecondaryHeaderSize = sizeof(SecondaryHeader);
  const uptr HeadersSize = SecondaryHeaderSize + AlignedChunkHeaderSize;
  uptr PageSize;
  SpinMutex StatsMutex;
  atomic_uint8_t MayReturnNull;
};

#endif  // SCUDO_ALLOCATOR_SECONDARY_H_
