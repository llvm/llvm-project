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
  void Init() {
    PageSizeCached = GetPageSizeCached();
  }

  void *Allocate(AllocatorStats *Stats, uptr Size, uptr Alignment) {
    const uptr UserSize = Size - Chunk::getHeaderSize();
    // The Scudo frontend prevents us from allocating more than
    // MaxAllowedMallocSize, so integer overflow checks would be superfluous.
    uptr MapSize = Size + AlignedReservedAddressRangeSize;
    if (Alignment > MinAlignment)
      MapSize += Alignment;
    const uptr PageSize = PageSizeCached;
    MapSize = RoundUpTo(MapSize, PageSize);
    // Account for 2 guard pages, one before and one after the chunk.
    MapSize += 2 * PageSize;

    ReservedAddressRange AddressRange;
    uptr MapBeg = AddressRange.Init(MapSize);
    if (UNLIKELY(MapBeg == ~static_cast<uptr>(0)))
      return ReturnNullOrDieOnFailure::OnOOM();
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
        DCHECK_GE(UserBeg, MapBeg);
        const uptr NewMapBeg = RoundDownTo(UserBeg - HeadersSize, PageSize) -
            PageSize;
        DCHECK_GE(NewMapBeg, MapBeg);
        if (NewMapBeg != MapBeg) {
          AddressRange.Unmap(MapBeg, NewMapBeg - MapBeg);
          MapBeg = NewMapBeg;
        }
        UserEnd = UserBeg + UserSize;
      }
      const uptr NewMapEnd = RoundUpTo(UserEnd, PageSize) + PageSize;
      if (NewMapEnd != MapEnd) {
        AddressRange.Unmap(NewMapEnd, MapEnd - NewMapEnd);
        MapEnd = NewMapEnd;
      }
      MapSize = MapEnd - MapBeg;
    }

    DCHECK_LE(UserEnd, MapEnd - PageSize);
    // Actually mmap the memory, preserving the guard pages on either side
    CHECK_EQ(MapBeg + PageSize,
             AddressRange.Map(MapBeg + PageSize, MapSize - 2 * PageSize));
    const uptr Ptr = UserBeg - Chunk::getHeaderSize();
    ReservedAddressRange *StoredRange = getReservedAddressRange(Ptr);
    *StoredRange = AddressRange;

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

  void Deallocate(AllocatorStats *Stats, void *Ptr) {
    // Since we're unmapping the entirety of where the ReservedAddressRange
    // actually is, copy onto the stack.
    const uptr PageSize = PageSizeCached;
    ReservedAddressRange AddressRange = *getReservedAddressRange(Ptr);
    {
      SpinMutexLock l(&StatsMutex);
      Stats->Sub(AllocatorStatAllocated, AddressRange.size() - 2 * PageSize);
      Stats->Sub(AllocatorStatMapped, AddressRange.size() - 2 * PageSize);
    }
    AddressRange.Unmap(reinterpret_cast<uptr>(AddressRange.base()),
                       AddressRange.size());
  }

  uptr GetActuallyAllocatedSize(void *Ptr) {
    const ReservedAddressRange *StoredRange = getReservedAddressRange(Ptr);
    // Deduct PageSize as ReservedAddressRange size includes the trailing guard
    // page.
    const uptr MapEnd = reinterpret_cast<uptr>(StoredRange->base()) +
        StoredRange->size() - PageSizeCached;
    return MapEnd - reinterpret_cast<uptr>(Ptr);
  }

 private:
  ReservedAddressRange *getReservedAddressRange(uptr Ptr) {
    return reinterpret_cast<ReservedAddressRange*>(
        Ptr - sizeof(ReservedAddressRange));
  }
  ReservedAddressRange *getReservedAddressRange(const void *Ptr) {
    return getReservedAddressRange(reinterpret_cast<uptr>(Ptr));
  }

  static constexpr uptr AlignedReservedAddressRangeSize =
      RoundUpTo(sizeof(ReservedAddressRange), MinAlignment);
  static constexpr uptr HeadersSize =
      AlignedReservedAddressRangeSize + Chunk::getHeaderSize();

  uptr PageSizeCached;
  SpinMutex StatsMutex;
};

#endif  // SCUDO_ALLOCATOR_SECONDARY_H_
