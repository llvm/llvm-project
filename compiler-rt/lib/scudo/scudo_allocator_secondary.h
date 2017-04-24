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
    atomic_store(&MayReturnNull, AllocatorMayReturnNull, memory_order_relaxed);
  }

  void *Allocate(AllocatorStats *Stats, uptr Size, uptr Alignment) {
    // The Scudo frontend prevents us from allocating more than
    // MaxAllowedMallocSize, so integer overflow checks would be superfluous.
    uptr MapSize = Size + SecondaryHeaderSize;
    MapSize = RoundUpTo(MapSize, PageSize);
    // Account for 2 guard pages, one before and one after the chunk.
    MapSize += 2 * PageSize;
    // The size passed to the Secondary comprises the alignment, if large
    // enough. Subtract it here to get the requested size, including header.
    if (Alignment > MinAlignment)
      Size -= Alignment;

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

    // In the rare event of larger alignments, we will attempt to fit the mmap
    // area better and unmap extraneous memory. This will also ensure that the
    // offset and unused bytes field of the header stay small.
    if (Alignment > MinAlignment) {
      if (UserBeg & (Alignment - 1))
        UserBeg += Alignment - (UserBeg & (Alignment - 1));
      CHECK_GE(UserBeg, MapBeg);
      uptr NewMapBeg = RoundDownTo(UserBeg - HeadersSize, PageSize) - PageSize;
      CHECK_GE(NewMapBeg, MapBeg);
      uptr NewMapEnd = RoundUpTo(UserBeg + (Size - AlignedChunkHeaderSize),
                                 PageSize) + PageSize;
      CHECK_LE(NewMapEnd, MapEnd);
      // Unmap the extra memory if it's large enough, on both sides.
      uptr Diff = NewMapBeg - MapBeg;
      if (Diff > PageSize)
        UnmapOrDie(reinterpret_cast<void *>(MapBeg), Diff);
      Diff = MapEnd - NewMapEnd;
      if (Diff > PageSize)
        UnmapOrDie(reinterpret_cast<void *>(NewMapEnd), Diff);
      MapBeg = NewMapBeg;
      MapEnd = NewMapEnd;
      MapSize = NewMapEnd - NewMapBeg;
    }

    uptr UserEnd = UserBeg + (Size - AlignedChunkHeaderSize);
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

    return reinterpret_cast<void *>(UserBeg);
  }

  void *ReturnNullOrDieOnBadRequest() {
    if (atomic_load(&MayReturnNull, memory_order_acquire))
      return nullptr;
    ReportAllocatorCannotReturnNull(false);
  }

  void *ReturnNullOrDieOnOOM() {
    if (atomic_load(&MayReturnNull, memory_order_acquire))
      return nullptr;
    ReportAllocatorCannotReturnNull(true);
  }

  void SetMayReturnNull(bool AllocatorMayReturnNull) {
    atomic_store(&MayReturnNull, AllocatorMayReturnNull, memory_order_release);
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

  uptr TotalMemoryUsed() {
    UNIMPLEMENTED();
  }

  bool PointerIsMine(const void *Ptr) {
    UNIMPLEMENTED();
  }

  uptr GetActuallyAllocatedSize(void *Ptr) {
    SecondaryHeader *Header = getHeader(Ptr);
    // Deduct PageSize as MapSize includes the trailing guard page.
    uptr MapEnd = Header->MapBeg + Header->MapSize - PageSize;
    return MapEnd - reinterpret_cast<uptr>(Ptr);
  }

  void *GetMetaData(const void *Ptr) {
    UNIMPLEMENTED();
  }

  void *GetBlockBegin(const void *Ptr) {
    UNIMPLEMENTED();
  }

  void *GetBlockBeginFastLocked(void *Ptr) {
    UNIMPLEMENTED();
  }

  void PrintStats() {
    UNIMPLEMENTED();
  }

  void ForceLock() {
    UNIMPLEMENTED();
  }

  void ForceUnlock() {
    UNIMPLEMENTED();
  }

  void ForEachChunk(ForEachChunkCallback Callback, void *Arg) {
    UNIMPLEMENTED();
  }

 private:
  // A Secondary allocated chunk header contains the base of the mapping and
  // its size. Currently, the base is always a page before the header, but
  // we might want to extend that number in the future based on the size of
  // the allocation.
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
