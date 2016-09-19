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

namespace __scudo {

class ScudoLargeMmapAllocator {
 public:

  void Init(bool AllocatorMayReturnNull) {
    PageSize = GetPageSizeCached();
    atomic_store(&MayReturnNull, AllocatorMayReturnNull, memory_order_relaxed);
  }

  void *Allocate(AllocatorStats *Stats, uptr Size, uptr Alignment) {
    // The Scudo frontend prevents us from allocating more than
    // MaxAllowedMallocSize, so integer overflow checks would be superfluous.
    uptr MapSize = RoundUpTo(Size + sizeof(SecondaryHeader), PageSize);
    // Account for 2 guard pages, one before and one after the chunk.
    uptr MapBeg = reinterpret_cast<uptr>(MmapNoAccess(MapSize + 2 * PageSize));
    CHECK_NE(MapBeg, ~static_cast<uptr>(0));
    // A page-aligned pointer is assumed after that, so check it now.
    CHECK(IsAligned(MapBeg, PageSize));
    MapBeg += PageSize;
    CHECK_EQ(MapBeg, reinterpret_cast<uptr>(MmapFixedOrDie(MapBeg, MapSize)));
    uptr MapEnd = MapBeg + MapSize;
    uptr Ptr = MapBeg + sizeof(SecondaryHeader);
    // TODO(kostyak): add a random offset to Ptr.
    CHECK_GT(Ptr + Size, MapBeg);
    CHECK_LE(Ptr + Size, MapEnd);
    SecondaryHeader *Header = getHeader(Ptr);
    Header->MapBeg = MapBeg - PageSize;
    Header->MapSize = MapSize + 2 * PageSize;
    Stats->Add(AllocatorStatAllocated, MapSize);
    Stats->Add(AllocatorStatMapped, MapSize);
    return reinterpret_cast<void *>(Ptr);
  }

  void *ReturnNullOrDie() {
    if (atomic_load(&MayReturnNull, memory_order_acquire))
      return nullptr;
    ReportAllocatorCannotReturnNull();
  }

  void SetMayReturnNull(bool AllocatorMayReturnNull) {
    atomic_store(&MayReturnNull, AllocatorMayReturnNull, memory_order_release);
  }

  void Deallocate(AllocatorStats *Stats, void *Ptr) {
    SecondaryHeader *Header = getHeader(Ptr);
    Stats->Sub(AllocatorStatAllocated, Header->MapSize);
    Stats->Sub(AllocatorStatMapped, Header->MapSize);
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
    uptr MapEnd = Header->MapBeg + Header->MapSize;
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
  // Check that sizeof(SecondaryHeader) is a multiple of 16.
  COMPILER_CHECK((sizeof(SecondaryHeader) & 0xf) == 0);

  SecondaryHeader *getHeader(uptr Ptr) {
    return reinterpret_cast<SecondaryHeader*>(Ptr - sizeof(SecondaryHeader));
  }
  SecondaryHeader *getHeader(const void *Ptr) {
    return getHeader(reinterpret_cast<uptr>(Ptr));
  }

  uptr PageSize;
  atomic_uint8_t MayReturnNull;
};

} // namespace __scudo

#endif  // SCUDO_ALLOCATOR_SECONDARY_H_
