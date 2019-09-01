// -*- C++ -*-
#ifndef __SHADOW_MEM_ALLOCATOR_H__
#define __SHADOW_MEM_ALLOCATOR_H__

#include <cstdlib>
#include <iostream>
#include <inttypes.h>

#include "dictionary.h"

// The memory-access-line allocator is dedicated to allocating specific
// fixed-size arrays of MemoryAccess_t objects, e.g., MemoryAccess_t[1],
// MemoryAccess_t[2], MemoryAccess_t[4], and MemoryAccess_t[8].  For each array
// size, the allocator defines a slab type to be a system page of such objects.
// A slab contains some metadata for the collection of objects, followed by
// allocated storage for the objects themselves.  The metadata for a slab
// consists of the following:
//
// 1) A header, which stores a pointer to a subsequent slab as well as the size
// of the array of MemoryAccess_t objects in the slab.
//
// 2) A back pointer to a previous slab.  Together with the pointer in the
// header, this back pointer allows for doubly-linked lists of slabs.
//
// 3) A bit map identifying used and free MemoryAccess_t arrays in the slab.
// Slabs for different fixed-size arrays of MemoryAccess_t objects have
// different lengths, since different numbers of such arrays can fit within a
// single system page.

// Constants for the memory-access-line allocator.
//
// System-page size.
const unsigned SYS_PAGE_SIZE = 4096;
// Mask to get sub-system-page portion of a memory address.
const uintptr_t SYS_PAGE_DATA_MASK = SYS_PAGE_SIZE - 1;
// Mask to get the system page of a memory address.
const uintptr_t SYS_PAGE_MASK = ~SYS_PAGE_DATA_MASK;

// Helper macro to get the size of a struct field.
#define member_size(type, member) sizeof(((type *)0)->member)

// Template class for the slab header.
template<typename SlabType, uint64_t Size>
struct SlabHead_t {
  // Pointer to another Slab_t and the line size for this slab.
  SlabType *NextAndSize = reinterpret_cast<SlabType *>(Size);

  // Method to get the size associated with the slab.
  unsigned getSize() const {
    return reinterpret_cast<uintptr_t>(NextAndSize) & SYS_PAGE_DATA_MASK;
  }

  // Method to get the next pointer from the slab header.
  SlabType *getNext() const {
    return reinterpret_cast<SlabType *>(
        reinterpret_cast<uintptr_t>(NextAndSize) & SYS_PAGE_MASK);
  }

  // Method to set the next pointer in the slab header.
  void setNext(SlabType *Ptr) {
    assert((reinterpret_cast<uintptr_t>(Ptr) & SYS_PAGE_DATA_MASK) == 0 &&
           "Given pointer is not aligned.");
    NextAndSize = reinterpret_cast<SlabType *>(
        reinterpret_cast<uintptr_t>(Ptr) | getSize());
  }
};

// Template class for a slab.  The template arguments identify the size of the
// fixed-size array of MemoryAccess_t objects in the slab, as well as NumLines,
// the number of such objects in the slab.
template<unsigned Size, uint64_t NumLines>
struct Slab_t {
  using SlabType = Slab_t<Size, NumLines>;
  using LineType = MemoryAccess_t[Size];
  // static constexpr uint64_t NumLines =
  //   (8 * (SYS_PAGE_SIZE - sizeof(SlabHead_t<SlabType, Size>)
  //         - sizeof(SlabType *)) + 63) / ((8 * sizeof(LineType)) + 1);
  static constexpr int UsedMapSize = (NumLines + 63) / 64;

  // Slab header.
  SlabHead_t<SlabType, Size> Head;
  // Slab back pointer, for creating doubly-linked lists of slabs.
  SlabType *Back = nullptr;

  // Bit map of used lines.
  uint64_t UsedMap[UsedMapSize] = { 0 };

  // Line data structures.
  LineType Lines[NumLines];

  static_assert(sizeof(Lines) == sizeof(LineType) * NumLines,
                "Unexpected sizeof(Lines)");

  Slab_t() {
    // Not all bits in the allocated bit map correspond to lines in the slab.
    // Initialize the slab by setting equal to 1 the bits in the bit map that
    // don't correspond to valid lines in the slab.
    UsedMap[UsedMapSize-1] |= ~((1UL << (NumLines % 64)) - 1);
  }

  // Returns true if this slab contains no free lines.
  bool isFull() const {
    for (int i = 0; i < UsedMapSize; ++i)
      if (UsedMap[i] != static_cast<uint64_t>(-1))
        return false;
    return true;
  }

  // Get a free line from the slab, marking that line as used in the process.
  // Returns nullptr if no free line is available.
  LineType *getFreeLine() {
    for (int i = 0; i < UsedMapSize; ++i) {
      if (UsedMap[i] == static_cast<uint64_t>(-1))
        continue;

      // Get the free line.
      LineType *Line = &Lines[64 * i + __builtin_ctzl(UsedMap[i] + 1)];
      // Mark the line as used.
      UsedMap[i] |= UsedMap[i] + 1;

      return Line;
    }
    // No free lines in this slab.
    return nullptr;
  }

  // Returns a line to this slab, marking that line as available.
  void returnLine(LineType *Line) {
    uintptr_t LinePtr = reinterpret_cast<uintptr_t>(Line);
    assert((LinePtr & SYS_PAGE_MASK) == reinterpret_cast<uintptr_t>(this) &&
           "Line does not belong to this slab.");

    // Compute the index of this line in the array.
    uint64_t LineIdx = LinePtr & SYS_PAGE_DATA_MASK;
    LineIdx -= offsetof(SlabType, Lines);
    LineIdx /= sizeof(LineType);

    // Mark the line as available in the map.
    uint64_t MapIdx = LineIdx / 64;
    uint64_t MapBit = LineIdx % 64;

    assert(MapIdx < UsedMapSize && "Invalid MapIdx.");
    assert(0 != (UsedMap[MapIdx] & (1UL << MapBit)) &&
           "Line is not marked used.");
    UsedMap[MapIdx] &= ~(1UL << MapBit);
  }
};

// Template instantiations for slabs for different fixed-size arrays of
// MemoryAccess_t's.

// Slab of MemoryAccess_t[1].
using Slab1_t =
  Slab_t<1, (SYS_PAGE_SIZE - sizeof(uintptr_t[2]) - sizeof(uint64_t[4]))/
         sizeof(MemoryAccess_t[1])>;

static_assert(sizeof(SlabHead_t<Slab1_t, 1>) == sizeof(uintptr_t),
              "Unexpected SlabHead_t size.");
static_assert(sizeof(Slab1_t) <= SYS_PAGE_SIZE, "Invalid Slab1_t");
static_assert(member_size(Slab1_t, UsedMap) * 8 >=
              (member_size(Slab1_t, Lines)/sizeof(MemoryAccess_t[1])),
              "Bad size for Slab1_t.UsedMap");
static_assert((member_size(Slab1_t, UsedMap) * 8) - 64 <
              (member_size(Slab1_t, Lines)/sizeof(MemoryAccess_t[1])),
              "Inefficient size for Slab1_t.UsedMap");

// Slab of MemoryAccess_t[2].
using Slab2_t =
  Slab_t<2, (SYS_PAGE_SIZE - sizeof(uintptr_t[2]) - sizeof(uint64_t[2]))/
         sizeof(MemoryAccess_t[2])>;

static_assert(sizeof(SlabHead_t<Slab2_t, 2>) == sizeof(uintptr_t),
              "Unexpected SlabHead_t size.");
static_assert(sizeof(Slab2_t) <= SYS_PAGE_SIZE, "Invalid Slab2_t");
static_assert(member_size(Slab2_t, UsedMap) * 8 >=
              (member_size(Slab2_t, Lines)/sizeof(MemoryAccess_t[2])),
              "Bad size for Slab2_t.UsedMap");
static_assert((member_size(Slab2_t, UsedMap) * 8) - 64 <
              (member_size(Slab2_t, Lines)/sizeof(MemoryAccess_t[2])),
              "Inefficient size for Slab2_t.UsedMap");

// Slab of MemoryAccess_t[4].
using Slab4_t =
  Slab_t<4, (SYS_PAGE_SIZE - sizeof(uintptr_t[2]) - sizeof(uint64_t[1]))/
         sizeof(MemoryAccess_t[4])>;

static_assert(sizeof(SlabHead_t<Slab4_t, 4>) == sizeof(uintptr_t),
              "Unexpected SlabHead_t size.");
static_assert(sizeof(Slab4_t) <= SYS_PAGE_SIZE, "Invalid Slab4_t");
static_assert(member_size(Slab4_t, UsedMap) * 8 >=
              (member_size(Slab4_t, Lines)/sizeof(MemoryAccess_t[4])),
              "Bad size for Slab4_t.UsedMap");
static_assert((member_size(Slab4_t, UsedMap) * 8) - 64 <
              (member_size(Slab4_t, Lines)/sizeof(MemoryAccess_t[4])),
              "Inefficient size for Slab4_t.UsedMap");

// Slab of MemoryAccess_t[8].
using Slab8_t =
  Slab_t<8, (SYS_PAGE_SIZE - sizeof(uintptr_t[2]) - sizeof(uint64_t[1]))/
         sizeof(MemoryAccess_t[8])>;

static_assert(sizeof(SlabHead_t<Slab8_t, 8>) == sizeof(uintptr_t),
              "Unexpected SlabHead_t size.");
static_assert(sizeof(Slab8_t) <= SYS_PAGE_SIZE, "Invalid Slab8_t");
static_assert(member_size(Slab8_t, UsedMap) * 8 >=
              (member_size(Slab8_t, Lines)/sizeof(MemoryAccess_t[8])),
              "Bad size for Slab8_t.UsedMap");
static_assert((member_size(Slab8_t, UsedMap) * 8) - 64 <
              (member_size(Slab8_t, Lines)/sizeof(MemoryAccess_t[8])),
              "Bad size for Slab8_t.UsedMap");

// Top-level class for the allocating lines of memory accesses.
class MALineAllocator {
  // The types of lines -- fixed-size arrays of MemoryAccess_t objects --
  // supported by this allocator.
  using LineType1 = MemoryAccess_t[1];
  using LineType2 = MemoryAccess_t[2];
  using LineType4 = MemoryAccess_t[4];
  using LineType8 = MemoryAccess_t[8];

  // Linked lists of slabs with available lines.
  Slab1_t *MA1Lines = nullptr;
  Slab2_t *MA2Lines = nullptr;
  Slab4_t *MA4Lines = nullptr;
  Slab8_t *MA8Lines = nullptr;

  // Doubly-linked lists of full slabs.
  Slab1_t *FullMA1 = nullptr;
  Slab2_t *FullMA2 = nullptr;
  Slab4_t *FullMA4 = nullptr;
  Slab8_t *FullMA8 = nullptr;

public:
  MALineAllocator() {
    // Initialize the allocator with 1 page of each type of line.
    MA1Lines = new (aligned_alloc(SYS_PAGE_SIZE, sizeof(Slab1_t))) Slab1_t;
    MA2Lines = new (aligned_alloc(SYS_PAGE_SIZE, sizeof(Slab2_t))) Slab2_t;
    MA4Lines = new (aligned_alloc(SYS_PAGE_SIZE, sizeof(Slab4_t))) Slab4_t;
    MA8Lines = new (aligned_alloc(SYS_PAGE_SIZE, sizeof(Slab8_t))) Slab8_t;
  }

  // Free the slabs back to system memory.
  template <typename ST>
  void freeSlabs(ST *&List) {
    ST *Slab = List;
    ST *PrevSlab = nullptr;
    while (Slab) {
      PrevSlab = Slab;
      Slab = Slab->Head.getNext();
      PrevSlab->~ST();
      free(PrevSlab);
    }
    List = nullptr;
  }

  ~MALineAllocator() {
    assert(!FullMA1 && "Full slabs remaining.");
    assert(!FullMA2 && "Full slabs remaining.");
    assert(!FullMA4 && "Full slabs remaining.");
    assert(!FullMA8 && "Full slabs remaining.");
    freeSlabs<Slab1_t>(MA1Lines);
    freeSlabs<Slab2_t>(MA2Lines);
    freeSlabs<Slab4_t>(MA4Lines);
    freeSlabs<Slab8_t>(MA8Lines);
  }

  // Call the destructor on a line.
  template <typename LT>
  LT *destruct(LT *Line, unsigned Size) {
    for (unsigned i = 0; i < Size; ++i)
      (*Line)[i].~MemoryAccess_t();
    return Line;
  }

  // Call the destructor on Line, then return it to Slab.
  template <typename LT, typename ST>
  void freeLine(LT *Line, ST *Slab, ST *&List, ST *&Full, unsigned Size) {
    // Destruct the line.
    Line = destruct<LT>(Line, Size);

    if (Slab->isFull()) {
      // Slab is no longer full, so move it back to List.

      // Make Slab's predecessor point to Slab's successor.
      if (Slab->Back)
        Slab->Back->Head.setNext(Slab->Head.getNext());
      else
        Full = Slab->Head.getNext();

      // Make Slab's successor point to Slab's predecessor.
      if (Slab->Head.getNext())
        Slab->Head.getNext()->Back = Slab->Back;

      // Push Slab to the start of List.
      Slab->Back = nullptr;
      Slab->Head.setNext(List);
      List = Slab;
    }

    Slab->returnLine(Line);
  }

  // Deallocate the line pointed to by Ptr.
  bool deallocate(void *Ptr) {
    // Get the Line size from the page containing Ptr.
    uintptr_t PagePtr = reinterpret_cast<uintptr_t>(Ptr) & SYS_PAGE_MASK;
    // For getting the size associated with a particular slab, there's no
    // difference between the headers for different slab types.
    unsigned Size =
      reinterpret_cast<SlabHead_t<Slab1_t, 1> *>(PagePtr)->getSize();

    // Dispatch to the appropriate returnLine method, based on Size.
    switch (Size) {
    default: return false;
    case 1:
      freeLine(reinterpret_cast<LineType1 *>(Ptr),
               reinterpret_cast<Slab1_t *>(PagePtr), MA1Lines, FullMA1, 1);
      break;
    case 2:
      freeLine(reinterpret_cast<LineType2 *>(Ptr),
               reinterpret_cast<Slab2_t *>(PagePtr), MA2Lines, FullMA2, 2);
      break;
    case 4:
      freeLine(reinterpret_cast<LineType4 *>(Ptr),
               reinterpret_cast<Slab4_t *>(PagePtr), MA4Lines, FullMA4, 4);
      break;
    case 8:
      freeLine(reinterpret_cast<LineType8 *>(Ptr),
               reinterpret_cast<Slab8_t *>(PagePtr), MA8Lines, FullMA8, 8);
      break;
    }
    return true;
  }

  // Get the storage for a line out of a slab in List.  Update List and Full
  // appropriately if the slab in List becomes full.
  template<typename LT, typename ST>
  LT *getLine(ST *&List, ST *&Full) {
    // TODO: Consider getting a Line from the fullest slab.  We still want this
    // process to be fast in the common case.
    ST *Slab = List;
    LT *Line = Slab->getFreeLine();

    // If Slab is now full, move it to the Full list.
    if (Slab->isFull()) {
      if (!Slab->Head.getNext())
        List = new (aligned_alloc(SYS_PAGE_SIZE, sizeof(ST))) ST;
      else
        List = Slab->Head.getNext();
      Slab->Head.setNext(Full);
      if (Full)
        Full->Back = Slab;
      Full = Slab;
    }

    assert(Line && "No line found.");
    return Line;
  }

  // Instantiations of getLine<> to get lines of specific sizes.
  LineType1 *getMA1Line() {
    return getLine<LineType1, Slab1_t>(MA1Lines, FullMA1);
  }
  LineType2 *getMA2Line() {
    return getLine<LineType2, Slab2_t>(MA2Lines, FullMA2);
  }
  LineType4 *getMA4Line() {
    return getLine<LineType4, Slab4_t>(MA4Lines, FullMA4);
  }
  LineType8 *getMA8Line() {
    return getLine<LineType8, Slab8_t>(MA8Lines, FullMA8);
  }

  // Call the constructor on all entries of Line.
  template <typename LT>
  MemoryAccess_t *construct(LT *Line, unsigned Size) {
    for (unsigned i = 0; i < Size; ++i)
      new (&((*Line)[i])) MemoryAccess_t;
    return *Line;
  }

  // Allocate a line with size entries by getting a line from an appropriate
  // slab and then calling the constructor on that line.
  MemoryAccess_t *allocate(size_t size) {
    switch (size) {
    default: return nullptr;
    case 1: return construct<LineType1>(getMA1Line(), 1);
    case 2: return construct<LineType2>(getMA2Line(), 2);
    case 4: return construct<LineType4>(getMA4Line(), 4);
    case 8: return construct<LineType8>(getMA8Line(), 8);
    }
  }    
};

#endif // __SHADOW_MEM_ALLOCATOR__
