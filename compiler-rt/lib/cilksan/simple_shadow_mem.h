// -*- C++ -*-
#ifndef __SIMPLE_SHADOW_MEM__
#define __SIMPLE_SHADOW_MEM__

#include <cstdlib>
#include <iostream>
#include <inttypes.h>

#include "cilksan_internal.h"
#include "debug_util.h"
#include "dictionary.h"
#include "shadow_mem_allocator.h"

static const unsigned ReadMAAllocator = 0;
static const unsigned WriteMAAllocator = 1;
static const unsigned AllocMAAllocator = 2;

// A simple dictionary implementation that uses a two-level table structure.
// The table structure involves a table of pages, where each page represents a
// line of memory locations.  A line of memory accesses is represented as an
// array of MemoryAccess_t objects.
//
// Pages and lines in this representation do not necessarily correspond with
// OS or hardware notions of pages or cache lines.
//
// The template parameter identifies which memory allocator to use to allocate
// lines.
template <unsigned AllocIdx>
class SimpleDictionary {
private:
  // Constant parameters for the table structure.
  // log_2 of bytes per line.
  static constexpr unsigned LG_LINE_SIZE = 3;
  // log_2 of lines per page.
  static constexpr unsigned LG_PAGE_SIZE = 24 - LG_LINE_SIZE;
  // log_2 of number of pages in the top-level table.
  static constexpr unsigned LG_TABLE_SIZE = 48 - LG_PAGE_SIZE - LG_LINE_SIZE;

  // Bytes per line.
  static constexpr uintptr_t LINE_SIZE = (1UL << LG_LINE_SIZE);
  // Low-order bit of address identifying the page.
  static constexpr uintptr_t PAGE_OFF = (1UL << (LG_PAGE_SIZE + LG_LINE_SIZE));

  // Mask to identify the byte within a line.
  static constexpr uintptr_t BYTE_MASK = (LINE_SIZE - 1);
  // Mask to identify the line.
  static constexpr uintptr_t LINE_MASK = ~BYTE_MASK;
  // Mask to identify the page.
  static constexpr uintptr_t PAGE_MASK = ~(PAGE_OFF - 1);
  // Mask to identify the index of a line in a page.
  static constexpr uintptr_t LINE_IDX_MASK = LINE_MASK ^ PAGE_MASK;

  // Helper methods to get the indices into the dictionary from a given address.
  // We used these helper methods, rather than a bitfiled struct, because the
  // language standard provides too few guarantees on the order of fields in a
  // bitfield struct.
  __attribute__((always_inline))
  static uintptr_t byte(uintptr_t addr) {
    return addr & BYTE_MASK;
  }
  __attribute__((always_inline))
  static uintptr_t line(uintptr_t addr) {
    return (addr & LINE_IDX_MASK) >> LG_LINE_SIZE;
  }
  __attribute__((always_inline))
  static uintptr_t page(uintptr_t addr) {
    return (addr >> (LG_PAGE_SIZE + LG_LINE_SIZE));
  }

  // Helper methods for computing aligned addresses from a given address.  These
  // are used to iterate through the different parts of the shadow-memory
  // structure.
  __attribute__((always_inline))
  static uintptr_t alignByPrevGrainsize(uintptr_t addr,
                                        unsigned lgGrainsize) {
    uintptr_t grainsize = 1 << lgGrainsize;
    uintptr_t mask = ~(grainsize - 1);
    return addr & mask;
  }
  __attribute__((always_inline))
  static uintptr_t alignByNextGrainsize(uintptr_t addr,
                                        unsigned lgGrainsize) {
    uintptr_t grainsize = 1 << lgGrainsize;
    uintptr_t mask = ~(grainsize - 1);
    return (addr + grainsize) & mask;
  }
  __attribute__((always_inline))
  static bool isLineStart(uintptr_t addr) {
    return byte(addr) == 0;
  }
  __attribute__((always_inline))
  static bool isPageStart(uintptr_t addr) {
    return (addr & ~PAGE_MASK) == 0;
  }

  // Pair-like data structure to represent a continuous region of memory.
  struct Chunk_t {
    uintptr_t addr;
    size_t size;

    Chunk_t(uintptr_t addr, size_t size) : addr(addr), size(size) {}

    // Returns true if this chunk represents an empty region of memory.
    bool isEmpty() const {
      return 0 == size;
    }

    // Get the chunk after this chunk whose address is grainsize-aligned.
    __attribute__((always_inline))
    Chunk_t next(unsigned lgGrainsize) const {
      cilksan_assert(((lgGrainsize == (LG_PAGE_SIZE + LG_LINE_SIZE)) ||
                      (lgGrainsize <= LG_LINE_SIZE)) && "Invalid lgGrainsize");
      
      uintptr_t nextAddr = alignByNextGrainsize(addr, lgGrainsize);
      size_t chunkSize = nextAddr - addr;
      if (chunkSize > size)
        return Chunk_t(nextAddr, 0);
      return Chunk_t(nextAddr, size - chunkSize);
    }

    // Returns true if this Chunk_t is entirely contained within the line.
    __attribute__((always_inline))
    bool withinLine() const {
      uintptr_t nextLineAddr = alignByNextGrainsize(addr, LG_LINE_SIZE);
      return ((addr + size) < nextLineAddr);
    }

    // Returns the last byte address within this Chunk_t that lies within the
    // line.
    __attribute__((always_inline))
    uintptr_t endAddrForLine() const {
      if (!withinLine())
        return alignByNextGrainsize(addr, LG_LINE_SIZE) - 1;
      return addr + size;
    }

    // Computes a LgGrainsize for this Chunk_t based on its start and end.
    __attribute__((always_inline))
    uintptr_t getLgGrainsize() const {
      cilksan_assert(0 != addr && "Chunk defined on null address.");
      // Compute the lg grainsize implied by addr.
      unsigned lgGrainsize = __builtin_ctzl(addr);
      // Cap the lg grainsize at LG_LINE_SIZE.
      if (lgGrainsize > LG_LINE_SIZE)
        lgGrainsize = LG_LINE_SIZE;

      // Quick test to see if we need to check the end address of this chunk.
      if (size >= LINE_SIZE)
        return lgGrainsize;

      // Check if the end of the chunk is in the same line.
      // uintptr_t nextLineAddr = alignByNextGrainsize(addr, LG_LINE_SIZE);
      if (withinLine()) {
        // Compute the lg grainsize implied by the end of the chunk.
        unsigned endLgGrainsize = __builtin_ctzl(addr + size);
        // Take the smaller of the two grainsizes.
        if (endLgGrainsize < lgGrainsize)
          lgGrainsize = endLgGrainsize;
      }
      return lgGrainsize;
    }

  private:
    // DEBUGGING: No default constructor for Chunk_t.
    Chunk_t() = delete;
  };

  // Helper methods to check if chunk startsat a line or page boundary.
  __attribute__((always_inline))
  static bool isLineStart(Chunk_t chunk) {
    return isLineStart(chunk.addr);
  }
  __attribute__((always_inline))
  static bool isPageStart(Chunk_t chunk) {
    return isPageStart(chunk.addr);
  }

  // Custom memory allocator for lines.
  static MALineAllocator &MAAlloc;

  // Data structure for a line of memory locations.  A line is represented as an
  // array of MemoryAccess_t objects, where each object represents an aligned
  // set of (1 << LgGrainsize) bytes.  LgGrainsize must be in
  // [0, LG_LINE_SIZE].
  struct Line_t {
  private:
    static const uintptr_t MemAccsMask = (1UL << 48) - 1;
    static const uintptr_t NumNonNullAccsRShift = 48;
    static const uintptr_t LgGrainsizeRShift = 48 + 8;
    static const uintptr_t MetadataFieldMask = (1UL << 8) - 1;
    // // If we need to store more metadata for each line, we can shrink these
    // // fields.
    // unsigned LgGrainsize;
    // int NumNonNullAccs = 0;
    // The array of MemoryAccess_t objects in this line is allocated lazily.
    // MemoryAccess_t *MemAccs = nullptr;
    MemoryAccess_t *MemAccsPtr = nullptr;

    __attribute__((always_inline))
    MemoryAccess_t *getMemAccs() const {
      return reinterpret_cast<MemoryAccess_t *>(
          reinterpret_cast<uintptr_t>(MemAccsPtr) & MemAccsMask);
    }
    __attribute__((always_inline))
    void setMemAccs(MemoryAccess_t *newMemAccs) {
      MemAccsPtr = reinterpret_cast<MemoryAccess_t *>(
          (reinterpret_cast<uintptr_t>(MemAccsPtr) & ~MemAccsMask) |
          (reinterpret_cast<uintptr_t>(newMemAccs) & MemAccsMask));
    }
    __attribute__((always_inline))
    void setLgGrainsize(unsigned newLgGrainsize) {
      MemAccsPtr = reinterpret_cast<MemoryAccess_t *>(
          (reinterpret_cast<uintptr_t>(MemAccsPtr) &
           ~(MetadataFieldMask << LgGrainsizeRShift)) | 
          (static_cast<uintptr_t>(newLgGrainsize) << LgGrainsizeRShift));
    }
    __attribute__((always_inline))
    void scaleNumNonNullAccs(int replFactor) {
      MemAccsPtr = reinterpret_cast<MemoryAccess_t *>(
          (reinterpret_cast<uintptr_t>(MemAccsPtr) &
           ~(MetadataFieldMask << NumNonNullAccsRShift)) |
          ((reinterpret_cast<uintptr_t>(MemAccsPtr) &
            (MetadataFieldMask << NumNonNullAccsRShift)) * replFactor));
    }
  public:
    // Line_t(unsigned LgGrainsize) : LgGrainsize(LgGrainsize) {
    //   assert(LgGrainsize >= 0 && LgGrainsize <= LG_LINE_SIZE &&
    //          "Invalid grainsize for Line_t");
    // }
    // // By default, a Line_t contains entries of (1 << LG_LINE_SIZE) bytes.
    // Line_t() : LgGrainsize(LG_LINE_SIZE) { }
    Line_t(unsigned LgGrainsize) {
      cilksan_assert(LgGrainsize >= 0 && LgGrainsize <= LG_LINE_SIZE &&
                     "Invalid grainsize for Line_t");
      setLgGrainsize(LgGrainsize);
    }
    Line_t() {
      // By default, a Line_t contains entries of (1 << LG_LINE_SIZE) bytes.
      setLgGrainsize(LG_LINE_SIZE);
    }
    ~Line_t() {
      if (isMaterialized()) {
        MAAlloc.deallocate(getMemAccs());
        // MemAccs = nullptr;
        MemAccsPtr = nullptr;
      }
    }

    __attribute__((always_inline))
    unsigned getLgGrainsize() const {
      return static_cast<unsigned>(
          (reinterpret_cast<uintptr_t>(MemAccsPtr) >> LgGrainsizeRShift)
          & MetadataFieldMask);
    }

    __attribute__((always_inline))
    bool isEmpty() const {
      // return (0 == NumNonNullAccs);
      return noNonNullAccs();
    }

    __attribute__((always_inline))
    int getNumNonNullAccs() const {
      return static_cast<int>(
          (reinterpret_cast<uintptr_t>(MemAccsPtr) >> NumNonNullAccsRShift)
          & MetadataFieldMask);
    }
    __attribute__((always_inline))
    bool noNonNullAccs() const {
      return (0 == (reinterpret_cast<uintptr_t>(MemAccsPtr) &
                    (MetadataFieldMask << NumNonNullAccsRShift)));
    }
    __attribute__((always_inline))
    void zeroNumNonNullAccs() {
      MemAccsPtr = reinterpret_cast<MemoryAccess_t *>(
          reinterpret_cast<uintptr_t>(MemAccsPtr) &
          ~(MetadataFieldMask << NumNonNullAccsRShift));
    }
    __attribute__((always_inline))
    void incNumNonNullAccs() {
      MemAccsPtr = reinterpret_cast<MemoryAccess_t *>(
          reinterpret_cast<uintptr_t>(MemAccsPtr) +
          (1UL << NumNonNullAccsRShift));
    }
    __attribute__((always_inline))
    void decNumNonNullAccs() {
      cilksan_assert(!noNonNullAccs() && "Decrementing NumNonNullAccs below 0");
      MemAccsPtr = reinterpret_cast<MemoryAccess_t *>(
          reinterpret_cast<uintptr_t>(MemAccsPtr) -
          (1UL << NumNonNullAccsRShift));
    }

    // Check if the array of MemoryAccess_t's has been allocated.
    __attribute__((always_inline))
    bool isMaterialized() const {
      return (nullptr != getMemAccs());
    }

    // Allocate the array of MemoryAccess_t's for this line.
    void materialize() {
      cilksan_assert(!getMemAccs() && "MemAccs already materialized.");
      int NumMemAccs = (1 << LG_LINE_SIZE) / (1 << getLgGrainsize());
      // MemAccs = MAAlloc.allocate(NumMemAccs);
      setMemAccs(MAAlloc.allocate(NumMemAccs));
    }

    // Reduce the grainsize of this line to newLgGrainsize, which must fall
    // within [0, LgGrainsize].
    void refine(unsigned newLgGrainsize) {
      cilksan_assert(newLgGrainsize < getLgGrainsize() &&
                     "Invalid grainsize for refining Line.");
      // If MemAccs hasn't been materialzed yet, then just update LgGrainsize.
      if (!isMaterialized()) {
        // LgGrainsize = newLgGrainsize;
        setLgGrainsize(newLgGrainsize);
        return;
      }

      MemoryAccess_t *MemAccs = getMemAccs();
      // Create a new array of MemoryAccess_t's.
      int newNumMemAccs = (1 << LG_LINE_SIZE) / (1 << newLgGrainsize);
      MemoryAccess_t *NewMemAccs = MAAlloc.allocate(newNumMemAccs);

      // Copy the old MemoryAccess_t's into the new array with replication.
      if (!noNonNullAccs()) {
        unsigned LgGrainsize = getLgGrainsize();
        int oldNumMemAccs = (1 << LG_LINE_SIZE) / (1 << LgGrainsize);
        int replFactor = (1 << LgGrainsize) / (1 << newLgGrainsize);
        for (int i = 0; i < oldNumMemAccs; ++i)
          if (MemAccs[i].isValid())
            for (int j = replFactor * i; j < replFactor * (i+1); ++j)
              NewMemAccs[j] = MemAccs[i];
        // NumNonNullAccs *= replFactor;
#if CILKSAN_DEBUG
        int oldNumNonNullAccs = getNumNonNullAccs();
#endif
        scaleNumNonNullAccs(replFactor);
        cilksan_assert(oldNumNonNullAccs * replFactor == getNumNonNullAccs());
      }

      // Replace the old MemAccs array and LgGrainsize value.
      MAAlloc.deallocate(MemAccs);
      // MemAccs = NewMemAccs;
      // LgGrainsize = newLgGrainsize;
      setMemAccs(NewMemAccs);
      setLgGrainsize(newLgGrainsize);
    }

    // Reset this Line_t object with a default LgGrainsize and no valid
    // MemoryAccess_t's.
    void reset() {
      if (isMaterialized()) {
        MAAlloc.deallocate(getMemAccs());
        MemAccsPtr = nullptr;
      }
      setLgGrainsize(LG_LINE_SIZE);
      // LgGrainsize = LG_LINE_SIZE;
      // if (isMaterialized()) {
      //   MAAlloc.deallocate(MemAccs);
      //   MemAccs = nullptr;
      //   NumNonNullAccs = 0;
      // }
    }

    // Helper method to convert a byte address into an index into this line.
    __attribute__((always_inline))
    uintptr_t getIdx(uintptr_t byte) const {
      return byte >> getLgGrainsize();
    }

    // Access the MemoryAccess_t object in this line for the byte address.
    __attribute__((always_inline))
    MemoryAccess_t &operator[] (uintptr_t byte) {
      cilksan_assert(getMemAccs() && "MemAccs not materialized");
      return getMemAccs()[getIdx(byte)];
    }
    __attribute__((always_inline))
    const MemoryAccess_t &operator[] (uintptr_t byte) const {
      cilksan_assert(getMemAccs() && "MemAccs not materialized");
      return getMemAccs()[getIdx(byte)];
    }

    // Set all entries in this line covered by Accessed to be MA, which must be
    // a valid MemoryAccess_t.
    //
    // TODO: Check if C++ copy elision avoid unneccesary calls to the copy
    // constructor for the MemoryAccess_t, either in C++11, C++14, or C++17.
    __attribute__((always_inline))
    void set(Chunk_t &Accessed, const MemoryAccess_t &MA) {
      cilksan_assert(MA.isValid() && "Setting to invalid MemoryAccess_t");
      // Get the grainsize of the access.
      unsigned AccessedLgGrainsize = Accessed.getLgGrainsize();

      // If we're overwritting the entire line, then we can coalesce the line.
      if (AccessedLgGrainsize == LG_LINE_SIZE) {
        // Reset the line if necessary.
        if (getLgGrainsize() != LG_LINE_SIZE)
          reset();

        // Materialize the line, if necessary.
        if (!isMaterialized())
          materialize();

        MemoryAccess_t *MemAccs = getMemAccs();
        // Check if we're adding a new valid entry.
        if (!MemAccs[0].isValid())
          // ++NumNonNullAccs;
          incNumNonNullAccs();

        // Add the entry.
        MemAccs[0] = MA;

        // Updated Accessed.
        Accessed = Accessed.next(AccessedLgGrainsize);
        return;
      }

      // We're updating the content of the line in a refined manner, meaning
      // that either Accessed or this Line_t store MemoryAccess_t's at a finer
      // granularity than 8-byte chunks.

      // Pick the smaller of the line's existing grainsize or the grainsize of
      // the access.
      unsigned LgGrainsize = getLgGrainsize();
      if (LgGrainsize > AccessedLgGrainsize) {
        // The access has a smaller grainsize, so first refine the line to match
        // that grainsize.
        refine(AccessedLgGrainsize);
      } else if (LgGrainsize < AccessedLgGrainsize) {
        AccessedLgGrainsize = LgGrainsize;
      }

      // Materialize the line, if necessary.
      if (!isMaterialized())
        materialize();

      // Update the accesses in the line, until we find a new non-null Entry.
      MemoryAccess_t *MemAccs = getMemAccs();
      do {
        uintptr_t Idx = getIdx(byte(Accessed.addr));
        // Increase the count of non-null memory accesses, if necessary.
        // if (!(*this)[byte(Accessed.addr)].isValid())
        //   ++NumNonNullAccs;
        if (!MemAccs[Idx].isValid())
          incNumNonNullAccs();

        // Copy the memory access
        // (*this)[byte(Accessed.addr)] = MA;
        MemAccs[Idx] = MA;

        // Get the next location.
        Accessed = Accessed.next(AccessedLgGrainsize);
        if (Accessed.isEmpty())
          return;

      } while (!isLineStart(Accessed));
    }

    // Starting from the first address in Accessed, insert MA into entries in
    // this Line_t until either the end of this line is reached or a change is
    // detected in the MemoryAccess_t object.
    //
    // TODO: Check if C++ copy elision avoid unneccesary calls to the copy
    // constructor for the MemoryAccess_t, either in C++11, C++14, or C++17.
    __attribute__((always_inline))
    void insert(Chunk_t &Accessed, unsigned PrevIdx, const MemoryAccess_t &MA) {
      cilksan_assert(MA.isValid() && "Setting to invalid MemoryAccess_t");
      // Get the grainsize of the access.
      unsigned AccessedLgGrainsize = Accessed.getLgGrainsize();

      // If neither the line nor the access are refined, then we can optimize
      // the insert process.
      if ((AccessedLgGrainsize == LG_LINE_SIZE) &&
          (getLgGrainsize() == LG_LINE_SIZE)) {
        // Materialize the line, if necessary.
        if (!isMaterialized())
          materialize();

        MemoryAccess_t *MemAccs = getMemAccs();
        // Check if we're adding a new valid entry.
        if (!MemAccs[0].isValid())
          // ++NumNonNullAccs;
          incNumNonNullAccs();

        // Add the entry.
        MemAccs[0] = MA;

        // Updated Accessed.
        Accessed = Accessed.next(AccessedLgGrainsize);
        return;
      }

      // We're updating the content of the line in a refined manner, meaning
      // that either Accessed or this Line_t store MemoryAccess_t's at a finer
      // granularity than 8-byte chunks.

      // Pick the smaller of the line's existing grainsize or the grainsize of
      // the access.
      unsigned LgGrainsize = getLgGrainsize();
      if (LgGrainsize > AccessedLgGrainsize) {
        // The access has a smaller grainsize, so first refine the line to match
        // that grainsize.
        refine(AccessedLgGrainsize);
      } else if (LgGrainsize < AccessedLgGrainsize) {
        AccessedLgGrainsize = LgGrainsize;
      }

      // Materialize the line, if necessary.
      if (!isMaterialized())
        materialize();

      MemoryAccess_t *MemAccs = getMemAccs();
      const MemoryAccess_t Previous = MemAccs[PrevIdx];
      unsigned EntryIdx;
      // Update the accesses in the line, until we find a new non-null Entry.
      do {
        uintptr_t Idx = getIdx(byte(Accessed.addr));
        // Increase the count of non-null memory accesses, if necessary.
        // if (!(*this)[byte(Accessed.addr)].isValid())
        //   ++NumNonNullAccs;
        if (!MemAccs[Idx].isValid())
          incNumNonNullAccs();

        // Copy the memory access
        // (*this)[byte(Accessed.addr)] = MA;
        MemAccs[Idx] = MA;

        // Get the next location.
        Accessed = Accessed.next(AccessedLgGrainsize);
        if (Accessed.isEmpty())
          return;

        // If we're not at the end of the line yet, update PrevAccess.
        if (!isLineStart(Accessed))
          EntryIdx = getIdx(byte(Accessed.addr));
      } while (!isLineStart(Accessed) &&
               (!MemAccs[EntryIdx].isValid() ||
                (Previous.isValid() && (Previous == MemAccs[EntryIdx]))));
    }

    // Reset all the entries of this line covered by Accessed.
    __attribute__((always_inline))
    void clear(Chunk_t &Accessed) {
      // Get the grainsize of the access.
      unsigned AccessedLgGrainsize = Accessed.getLgGrainsize();

      if (LG_LINE_SIZE == AccessedLgGrainsize) {
        if (!isEmpty())
          // Reset the line.
          reset();

        // Updated Accessed.
        Accessed = Accessed.next(AccessedLgGrainsize);
        return;
      }

      // Pick the smaller of the line's existing grainsize or the grainsize of
      // the access.
      unsigned LgGrainsize = getLgGrainsize();
      if (LgGrainsize > AccessedLgGrainsize) {
        // The access has a smaller grainsize, so first refine the line to match
        // that grainsize.
        refine(AccessedLgGrainsize);
      } else if (LgGrainsize < AccessedLgGrainsize) {
        AccessedLgGrainsize = LgGrainsize;
      }

      // If the line is already empty, then there's nothing to clear.
      if (isEmpty()) {
        Accessed = Accessed.next(LG_LINE_SIZE);
        return;
      }

      MemoryAccess_t *MemAccs = getMemAccs();
      do {
        uintptr_t Idx = getIdx(byte(Accessed.addr));
        // If we find a valid MemoryAccess_t, invalidate it.
        // if ((*this)[byte(Accessed.addr)].isValid()) {
        //   (*this)[byte(Accessed.addr)].invalidate();
        if (MemAccs[Idx].isValid()) {
          MemAccs[Idx].invalidate();

          // Decrement the number of non-null accesses.
          // --NumNonNullAccs;
          decNumNonNullAccs();

          // Skip to the end of the line if it becomes empty.
          // if (0 == NumNonNullAccs) {
          if (noNonNullAccs()) {
            Accessed = Accessed.next(LG_LINE_SIZE);
            // Reset the line to forget about any refinement.
            if (getLgGrainsize() != LG_LINE_SIZE)
              reset();
          } else
            // Advance to the next entry in the line.
            Accessed = Accessed.next(AccessedLgGrainsize);
        } else
          // Advance to the next entry in the line.
          Accessed = Accessed.next(AccessedLgGrainsize);
      } while(!Accessed.isEmpty() && !isLineStart(Accessed.addr));
    }
  };

  // A page is an array of lines.
  struct Page_t {
    Line_t lines[1 << LG_PAGE_SIZE];

    Line_t &operator[] (uintptr_t line) {
      return lines[line];
    }
    const Line_t &operator[] (uintptr_t line) const {
      return lines[line];
    }
  };

  // A table is an array of pages.
  Page_t *Table[1 << LG_TABLE_SIZE] = { nullptr };

public:
  SimpleDictionary() {}
  ~SimpleDictionary() {
    for (int i = 0; i < (1 << LG_TABLE_SIZE); ++i)
      if (Table[i]) {
        delete Table[i];
        Table[i] = nullptr;
      }
  }

  // Helper class to store a particular location in the dictionary.  This class
  // makes it easy to re-retrieve the MemoryAccess_t object at a given location,
  // even if the underlying line structure in the shadow memory might change.
  struct Entry_t {
    uintptr_t Address;
    Page_t *Page = nullptr;

    Entry_t() {}
    Entry_t(const SimpleDictionary &D, uintptr_t Address) : Address(Address) {
      Page = D.Table[page(Address)];
    }

    // Get the MemoryAcces_t object at this location.  Returns nullptr if no
    // valid MemoryAccess_t object exists at the current address.
    __attribute__((always_inline))
    const MemoryAccess_t *get() const {
      // If there's no page, return nullptr.
      if (!Page)
        return nullptr;

      // If the line is empty, return nullptr.
      if ((*Page)[line(Address)].isEmpty())
        return nullptr;

      // Return the MemoryAccess_t at this address if it's valid, nullptr
      // otherwise.
      const MemoryAccess_t *Acc = &((*Page)[line(Address)][byte(Address)]);
      if (!Acc->isValid())
        return nullptr;
      return Acc;
    }
  };

  // Iterator class for querying the entries of the shadow memory corresponding
  // to a given accessed chunk.
  class Query_iterator {
    const SimpleDictionary &Dict;
    Chunk_t Accessed;
    Page_t *Page = nullptr;
    Line_t *Line = nullptr;
    Entry_t Entry;

  public:
    Query_iterator(const SimpleDictionary &Dict, Chunk_t Accessed)
        : Dict(Dict), Accessed(Accessed) {
      // Initialize the iterator to point to the first valid entry covered by
      // Accessed.
      if (Accessed.isEmpty())
        return;

      // Get the first non-null page for this access.
      if (!nextPage())
        return;

      // Get the first non-null line for this access.
      if (!nextLine())
        return;

      Entry = Entry_t(Dict, Accessed.addr);
    }

    // Returns true if this iterator has reached the end of the chunk Accessed.
    __attribute__((always_inline))
    bool isEnd() const {
      return Accessed.isEmpty();
    }

    // Get the MemoryAccess_t object at the current address.  Returns nullptr if
    // no valid MemoryAccess_t object exists at the current address.
    __attribute__((always_inline))
    const MemoryAccess_t *get() const {
      if (isEnd())
        return nullptr;

      cilksan_assert(Line && "Null Line for Query_iterator not at end.");
      if (Line->isEmpty())
        return nullptr;

      const MemoryAccess_t *Access = &(*Line)[byte(Accessed.addr)];
      if (!Access->isValid())
        return nullptr;
      return Access;
    }

    // Get the current starting address being queried.
    uintptr_t getAddress() const {
      return Accessed.addr;
    }

    // Scan the entries from Accessed until an entry with a new non-null
    // MemoryAccess_t is found.
    __attribute__((always_inline))
    void next() {
      cilksan_assert(!isEnd() &&
                     "Cannot call next() on an empty Line iterator");
      const Entry_t Previous = Entry;
      do {
        if (Line->isEmpty())
          Accessed = Accessed.next(LG_LINE_SIZE);
        else
          Accessed = Accessed.next(Line->getLgGrainsize());

        if (Accessed.isEmpty())
          return;

        // Update the page, if necessary
        if (isPageStart(Accessed.addr))
          if (!nextPage())
            return;

        // Update the line, if necessary
        if (isLineStart(Accessed.addr))
          if (!nextLine())
            return;

        Entry = Entry_t(Dict, Accessed.addr);
      } while(!Entry.get() ||
              (Previous.get() && (*Previous.get() == *Entry.get())));
    }

  private:
    // Helper method to get the next non-null page covered by Accessed.  Returns
    // true if a page is found, false otherwise.
    bool nextPage() {
      cilksan_assert(!isEnd() && "Cannot call nextPage() on an empty Line iterator");
      // Scan to find the non-null page.
      Page = Dict.Table[page(Accessed.addr)];
      while (!Page) {
        Accessed = Accessed.next(LG_PAGE_SIZE + LG_LINE_SIZE);
        // Return early if the access becomes empty.
        if (Accessed.isEmpty())
          return false;
        Page = Dict.Table[page(Accessed.addr)];
      }
      return true;
    }

    // Helper method to get the next non-null line covered by Accessed.  Returns
    // true if a line is found, false otherwise.
    bool nextLine() {
      cilksan_assert(!isEnd() &&
                     "Cannot call nextLine() on an empty Line iterator");
      cilksan_assert(Page && "nextLine() called with null page");
      // Scan to find the non-null line.
      Line = &(*Page)[line(Accessed.addr)];
      while (!Line || Line->isEmpty()) {
        Accessed = Accessed.next(LG_LINE_SIZE);
        // Return early if the access becomes empty.
        if (Accessed.isEmpty())
          return false;

        // If this search reaches the end of the page, get the next page.
        if (isPageStart(Accessed.addr))
          if (!nextPage())
            return false;

        Line = &(*Page)[line(Accessed.addr)];
      }
      return true;
    }
  };

  // Iterator class for updating the entries of the shadow memory corresponding
  // to a given chunk Accessed.
  class Update_iterator {
    SimpleDictionary &Dict;
    Chunk_t Accessed;
    Page_t *Page = nullptr;
    Line_t *Line = nullptr;
    Entry_t Entry;

  public:
    Update_iterator(SimpleDictionary &Dict, Chunk_t Accessed)
        : Dict(Dict), Accessed(Accessed) {
      // Initialize the iterator to point to the first page and line entries for
      // Accessed.
      if (Accessed.isEmpty())
        return;

      // Get the page for this access.
      if (!nextPage())
        return;

      // Get the line for this access.
      if (!nextLine())
        return;

      Entry = Entry_t(Dict, Accessed.addr);
    }

    // Returns true if this iterator has reached the end of the chunk Accessed.
    __attribute__((always_inline))
    bool isEnd() const {
      return Accessed.isEmpty();
    }

    // Get the MemoryAccess_t object at the current address.  Returns nullptr if
    // no valid MemoryAccess_t object exists at the current address.
    __attribute__((always_inline))
    MemoryAccess_t *get() const {
      if (isEnd() || !Page)
        return nullptr;

      if (Line->isEmpty())
        return nullptr;

      MemoryAccess_t *Access = &(*Line)[byte(Accessed.addr)];
      if (!Access->isValid())
        return nullptr;
      return Access;
    }

    // Get the current starting address being queried.
    __attribute__((always_inline))
    uintptr_t getAddress() const {
      return Accessed.addr;
    }

    // Scan the entries from Accessed until we find a location with an invalid
    // MemoryAccess_t or a MemoryAccess_t that does not match the previous one.
    __attribute__((always_inline))
    void next() {
      cilksan_assert(!isEnd() &&
                     "Cannot call next() on an empty Line iterator");
      cilksan_assert(Page && "Cannot call next() with null Page");
      cilksan_assert(Line && "Cannot call next() with null Line");

      // Remember the previous Entry.
      const Entry_t Previous = Entry;
      do {
        Accessed = Accessed.next(Line->getLgGrainsize());
        if (Accessed.isEmpty())
          return;

        // Update the page, if necessary
        if (isPageStart(Accessed.addr))
          if (!nextPage())
            return;

        // Update the line, if necessary
        if (isLineStart(Accessed.addr))
          if (!nextLine())
            return;

        Entry = Entry_t(Dict, Accessed.addr);
      } while (Entry.get() && Previous.get() &&
               *Previous.get() == *Entry.get());
    }

    // Insert MemoryAccess_t into all entries from Accessed.
    __attribute__((always_inline))
    void set(const MemoryAccess_t &MA) {
      do {
        // Create a new page, if necessary.
        if (!Page) {
          Page = new Page_t;
          Dict.Table[page(Accessed.addr)] = Page;
          Line = &(*Page)[line(Accessed.addr)];
        }

        // Set MemoryAccess_t objects in the current line.
        Line->set(Accessed, MA);

        // Return early if we've handled the whole access.
        if (Accessed.isEmpty())
          return;

        // Update the current page, if necessary.
        if (isPageStart(Accessed))
          nextPage();

        // Update the current line, if necessary.
        if (isLineStart(Accessed))
          nextLine();

      } while(true);
    }

    // Insert MemoryAccess_t into all entries from Accessed until a new valid
    // MemoryAccess_t is discovered.
    __attribute__((always_inline))
    void insert(const MemoryAccess_t &MA) {
      // Copy the MemoryAccess_t at the previous entry.  In case this method
      // changes the MemoryAccess_t at this previous entry, this copy ensures
      // that comparisons use the previous MemoryAccess_t value.
      const MemoryAccess_t Previous(
          Entry.get() ? *Entry.get() : MemoryAccess_t());
      do {
        // Create a new page, if necessary.
        if (!Page) {
          Page = new Page_t;
          Dict.Table[page(Accessed.addr)] = Page;
          Line = &(*Page)[line(Accessed.addr)];
        }

        // Set MemoryAccess_t objects in the current line.
        Line->insert(Accessed, Line->getIdx(byte(Accessed.addr)), MA);

        // Return early if we've handled the whole access.
        if (Accessed.isEmpty())
          return;

        // Update the current page, if necessary.
        if (isPageStart(Accessed))
          nextPage();

        // Update the current line, if necessary.
        if (isLineStart(Accessed))
          nextLine();

        Entry = Entry_t(Dict, Accessed.addr);
      } while (!Entry.get() ||
               (Previous.isValid() && (Previous == *Entry.get())));
    }

    // Clear all entries covered by Accessed.
    __attribute__((always_inline))
    void clear() {
      do {
        // Scan for a non-null Page.
        if (!nextNonNullPage())
          return;

        // Scan for a non-null Line.
        if (!nextNonNullLine())
          return;

        Line->clear(Accessed);

        // Return early if we've handled the whole access.
        if (Accessed.isEmpty())
          return;
      } while (true);
    }

  private:
    // In contrast to Query iterators, Update iterators should typically get
    // pointers to null pages and lines, not skip them.
    bool nextPage() {
      cilksan_assert(!isEnd() &&
                     "Cannot call nextPage() on an empty Line iterator");
      Page = Dict.Table[page(Accessed.addr)];
      return true;
    }
    bool nextLine() {
      cilksan_assert(!isEnd() &&
                     "Cannot call nextLine() on an empty Line iterator");
      if (!Page) {
        Line = nullptr;
        return false;
      }
      Line = &(*Page)[line(Accessed.addr)];
      return true;
    }

    // Helper method to get the next non-null page, similar to the nextPage
    // method for Query_iterators.
    bool nextNonNullPage() {
      cilksan_assert(!isEnd() &&
                     "Cannot call nextPage() on an empty Line iterator");
      // Scan to find the non-null page.
      Page = Dict.Table[page(Accessed.addr)];
      while (!Page) {
        Accessed = Accessed.next(LG_PAGE_SIZE + LG_LINE_SIZE);
        // Return early if the access becomes empty.
        if (Accessed.isEmpty())
          return false;
        Page = Dict.Table[page(Accessed.addr)];
      }
      return true;
    }

    // Helper method to get the next non-null line, similar to the nextLine
    // method for Query_iterators.
    bool nextNonNullLine() {
      cilksan_assert(!isEnd() &&
                     "Cannot call nextLine() on an empty Line iterator");
      cilksan_assert(Page && "nextLine() called with null page");
      // Scan to find the non-null line.
      Line = &(*Page)[line(Accessed.addr)];
      while (!Line || Line->isEmpty()) {
        Accessed = Accessed.next(LG_LINE_SIZE);
        // Return early if the access becomes empty.
        if (Accessed.isEmpty())
          return false;

        // If this search reaches the end of the page, get the next page.
        if (isPageStart(Accessed.addr))
          if (!nextNonNullPage())
            return false;

        Line = &(*Page)[line(Accessed.addr)];
      }
      return true;
    }
  };

  // High-level method to check if this dictionary contains any entries covered
  // by the specified chunk of memory.
  __attribute__((always_inline))
  bool includes(uintptr_t addr, size_t size) const {
    Query_iterator QI(*this, Chunk_t(addr, size));
    return !QI.isEnd();
  }

  // High-level method to find a MemoryAccess_t object at the specified address.
  const MemoryAccess_t *find(uintptr_t addr) const {
    Query_iterator QI(*this, Chunk_t(addr, 1));
    return QI.get();
  }

  // High-level method to set shadow of the specified chunk of memory to match
  // the MemoryAccess_t MA.
  //
  // TODO: Check if C++ copy elision avoid unneccesary calls to the copy
  // constructor for the MemoryAccess_t, either in C++11, C++14, or C++17.
  void set(uintptr_t addr, size_t size, const MemoryAccess_t &MA) {
    Update_iterator UI(*this, Chunk_t(addr, size));
    UI.set(MA);
  }

  // Get a query iterator for the specified chunk of memory.
  Query_iterator getQueryIterator(uintptr_t addr, size_t size) const {
    return Query_iterator(*this, Chunk_t(addr, size));
  }

  // Get an update iterator for the specified chunk of memory.
  Update_iterator getUpdateIterator(uintptr_t addr, size_t size) {
    return Update_iterator(*this, Chunk_t(addr, size));
  }

  // Clear all entries for the specified chunk of memory.
  __attribute__((always_inline))
  void clear(uintptr_t addr, size_t size) {
    Update_iterator UI(*this, Chunk_t(addr, size));
    UI.clear();
  }
};

class SimpleShadowMem {
private:
  CilkSanImpl_t &CilkSanImpl;
  // The shadow memory involves three dictionaries to separately handle reads,
  // writes, and allocations.  The template parameter allows each dictionary to
  // use a different memory allocator.
  SimpleDictionary<ReadMAAllocator> Reads;
  SimpleDictionary<WriteMAAllocator> Writes;
  SimpleDictionary<AllocMAAllocator> Allocs;

public:
  SimpleShadowMem(CilkSanImpl_t &CilkSanImpl) : CilkSanImpl(CilkSanImpl) {}
  ~SimpleShadowMem() { destruct(); }

  template<bool is_read>
  __attribute__((always_inline))
  bool does_access_exists(uintptr_t addr, size_t mem_size) const {
    if (is_read)
      return Reads.includes(addr, mem_size);
    else
      return Writes.includes(addr, mem_size);
  }

  template<bool is_read>
  void insert_access(const csi_id_t acc_id, uintptr_t addr, size_t mem_size,
                     FrameData_t *f) {
    // std::cerr << "insert_access " << (is_read ? "read " : "write ")
    //           << reinterpret_cast<void*>(addr) << ", " << mem_size
    //           << ", func node " << (void*)f->Sbag->get_node() << "\n";
    if (is_read)
      Reads.set(addr, mem_size, MemoryAccess_t(f->getSbagForAccess(), acc_id,
                                               MAType_t::RW));
    else
      Writes.set(addr, mem_size, MemoryAccess_t(f->getSbagForAccess(), acc_id,
                                                MAType_t::RW));
  }

  template<typename QITy, bool prev_read, bool is_read>
  __attribute__((always_inline))
  void check_race(QITy &QI, const csi_id_t acc_id, MAType_t type,
                  uintptr_t addr, size_t mem_size, bool on_stack,
                  FrameData_t *f) const {
    // std::cerr << "check_race " << (prev_read ? "vs. read " : "vs. write ")
    //           << std::hex << addr << std::dec << ", " << mem_size << "\n";
    while (!QI.isEnd()) {
      // Find a previous access
      const MemoryAccess_t *PrevAccess = QI.get();
      if (PrevAccess && PrevAccess->isValid()) {
        // Get the function for this previous access
        auto Func = PrevAccess->getFunc();
        cilksan_assert(Func);

        // Get the bag for the previous access
        SPBagInterface *LCA = Func->get_set_node();
        // If it's a P-bag, then we have a race.
        if (LCA->is_PBag() ||
            f->check_parallel_iter(LCA, PrevAccess->getVersion())) {
          uintptr_t AccAddr = QI.getAddress();
          // If memory is allocated on stack, the accesses race with each other
          // only if the mem location is allocated in shared ancestor's stack.
          // We don't need to check for this because we clear shadow memory;
          // non-shared stack can't race because earlier one would've been
          // cleared

          // Try to get information on the allocation for this memory access.
          auto AllocFind = Allocs.find(AccAddr);
          AccessLoc_t AllocAccess =
            (nullptr == AllocFind) ? AccessLoc_t() : AllocFind->getLoc();

          // Report the race
          if (prev_read)
            CilkSanImpl.report_race(
                PrevAccess->getLoc(),
                AccessLoc_t(acc_id, type, CilkSanImpl.get_current_call_stack()),
                AllocAccess, AccAddr, RW_RACE);
          else {
            if (is_read)
              CilkSanImpl.report_race(
                  PrevAccess->getLoc(),
                  AccessLoc_t(acc_id, type,
                              CilkSanImpl.get_current_call_stack()),
                  AllocAccess, AccAddr, WR_RACE);
            else
              CilkSanImpl.report_race(
                  PrevAccess->getLoc(),
                  AccessLoc_t(acc_id, type,
                              CilkSanImpl.get_current_call_stack()),
                  AllocAccess, AccAddr, WW_RACE);
          }
        }
      }
      QI.next();
    }
  }

  __attribute__((always_inline))
  void check_race_with_prev_read(const csi_id_t acc_id, uintptr_t addr,
                                 size_t mem_size, bool on_stack,
                                 FrameData_t *f) const {
    using QITy = SimpleDictionary<ReadMAAllocator>::Query_iterator;
    QITy QI = Reads.getQueryIterator(addr, mem_size);
    // The second argument does not matter here.
    check_race<QITy, true, false>(QI, acc_id, MAType_t::RW, addr, mem_size,
                                  on_stack, f);
  }

  template<bool is_read>
  __attribute__((always_inline))
  void check_race_with_prev_write(const csi_id_t acc_id, MAType_t type,
                                  uintptr_t addr, size_t mem_size,
                                  bool on_stack, FrameData_t *f) const {
    using QITy = SimpleDictionary<WriteMAAllocator>::Query_iterator;
    QITy QI = Writes.getQueryIterator(addr, mem_size);
    // check_race(QI, false, is_read, acc_id, addr, mem_size, on_stack, f,
    //            call_stack);
    check_race<QITy, false, is_read>(QI, acc_id, type, addr, mem_size, on_stack,
                                     f);
  }

  template <typename UITy, bool with_read>
  __attribute__((always_inline))
  void update(UITy &UI, const csi_id_t acc_id, MAType_t type, uintptr_t addr,
              size_t mem_size, bool on_stack, FrameData_t *f) {
    // std::cerr << "update " << (with_read ? "read " : "write ")
    //           << reinterpret_cast<void*>(addr) << ", " << mem_size << "\n";
    while (!UI.isEnd()) {
      // Find a previous access
      MemoryAccess_t *PrevAccess = UI.get();
      if (!PrevAccess || !PrevAccess->isValid()) {
        // std::cerr << "update::insert: " << reinterpret_cast<void*>(UI.getAddress())
        //           << ", call_stack " << (void*)(call_stack.getTail()) << "\n";
        // This is the first access to this location.
        UI.insert(MemoryAccess_t(f->getSbagForAccess(), acc_id, type));
      } else {
        auto Func = PrevAccess->getFunc();
        cilksan_assert(Func);
        SPBagInterface *lastRSet = Func->get_set_node();
        uintptr_t AccAddr = UI.getAddress();

        // replace it only if it is in series with this access, i.e., if it's
        // one of the following:
        // a) in a SBag
        // b) in a PBag but should have been replaced because the access is
        // actually on the newly allocated stack frame (i.e., cactus stack abstraction)
        if ((lastRSet->is_SBag() &&
             !f->check_parallel_iter(lastRSet, PrevAccess->getVersion())) ||
            (on_stack && lastRSet->get_rsp() >= AccAddr)) {
          // std::cerr << "update::insert: " << reinterpret_cast<void*>(UI.getAddress())
          //           << ", call_stack " << (void*)(call_stack.getTail()) << "\n";
          UI.insert(MemoryAccess_t(f->getSbagForAccess(), acc_id, type));
        } else
          UI.next();
      }
    }    
  }

  __attribute__((always_inline))
  void update_with_read(const csi_id_t acc_id, uintptr_t addr, size_t mem_size,
                        bool on_stack, FrameData_t *f) {
    using UITy = SimpleDictionary<ReadMAAllocator>::Update_iterator;
    UITy UI = Reads.getUpdateIterator(addr, mem_size);
    update<UITy, true>(UI, acc_id, MAType_t::RW, addr, mem_size, on_stack, f);
  }

  __attribute__((always_inline))
  void update_with_write(const csi_id_t acc_id, MAType_t type, uintptr_t addr,
                         size_t mem_size, bool on_stack, FrameData_t *f) {
    using UITy = SimpleDictionary<WriteMAAllocator>::Update_iterator;
    UITy UI =  Writes.getUpdateIterator(addr, mem_size);
    update<UITy, false>(UI, acc_id, type, addr, mem_size, on_stack, f);
  }

  __attribute__((always_inline))
  void check_and_update_write(const csi_id_t acc_id, MAType_t type,
                              uintptr_t addr, size_t mem_size, bool on_stack,
                              FrameData_t *f) {
    // std::cerr << "check_and_update_write "
    //           << reinterpret_cast<void*>(addr) << ", " << mem_size << "\n";
    SimpleDictionary<WriteMAAllocator>::Update_iterator UI =
      Writes.getUpdateIterator(addr, mem_size);

    while (!UI.isEnd()) {
      // Find a previous access
      MemoryAccess_t *PrevAccess = UI.get();
      if (!PrevAccess || !PrevAccess->isValid()) {
        // std::cerr << "CAUW::insert: " << reinterpret_cast<void*>(UI.getAddress())
        //           << ", call_stack " << (void*)(call_stack.getTail()) << "\n";
        // This is the first access to this location.
        UI.insert(MemoryAccess_t(f->getSbagForAccess(), acc_id, type));
      } else {
        auto Func = PrevAccess->getFunc();
        cilksan_assert(Func);
        // std::cerr << "PrevAccess Func node: " << (void*)Func->get_node() << "\n";
        SPBagInterface *LCA = Func->get_set_node();
        uintptr_t AccAddr = UI.getAddress();

        // Check for races
        if (LCA->is_PBag() ||
            f->check_parallel_iter(LCA, PrevAccess->getVersion())) {
          // If memory is allocated on stack, the accesses race with each other
          // only if the mem location is allocated in shared ancestor's stack.
          // We don't need to check for this because we clear shadow memory;
          // non-shared stack can't race because earlier one would've been
          // cleared.

          // Try to get information on the allocation for this memory access.
          auto AllocFind = Allocs.find(AccAddr);
          AccessLoc_t AllocAccess =
            (nullptr == AllocFind) ? AccessLoc_t() : AllocFind->getLoc();
            
          // Report the race
          CilkSanImpl.report_race(
              PrevAccess->getLoc(),
              AccessLoc_t(acc_id, type, CilkSanImpl.get_current_call_stack()),
              AllocAccess, AccAddr, WW_RACE);
        }

        // Update the table
        //
        // replace it only if it is in series with this access, i.e., if it's
        // one of the following:
        // a) in a SBag
        // b) in a PBag but should have been replaced because the access is
        // actually on the newly allocated stack frame (i.e., cactus stack abstraction)
        if ((LCA->is_SBag() &&
             !f->check_parallel_iter(LCA, PrevAccess->getVersion())) ||
            (on_stack && LCA->get_rsp() >= AccAddr)) {
          // std::cerr << "CAUW::insert: " << reinterpret_cast<void*>(UI.getAddress())
          //           << ", call_stack " << (void*)(call_stack.getTail()) << "\n";
          UI.insert(MemoryAccess_t(f->getSbagForAccess(), acc_id, type));
        } else
          UI.next();
      }
    }    
  }

  __attribute__((always_inline))
  void clear(size_t start, size_t size) {
    Reads.clear(start, size);
    Writes.clear(start, size);
  }

  void record_alloc(size_t start, size_t size, FrameData_t *f,
                    csi_id_t alloca_id) {
    Allocs.set(start, size, MemoryAccess_t(f->getSbagForAccess(),
                                           alloca_id, MAType_t::ALLOC));
  }

  void record_free(size_t start, size_t size, FrameData_t *f,
                   csi_id_t free_id, MAType_t type) {
    Allocs.clear(start, size);
    Writes.set(start, size, MemoryAccess_t(f->getSbagForAccess(),
                                           free_id, type));
  }

  __attribute__((always_inline))
  void clear_alloc(size_t start, size_t size) {
    Allocs.clear(start, size);
  }

  void destruct() {}
};


void Shadow_Memory::init(CilkSanImpl_t &CilkSanImpl) {
  shadow_mem = new SimpleShadowMem(CilkSanImpl);
}

// Inserts access, and replaces any that are already in the shadow memory.
template<bool is_read>
void Shadow_Memory::insert_access(const csi_id_t acc_id, uintptr_t addr,
                                  size_t mem_size, FrameData_t *f) {
  shadow_mem->insert_access<is_read>(acc_id, addr, mem_size, f);
}

// Returns true if ANY bytes between addr and addr+mem_size are in the shadow
// memory.
template<bool is_read>
__attribute__((always_inline))
bool Shadow_Memory::does_access_exists(uintptr_t addr, size_t mem_size) const {
  return shadow_mem->does_access_exists<is_read>(addr, mem_size);
}

__attribute__((always_inline))
void Shadow_Memory::clear(size_t start, size_t size) {
  shadow_mem->clear(start, size);
}

void Shadow_Memory::record_alloc(size_t start, size_t size, FrameData_t *f,
                                 csi_id_t alloca_id) {
  shadow_mem->record_alloc(start, size, f, alloca_id);
}

void Shadow_Memory::record_free(size_t start, size_t size, FrameData_t *f,
                                csi_id_t free_id, MAType_t type) {
  shadow_mem->record_free(start, size, f, free_id, type);
}

__attribute__((always_inline))
void Shadow_Memory::clear_alloc(size_t start, size_t size) {
  shadow_mem->clear_alloc(start, size);
}

void Shadow_Memory::check_race_with_prev_read(const csi_id_t acc_id,
                                              uintptr_t addr, size_t mem_size,
                                              bool on_stack,
                                              FrameData_t *f) const {
  shadow_mem->check_race_with_prev_read(acc_id, addr, mem_size, on_stack, f);
}

template<bool is_read>
void Shadow_Memory::check_race_with_prev_write(const csi_id_t acc_id,
                                               MAType_t type, uintptr_t addr,
                                               size_t mem_size, bool on_stack,
                                               FrameData_t *f) const {
  shadow_mem->check_race_with_prev_write<is_read>(acc_id, type, addr, mem_size,
                                                  on_stack, f);
}

__attribute__((always_inline))
void Shadow_Memory::update_with_write(const csi_id_t acc_id, MAType_t type,
                                      uintptr_t addr, size_t mem_size,
                                      bool on_stack, FrameData_t *f) {
  shadow_mem->update_with_write(acc_id, type, addr, mem_size, on_stack, f);
}

__attribute__((always_inline))
void Shadow_Memory::update_with_read(const csi_id_t acc_id, uintptr_t addr,
                                     size_t mem_size, bool on_stack,
                                     FrameData_t *f) {
  shadow_mem->update_with_read(acc_id, addr, mem_size, on_stack, f);
}

__attribute__((always_inline))
void Shadow_Memory::check_and_update_write(const csi_id_t acc_id, MAType_t type,
                                           uintptr_t addr, size_t mem_size,
                                           bool on_stack, FrameData_t *f) {
  shadow_mem->check_and_update_write(acc_id, type, addr, mem_size, on_stack, f);
}

void Shadow_Memory::destruct() {
  if (shadow_mem) {
    delete shadow_mem;
    shadow_mem = nullptr;
  }
}

#endif // __SIMPLE_SHADOW_MEM__
