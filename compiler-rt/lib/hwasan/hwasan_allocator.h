//===-- hwasan_allocator.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of HWAddressSanitizer.
//
//===----------------------------------------------------------------------===//

#ifndef HWASAN_ALLOCATOR_H
#define HWASAN_ALLOCATOR_H

#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_ring_buffer.h"

namespace __hwasan {

struct HwasanThreadLocalMallocStorage {
  // Allocator cache contains atomic_uint64_t which must be 8-byte aligned.
  ALIGNED(8) uptr allocator_cache[96 * (512 * 8 + 16)];  // Opaque.
  void CommitBack();

 private:
  // These objects are allocated via mmap() and are zero-initialized.
  HwasanThreadLocalMallocStorage() {}
};

struct Metadata;

class HwasanChunkView {
 public:
  HwasanChunkView() : block_(0), metadata_(nullptr) {}
  HwasanChunkView(uptr block, Metadata *metadata)
      : block_(block), metadata_(metadata) {}
  bool IsValid() const;        // Checks if it points to a valid allocated chunk
  bool IsAllocated() const;    // Checks if the memory is currently allocated
  uptr Beg() const;            // First byte of user memory
  uptr End() const;            // Last byte of user memory
  uptr UsedSize() const;       // Size requested by the user
  u32 GetAllocStackId() const;
  u32 GetFreeStackId() const;
 private:
  uptr block_;
  Metadata *const metadata_;
};

HwasanChunkView FindHeapChunkByAddress(uptr address);

// Information about one (de)allocation that happened in the past.
// These are recorded in a thread-local ring buffer.
struct HeapAllocationRecord {
  uptr tagged_addr;
  u32  free_context_id;
  u32  requested_size;
};

typedef RingBuffer<HeapAllocationRecord> HeapAllocationsRingBuffer;


} // namespace __hwasan

#endif // HWASAN_ALLOCATOR_H
