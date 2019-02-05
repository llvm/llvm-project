//===-- hwasan_thread_list.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of HWAddressSanitizer.
//
//===----------------------------------------------------------------------===//

// HwasanThreadList is a registry for live threads, as well as an allocator for
// HwasanThread objects and their stack history ring buffers. There are
// constraints on memory layout of the shadow region and CompactRingBuffer that
// are part of the ABI contract between compiler-rt and llvm.
//
// * Start of the shadow memory region is aligned to 2**kShadowBaseAlignment.
// * All stack ring buffers are located within (2**kShadowBaseAlignment)
// sized region below and adjacent to the shadow region.
// * Each ring buffer has a size of (2**N)*4096 where N is in [0, 8), and is
// aligned to twice its size. The value of N can be different for each buffer.
//
// These constrains guarantee that, given an address A of any element of the
// ring buffer,
//     A_next = (A + sizeof(uptr)) & ~((1 << (N + 13)) - 1)
//   is the address of the next element of that ring buffer (with wrap-around).
// And, with K = kShadowBaseAlignment,
//     S = (A | ((1 << K) - 1)) + 1
//   (align up to kShadowBaseAlignment) is the start of the shadow region.
//
// These calculations are used in compiler instrumentation to update the ring
// buffer and obtain the base address of shadow using only two inputs: address
// of the current element of the ring buffer, and N (i.e. size of the ring
// buffer). Since the value of N is very limited, we pack both inputs into a
// single thread-local word as
//   (1 << (N + 56)) | A
// See the implementation of class CompactRingBuffer, which is what is stored in
// said thread-local word.
//
// Note the unusual way of aligning up the address of the shadow:
//   (A | ((1 << K) - 1)) + 1
// It is only correct if A is not already equal to the shadow base address, but
// it saves 2 instructions on AArch64.

#include "hwasan.h"
#include "hwasan_allocator.h"
#include "hwasan_flags.h"
#include "hwasan_thread.h"

#include "sanitizer_common/sanitizer_placement_new.h"

namespace __hwasan {

static uptr RingBufferSize() {
  uptr desired_bytes = flags()->stack_history_size * sizeof(uptr);
  // FIXME: increase the limit to 8 once this bug is fixed:
  // https://bugs.llvm.org/show_bug.cgi?id=39030
  for (int shift = 1; shift < 7; ++shift) {
    uptr size = 4096 * (1ULL << shift);
    if (size >= desired_bytes)
      return size;
  }
  Printf("stack history size too large: %d\n", flags()->stack_history_size);
  CHECK(0);
  return 0;
}

struct ThreadListHead {
  Thread *list_;

  ThreadListHead() : list_(nullptr) {}

  void Push(Thread *t) {
    t->next_ = list_;
    list_ = t;
  }

  Thread *Pop() {
    Thread *t = list_;
    if (t)
      list_ = t->next_;
    return t;
  }

  void Remove(Thread *t) {
    Thread **cur = &list_;
    while (*cur != t) cur = &(*cur)->next_;
    CHECK(*cur && "thread not found");
    *cur = (*cur)->next_;
  }

  template <class CB>
  void ForEach(CB cb) {
    Thread *t = list_;
    while (t) {
      cb(t);
      t = t->next_;
    }
  }
};

struct ThreadStats {
  uptr n_live_threads;
  uptr total_stack_size;
};

class HwasanThreadList {
 public:
  HwasanThreadList(uptr storage, uptr size)
      : free_space_(storage), free_space_end_(storage + size) {
    // [storage, storage + size) is used as a vector of
    // thread_alloc_size_-sized, ring_buffer_size_*2-aligned elements.
    // Each element contains
    // * a ring buffer at offset 0,
    // * a Thread object at offset ring_buffer_size_.
    ring_buffer_size_ = RingBufferSize();
    thread_alloc_size_ =
        RoundUpTo(ring_buffer_size_ + sizeof(Thread), ring_buffer_size_ * 2);
  }

  Thread *CreateCurrentThread() {
    Thread *t;
    {
      SpinMutexLock l(&list_mutex_);
      t = free_list_.Pop();
      if (t) {
        uptr start = (uptr)t - ring_buffer_size_;
        internal_memset((void *)start, 0, ring_buffer_size_ + sizeof(Thread));
      } else {
        t = AllocThread();
      }
      live_list_.Push(t);
    }
    t->Init((uptr)t - ring_buffer_size_, ring_buffer_size_);
    AddThreadStats(t);
    return t;
  }

  void DontNeedThread(Thread *t) {
    uptr start = (uptr)t - ring_buffer_size_;
    ReleaseMemoryPagesToOS(start, start + thread_alloc_size_);
  }

  void ReleaseThread(Thread *t) {
    RemoveThreadStats(t);
    t->Destroy();
    SpinMutexLock l(&list_mutex_);
    live_list_.Remove(t);
    free_list_.Push(t);
    DontNeedThread(t);
  }

  Thread *GetThreadByBufferAddress(uptr p) {
    return (Thread *)(RoundDownTo(p, ring_buffer_size_ * 2) +
                      ring_buffer_size_);
  }

  uptr MemoryUsedPerThread() {
    uptr res = sizeof(Thread) + ring_buffer_size_;
    if (auto sz = flags()->heap_history_size)
      res += HeapAllocationsRingBuffer::SizeInBytes(sz);
    return res;
  }

  template <class CB>
  void VisitAllLiveThreads(CB cb) {
    SpinMutexLock l(&list_mutex_);
    live_list_.ForEach(cb);
  }

  void AddThreadStats(Thread *t) {
    SpinMutexLock l(&stats_mutex_);
    stats_.n_live_threads++;
    stats_.total_stack_size += t->stack_size();
  }

  void RemoveThreadStats(Thread *t) {
    SpinMutexLock l(&stats_mutex_);
    stats_.n_live_threads--;
    stats_.total_stack_size -= t->stack_size();
  }

  ThreadStats GetThreadStats() {
    SpinMutexLock l(&stats_mutex_);
    return stats_;
  }

 private:
  Thread *AllocThread() {
    uptr align = ring_buffer_size_ * 2;
    CHECK(IsAligned(free_space_, align));
    Thread *t = (Thread *)(free_space_ + ring_buffer_size_);
    free_space_ += thread_alloc_size_;
    CHECK(free_space_ <= free_space_end_ && "out of thread memory");
    return t;
  }

  uptr free_space_;
  uptr free_space_end_;
  uptr ring_buffer_size_;
  uptr thread_alloc_size_;

  ThreadListHead free_list_;
  ThreadListHead live_list_;
  SpinMutex list_mutex_;

  ThreadStats stats_;
  SpinMutex stats_mutex_;
};

void InitThreadList(uptr storage, uptr size);
HwasanThreadList &hwasanThreadList();

} // namespace
