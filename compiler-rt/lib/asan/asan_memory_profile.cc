//===-- asan_memory_profile.cc.cc -----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// This file implements __sanitizer_print_memory_profile.
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_stackdepot.h"
#include "sanitizer_common/sanitizer_stacktrace.h"
#include "sanitizer_common/sanitizer_stoptheworld.h"
#include "lsan/lsan_common.h"
#include "asan/asan_allocator.h"

#if CAN_SANITIZE_LEAKS

namespace __asan {

struct AllocationSite {
  u32 id;
  uptr total_size;
  uptr count;
};

class HeapProfile {
 public:
  HeapProfile() : allocations_(1024) {}
  void Insert(u32 id, uptr size) {
    total_allocated_ += size;
    total_count_++;
    // Linear lookup will be good enough for most cases (although not all).
    for (uptr i = 0; i < allocations_.size(); i++) {
      if (allocations_[i].id == id) {
        allocations_[i].total_size += size;
        allocations_[i].count++;
        return;
      }
    }
    allocations_.push_back({id, size, 1});
  }

  void Print(uptr top_percent) {
    InternalSort(&allocations_, allocations_.size(),
                 [](const AllocationSite &a, const AllocationSite &b) {
                   return a.total_size > b.total_size;
                 });
    CHECK(total_allocated_);
    uptr total_shown = 0;
    Printf("Live Heap Allocations: %zd bytes from %zd allocations; "
           "showing top %zd%%\n", total_allocated_, total_count_, top_percent);
    for (uptr i = 0; i < allocations_.size(); i++) {
      auto &a = allocations_[i];
      Printf("%zd byte(s) (%zd%%) in %zd allocation(s)\n", a.total_size,
             a.total_size * 100 / total_allocated_, a.count);
      StackDepotGet(a.id).Print();
      total_shown += a.total_size;
      if (total_shown * 100 / total_allocated_ > top_percent)
        break;
    }
  }

 private:
  uptr total_allocated_ = 0;
  uptr total_count_ = 0;
  InternalMmapVector<AllocationSite> allocations_;
};

static void ChunkCallback(uptr chunk, void *arg) {
  HeapProfile *hp = reinterpret_cast<HeapProfile*>(arg);
  AsanChunkView cv = FindHeapChunkByAllocBeg(chunk);
  if (!cv.IsAllocated()) return;
  u32 id = cv.GetAllocStackId();
  if (!id) return;
  hp->Insert(id, cv.UsedSize());
}

static void MemoryProfileCB(const SuspendedThreadsList &suspended_threads_list,
                            void *argument) {
  HeapProfile hp;
  __lsan::ForEachChunk(ChunkCallback, &hp);
  hp.Print(reinterpret_cast<uptr>(argument));
}

}  // namespace __asan

extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE
void __sanitizer_print_memory_profile(uptr top_percent) {
  __sanitizer::StopTheWorld(__asan::MemoryProfileCB, (void*)top_percent);
}
}  // extern "C"

#endif  // CAN_SANITIZE_LEAKS
