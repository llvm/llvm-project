// Test thread_local_quarantine_size_kb

// RUN: %clangxx_asan  %s -o %t
// RUN: %env_asan_opts=thread_local_quarantine_size_kb=256:verbosity=1 %run %t 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-VALUE
// RUN: %env_asan_opts=thread_local_quarantine_size_kb=64:quarantine_size_mb=64 %run %t 2>&1 | \
// RUN:   FileCheck %s --allow-empty --check-prefix=CHECK-SMALL-LOCAL-CACHE-SMALL-OVERHEAD
// RUN: %env_asan_opts=thread_local_quarantine_size_kb=0:quarantine_size_mb=64 %run %t 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-NO-LOCAL-CACHE-HUGE-OVERHEAD

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sanitizer/allocator_interface.h>

// The idea is allocate a lot of small blocks, totaling 5Mb of user memory
// total, and verify that quarantine does not incur too much memory overhead.
// There's always an overhead for red zones, shadow memory and such, but
// quarantine accounting should not significantly contribute to that.
static const int kNumAllocs = 20000;
static const int kAllocSize = 256;
static const size_t kHeapSizeLimit = 12 << 20;

int main() {
  size_t old_heap_size = __sanitizer_get_heap_size();
  for (int i = 0; i < kNumAllocs; i++) {
    char *g = new char[kAllocSize];
    memset(g, -1, kAllocSize);
    delete [] (g);
  }
  size_t new_heap_size = __sanitizer_get_heap_size();
  fprintf(stderr, "heap size: new: %zd old: %zd\n", new_heap_size,
          old_heap_size);
  if (new_heap_size - old_heap_size > kHeapSizeLimit)
    fprintf(stderr, "Heap size limit exceeded");
}

// CHECK-VALUE: thread_local_quarantine_size_kb=256K
// CHECK-SMALL-LOCAL-CACHE-SMALL-OVERHEAD-NOT: Heap size limit exceeded
// CHECK-NO-LOCAL-CACHE-HUGE-OVERHEAD: Heap size limit exceeded
