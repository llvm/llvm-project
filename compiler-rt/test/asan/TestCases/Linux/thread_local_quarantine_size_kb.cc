// Test thread_local_quarantine_size_kb

// RUN: %clangxx_asan  %s -o %t
// RUN: %env_asan_opts=thread_local_quarantine_size_kb=64:quarantine_size_mb=64:verbosity=1 %run %t 2>&1 | \
// RUN:   FileCheck %s --allow-empty --check-prefix=CHECK-SMALL-LOCAL-CACHE-SMALL-OVERHEAD
// RUN: %env_asan_opts=thread_local_quarantine_size_kb=0:quarantine_size_mb=0 %run %t 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-QUARANTINE-DISABLED-SMALL-OVERHEAD
// RUN: %env_asan_opts=thread_local_quarantine_size_kb=0:quarantine_size_mb=64 not %run %t 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-FOR-PARAMETER-ERROR

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sanitizer/allocator_interface.h>

// The idea is allocate a lot of small blocks, totaling 5Mb of user memory,
// and verify that quarantine does not incur too much memory overhead.
// There's always an overhead for red zones, shadow memory and such, but
// quarantine accounting should not significantly contribute to that.
// The zero sized thread local cache is specifically tested since it used to
// generate a huge overhead of almost empty quarantine batches.
static const size_t kHeapSizeIncrementLimit = 12 << 20;
static const int kNumAllocs = 20000;
static const int kAllocSize = 256;

int main() {
  size_t heap_size = __sanitizer_get_heap_size();
  fprintf(stderr, "Heap size: %zd\n", heap_size);

  for (int i = 0; i < kNumAllocs; i++) {
    char *temp = new char[kAllocSize];
    memset(temp, -1, kAllocSize);
    delete [] (temp);
  }

  size_t new_heap_size = __sanitizer_get_heap_size();
  fprintf(stderr, "New heap size: %zd\n", new_heap_size);
  if (new_heap_size - heap_size < kHeapSizeIncrementLimit)
    fprintf(stderr, "Heap growth is within limits\n");
}

// CHECK-SMALL-LOCAL-CACHE-SMALL-OVERHEAD: thread_local_quarantine_size_kb=64K
// CHECK-SMALL-LOCAL-CACHE-SMALL-OVERHEAD: Heap growth is within limits
// CHECK-QUARANTINE-DISABLED-SMALL-OVERHEAD: Heap growth is within limits
// CHECK-FOR-PARAMETER-ERROR: thread_local_quarantine_size_kb can be set to 0 only when quarantine_size_mb is set to 0
