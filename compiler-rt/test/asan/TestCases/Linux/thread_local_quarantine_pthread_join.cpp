// Test how creating and joining a lot of threads making only a few allocations
// each affect total quarantine (and overall heap) size.

// RUN: %clangxx_asan  %s -o %t
// RUN: %env_asan_opts=thread_local_quarantine_size_kb=64:quarantine_size_mb=1:allocator_release_to_os_interval_ms=-1 %run %t 2>&1 | \
// RUN:   FileCheck %s --allow-empty --check-prefix=CHECK-SMALL-LOCAL-CACHE-SMALL-OVERHEAD

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sanitizer/allocator_interface.h>

// Thread local quarantine is merged to the global one when thread exits and
// this scenario (a few allocations per thread) used to generate a huge overhead
// of practically empty quarantine batches (one per thread).
static const size_t kHeapSizeIncrementLimit = 2 << 20;
static const int kNumThreads = 2048;
// The allocation size is so small because all we want to test is that
// quarantine block merging process does not leak memory used for quarantine
// blocks.
// TODO(alekseyshl): Add more comprehensive test verifying quarantine size
// directly (requires quarantine stats exposed in allocator stats and API).
static const int kAllocSize = 1;

void *ThreadFn(void *unused) {
  char *temp = new char[kAllocSize];
  memset(temp, -1, kAllocSize);
  delete [] (temp);
  return NULL;
}

int main() {
  // Warm up all internal structures.
  pthread_t t;
  pthread_create(&t, 0, ThreadFn, 0);
  pthread_join(t, 0);

  size_t heap_size = __sanitizer_get_heap_size();
  fprintf(stderr, "Heap size: %zd\n", heap_size);

  for (int i = 0; i < kNumThreads; i++) {
    pthread_t t;
    pthread_create(&t, 0, ThreadFn, 0);
    pthread_join(t, 0);

    size_t new_heap_size = __sanitizer_get_heap_size();
  }

  size_t new_heap_size = __sanitizer_get_heap_size();
  fprintf(stderr, "New heap size: %zd\n", new_heap_size);
  if (new_heap_size - heap_size < kHeapSizeIncrementLimit)
    fprintf(stderr, "Heap growth is within limits\n");
}

// CHECK-SMALL-LOCAL-CACHE-SMALL-OVERHEAD: Heap growth is within limits
