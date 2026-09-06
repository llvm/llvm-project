// The side table for large allocations is bounded by double_free_max_entries
// and evicts in FIFO order, so a bounded table always remembers the most recent
// frees. With room for a single entry, freeing the newest address again is
// still reported as a double free, while the evicted address is no longer
// recognized as one and falls back to the invalid-free diagnostic.
//
// RUN: %clangxx_lsan -O0 %s -o %t
// RUN: %env_lsan_opts=detect_leaks=0:detect_double_free=1:double_free_max_entries=1 not %run %t remembered 2>&1 | FileCheck %s --check-prefix=CHECK-REMEMBERED
// RUN: %env_lsan_opts=detect_leaks=0:detect_double_free=1:double_free_max_entries=1 not %run %t evicted 2>&1 | FileCheck %s --check-prefix=CHECK-EVICTED
// REQUIRES: lsan-standalone
// UNSUPPORTED: darwin, target={{.*netbsd.*}}

#include <cstdio>
#include <cstdlib>
#include <cstring>

__attribute__((noinline)) static void Free(void *p) { free(p); }

int main(int argc, char **argv) {
  if (argc != 2)
    return 1;

  void *old_chunk = malloc(2 << 20);
  void *new_chunk = malloc(3 << 20);
  Free(old_chunk);
  // Only one record fits, so this evicts the one for old_chunk.
  Free(new_chunk);

  if (!strcmp(argv[1], "remembered")) {
    fprintf(stderr, "freeing remembered chunk\n");
    Free(new_chunk);
  } else {
    fprintf(stderr, "freeing evicted chunk\n");
    Free(old_chunk);
  }

  fprintf(stderr, "not reached\n");
  return 0;
}

// CHECK-REMEMBERED: freeing remembered chunk
// CHECK-REMEMBERED: ERROR: LeakSanitizer: attempting double-free on
// CHECK-REMEMBERED: SUMMARY: LeakSanitizer: double-free
// CHECK-REMEMBERED-NOT: not reached

// The evicted address is still diagnosed instead of being allowed to fault
// inside the runtime, just no longer as a double free.
// CHECK-EVICTED: freeing evicted chunk
// CHECK-EVICTED: ERROR: LeakSanitizer: attempting free on address which was not
// CHECK-EVICTED: SUMMARY: LeakSanitizer: bad-free
// CHECK-EVICTED-NOT: not reached
