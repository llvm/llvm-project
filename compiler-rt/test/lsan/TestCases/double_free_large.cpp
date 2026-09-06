// Double-free detection for a chunk served by the secondary allocator. Such a
// chunk is unmapped as soon as it is freed, so its metadata is gone and the
// second free can only be recognized from the side table.
//
// RUN: %clangxx_lsan -O0 %s -o %t
// RUN: %env_lsan_opts=detect_leaks=0:detect_double_free=1 not %run %t 2>&1 | FileCheck %s
// RUN: %env_lsan_opts=detect_leaks=0:detect_double_free=1:double_free_max_entries=0 not %run %t 2>&1 | FileCheck %s
// REQUIRES: lsan-standalone
// UNSUPPORTED: darwin, target={{.*netbsd.*}}

#include <cstdlib>

__attribute__((noinline)) static void Free(void *p) { free(p); }

int main() {
  void *p = malloc(2 << 20);
  Free(p);
  Free(p);
  return 0;
}

// CHECK: ERROR: LeakSanitizer: attempting double-free on
// CHECK: The second free occurred here:
// CHECK: The first free occurred here:
// CHECK: The memory was allocated here:
// CHECK: SUMMARY: LeakSanitizer: double-free
