// Basic double-free detection for a chunk served by the primary allocator.
//
// The double free is undefined behavior, so build at -O0 and route the frees
// through an opaque function: an optimizing compiler is otherwise allowed to
// drop the second free() before LeakSanitizer can observe it.
//
// RUN: %clangxx_lsan -O0 %s -o %t
// RUN: %env_lsan_opts=detect_leaks=0 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-OFF
// RUN: %env_lsan_opts=detect_leaks=0:detect_double_free=0 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-OFF
// RUN: %env_lsan_opts=detect_leaks=0:detect_double_free=1 not %run %t 2>&1 | FileCheck %s
// RUN: %env_lsan_opts=detect_leaks=0:detect_double_free=1:double_free_max_entries=0 not %run %t 2>&1 | FileCheck %s
// REQUIRES: lsan-standalone
// UNSUPPORTED: darwin, target={{.*netbsd.*}}

#include <cstdio>
#include <cstdlib>

__attribute__((noinline)) static void Free(void *p) { free(p); }

int main() {
  void *p = malloc(16);
  Free(p);
  Free(p);
  puts("completed");
  return 0;
}

// The reported stack starts at the intercepted free(), not inside the runtime
// helper that captures it.
// CHECK: ERROR: LeakSanitizer: attempting double-free on
// CHECK: The second free occurred here:
// CHECK-NEXT: #0 {{.*}}free
// CHECK: The first free occurred here:
// CHECK: The memory was allocated here:
// CHECK: SUMMARY: LeakSanitizer: double-free

// CHECK-OFF: completed
// CHECK-OFF-NOT: double-free
