// double_free_max_entries is validated at startup. A negative value falls back
// to the unlimited table and an oversized one is capped, both with a warning,
// rather than reaching mmap as a multi-gigabyte request. Detection keeps
// working either way.
//
// RUN: %clangxx_lsan -O0 %s -o %t
// RUN: %env_lsan_opts=detect_leaks=0:detect_double_free=1:double_free_max_entries=-1 not %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-NEGATIVE
// RUN: %env_lsan_opts=detect_leaks=0:detect_double_free=1:double_free_max_entries=2000000000 not %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-HUGE
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

// CHECK-NEGATIVE: WARNING: LeakSanitizer: double_free_max_entries=-1 is negative
// CHECK-HUGE: WARNING: LeakSanitizer: double_free_max_entries=2000000000 is too large, capping it at 4194304
// CHECK: ERROR: LeakSanitizer: attempting double-free on
// CHECK: SUMMARY: LeakSanitizer: double-free
