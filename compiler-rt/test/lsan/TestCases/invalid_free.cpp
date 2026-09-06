// An invalid free must produce a diagnostic instead of faulting inside the
// runtime while it dereferences metadata for an address the allocator never
// handed out.
//
// RUN: %clangxx_lsan -O0 %s -o %t
// RUN: %env_lsan_opts=detect_leaks=0:detect_double_free=1 not %run %t heap 2>&1 | FileCheck %s
// RUN: %env_lsan_opts=detect_leaks=0:detect_double_free=1 not %run %t global 2>&1 | FileCheck %s
// REQUIRES: lsan-standalone
// UNSUPPORTED: darwin, target={{.*netbsd.*}}

#include <cstdio>
#include <cstdlib>
#include <cstring>

static long g_global[8];

__attribute__((noinline)) static void Free(void *p) { free(p); }

int main(int argc, char **argv) {
  if (argc != 2)
    return 1;

  void *p;
  if (!strcmp(argv[1], "heap")) {
    // Interior pointer into a live allocation.
    p = (char *)malloc(64) + 16;
  } else {
    p = g_global;
  }

  fprintf(stderr, "freeing invalid pointer\n");
  Free(p);
  fprintf(stderr, "not reached\n");
  return 0;
}

// CHECK: freeing invalid pointer
// CHECK: ERROR: LeakSanitizer: attempting free on address which was not
// CHECK: SUMMARY: LeakSanitizer: bad-free
// CHECK-NOT: not reached
