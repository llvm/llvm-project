// free_sized() and free_aligned_sized() go through the same double-free path as
// free(), including the out-of-line stack capture, so they need the same
// coverage: a matching sized free must be accepted, and a second one must be
// reported.
//
// RUN: %clangxx_lsan -O0 %s -o %t
// RUN: %env_lsan_opts=detect_leaks=0:detect_double_free=1 %run %t ok 2>&1 | FileCheck %s --check-prefix=CHECK-OK
// RUN: %env_lsan_opts=detect_leaks=0:detect_double_free=1 not %run %t sized 2>&1 | FileCheck %s
// RUN: %env_lsan_opts=detect_leaks=0:detect_double_free=1 not %run %t aligned 2>&1 | FileCheck %s
// REQUIRES: lsan-standalone
// UNSUPPORTED: darwin, target={{.*netbsd.*}}

#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" void free_sized(void *p, size_t size);
extern "C" void free_aligned_sized(void *p, size_t alignment, size_t size);

__attribute__((noinline)) static void FreeSized(void *p, size_t size) {
  free_sized(p, size);
}

__attribute__((noinline)) static void
FreeAlignedSized(void *p, size_t alignment, size_t size) {
  free_aligned_sized(p, alignment, size);
}

int main(int argc, char **argv) {
  if (argc != 2)
    return 1;

  if (!strcmp(argv[1], "ok")) {
    FreeSized(malloc(64), 64);
    FreeAlignedSized(aligned_alloc(64, 128), 64, 128);
    fprintf(stderr, "completed\n");
    return 0;
  }

  if (!strcmp(argv[1], "sized")) {
    void *p = malloc(64);
    FreeSized(p, 64);
    FreeSized(p, 64);
  } else {
    void *p = aligned_alloc(64, 128);
    FreeAlignedSized(p, 64, 128);
    FreeAlignedSized(p, 64, 128);
  }

  fprintf(stderr, "not reached\n");
  return 0;
}

// CHECK-OK: completed
// CHECK-OK-NOT: LeakSanitizer:

// CHECK: ERROR: LeakSanitizer: attempting double-free on
// CHECK: SUMMARY: LeakSanitizer: double-free
// CHECK-NOT: not reached
