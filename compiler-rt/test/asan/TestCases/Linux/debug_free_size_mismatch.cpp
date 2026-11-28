// Check that __asan_get_report_dealloc_address handles free-size-mismatch, the
// error reported by free_sized and free_aligned_sized. ASan does not intercept
// these on Darwin and Windows.

// RUN: %clangxx_asan -O0 %s -o %t
// RUN: %env_asan_opts=free_size_mismatch=1 not %run %t free_sized 2>&1 | \
// RUN:   FileCheck %s -DFN=free_sized
// RUN: %env_asan_opts=free_size_mismatch=1 not %run %t free_aligned_sized \
// RUN:   2>&1 | FileCheck %s -DFN=free_aligned_sized

/// aligned_alloc is not supported on android's libc.
// UNSUPPORTED: android

#include <sanitizer/asan_interface.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" void free_sized(void *p, size_t size);
extern "C" void free_aligned_sized(void *p, size_t alignment, size_t size);

extern "C" void __asan_on_error() {
  const void *addr = nullptr;
  size_t size = 0xbad;
  int found = __asan_get_report_dealloc_address(&addr, &size);
  fprintf(stderr, "description: %s\n", __asan_get_report_description());
  fprintf(stderr, "found: %d, addr: %p, size: %zu\n", found, addr, size);
}

int main(int argc, char **argv) {
  if (argc != 2)
    return 1;
  bool aligned = !strcmp(argv[1], "free_aligned_sized");

  void *p = aligned ? aligned_alloc(128, 256) : malloc(256);
  if (!p)
    return 1;
  fprintf(stderr, "p: %p\n", p);
  // CHECK: p: 0x[[ADDR:[0-9a-f]+]]

  if (aligned)
    free_aligned_sized(p, 128, 64);
  else
    free_sized(p, 64);
  // CHECK: description: free-size-mismatch
  // CHECK: found: 1, addr: 0x[[ADDR]], size: 64
  // CHECK: ERROR: AddressSanitizer: free-size-mismatch on {{0x0*}}[[ADDR]]
  // CHECK: object passed to [[FN]] has wrong size or alignment:

  return 0;
}
