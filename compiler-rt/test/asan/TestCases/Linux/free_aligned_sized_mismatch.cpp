// RUN: %clangxx_asan -O0 %s -o %t
// RUN: %env_asan_opts=free_size_mismatch=1 not %run %t 128 128 2>&1 | \
// RUN:  FileCheck %s --check-prefixes=CHECK,CHECK-SIZE
// RUN: %env_asan_opts=free_size_mismatch=1 not %run %t 64 256 2>&1 | \
// RUN:  FileCheck %s --check-prefixes=CHECK,CHECK-ALIGN
// RUN: %env_asan_opts=free_size_mismatch=1 not %run %t 64 128 2>&1 | \
// RUN:  FileCheck %s --check-prefixes=CHECK,CHECK-ALIGN,CHECK-SIZE

/// Checks disabled here.
// RUN: %env_asan_opts=free_size_mismatch=0 %run %t 128 128
// RUN: %env_asan_opts=free_size_mismatch=0 %run %t 64 256
// RUN: %env_asan_opts=free_size_mismatch=0 %run %t 64 128

/// The test should not fail for these cases since the size and alignment match.
// RUN: %env_asan_opts=free_size_mismatch=1 %run %t 128 256

#include <stdio.h>
#include <stdlib.h>

extern "C" void free_aligned_sized(void *p, size_t alignment, size_t size);

int main(int argc, char **argv) {
  if (argc != 3)
    return 1;

  size_t free_alignment = atoi(argv[1]);
  size_t free_size = atoi(argv[2]);

  // TODO: free_aligned_sized is only usable with aligned_alloc, but this isn't
  // checked here, so we can probably add support for checking that later.
  void *p = aligned_alloc(128, 256);
  if (!p)
    return 1;

  free_aligned_sized(p, free_alignment, free_size);
  // CHECK:       ERROR: AddressSanitizer: free-size-mismatch on
  // CHECK:         object passed to free_aligned_sized has wrong size or alignment:
  // CHECK-SIZE:    size of the allocation:   256 bytes;
  // CHECK-SIZE:    size of the deallocation: 128 bytes.
  // CHECK-ALIGN:   alignment of the allocation:   128 bytes;
  // CHECK-ALIGN:   alignment of the deallocation:  64 bytes.
  // CHECK:         is located 0 bytes inside of 256-byte region
  // CHECK:         allocated by thread T0 here:
  // CHECK:         #0{{.*}}aligned_alloc

  return 0;
}
