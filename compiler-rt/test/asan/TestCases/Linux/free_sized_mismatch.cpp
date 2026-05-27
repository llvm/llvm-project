// RUN: %clangxx_asan -O0 %s -o %t
// RUN: %env_asan_opts=free_size_mismatch=1 not %run %t 1 32 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CHECK,CHECK-MALLOC
// RUN: %env_asan_opts=free_size_mismatch=1 not %run %t 2 32 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CHECK,CHECK-CALLOC
// RUN: %env_asan_opts=free_size_mismatch=1 not %run %t 3 32 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CHECK,CHECK-REALLOC

/// Checks disabled here.
// RUN: %env_asan_opts=free_size_mismatch=0 %run %t 1 32
// RUN: %env_asan_opts=free_size_mismatch=0 %run %t 2 32
// RUN: %env_asan_opts=free_size_mismatch=0 %run %t 3 32

/// The test should not fail for these cases since the sizes match.
// RUN: %env_asan_opts=free_size_mismatch=1 %run %t 1 64
// RUN: %env_asan_opts=free_size_mismatch=1 %run %t 2 64
// RUN: %env_asan_opts=free_size_mismatch=1 %run %t 3 64

#include <stdio.h>
#include <stdlib.h>

extern "C" void free_sized(void *p, size_t size);

int main(int argc, char **argv) {
  if (argc != 3)
    return 1;
  int alloc_type = atoi(argv[1]);

  void *p = nullptr;
  if (alloc_type == 1) {
    p = malloc(64);
  } else if (alloc_type == 2) {
    p = calloc(8, 8);
  } else if (alloc_type == 3) {
    p = malloc(32);
    p = realloc(p, 64);
  }

  if (!p)
    return 1;

  size_t free_size = atoi(argv[2]);
  free_sized(p, free_size);
  // CHECK:        ERROR: AddressSanitizer: free-size-mismatch on
  // CHECK:          object passed to free_sized has wrong size or alignment:
  // CHECK:          size of the allocation:   64 bytes;
  // CHECK:          size of the deallocation: 32 bytes.
  // CHECK:          is located 0 bytes inside of 64-byte region
  // CHECK:          allocated by thread T0 here:
  // CHECK-MALLOC:   #0{{.*}}malloc
  // CHECK-CALLOC:   #0{{.*}}calloc
  // CHECK-REALLOC:  #0{{.*}}realloc

  return 0;
}
