// Checks that the debugging API returns correct shadow scale and offset.
// RUN: %clangxx_asan -O %s -o %t
// RUN: %env_asan_opts=verbosity=1 %run %t 2>&1 | FileCheck %s

#include <sanitizer/asan_interface.h>
#include <stdio.h>
#include <stdlib.h>

#if _WIN64
#define PTR "%llx"
#else
#define PTR "%lx"
#endif

// printed because of verbosity=1
// CHECK: SHADOW_SCALE: [[SCALE:[0-9]+]]
// CHECK: SHADOW_OFFSET: 0x{{0*}}[[OFFSET:[0-9a-f]+]]

int main() {
  size_t scale, offset;
  __asan_get_shadow_mapping(&scale, &offset);

  fprintf(stderr, "scale: %d\n", (int)scale);
  fprintf(stderr, "offset: 0x" PTR "\n", (void*)offset);

  // CHECK: scale: [[SCALE]]
  // CHECK: offset: 0x{{0*}}[[OFFSET]]

  return 0;
}
