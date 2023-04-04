// RUN: %clang_hwasan %s -o %t
// RUN: not %run %t 50 2>&1 | FileCheck %s
// RUN: not %run %t 40 2>&1 | FileCheck %s
// RUN: not %run %t 30 2>&1 | FileCheck %s

// REQUIRES: stable-runtime

#include <stdlib.h>
#include <stdio.h>
#include <sanitizer/hwasan_interface.h>

int main(int argc, char **argv) {
  __hwasan_enable_allocator_tagging();
  if (argc != 2) return 0;
  int realloc_size = atoi(argv[1]);
  char * volatile x = (char*)malloc(40);
  free(x);
  x = realloc(x, realloc_size);
  // CHECK: ERROR: HWAddressSanitizer: invalid-free on address
  // CHECK: tags: [[PTR_TAG:..]]/[[MEM_TAG:..]] (ptr/mem)
  // CHECK: freed by thread {{.*}} here:
  // CHECK: previously allocated by thread {{.*}} here:
  // CHECK: Memory tags around the buggy address (one tag corresponds to 16 bytes):
  // CHECK: =>{{.*}}[[MEM_TAG]]
  fprintf(stderr, "DONE\n");
  __hwasan_disable_allocator_tagging();
// CHECK-NOT: DONE
}
