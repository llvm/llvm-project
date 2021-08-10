// RUN: %clang_hwasan %s -o %t && not %run %t 2>&1 | FileCheck %s

// REQUIRES: stable-runtime

#include <stdlib.h>
#include <stdio.h>
#include <sanitizer/hwasan_interface.h>

int main() {
  __hwasan_enable_allocator_tagging();
  char * volatile x = (char*)malloc(40);
  free(x);
  free(x);
  // CHECK: ERROR: HWAddressSanitizer: invalid-free on address {{.*}} at pc {{[0x]+}}[[PC:.*]] on thread T{{[0-9]+}}
  // CHECK: tags: [[PTR_TAG:..]]/[[MEM_TAG:..]] (ptr/mem)
  // CHECK: #0 {{[0x]+}}{{.*}}[[PC]] in free
  // CHECK: freed by thread {{.*}} here:
  // CHECK: previously allocated here:
  // CHECK: Memory tags around the buggy address (one tag corresponds to 16 bytes):
  // CHECK: =>{{.*}}[[MEM_TAG]]
  fprintf(stderr, "DONE\n");
  __hwasan_disable_allocator_tagging();
// CHECK-NOT: DONE
}
