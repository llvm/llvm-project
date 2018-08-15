// RUN: %clangxx_hwasan -DSIZE=16 -O0 %s -o %t && %run %t 2>&1 | FileCheck %s

// REQUIRES: stable-runtime

#include <assert.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sanitizer/hwasan_interface.h>

int main() {
  char *p = (char *)mmap(nullptr, 4096, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, 0, 0);
  assert(p);

  __hwasan_tag_memory(p, 1, 32);
  __hwasan_tag_memory(p + 32, 3, 16);
  __hwasan_tag_memory(p + 48, 0, 32);
  __hwasan_tag_memory(p + 80, 4, 16);

  char *q = (char *)__hwasan_tag_pointer(p, 7);
  __hwasan_print_shadow(q + 5, 89 - 5);
  // CHECK:      HWASan shadow map for {{.*}}5 .. {{.*}}9 (pointer tag 7)
  // CHECK-NEXT:   {{.*}}0: 1
  // CHECK-NEXT:   {{.*}}0: 1
  // CHECK-NEXT:   {{.*}}0: 3
  // CHECK-NEXT:   {{.*}}0: 0
  // CHECK-NEXT:   {{.*}}0: 0
  // CHECK-NEXT:   {{.*}}0: 4
}
