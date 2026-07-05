// RUN: %clangxx_hwasan -DSIZE=16 -O0 %s -o %t && %run %t 2>&1 | FileCheck %s

#include <assert.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sanitizer/hwasan_interface.h>

int main() {
  char *alloc = (char *)malloc(4096);

  // Simulate short granule tags.
  alloc[15] = 0x00;
  alloc[31] = 0xbb;
  alloc[47] = 0xcc;
  alloc[63] = 0xdd;
  alloc[79] = 0xee;
  alloc[95] = 0xff;

  char *p = alloc;

  // Write tags to shadow.
  __hwasan_tag_memory(p, 1, 32);
  __hwasan_tag_memory(p + 32, 16, 16);
  __hwasan_tag_memory(p + 48, 0, 32);
  __hwasan_tag_memory(p + 80, 4, 16);

  char *q = (char *)__hwasan_tag_pointer(p, 7);
  __hwasan_print_shadow(q + 5, 89 - 5);
  // CHECK:      HWASan shadow map for {{.*}}5 .. {{.*}}9 (pointer tag 7)
  // CHECK-NEXT:   {{.*}}0: 01(00)
  // CHECK-NEXT:   {{.*}}0: 01(bb)
  // CHECK-NEXT:   {{.*}}0: 10
  // CHECK-NEXT:   {{.*}}0: 00
  // CHECK-NEXT:   {{.*}}0: 00
  // CHECK-NEXT:   {{.*}}0: 04(ff)

  free(alloc);
}
