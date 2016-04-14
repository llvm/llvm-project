// Test that malloc_zone_from_ptr returns a valid zone for a 0-sized allocation.

// RUN: %clang_tsan %s -o %t -framework Foundation
// RUN: %run %t 2>&1 | FileCheck %s

#import <Foundation/Foundation.h>
#include <malloc/malloc.h>

int main() {
  void *p = malloc(0);

  size_t s = malloc_size(p);
  printf("size = 0x%zx\n", s);

  malloc_zone_t *z = malloc_zone_from_ptr(p);
  if (z)
    printf("z = %p\n", z);
  else
    printf("no zone\n");
}

// CHECK: z = 0x{{[0-9a-f]+}}
// CHECK-NOT: no zone
