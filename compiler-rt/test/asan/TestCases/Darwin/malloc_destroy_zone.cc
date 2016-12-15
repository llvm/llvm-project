// RUN: %clangxx_asan %s -o %t && %run %t 2>&1 | FileCheck %s

#include <malloc/malloc.h>
#include <stdlib.h>
#include <stdio.h>

int main() {
  fprintf(stderr, "start\n");
  malloc_zone_t *zone = malloc_create_zone(0, 0);
  fprintf(stderr, "zone = %p\n", zone);
  malloc_set_zone_name(zone, "myzone");
  fprintf(stderr, "name changed\n");
  malloc_destroy_zone(zone);
  fprintf(stderr, "done\n");
  return 0;
}

// CHECK: start
// CHECK-NEXT: zone = 0x{{.*}}
// CHECK-NEXT: name changed
// CHECK-NEXT: done
