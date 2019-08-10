// Check that malloc_default_zone and malloc_zone_from_ptr return the
// sanitizer-installed malloc zone even when MallocStackLogging (MSL) is
// requested. This prevents crashes in certain situations. Note that the
// sanitizers and MSL cannot be used together. If both are enabled, MSL
// functionality is essentially deactivated since it only hooks the default
// allocator which is replaced by a custom sanitizer allocator.
//
// MSL=lite creates its own special malloc zone, copies the passed zone name,
// and leaks it.
// RUN: echo "leak:create_and_insert_msl_lite_zone" >> lsan.supp
//
// RUN: %clangxx -g %s -o %t
// RUN:                                                                   %run %t | FileCheck %s
// RUN: %env MallocStackLogging=lite LSAN_OPTIONS=suppressions=lsan.supp  %run %t | FileCheck %s
// RUN: %env MallocStackLogging=full                                      %run %t | FileCheck %s
//
// UBSan does not install a malloc zone.
// XFAIL: ubsan
//

#include <malloc/malloc.h>
#include <stdlib.h>
#include <stdio.h>

int main(void) {
  malloc_zone_t *default_zone = malloc_default_zone();
  printf("default zone name: %s\n", malloc_get_zone_name(default_zone));
// CHECK: default zone name: {{a|l|t}}san

  void *ptr1 = malloc(10);
  void *ptr2 = malloc_zone_malloc(default_zone, 10);

  malloc_zone_t* zone1 = malloc_zone_from_ptr(ptr1);
  malloc_zone_t* zone2 = malloc_zone_from_ptr(ptr2);

  printf("zone1: %d\n", zone1 == default_zone);
  printf("zone2: %d\n", zone2 == default_zone);
// CHECK: zone1: 1
// CHECK: zone2: 1

  free(ptr1);
  malloc_zone_free(zone2, ptr2);

  return 0;
}
