// Make sure the zones created by malloc_create_zone() are write-protected.
#include <malloc/malloc.h>
#include <stdio.h>

// RUN: %clangxx_asan %s -o %t
// RUN: ASAN_OPTIONS="abort_on_error=1" not --crash %run %t 2>&1 | FileCheck %s


void *pwn(malloc_zone_t *unused_zone, size_t unused_size) {
  printf("PWNED\n");
  return NULL;
}

int main() {
  malloc_zone_t *zone = malloc_create_zone(0, 0);
  zone->malloc = pwn;
  void *v = malloc_zone_malloc(zone, 1);
  // CHECK-NOT: PWNED
  return 0;
}
