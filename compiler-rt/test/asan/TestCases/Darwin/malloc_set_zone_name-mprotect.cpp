// Regression test for a bug in malloc_create_zone()
// (https://code.google.com/p/address-sanitizer/issues/detail?id=203)
// The old implementation of malloc_create_zone() didn't always return a
// page-aligned address, so we can only test on a best-effort basis.

// RUN: %clangxx_asan %s -o %t
// RUN: %run %t 2>&1

#include <malloc/malloc.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

const int kNumIter = 4096;
const int kNumZones = 100;
int main() {
  char *mem[kNumIter * 2];
  // Allocate memory chunks from different size classes up to 1 page.
  // (For the case malloc() returns memory chunks in descending order)
  for (int i = 0; i < kNumIter; i++) {
    mem[i] = (char*)malloc(8 * i);
  }
  // Try to allocate a page-aligned malloc zone. Otherwise the mprotect() call
  // in malloc_set_zone_name() will silently fail.
  malloc_zone_t *zone = NULL;
  bool aligned = false;
  for (int i = 0; i < kNumZones; i++) {
    zone = malloc_create_zone(0, 0);
    if (((uintptr_t)zone & (~0xfff)) == (uintptr_t)zone) {
      aligned = true;
      break;
    }
  }
  if (!aligned) {
    printf("Warning: couldn't allocate a page-aligned zone.");
    return 0;
  }
  // malloc_set_zone_name() calls mprotect(zone, 4096, PROT_READ | PROT_WRITE),
  // modifies the zone contents and then calls mprotect(zone, 4096, PROT_READ).
  malloc_set_zone_name(zone, "foobar");
  // Allocate memory chunks from different size classes again.
  for (int i = 0; i < kNumIter; i++) {
    mem[i + kNumIter] = (char*)malloc(8 * i);
  }
  // Access the allocated memory chunks and free them.
  for (int i = 0; i < kNumIter * 2; i++) {
    memset(mem[i], 'a', 8 * (i % kNumIter));
    free(mem[i]);
  }
  malloc_destroy_zone(zone);
  return 0;
}
