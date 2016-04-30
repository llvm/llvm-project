// Test that malloc_zone_from_ptr returns a valid zone for a 0-sized allocation.

// RUN: %clang_tsan %s -o %t -framework Foundation
// RUN: %run %t 2>&1 | FileCheck %s

#import <Foundation/Foundation.h>
#include <malloc/malloc.h>

int some_global;

void describe_zone(void *p) {
  malloc_zone_t *z = malloc_zone_from_ptr(p);
  if (z) {
    fprintf(stderr, "zone = %p\n", z);
  }	else {
  	fprintf(stderr, "zone = no zone\n");
  }
}

int main() {
  void *p;
  size_t s;

  p = malloc(0x40);
  s = malloc_size(p);
  fprintf(stderr, "size = 0x%zx\n", s);
  // CHECK: size = 0x40
  describe_zone(p);
  // CHECK: zone = 0x{{[0-9a-f]+}}

  p = malloc(0);
  s = malloc_size(p);
  fprintf(stderr, "size = 0x%zx\n", s);
  // CHECK: size = 0x1
  describe_zone(p);
  // CHECK: zone = 0x{{[0-9a-f]+}}

  p = &some_global;
  s = malloc_size(p);
  fprintf(stderr, "size = 0x%zx\n", s);
  // CHECK: size = 0x0
  describe_zone(p);
  // CHECK: zone = no zone

  p = mmap(0, 0x1000, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANON, -1, 0);
  if (!p) {
  	fprintf(stderr, "mmap failed\n");
  	exit(1);
  }
  s = malloc_size(p);
  fprintf(stderr, "size = 0x%zx\n", s);
  // CHECK: size = 0x0
  describe_zone(p);
  // CHECK: zone = no zone
}
