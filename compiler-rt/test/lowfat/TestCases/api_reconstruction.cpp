// RUN: %clangxx_lowfat -O0 %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_lowfat -O2 %s -o %t && %run %t 2>&1 | FileCheck %s

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>

typedef uintptr_t uptr;

extern "C" {
uptr __lf_get_base(uptr ptr);
uptr __lf_get_size(uptr ptr);
uptr __lf_get_offset(uptr ptr);
uptr __lf_get_usable_size(uptr ptr);
}

#define lowfat_base(p)        __lf_get_base((uptr)(p))
#define lowfat_size(p)        __lf_get_size((uptr)(p))
#define lowfat_offset(p)      __lf_get_offset((uptr)(p))
#define lowfat_usable_size(p) __lf_get_usable_size((uptr)(p))

int main() {
  size_t request_size = 100;
  char *p = (char *)malloc(request_size);
  uptr base = lowfat_base(p);
  uptr size = lowfat_size(p);

  printf("Base and Interior Pointers:\n");
  // Test start of object
  if (lowfat_base(p) == (uptr)p) printf("  start: ok\n");
  // Test interior pointer
  if (lowfat_base(p + 50) == (uptr)p) printf("  interior: ok\n");

  printf("Size Operations:\n");
  // Size should be >= requested size and usually a power of 2 (or from config)
  if (size >= request_size) printf("  size_ge: ok\n");

  printf("Usable Size & Offsets:\n");
  if (lowfat_offset(p) == 0) printf("  offset_0: ok\n");
  if (lowfat_offset(p + 50) == 50) printf("  offset_50: ok\n");
  if (lowfat_usable_size(p) == size) printf("  usable_start: ok\n");
  if (lowfat_usable_size(p + 50) == size - 50) printf("  usable_50: ok\n");

  printf("Non-Fat Pointer Handling:\n");
  // Standard stack pointer (not instrumented yet in this test)
  int x;
  if (lowfat_base(&x) == 0) printf("  stack_base: ok\n");
  if (lowfat_size(&x) == (uptr)-1) printf("  stack_size: ok\n");

  // NULL pointer
  if (lowfat_base(NULL) == 0) printf("  null_base: ok\n");
  if (lowfat_size(NULL) == (uptr)-1) printf("  null_size: ok\n");

  free(p);
  
  // CHECK: Base and Interior Pointers:
  // CHECK:   start: ok
  // CHECK:   interior: ok
  // CHECK: Size Operations:
  // CHECK:   size_ge: ok
  // CHECK: Usable Size & Offsets:
  // CHECK:   offset_0: ok
  // CHECK:   offset_50: ok
  // CHECK:   usable_start: ok
  // CHECK:   usable_50: ok
  // CHECK: Non-Fat Pointer Handling:
  // CHECK:   stack_base: ok
  // CHECK:   stack_size: ok
  // CHECK:   null_base: ok
  // CHECK:   null_size: ok

  return 0;
}
