// RUN: %clang_asan -O0 %s -o %t
// RUN: %run %t 0x00 2>&1 | FileCheck %s -check-prefix=X00
// RUN: not %run %t 0xf1 2>&1 | FileCheck %s -check-prefix=XF1
// RUN: not %run %t 0xf2 2>&1 | FileCheck %s -check-prefix=XF2
// RUN: not %run %t 0xf3 2>&1 | FileCheck %s -check-prefix=XF3
// RUN: not %run %t 0xf5 2>&1 | FileCheck %s -check-prefix=XF5
// RUN: not %run %t 0xf8 2>&1 | FileCheck %s -check-prefix=XF8

#include <assert.h>
#include <sanitizer/asan_interface.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

void __asan_set_shadow_00(size_t addr, size_t size);
void __asan_set_shadow_f1(size_t addr, size_t size);
void __asan_set_shadow_f2(size_t addr, size_t size);
void __asan_set_shadow_f3(size_t addr, size_t size);
void __asan_set_shadow_f5(size_t addr, size_t size);
void __asan_set_shadow_f8(size_t addr, size_t size);

char a __attribute__((aligned(8)));

void f(long arg) {
  size_t shadow_offset;
  size_t shadow_scale;
  __asan_get_shadow_mapping(&shadow_scale, &shadow_offset);
  size_t addr = (((size_t)&a) >> shadow_scale) + shadow_offset;

  switch (arg) {
  // X00-NOT: AddressSanitizer
  // X00: PASS
  case 0x00:
    return __asan_set_shadow_00(addr, 1);
  // XF1: AddressSanitizer: stack-buffer-underflow
  // XF1: [f1]
  case 0xf1:
    return __asan_set_shadow_f1(addr, 1);
  // XF2: AddressSanitizer: stack-buffer-overflow
  // XF2: [f2]
  case 0xf2:
    return __asan_set_shadow_f2(addr, 1);
  // XF3: AddressSanitizer: stack-buffer-overflow
  // XF3: [f3]
  case 0xf3:
    return __asan_set_shadow_f3(addr, 1);
  // XF5: AddressSanitizer: stack-use-after-return
  // XF5: [f5]
  case 0xf5:
    return __asan_set_shadow_f5(addr, 1);
  // XF8: AddressSanitizer: stack-use-after-scope
  // XF8: [f8]
  case 0xf8:
    return __asan_set_shadow_f8(addr, 1);
  }
  assert(0);
}

int main(int argc, char **argv) {
  assert(argc > 1);

  long arg = strtol(argv[1], 0, 16);
  f(arg);
  a = 1;
  printf("PASS\n");
  return 0;
}
