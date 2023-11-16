// Test that verifies that hwasan produces valid symbolizer markup when enabled.
// RUN: %clang_hwasan -O0 %s -o %t
// RUN: env HWASAN_OPTIONS=enable_symbolizer_markup=1 not %run %t 2>&1 | FileCheck %s
// RUN: %clang_hwasan -O1 %s -o %t
// RUN: env HWASAN_OPTIONS=enable_symbolizer_markup=1 not %run %t 2>&1 | FileCheck %s
// RUN: %clang_hwasan -O2 %s -o %t
// RUN: env HWASAN_OPTIONS=enable_symbolizer_markup=1 not %run %t 2>&1 | FileCheck %s
// RUN: %clang_hwasan -O3 %s -o %t
// RUN: env HWASAN_OPTIONS=enable_symbolizer_markup=1 not %run %t 2>&1 | FileCheck %s


#include <stdlib.h>
#include <stdio.h>
#include <sanitizer/hwasan_interface.h>

int main() {
  __hwasan_enable_allocator_tagging();
  char * volatile x = (char*)malloc(10);
  free(x);
  __hwasan_disable_allocator_tagging();

  int r = 0;
  r = x[5];
  return r;
}

// COM: For element syntax see: https://llvm.org/docs/SymbolizerMarkupFormat.html
// COM: OPEN is {{{ and CLOSE is }}}
// CHECK: [[OPEN:{{{]]reset[[CLOSE:}}}]]
// CHECK: [[OPEN]]module:[[MOD_ID:[0-9]+]]:{{.+}}:elf:{{[0-9a-fA-F]+}}[[CLOSE]]
// CHECK: [[OPEN]]mmap:{{0x[0-9a-fA-F]+:0x[0-9a-fA-F]+}}:load:[[MOD_ID]]:{{r[wx]{0,2}:0x[0-9a-fA-F]+}}[[CLOSE]]
// CHECK: [[OPEN]]bt:{{[0-9]+}}:0x{{[0-9a-fA-F]+}}[[CLOSE]]
