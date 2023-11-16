// Test that verifies that asan produces valid symbolizer markup when enabled.
// RUN: %clangxx_asan -O1 %s -o %t 
// RUN: env ASAN_OPTIONS=enable_symbolizer_markup=1 not %run %t 2>&1 | FileCheck %s
// REQUIRES: linux

#include <stdlib.h>
int main() {
  char *x = (char*)malloc(10 * sizeof(char));
  free(x);
  return x[5];
}
// COM: For element syntax see: https://llvm.org/docs/SymbolizerMarkupFormat.html
// COM: OPEN is {{{ and CLOSE is }}}
// CHECK: [[OPEN:{{{]]reset[[CLOSE:}}}]]
// CHECK: [[OPEN]]module:[[MOD_ID:[0-9]+]]:{{.+}}:elf:{{[0-9a-fA-F]+}}[[CLOSE]]
// CHECK: [[OPEN]]mmap:{{0x[0-9a-fA-F]+:0x[0-9a-fA-F]+}}:load:[[MOD_ID]]:{{r[wx]{0,2}:0x[0-9a-fA-F]+}}[[CLOSE]]
// CHECK: [[OPEN]]bt:{{[0-9]+}}:0x{{[0-9a-fA-F]+}}[[CLOSE]]
