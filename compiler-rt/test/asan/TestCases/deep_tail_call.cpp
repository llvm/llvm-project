// RUN: %clangxx_asan -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O2 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O3 %s -o %t && not %run %t 2>&1 | FileCheck %s

// CHECK: AddressSanitizer: global-buffer-overflow
#include "defines.h"
int global[10];
// CHECK: {{#0.*call4}}
void ATTRIBUTE_NOINLINE call4(int i) { global[i + 10]++; }
// CHECK: {{#1.*call3}}
void ATTRIBUTE_NOINLINE call3(int i) { call4(i); }
// CHECK: {{#2.*call2}}
void ATTRIBUTE_NOINLINE call2(int i) { call3(i); }
// CHECK: {{#3.*call1}}
void ATTRIBUTE_NOINLINE call1(int i) { call2(i); }
// CHECK: {{#4.*main}}
int main(int argc, char **argv) {
  call1(argc);
  return global[0];
}
