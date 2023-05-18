// RUN: %clangxx_asan -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O2 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O3 %s -o %t && not %run %t 2>&1 | FileCheck %s

// CHECK: AddressSanitizer: global-buffer-overflow
int global[10];
// CHECK: {{#0.*call4}}
void __attribute__((noinline)) call4(int i) { global[i+10]++; }
// CHECK: {{#1.*call3}}
void __attribute__((noinline)) call3(int i) { call4(i); }
// CHECK: {{#2.*call2}}
void __attribute__((noinline)) call2(int i) { call3(i); }
// CHECK: {{#3.*call1}}
void __attribute__((noinline)) call1(int i) { call2(i); }
// CHECK: {{#4.*main}}
int main(int argc, char **argv) {
  call1(argc);
  return global[0];
}
