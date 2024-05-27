// REQUIRES: x86-registered-target

// Test that we get the module flag TLSMaxAlign on the PS platforms.
// RUN: %clang_cc1 -triple x86_64-scei-ps4 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-scei-ps5 -emit-llvm -o - %s | FileCheck %s

int main(void) {
  return 0;
}

// CHECK-DAG: ![[MDID:[0-9]+]] = !{i32 1, !"MaxTLSAlign", i32 256}
// CHECK-DAG: llvm.module.flags = {{.*}}![[MDID]]
