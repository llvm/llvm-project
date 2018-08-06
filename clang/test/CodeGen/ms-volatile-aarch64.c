// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-win32 -emit-llvm -fms-extensions -fms-volatile -o - < %s | FileCheck %s

void test1(int volatile *p, int v) {
  __iso_volatile_store32(p, v);
  // CHECK-LABEL: @test1
  // CHECK: store volatile {{.*}}, {{.*}}
}
int test2(const int volatile *p) {
  return __iso_volatile_load32(p);
  // CHECK-LABEL: @test2
  // CHECK: load volatile {{.*}}
}
