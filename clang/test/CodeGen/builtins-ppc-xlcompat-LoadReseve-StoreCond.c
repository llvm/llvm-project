// RUN: %clang_cc1 -O2 -triple=powerpc-unknown-aix -emit-llvm %s -o - | \
// RUN: FileCheck %s
// RUN: %clang_cc1 -O2 -triple=powerpc64-unknown-aix -emit-llvm %s -o - | \
// RUN: FileCheck %s
// RUN: %clang_cc1 -O2 -triple=powerpc64le-unknown-unknown -emit-llvm %s \
// RUN:  -o - | FileCheck %s
// RUN: %clang_cc1 -O2 -triple=powerpc64-unknown-unknown -emit-llvm %s \
// RUN:  -o - | FileCheck %s

int test_lwarx(volatile int* a) {
  // CHECK: @test_lwarx
  // CHECK: %0 = tail call i32 asm sideeffect "lwarx $0, ${1:y}", "=r,*Z,~{memory}"(i32* %a)
  return __lwarx(a);
}
int test_stwcx(volatile int* a, int val) {
  // CHECK: @test_stwcx
  // CHECK: %0 = bitcast i32* %a to i8*
  // CHECK: %1 = tail call i32 @llvm.ppc.stwcx(i8* %0, i32 %val)
  return __stwcx(a, val);
}
