// RUN: %clang_cc1 -triple=powerpc-unknown-aix -emit-llvm %s -o - | \
// RUN: FileCheck %s
// RUN: %clang_cc1 -triple=powerpc64-unknown-aix -emit-llvm %s -o - | \
// RUN: FileCheck %s
// RUN: %clang_cc1 -triple=powerpc64le-unknown-unknown -emit-llvm %s \
// RUN:  -o - | FileCheck %s
// RUN: %clang_cc1 -triple=powerpc64-unknown-unknown -emit-llvm %s \
// RUN:  -o - | FileCheck %s

int test_lwarx(volatile int* a) {
  // CHECK: @test_lwarx
  // CHECK: %1 = bitcast i32* %0 to i8*
  // CHECK: %2 = call i32 @llvm.ppc.lwarx(i8* %1)
  return __lwarx(a);
}
int test_stwcx(volatile int* a, int val) {
  // CHECK: @test_stwcx
  // CHECK: %1 = bitcast i32* %0 to i8*
  // CHECK: %2 = load i32, i32* %val.addr, align 4
  // CHECK: %3 = call i32 @llvm.ppc.stwcx(i8* %1, i32 %2)
  return __stwcx(a, val);
}
