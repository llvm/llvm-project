// RUN: %clang_cc1 -ffreestanding -triple x86_64-unknown-unknown -emit-llvm -o - %s | FileCheck %s


#include <x86intrin.h>

unsigned long long test_rdpmc(int a) {
  return _rdpmc(a);
// CHECK: @test_rdpmc
// CHECK: call i64 @llvm.x86.rdpmc
}

int test_rdtsc() {
  return _rdtsc();
// CHECK: @test_rdtsc
// CHECK: call i64 @llvm.x86.rdtsc
}
