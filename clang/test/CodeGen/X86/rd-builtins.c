// RUN: %clang_cc1 -ffreestanding -triple x86_64-unknown-unknown -emit-llvm -o - %s | FileCheck %s


#include <x86intrin.h>

unsigned long long test_rdpmc(int a) {
  return _rdpmc(a);
// CHECK: @test_rdpmc
// CHECK: call i64 @llvm.x86.rdpmc
}

int test_rdtsc(void) {
  return _rdtsc();
// CHECK: @test_rdtsc
// CHECK: call i64 @llvm.x86.rdtsc
}

unsigned long long test_rdtscp(unsigned int *a) {
// CHECK: @test_rdtscp
// CHECK: [[RDTSCP:%.*]] = call { i64, i32 } @llvm.x86.rdtscp
// CHECK: [[TSC_AUX:%.*]] = extractvalue { i64, i32 } [[RDTSCP]], 1
// CHECK: store i32 [[TSC_AUX]], i32* %{{.*}}
// CHECK: [[TSC:%.*]] = extractvalue { i64, i32 } [[RDTSCP]], 0
  return __rdtscp(a);
}
