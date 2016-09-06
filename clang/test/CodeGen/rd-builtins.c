// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o - %s | FileCheck %s

// Don't include mm_malloc.h, it's system specific.
#define __MM_MALLOC_H

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
