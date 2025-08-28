// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -Wall -Wextra -Wpedantic -fenable-ripple -S -O2 -emit-llvm %s -o - | FileCheck --implicit-check-not="warning:" %s

#include <ripple.h>

__attribute__((noinline)) int toBeSpecialized(int n, int m, int p) {
  return n * m * p * 32;
}

// CHECK-LABEL: @test
// CHECK: call fastcc <128 x i{{[0-9]+}}> @ripple.specialization.final.{{[0-9]+}}.toBeSpecialized(<128 x {{i32}}>{{.*}}, i{{[0-9]+}}{{.*}}, <128 x {{i32}}>

// CHECK-LABEL: define{{.*}}internal{{.*}}fastcc{{.*}}<128 x i{{[0-9]+}}> @ripple.specialization.final.{{[0-9]+}}.toBeSpecialized(<128 x i{{[0-9]+}}>{{.*}}%0, i{{[0-9]+}}{{.*}}%1, <128 x i32>{{.*}}%2)
// CHECK: %[[SplatInsert:[a-zA-Z0-9.]+]] = insertelement <128 x i32> poison, i32 %1
// CHECK-NEXT: %[[SplatArg1:[a-zA-Z0-9.]+]] = shufflevector <128 x i32> %[[SplatInsert]], <128 x i32> poison, <128 x i32> zeroinitializer
// CHECK-NEXT: %[[MulBy32:[A-Za-z0-9_.]+]] = shl <128 x i{{[0-9]+}}> %0, splat (i{{[0-9]+}} 5)
// CHECK-NEXT: %[[MulBySplat:[A-Za-z0-9_.]+]] = mul <128 x i32> %[[MulBy32]], %[[SplatArg1]]
// CHECK-NEXT: %[[MulByArg2:[A-Za-z0-9_.]+]] = mul <128 x i32> %[[MulBySplat]], %2
// CHECK-NEXT: ret <128 x i{{[0-9]+}}> %[[MulByArg2]]

void test(int *in, int *out) {
  ripple_block_t BS = ripple_set_block_shape(0, 128);
  size_t idx = ripple_id(BS, 0);
  out[idx] = toBeSpecialized(in[idx], in[0], out[idx]);
}
