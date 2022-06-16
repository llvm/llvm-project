// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -flax-vector-conversions=none -target-feature +altivec -target-feature +vsx \
// RUN:   -triple powerpc64-unknown-unknown -target-cpu pwr8 \
// RUN:   -emit-llvm %s -o - -U__XL_COMPAT_ALTIVEC__ | FileCheck %s
// RUN: %clang_cc1 -flax-vector-conversions=none -target-feature +altivec -target-feature +vsx \
// RUN:   -triple powerpc64le-unknown-unknown -target-cpu pwr8 \
// RUN:   -emit-llvm %s -o - -U__XL_COMPAT_ALTIVEC__ | \
// RUN:   FileCheck %s -check-prefix=CHECK-LE
#include <altivec.h>

vector signed __int128 res_vslll;
unsigned long long aull[2] = { 1L, 2L };

void testVectorInt128Pack(){
// CHECK-LABEL: testVectorInt128Pack
// CHECK-LABEL-LE: testVectorInt128Pack
  res_vslll = __builtin_pack_vector_int128(aull[0], aull[1]);
// CHECK: %[[V1:[0-9]+]] = insertelement <2 x i64> undef, i64 %{{[0-9]+}}, i64 0
// CHECK-NEXT: %[[V2:[0-9]+]] = insertelement <2 x i64> %[[V1]], i64 %{{[0-9]+}}, i64 1
// CHECK-NEXT:  bitcast <2 x i64> %[[V2]] to <1 x i128>

// CHECK-LE: %[[V1:[0-9]+]] = insertelement <2 x i64> undef, i64 %{{[0-9]+}}, i64 1
// CHECK-NEXT-LE: %[[V2:[0-9]+]] = insertelement <2 x i64> %[[V1]], i64 %{{[0-9]+}}, i64 0
// CHECK-NEXT-LE:  bitcast <2 x i64> %[[V2]] to <1 x i128>

  __builtin_unpack_vector_int128(res_vslll, 0);
// CHECK:  %[[V1:[0-9]+]] = bitcast <1 x i128> %{{[0-9]+}} to <2 x i64>
// CHECK-NEXT: %{{[0-9]+}} = extractelement <2 x i64> %[[V1]], i32 0

// CHECK-LE:  %[[V1:[0-9]+]] = bitcast <1 x i128> %{{[0-9]+}} to <2 x i64>
// CHECK-NEXT-LE: %{{[0-9]+}} = extractelement <2 x i64> %[[V1]], i32 1

  __builtin_unpack_vector_int128(res_vslll, 1);
// CHECK:  %[[V1:[0-9]+]] = bitcast <1 x i128> %{{[0-9]+}} to <2 x i64>
// CHECK-NEXT: %{{[0-9]+}} = extractelement <2 x i64> %[[V1]], i32 1

// CHECK-LE:  %[[V1:[0-9]+]] = bitcast <1 x i128> %{{[0-9]+}} to <2 x i64>
// CHECK-NEXT-LE: %{{[0-9]+}} = extractelement <2 x i64> %[[V1]], i32 0
}

