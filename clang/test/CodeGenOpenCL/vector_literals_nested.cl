// RUN: %clang_cc1 %s -emit-llvm -O3 -o - | FileCheck %s

typedef int int2 __attribute((ext_vector_type(2)));
typedef int int4 __attribute((ext_vector_type(4)));

__constant const int4 itest1 = (int4)(1, 2, ((int2)(3, 4)));
// CHECK: constant <4 x i32> <i32 1, i32 2, i32 3, i32 4>
__constant const int4 itest2 = (int4)(1, 2, ((int2)(3)));
// CHECK: constant <4 x i32> <i32 1, i32 2, i32 3, i32 3>

typedef float float2 __attribute((ext_vector_type(2)));
typedef float float4 __attribute((ext_vector_type(4)));

void ftest1(float4 *p) {
  *p = (float4)(1.1f, 1.2f, ((float2)(1.3f, 1.4f)));
// CHECK: store <4 x float> <float 0x3FF19999A0000000, float 0x3FF3333340000000, float 0x3FF4CCCCC0000000, float 0x3FF6666660000000>
}

float4 ftest2(float4 *p) {
   *p =  (float4)(1.1f, 1.2f, ((float2)(1.3f)));
// CHECK: store <4 x float> <float 0x3FF19999A0000000, float 0x3FF3333340000000, float 0x3FF4CCCCC0000000, float 0x3FF4CCCCC0000000>
}

