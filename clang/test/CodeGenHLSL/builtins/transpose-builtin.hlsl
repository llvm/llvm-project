// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s -fnative-half-type -fnative-int16-type -emit-llvm -disable-llvm-passes -o - | FileCheck %s
// RUN: %if cir-enabled %{ %clang_cc1 -fclangir -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s -fnative-half-type -fnative-int16-type -emit-llvm -disable-llvm-passes -o - | FileCheck %s %}

// NOTE: This test is only to confirm we can do codgen with the matrix alias.

// CHECK-LABEL: define {{.*}}transpose_half_2x2
void transpose_half_2x2(half2x2 a) {
  // CHECK: [[A_ADDR:%.*]] = alloca [2 x <2 x half>], align 2
  // CHECK: [[A_T:%.*]] = alloca [2 x <2 x half>], align 2
  // CHECK:        [[A:%.*]] = load <4 x half>, ptr [[A_ADDR]], align 2
  // CHECK-NEXT:   [[TRANS:%.*]] = call {{.*}}<4 x half> @llvm.matrix.transpose.v4f16(<4 x half> [[A]], i32 2, i32 2)
  // CHECK-NEXT:   store <4 x half> [[TRANS]], ptr [[A_T]], align 2

  half2x2 a_t = __builtin_matrix_transpose(a);
}

// CHECK-LABEL: define {{.*}}transpose_float_3x2
void transpose_float_3x2(float3x2 a) {
  // CHECK: [[A_ADDR:%.*]] = alloca [2 x <3 x float>], align 4
  // CHECK: [[A_T:%.*]] = alloca [3 x <2 x float>], align 4
  // CHECK:        [[A:%.*]] = load <6 x float>, ptr [[A_ADDR]], align 4
  // CHECK-NEXT:   [[TRANS:%.*]] = call {{.*}}<6 x float> @llvm.matrix.transpose.v6f32(<6 x float> [[A]], i32 3, i32 2)
  // CHECK-NEXT:   store <6 x float> [[TRANS]], ptr [[A_T]], align 4

  float2x3 a_t = __builtin_matrix_transpose(a);
}

// CHECK-LABEL: define {{.*}}transpose_int_4x3
void transpose_int_4x3(int4x3 a) {
  // CHECK: [[A_ADDR:%.*]] = alloca [3 x <4 x i32>], align 4
  // CHECK: [[A_T:%.*]] = alloca [4 x <3 x i32>], align 4
  // CHECK:         [[A:%.*]] = load <12 x i32>, ptr [[A_ADDR]], align 4
  // CHECK-NEXT:    [[TRANS:%.*]] = call <12 x i32> @llvm.matrix.transpose.v12i32(<12 x i32> [[A]], i32 4, i32 3)
  // CHECK-NEXT:    store <12 x i32> [[TRANS]], ptr [[A_T]], align 4

  int3x4 a_t = __builtin_matrix_transpose(a);
}
