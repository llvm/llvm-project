// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// ==================================================================
// Float Matrix vs Float Matrix
// ==================================================================

// CHECK-LABEL: define {{.*}}test_float_lt
// CHECK: fcmp {{.*}}olt <4 x float>
bool2x2 test_float_lt(float2x2 A, float2x2 B) {
  return A < B;
}

// CHECK-LABEL: define {{.*}}test_float_gt
// CHECK: fcmp {{.*}}ogt <9 x float>
bool3x3 test_float_gt(float3x3 A, float3x3 B) {
  return A > B;
}

// CHECK-LABEL: define {{.*}}test_float_le
// CHECK: fcmp {{.*}}ole <16 x float>
bool4x4 test_float_le(float4x4 A, float4x4 B) {
  return A <= B;
}

// CHECK-LABEL: define {{.*}}test_float_ge
// CHECK: fcmp {{.*}}oge <6 x float>
bool2x3 test_float_ge(float2x3 A, float2x3 B) {
  return A >= B;
}

// CHECK-LABEL: define {{.*}}test_float_eq
// CHECK: fcmp {{.*}}oeq <4 x float>
bool2x2 test_float_eq(float2x2 A, float2x2 B) {
  return A == B;
}

// CHECK-LABEL: define {{.*}}test_float_neq
// CHECK: fcmp {{.*}}une <4 x float>
bool2x2 test_float_neq(float2x2 A, float2x2 B) {
  return A != B;
}

// ==================================================================
// Integer Matrix vs Integer Matrix
// ==================================================================

// CHECK-LABEL: define {{.*}}test_int_lt
// CHECK: icmp slt <4 x i32>
bool2x2 test_int_lt(int2x2 A, int2x2 B) {
  return A < B;
}

// CHECK-LABEL: define {{.*}}test_int_ge
// CHECK: icmp sge <4 x i32>
bool2x2 test_int_ge(int2x2 A, int2x2 B) {
  return A >= B;
}

// CHECK-LABEL: define {{.*}}test_int_eq
// CHECK: icmp eq <4 x i32>
bool2x2 test_int_eq(int2x2 A, int2x2 B) {
  return A == B;
}

// ==================================================================
// Matrix vs Scalar (Broadcast)
// ==================================================================

// CHECK-LABEL: define {{.*}}test_scalar_lt
// CHECK: [[SPLAT:%.*]] = insertelement <4 x float> poison, float {{%.*}}, i64 0
// CHECK: [[B_MAT:%.*]] = shufflevector <4 x float> [[SPLAT]], <4 x float> poison, <4 x i32> zeroinitializer
// CHECK: fcmp {{.*}}olt <4 x float> {{%.*}}, [[B_MAT]]
bool2x2 test_scalar_lt(float2x2 A, float B) {
  return A < B;
}

// CHECK-LABEL: define {{.*}}test_scalar_neq
// CHECK: [[SPLAT:%.*]] = insertelement <4 x float> poison, float {{%.*}}, i64 0
// CHECK: [[B_MAT:%.*]] = shufflevector <4 x float> [[SPLAT]], <4 x float> poison, <4 x i32> zeroinitializer
// CHECK: fcmp {{.*}}une <4 x float> {{%.*}}, [[B_MAT]]
bool2x2 test_scalar_neq(float2x2 A, float B) {
  return A != B;
}
