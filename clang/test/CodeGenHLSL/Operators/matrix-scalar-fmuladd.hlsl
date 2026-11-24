// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -ffp-contract=on -emit-llvm -disable-llvm-passes -o - | FileCheck %s
// RUN: %clang_cc1 -finclude-default-header -triple spirv-unknown-vulkan-library %s -ffp-contract=on -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// Regression test for issue llvm/llvm-project#244340

// CHECK-LABEL: test_matrix_mul_plus_scalar
// CHECK: [[MUL:%.*]] = fmul {{.*}}<6 x float> %{{.*}}, splat (float 2.000000e+00)
// CHECK: fadd {{.*}}<6 x float> [[MUL]], splat (float 3.000000e+00)
float2x3 test_matrix_mul_plus_scalar(float2x3 m) {
  return m * 2 + 3;
}

// CHECK-LABEL: test_matrix_mul_minus_scalar
// CHECK: [[MUL:%.*]] = fmul {{.*}}<6 x float> %{{.*}}, splat (float 2.000000e+00)
// CHECK: fsub {{.*}}<6 x float> [[MUL]], splat (float 3.000000e+00)
float2x3 test_matrix_mul_minus_scalar(float2x3 m) {
  return m * 2 - 3;
}

// CHECK-LABEL: test_scalar_plus_matrix_mul
// CHECK: [[MUL:%.*]] = fmul {{.*}}<6 x float> splat (float 2.000000e+00), %{{.*}}
// CHECK: fadd {{.*}}<6 x float> splat (float 3.000000e+00), [[MUL]]
float2x3 test_scalar_plus_matrix_mul(float2x3 m) {
  return 3 + 2 * m;
}

// CHECK-LABEL: test_scalar_minus_matrix_mul
// CHECK: [[MUL:%.*]] = fmul {{.*}}<6 x float> splat (float 2.000000e+00), %{{.*}}
// CHECK: fsub {{.*}}<6 x float> splat (float 3.000000e+00), [[MUL]]
float2x3 test_scalar_minus_matrix_mul(float2x3 m) {
  return 3 - 2 * m;
}

// CHECK-LABEL: test_scalar_plus_neg_matrix_mul
// CHECK: [[MUL:%.*]] = fmul {{.*}}<6 x float> %{{.*}}, %{{.*}}
// CHECK: [[NEG:%.*]] = fneg {{.*}}<6 x float> [[MUL]]
// CHECK: fadd {{.*}}<6 x float> splat (float 3.000000e+00), [[NEG]]
float2x3 test_scalar_plus_neg_matrix_mul(float2x3 m) {
  return 3 + -(m * m);
}

// CHECK-LABEL: test_scalar_mul_plus_matrix
// CHECK: [[MUL:%.*]] = fmul {{.*}}float %{{.*}}, %{{.*}}
// CHECK: [[INSERT:%.*]] = insertelement <6 x float> poison, float [[MUL]], i64 0
// CHECK: [[SPLAT:%.*]] = shufflevector <6 x float> [[INSERT]], <6 x float> poison, <6 x i32> zeroinitializer
// CHECK: fadd {{.*}}<6 x float> [[SPLAT]], %{{.*}}
float2x3 test_scalar_mul_plus_matrix(float2x3 m, float a, float b) {
  return a * b + m;
}

// CHECK-LABEL: test_scalar_mul_minus_matrix
// CHECK: [[MUL:%.*]] = fmul {{.*}}float %{{.*}}, %{{.*}}
// CHECK: [[INSERT:%.*]] = insertelement <6 x float> poison, float [[MUL]], i64 0
// CHECK: [[SPLAT:%.*]] = shufflevector <6 x float> [[INSERT]], <6 x float> poison, <6 x i32> zeroinitializer
// CHECK: fsub {{.*}}<6 x float> [[SPLAT]], %{{.*}}
float2x3 test_scalar_mul_minus_matrix(float2x3 m, float a, float b) {
  return a * b - m;
}

// CHECK-LABEL: test_matrix_plus_scalar_mul
// CHECK: [[MUL:%.*]] = fmul {{.*}}float %{{.*}}, %{{.*}}
// CHECK: [[INSERT:%.*]] = insertelement <6 x float> poison, float [[MUL]], i64 0
// CHECK: [[SPLAT:%.*]] = shufflevector <6 x float> [[INSERT]], <6 x float> poison, <6 x i32> zeroinitializer
// CHECK: fadd {{.*}}<6 x float> %{{.*}}, [[SPLAT]]
float2x3 test_matrix_plus_scalar_mul(float2x3 m, float a, float b) {
  return m + a * b;
}

// CHECK-LABEL: test_matrix_minus_scalar_mul
// CHECK: [[MUL:%.*]] = fmul {{.*}}float %{{.*}}, %{{.*}}
// CHECK: [[INSERT:%.*]] = insertelement <6 x float> poison, float [[MUL]], i64 0
// CHECK: [[SPLAT:%.*]] = shufflevector <6 x float> [[INSERT]], <6 x float> poison, <6 x i32> zeroinitializer
// CHECK: fsub {{.*}}<6 x float> %{{.*}}, [[SPLAT]]
float2x3 test_matrix_minus_scalar_mul(float2x3 m, float a, float b) {
  return m - a * b;
}

// CHECK-LABEL: test_neg_scalar_mul_plus_matrix
// CHECK: [[MUL:%.*]] = fmul {{.*}}float %{{.*}}, %{{.*}}
// CHECK: [[NEG:%.*]] = fneg {{.*}}float [[MUL]]
// CHECK: [[INSERT:%.*]] = insertelement <6 x float> poison, float [[NEG]], i64 0
// CHECK: [[SPLAT:%.*]] = shufflevector <6 x float> [[INSERT]], <6 x float> poison, <6 x i32> zeroinitializer
// CHECK: fadd {{.*}}<6 x float> [[SPLAT]], %{{.*}}
float2x3 test_neg_scalar_mul_plus_matrix(float2x3 m, float a, float b) {
  return -(a * b) + m;
}

// CHECK-LABEL: test_add_assign_scalar_mul
// CHECK: [[MUL:%.*]] = fmul {{.*}}float %{{.*}}, %{{.*}}
// CHECK: [[INSERT:%.*]] = insertelement <6 x float> poison, float [[MUL]], i64 0
// CHECK: [[SPLAT:%.*]] = shufflevector <6 x float> [[INSERT]], <6 x float> poison, <6 x i32> zeroinitializer
// CHECK: [[ADD:%.*]] = fadd {{.*}}<6 x float> %{{.*}}, [[SPLAT]]
// CHECK: store <6 x float> [[ADD]], ptr %m.addr
float2x3 test_add_assign_scalar_mul(float2x3 m, float a, float b) {
  m += a * b;
  return m;
}

// CHECK-LABEL: test_sub_assign_scalar_mul
// CHECK: [[MUL:%.*]] = fmul {{.*}}float %{{.*}}, %{{.*}}
// CHECK: [[INSERT:%.*]] = insertelement <6 x float> poison, float [[MUL]], i64 0
// CHECK: [[SPLAT:%.*]] = shufflevector <6 x float> [[INSERT]], <6 x float> poison, <6 x i32> zeroinitializer
// CHECK: [[SUB:%.*]] = fsub {{.*}}<6 x float> %{{.*}}, [[SPLAT]]
// CHECK: store <6 x float> [[SUB]], ptr %m.addr
float2x3 test_sub_assign_scalar_mul(float2x3 m, float a, float b) {
  m -= a * b;
  return m;
}

// Make sure operands of the same type are still fused.
// CHECK-LABEL: test_matrix_mul_plus_matrix
// CHECK: call {{.*}}<6 x float> @llvm.fmuladd.v6f32(<6 x float> %{{.*}}, <6 x float> splat (float 2.000000e+00), <6 x float> %{{.*}})
float2x3 test_matrix_mul_plus_matrix(float2x3 m, float2x3 n) {
  return m * 2 + n;
}

// A vector add/sub's scalar operand is splatted, so it still fuses.
// CHECK-LABEL: test_vector_mul_plus_scalar
// CHECK: call {{.*}}<3 x float> @llvm.fmuladd.v3f32(<3 x float> %{{.*}}, <3 x float> splat (float 2.000000e+00), <3 x float> splat (float 3.000000e+00))
float3 test_vector_mul_plus_scalar(float3 v) {
  return v * 2 + 3;
}

// The float operand is converted to double, so it still fuses.
// CHECK-LABEL: test_double_mul_plus_float
// CHECK: [[CONV:%.*]] = fpext {{.*}}float %{{.*}} to double
// CHECK: call {{.*}}double @llvm.fmuladd.f64(double %{{.*}}, double %{{.*}}, double [[CONV]])
double test_double_mul_plus_float(double a, double b, float c) {
  return a * b + c;
}
