// RUN: %clang_cc1 -fenable-matrix -triple x86_64-apple-darwin %s -ffp-contract=on -emit-llvm -disable-llvm-passes -o - | FileCheck %s
// RUN: %clang_cc1 -fenable-matrix -triple x86_64-apple-darwin %s -ffp-contract=on -ffp-exception-behavior=strict -emit-llvm -disable-llvm-passes -o - | FileCheck %s --check-prefix=STRICT

// Regression test for issue llvm/llvm-project#244340

typedef float fx2x3_t __attribute__((matrix_type(2, 3)));
typedef float fx4_t __attribute__((ext_vector_type(4)));

// CHECK-LABEL: @test_matrix_mul_plus_scalar
// CHECK: [[MUL:%.*]] = fmul <6 x float> %{{.*}}, splat (float 2.000000e+00)
// CHECK: fadd <6 x float> [[MUL]], splat (float 3.000000e+00)
// STRICT-LABEL: @test_matrix_mul_plus_scalar
// STRICT: [[MUL:%.*]] = call <6 x float> @llvm.experimental.constrained.fmul.v6f32(<6 x float> %{{.*}}, <6 x float> splat (float 2.000000e+00), {{.*}})
// STRICT: call <6 x float> @llvm.experimental.constrained.fadd.v6f32(<6 x float> [[MUL]], <6 x float> splat (float 3.000000e+00), {{.*}})
fx2x3_t test_matrix_mul_plus_scalar(fx2x3_t m) {
  return m * 2.0f + 3.0f;
}

// CHECK-LABEL: @test_scalar_minus_matrix_mul
// CHECK: [[MUL:%.*]] = fmul <6 x float> splat (float 2.000000e+00), %{{.*}}
// CHECK: fsub <6 x float> splat (float 3.000000e+00), [[MUL]]
// STRICT-LABEL: @test_scalar_minus_matrix_mul
// STRICT: [[MUL:%.*]] = call <6 x float> @llvm.experimental.constrained.fmul.v6f32(<6 x float> splat (float 2.000000e+00), <6 x float> %{{.*}}, {{.*}})
// STRICT: call <6 x float> @llvm.experimental.constrained.fsub.v6f32(<6 x float> splat (float 3.000000e+00), <6 x float> [[MUL]], {{.*}})
fx2x3_t test_scalar_minus_matrix_mul(fx2x3_t m) {
  return 3.0f - 2.0f * m;
}

// CHECK-LABEL: @test_scalar_mul_plus_matrix
// CHECK: [[MUL:%.*]] = fmul float %{{.*}}, %{{.*}}
// CHECK: [[INSERT:%.*]] = insertelement <6 x float> poison, float [[MUL]], i64 0
// CHECK: [[SPLAT:%.*]] = shufflevector <6 x float> [[INSERT]], <6 x float> poison, <6 x i32> zeroinitializer
// CHECK: fadd <6 x float> [[SPLAT]], %{{.*}}
// STRICT-LABEL: @test_scalar_mul_plus_matrix
// STRICT: [[MUL:%.*]] = call float @llvm.experimental.constrained.fmul.f32(float %{{.*}}, float %{{.*}}, {{.*}})
// STRICT: [[INSERT:%.*]] = insertelement <6 x float> poison, float [[MUL]], i64 0
// STRICT: [[SPLAT:%.*]] = shufflevector <6 x float> [[INSERT]], <6 x float> poison, <6 x i32> zeroinitializer
// STRICT: call <6 x float> @llvm.experimental.constrained.fadd.v6f32(<6 x float> [[SPLAT]], <6 x float> %{{.*}}, {{.*}})
fx2x3_t test_scalar_mul_plus_matrix(fx2x3_t m, float a, float b) {
  return a * b + m;
}

// CHECK-LABEL: @test_matrix_plus_scalar_mul
// CHECK: [[MUL:%.*]] = fmul float %{{.*}}, %{{.*}}
// CHECK: [[INSERT:%.*]] = insertelement <6 x float> poison, float [[MUL]], i64 0
// CHECK: [[SPLAT:%.*]] = shufflevector <6 x float> [[INSERT]], <6 x float> poison, <6 x i32> zeroinitializer
// CHECK: fadd <6 x float> %{{.*}}, [[SPLAT]]
// STRICT-LABEL: @test_matrix_plus_scalar_mul
// STRICT: [[MUL:%.*]] = call float @llvm.experimental.constrained.fmul.f32(float %{{.*}}, float %{{.*}}, {{.*}})
// STRICT: [[INSERT:%.*]] = insertelement <6 x float> poison, float [[MUL]], i64 0
// STRICT: [[SPLAT:%.*]] = shufflevector <6 x float> [[INSERT]], <6 x float> poison, <6 x i32> zeroinitializer
// STRICT: call <6 x float> @llvm.experimental.constrained.fadd.v6f32(<6 x float> %{{.*}}, <6 x float> [[SPLAT]], {{.*}})
fx2x3_t test_matrix_plus_scalar_mul(fx2x3_t m, float a, float b) {
  return m + a * b;
}

// Make sure operands of the same type are still fused.
// CHECK-LABEL: @test_matrix_mul_plus_matrix
// CHECK: call <6 x float> @llvm.fmuladd.v6f32(<6 x float> %{{.*}}, <6 x float> splat (float 2.000000e+00), <6 x float> %{{.*}})
// STRICT-LABEL: @test_matrix_mul_plus_matrix
// STRICT: call <6 x float> @llvm.experimental.constrained.fmuladd.v6f32(<6 x float> %{{.*}}, <6 x float> splat (float 2.000000e+00), <6 x float> %{{.*}}, {{.*}})
fx2x3_t test_matrix_mul_plus_matrix(fx2x3_t m, fx2x3_t n) {
  return m * 2.0f + n;
}

// A vector add/sub's scalar operand is splatted, so it still fuses.
// CHECK-LABEL: @test_vector_mul_plus_scalar
// CHECK: call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %{{.*}}, <4 x float> splat (float 2.000000e+00), <4 x float> splat (float 3.000000e+00))
// STRICT-LABEL: @test_vector_mul_plus_scalar
// STRICT: call <4 x float> @llvm.experimental.constrained.fmuladd.v4f32(<4 x float> %{{.*}}, <4 x float> splat (float 2.000000e+00), <4 x float> splat (float 3.000000e+00), {{.*}})
fx4_t test_vector_mul_plus_scalar(fx4_t v) {
  return v * 2.0f + 3.0f;
}

// The float operand is converted to double, so it still fuses.
// CHECK-LABEL: @test_double_mul_plus_float
// CHECK: [[CONV:%.*]] = fpext float %{{.*}} to double
// CHECK: call double @llvm.fmuladd.f64(double %{{.*}}, double %{{.*}}, double [[CONV]])
// STRICT-LABEL: @test_double_mul_plus_float
// STRICT: [[CONV:%.*]] = call double @llvm.experimental.constrained.fpext.f64.f32(float %{{.*}}, {{.*}})
// STRICT: call double @llvm.experimental.constrained.fmuladd.f64(double %{{.*}}, double %{{.*}}, double [[CONV]], {{.*}})
double test_double_mul_plus_float(double a, double b, float c) {
  return a * b + c;
}
