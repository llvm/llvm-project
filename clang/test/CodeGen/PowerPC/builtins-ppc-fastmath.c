// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc64-unknown-unknown \
// RUN:   -emit-llvm %s -o - -target-cpu pwr7 | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64le-unknown-unknown \
// RUN:   -emit-llvm %s -o - -target-cpu pwr8 | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64-unknown-aix \
// RUN:   -emit-llvm %s -o - -target-cpu pwr7 | FileCheck %s
// RUN: %clang_cc1 -triple powerpc-unknown-aix \
// RUN:   -emit-llvm %s -o - -target-cpu pwr7 | FileCheck %s

extern vector float a;
extern vector float b;
extern vector float c;
extern vector double d;
extern vector double e;
extern vector double f;

// CHECK-LABEL: @test_flags_recipdivf(
// CHECK:    [[TMP0:%.*]] = load <4 x float>, ptr @a, align 16
// CHECK-NEXT:    [[TMP1:%.*]] = load <4 x float>, ptr @b, align 16
// CHECK-NEXT:    [[RECIPDIV:%.*]] = fdiv fast <4 x float> [[TMP0]], [[TMP1]]
// CHECK-NEXT:    [[TMP2:%.*]] = load <4 x float>, ptr @c, align 16
// CHECK-NEXT:    [[ADD:%.*]] = fadd <4 x float> [[RECIPDIV]], [[TMP2]]
// CHECK-NEXT:    ret <4 x float> [[ADD]]
//
vector float test_flags_recipdivf() {
  return __builtin_ppc_recipdivf(a, b) + c;
}

// CHECK-LABEL: @test_flags_recipdivd(
// CHECK:    [[TMP0:%.*]] = load <2 x double>, ptr @d, align 16
// CHECK-NEXT:    [[TMP1:%.*]] = load <2 x double>, ptr @e, align 16
// CHECK-NEXT:    [[RECIPDIV:%.*]] = fdiv fast <2 x double> [[TMP0]], [[TMP1]]
// CHECK-NEXT:    [[TMP2:%.*]] = load <2 x double>, ptr @f, align 16
// CHECK-NEXT:    [[ADD:%.*]] = fadd <2 x double> [[RECIPDIV]], [[TMP2]]
// CHECK-NEXT:    ret <2 x double> [[ADD]]
//
vector double test_flags_recipdivd() {
  return __builtin_ppc_recipdivd(d, e) + f;
}

// CHECK-LABEL: @test_flags_rsqrtf(
// CHECK:    [[TMP0:%.*]] = load <4 x float>, ptr @a, align 16
// CHECK-NEXT:    [[TMP1:%.*]] = call fast <4 x float> @llvm.sqrt.v4f32(<4 x float> [[TMP0]])
// CHECK-NEXT:    [[RSQRT:%.*]] = fdiv fast <4 x float> splat (float 1.000000e+00), [[TMP1]]
// CHECK-NEXT:    [[TMP2:%.*]] = load <4 x float>, ptr @b, align 16
// CHECK-NEXT:    [[ADD:%.*]] = fadd <4 x float> [[RSQRT]], [[TMP2]]
// CHECK-NEXT:    ret <4 x float> [[ADD]]
//
vector float test_flags_rsqrtf() {
  return __builtin_ppc_rsqrtf(a) + b;
}

// CHECK-LABEL: @test_flags_rsqrtd(
// CHECK:    [[TMP0:%.*]] = load <2 x double>, ptr @d, align 16
// CHECK-NEXT:    [[TMP1:%.*]] = call fast <2 x double> @llvm.sqrt.v2f64(<2 x double> [[TMP0]])
// CHECK-NEXT:    [[RSQRT:%.*]] = fdiv fast <2 x double> splat (double 1.000000e+00), [[TMP1]]
// CHECK-NEXT:    [[TMP2:%.*]] = load <2 x double>, ptr @e, align 16
// CHECK-NEXT:    [[ADD:%.*]] = fadd <2 x double> [[RSQRT]], [[TMP2]]
// CHECK-NEXT:    ret <2 x double> [[ADD]]
//
vector double test_flags_rsqrtd() {
  return __builtin_ppc_rsqrtd(d) + e;
}
