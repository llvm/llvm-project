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
// CHECK:    [[TMP0:%.*]] = load <4 x float>, <4 x float>* @a, align 16
// CHECK-NEXT:    [[TMP1:%.*]] = load <4 x float>, <4 x float>* @b, align 16
// CHECK-NEXT:    [[TMP2:%.*]] = load <4 x float>, <4 x float>* @a, align 16
// CHECK-NEXT:    [[TMP3:%.*]] = load <4 x float>, <4 x float>* @b, align 16
// CHECK-NEXT:    [[RECIPDIV:%.*]] = fdiv fast <4 x float> [[TMP2]], [[TMP3]]
// CHECK-NEXT:    [[TMP4:%.*]] = load <4 x float>, <4 x float>* @c, align 16
// CHECK-NEXT:    [[ADD:%.*]] = fadd <4 x float> [[RECIPDIV]], [[TMP4]]
// CHECK-NEXT:    ret <4 x float> [[ADD]]
//
vector float test_flags_recipdivf() {
  return __builtin_ppc_recipdivf(a, b) + c;
}

// CHECK-LABEL: @test_flags_recipdivd(
// CHECK:    [[TMP0:%.*]] = load <2 x double>, <2 x double>* @d, align 16
// CHECK-NEXT:    [[TMP1:%.*]] = load <2 x double>, <2 x double>* @e, align 16
// CHECK-NEXT:    [[TMP2:%.*]] = load <2 x double>, <2 x double>* @d, align 16
// CHECK-NEXT:    [[TMP3:%.*]] = load <2 x double>, <2 x double>* @e, align 16
// CHECK-NEXT:    [[RECIPDIV:%.*]] = fdiv fast <2 x double> [[TMP2]], [[TMP3]]
// CHECK-NEXT:    [[TMP4:%.*]] = load <2 x double>, <2 x double>* @f, align 16
// CHECK-NEXT:    [[ADD:%.*]] = fadd <2 x double> [[RECIPDIV]], [[TMP4]]
// CHECK-NEXT:    ret <2 x double> [[ADD]]
//
vector double test_flags_recipdivd() {
  return __builtin_ppc_recipdivd(d, e) + f;
}

// CHECK-LABEL: @test_flags_rsqrtf(
// CHECK:    [[TMP0:%.*]] = load <4 x float>, <4 x float>* @a, align 16
// CHECK-NEXT:    [[TMP1:%.*]] = load <4 x float>, <4 x float>* @a, align 16
// CHECK-NEXT:    [[TMP2:%.*]] = call fast <4 x float> @llvm.sqrt.v4f32(<4 x float> [[TMP1]])
// CHECK-NEXT:    [[RSQRT:%.*]] = fdiv fast <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, [[TMP2]]
// CHECK-NEXT:    [[TMP3:%.*]] = load <4 x float>, <4 x float>* @b, align 16
// CHECK-NEXT:    [[ADD:%.*]] = fadd <4 x float> [[RSQRT]], [[TMP3]]
// CHECK-NEXT:    ret <4 x float> [[ADD]]
//
vector float test_flags_rsqrtf() {
  return __builtin_ppc_rsqrtf(a) + b;
}

// CHECK-LABEL: @test_flags_rsqrtd(
// CHECK:    [[TMP0:%.*]] = load <2 x double>, <2 x double>* @d, align 16
// CHECK-NEXT:    [[TMP1:%.*]] = load <2 x double>, <2 x double>* @d, align 16
// CHECK-NEXT:    [[TMP2:%.*]] = call fast <2 x double> @llvm.sqrt.v2f64(<2 x double> [[TMP1]])
// CHECK-NEXT:    [[RSQRT:%.*]] = fdiv fast <2 x double> <double 1.000000e+00, double 1.000000e+00>, [[TMP2]]
// CHECK-NEXT:    [[TMP3:%.*]] = load <2 x double>, <2 x double>* @e, align 16
// CHECK-NEXT:    [[ADD:%.*]] = fadd <2 x double> [[RSQRT]], [[TMP3]]
// CHECK-NEXT:    ret <2 x double> [[ADD]]
//
vector double test_flags_rsqrtd() {
  return __builtin_ppc_rsqrtd(d) + e;
}
