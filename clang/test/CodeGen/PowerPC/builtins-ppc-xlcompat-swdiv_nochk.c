// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -no-opaque-pointers -triple powerpc64-unknown-linux-gnu \
// RUN:   -emit-llvm %s -o - -target-cpu pwr7 | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -triple powerpc64-unknown-aix \
// RUN:   -emit-llvm %s -o - -target-cpu pwr7 | FileCheck %s

extern double a;
extern double b;
extern double c;
extern float d;
extern float e;
extern float f;

// CHECK-LABEL: @test_swdiv_nochk(
// CHECK:    [[TMP0:%.*]] = load double, double* @a, align 8
// CHECK-NEXT:    [[TMP1:%.*]] = load double, double* @b, align 8
// CHECK-NEXT:    [[SWDIV_NOCHK:%.*]] = fdiv fast double [[TMP0]], [[TMP1]]
// CHECK-NEXT:    ret double [[SWDIV_NOCHK]]
//
double test_swdiv_nochk() {
  return __swdiv_nochk(a, b);
}

// CHECK-LABEL: @test_swdivs_nochk(
// CHECK:    [[TMP0:%.*]] = load float, float* @d, align 4
// CHECK-NEXT:    [[TMP1:%.*]] = load float, float* @e, align 4
// CHECK-NEXT:    [[SWDIV_NOCHK:%.*]] = fdiv fast float [[TMP0]], [[TMP1]]
// CHECK-NEXT:    ret float [[SWDIV_NOCHK]]
//
float test_swdivs_nochk() {
  return __swdivs_nochk(d, e);
}

// CHECK-LABEL: @test_flags_swdiv_nochk(
// CHECK:    [[TMP0:%.*]] = load double, double* @a, align 8
// CHECK-NEXT:    [[TMP1:%.*]] = load double, double* @b, align 8
// CHECK-NEXT:    [[SWDIV_NOCHK:%.*]] = fdiv fast double [[TMP0]], [[TMP1]]
// CHECK-NEXT:    [[TMP2:%.*]] = load double, double* @c, align 8
// CHECK-NEXT:    [[ADD:%.*]] = fadd double [[SWDIV_NOCHK]], [[TMP2]]
// CHECK-NEXT:    ret double [[ADD]]
//
double test_flags_swdiv_nochk() {
  return __swdiv_nochk(a, b) + c;
}

// CHECK-LABEL: @test_flags_swdivs_nochk(
// CHECK:    [[TMP0:%.*]] = load float, float* @d, align 4
// CHECK-NEXT:    [[TMP1:%.*]] = load float, float* @e, align 4
// CHECK-NEXT:    [[SWDIV_NOCHK:%.*]] = fdiv fast float [[TMP0]], [[TMP1]]
// CHECK-NEXT:    [[TMP2:%.*]] = load float, float* @f, align 4
// CHECK-NEXT:    [[ADD:%.*]] = fadd float [[SWDIV_NOCHK]], [[TMP2]]
// CHECK-NEXT:    ret float [[ADD]]
//
float test_flags_swdivs_nochk() {
  return __swdivs_nochk(d, e) + f;
}

// CHECK-LABEL: @test_builtin_ppc_swdiv_nochk(
// CHECK:    [[TMP0:%.*]] = load double, double* @a, align 8
// CHECK-NEXT:    [[TMP1:%.*]] = load double, double* @b, align 8
// CHECK-NEXT:    [[SWDIV_NOCHK:%.*]] = fdiv fast double [[TMP0]], [[TMP1]]
// CHECK-NEXT:    ret double [[SWDIV_NOCHK]]
//
double test_builtin_ppc_swdiv_nochk() {
  return __builtin_ppc_swdiv_nochk(a, b);
}

// CHECK-LABEL: @test_builtin_ppc_swdivs_nochk(
// CHECK:    [[TMP0:%.*]] = load float, float* @d, align 4
// CHECK-NEXT:    [[TMP1:%.*]] = load float, float* @e, align 4
// CHECK-NEXT:    [[SWDIV_NOCHK:%.*]] = fdiv fast float [[TMP0]], [[TMP1]]
// CHECK-NEXT:    ret float [[SWDIV_NOCHK]]
//
float test_builtin_ppc_swdivs_nochk() {
  return __builtin_ppc_swdivs_nochk(d, e);
}

// CHECK-LABEL: @test_flags_builtin_ppc_swdiv_nochk(
// CHECK:    [[TMP0:%.*]] = load double, double* @a, align 8
// CHECK-NEXT:    [[TMP1:%.*]] = load double, double* @b, align 8
// CHECK-NEXT:    [[SWDIV_NOCHK:%.*]] = fdiv fast double [[TMP0]], [[TMP1]]
// CHECK-NEXT:    [[TMP2:%.*]] = load double, double* @c, align 8
// CHECK-NEXT:    [[ADD:%.*]] = fadd double [[SWDIV_NOCHK]], [[TMP2]]
// CHECK-NEXT:    ret double [[ADD]]
//
double test_flags_builtin_ppc_swdiv_nochk() {
  return __builtin_ppc_swdiv_nochk(a, b) + c;
}

// CHECK-LABEL: @test_flags_builtin_ppc_swdivs_nochk(
// CHECK:    [[TMP0:%.*]] = load float, float* @d, align 4
// CHECK-NEXT:    [[TMP1:%.*]] = load float, float* @e, align 4
// CHECK-NEXT:    [[SWDIV_NOCHK:%.*]] = fdiv fast float [[TMP0]], [[TMP1]]
// CHECK-NEXT:    [[TMP2:%.*]] = load float, float* @f, align 4
// CHECK-NEXT:    [[ADD:%.*]] = fadd float [[SWDIV_NOCHK]], [[TMP2]]
// CHECK-NEXT:    ret float [[ADD]]
//
float test_flags_builtin_ppc_swdivs_nochk() {
  return __builtin_ppc_swdivs_nochk(d, e) + f;
}
