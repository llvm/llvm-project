// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc64-unknown-linux-gnu \
// RUN:   -emit-llvm %s -o - -target-cpu pwr7 | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu \
// RUN:   -emit-llvm %s -o - -target-cpu pwr8 | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64-unknown-aix \
// RUN:   -emit-llvm %s -o - -target-cpu pwr7 | FileCheck %s
// RUN: %clang_cc1 -triple powerpc-unknown-aix \
// RUN:   -emit-llvm %s -o - -target-cpu pwr7 | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64-unknown-linux-gnu -ffast-math -ffp-contract=fast \
// RUN:   -emit-llvm %s -o - -target-cpu pwr7 | FileCheck %s --check-prefix CHECK-OFAST

extern double a;
extern double b;
extern float c;
extern float d;

// CHECK-LABEL:   @test_swdiv(
// CHECK:         [[TMP0:%.*]] = load double, ptr @a
// CHECK-NEXT:    [[TMP1:%.*]] = load double, ptr @b
// CHECK-NEXT:    [[SWDIV:%.*]] = fdiv double [[TMP0]], [[TMP1]]
// CHECK-NEXT:    ret double [[SWDIV]]
//
// CHECK-OFAST-LABEL:   @test_swdiv(
// CHECK-OFAST:         [[TMP0:%.*]] = load double, ptr @a
// CHECK-OFAST-NEXT:    [[TMP1:%.*]] = load double, ptr @b
// CHECK-OFAST-NEXT:    [[SWDIV:%.*]] = fdiv fast double [[TMP0]], [[TMP1]]
// CHECK-OFAST-NEXT:    ret double [[SWDIV]]
//
double test_swdiv() {
  return __swdiv(a, b);
}

// CHECK-LABEL:   @test_swdivs(
// CHECK:         [[TMP0:%.*]] = load float, ptr @c
// CHECK-NEXT:    [[TMP1:%.*]] = load float, ptr @d
// CHECK-NEXT:    [[SWDIVS:%.*]] = fdiv float [[TMP0]], [[TMP1]]
// CHECK-NEXT:    ret float [[SWDIVS]]
//
// CHECK-OFAST-LABEL:   @test_swdivs(
// CHECK-OFAST:         [[TMP0:%.*]] = load float, ptr @c
// CHECK-OFAST-NEXT:    [[TMP1:%.*]] = load float, ptr @d
// CHECK-OFAST-NEXT:    [[SWDIVS:%.*]] = fdiv fast float [[TMP0]], [[TMP1]]
// CHECK-OFAST-NEXT:    ret float [[SWDIVS]]
//
float test_swdivs() {
  return __swdivs(c, d);
}

// CHECK-LABEL:   @test_builtin_ppc_swdiv(
// CHECK:         [[TMP0:%.*]] = load double, ptr @a
// CHECK-NEXT:    [[TMP1:%.*]] = load double, ptr @b
// CHECK-NEXT:    [[SWDIV:%.*]] = fdiv double [[TMP0]], [[TMP1]]
// CHECK-NEXT:    ret double [[SWDIV]]
//
// CHECK-OFAST-LABEL:   @test_builtin_ppc_swdiv(
// CHECK-OFAST:         [[TMP0:%.*]] = load double, ptr @a
// CHECK-OFAST-NEXT:    [[TMP1:%.*]] = load double, ptr @b
// CHECK-OFAST-NEXT:    [[SWDIV:%.*]] = fdiv fast double [[TMP0]], [[TMP1]]
// CHECK-OFAST-NEXT:    ret double [[SWDIV]]
//
double test_builtin_ppc_swdiv() {
  return __builtin_ppc_swdiv(a, b);
}

// CHECK-LABEL:   @test_builtin_ppc_swdivs(
// CHECK:         [[TMP0:%.*]] = load float, ptr @c
// CHECK-NEXT:    [[TMP1:%.*]] = load float, ptr @d
// CHECK-NEXT:    [[SWDIVS:%.*]] = fdiv float [[TMP0]], [[TMP1]]
// CHECK-NEXT:    ret float [[SWDIVS]]
//
// CHECK-OFAST-LABEL:   @test_builtin_ppc_swdivs(
// CHECK-OFAST:         [[TMP0:%.*]] = load float, ptr @c
// CHECK-OFAST-NEXT:    [[TMP1:%.*]] = load float, ptr @d
// CHECK-OFAST-NEXT:    [[SWDIVS:%.*]] = fdiv fast float [[TMP0]], [[TMP1]]
// CHECK-OFAST-NEXT:    ret float [[SWDIVS]]
//
float test_builtin_ppc_swdivs() {
  return __builtin_ppc_swdivs(c, d);
}
