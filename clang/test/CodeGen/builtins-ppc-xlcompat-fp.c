// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc64le-unknown-unknown \
// RUN:   -emit-llvm %s -o - -target-cpu pwr8 | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64-unknown-aix \
// RUN:   -emit-llvm %s -o - -target-cpu pwr7 | FileCheck %s
// RUN: %clang_cc1 -triple powerpc-unknown-aix \
// RUN:   -emit-llvm %s -o - -target-cpu pwr7 | FileCheck %s

extern double a;
extern double b;
extern double c;
extern float d;
extern float e;
extern float f;

// CHECK-LABEL: @test_fric(
// CHECK:    [[TMP0:%.*]] = load double, double* @a, align 8
// CHECK-NEXT:    [[TMP1:%.*]] = load double, double* @a, align 8
// CHECK-NEXT:    [[TMP2:%.*]] = call double @llvm.rint.f64(double [[TMP1]])
// CHECK-NEXT:    ret double [[TMP2]]
//
double test_fric() {
  return __fric(a);
}

// CHECK-LABEL: @test_frim(
// CHECK:    [[TMP0:%.*]] = load double, double* @a, align 8
// CHECK-NEXT:    [[TMP1:%.*]] = load double, double* @a, align 8
// CHECK-NEXT:    [[TMP2:%.*]] = call double @llvm.floor.f64(double [[TMP1]])
// CHECK-NEXT:    ret double [[TMP2]]
//
double test_frim() {
  return __frim(a);
}

// CHECK-LABEL: @test_frims(
// CHECK:    [[TMP0:%.*]] = load float, float* @d, align 4
// CHECK-NEXT:    [[TMP1:%.*]] = load float, float* @d, align 4
// CHECK-NEXT:    [[TMP2:%.*]] = call float @llvm.floor.f32(float [[TMP1]])
// CHECK-NEXT:    ret float [[TMP2]]
//
float test_frims() {
  return __frims(d);
}

// CHECK-LABEL: @test_frin(
// CHECK:    [[TMP0:%.*]] = load double, double* @a, align 8
// CHECK-NEXT:    [[TMP1:%.*]] = load double, double* @a, align 8
// CHECK-NEXT:    [[TMP2:%.*]] = call double @llvm.round.f64(double [[TMP1]])
// CHECK-NEXT:    ret double [[TMP2]]
//
double test_frin() {
  return __frin(a);
}

// CHECK-LABEL: @test_frins(
// CHECK:    [[TMP0:%.*]] = load float, float* @d, align 4
// CHECK-NEXT:    [[TMP1:%.*]] = load float, float* @d, align 4
// CHECK-NEXT:    [[TMP2:%.*]] = call float @llvm.round.f32(float [[TMP1]])
// CHECK-NEXT:    ret float [[TMP2]]
//
float test_frins() {
  return __frins(d);
}

// CHECK-LABEL: @test_frip(
// CHECK:    [[TMP0:%.*]] = load double, double* @a, align 8
// CHECK-NEXT:    [[TMP1:%.*]] = load double, double* @a, align 8
// CHECK-NEXT:    [[TMP2:%.*]] = call double @llvm.ceil.f64(double [[TMP1]])
// CHECK-NEXT:    ret double [[TMP2]]
//
double test_frip() {
  return __frip(a);
}

// CHECK-LABEL: @test_frips(
// CHECK:    [[TMP0:%.*]] = load float, float* @d, align 4
// CHECK-NEXT:    [[TMP1:%.*]] = load float, float* @d, align 4
// CHECK-NEXT:    [[TMP2:%.*]] = call float @llvm.ceil.f32(float [[TMP1]])
// CHECK-NEXT:    ret float [[TMP2]]
//
float test_frips() {
  return __frips(d);
}

// CHECK-LABEL: @test_friz(
// CHECK:    [[TMP0:%.*]] = load double, double* @a, align 8
// CHECK-NEXT:    [[TMP1:%.*]] = load double, double* @a, align 8
// CHECK-NEXT:    [[TMP2:%.*]] = call double @llvm.trunc.f64(double [[TMP1]])
// CHECK-NEXT:    ret double [[TMP2]]
//
double test_friz() {
  return __friz(a);
}

// CHECK-LABEL: @test_frizs(
// CHECK:    [[TMP0:%.*]] = load float, float* @d, align 4
// CHECK-NEXT:    [[TMP1:%.*]] = load float, float* @d, align 4
// CHECK-NEXT:    [[TMP2:%.*]] = call float @llvm.trunc.f32(float [[TMP1]])
// CHECK-NEXT:    ret float [[TMP2]]
//
float test_frizs() {
  return __frizs(d);
}

// CHECK-LABEL: @test_fsel(
// CHECK:    [[TMP0:%.*]] = load double, double* @a, align 8
// CHECK-NEXT:    [[TMP1:%.*]] = load double, double* @b, align 8
// CHECK-NEXT:    [[TMP2:%.*]] = load double, double* @c, align 8
// CHECK-NEXT:    [[TMP3:%.*]] = call double @llvm.ppc.fsel(double [[TMP0]], double [[TMP1]], double [[TMP2]])
// CHECK-NEXT:    ret double [[TMP3]]
//
double test_fsel() {
  return __fsel(a, b, c);
}

// CHECK-LABEL: @test_fsels(
// CHECK:    [[TMP0:%.*]] = load float, float* @d, align 4
// CHECK-NEXT:    [[TMP1:%.*]] = load float, float* @e, align 4
// CHECK-NEXT:    [[TMP2:%.*]] = load float, float* @f, align 4
// CHECK-NEXT:    [[TMP3:%.*]] = call float @llvm.ppc.fsels(float [[TMP0]], float [[TMP1]], float [[TMP2]])
// CHECK-NEXT:    ret float [[TMP3]]
//
float test_fsels() {
  return __fsels(d, e, f);
}

// CHECK-LABEL: @test_frsqrte(
// CHECK:    [[TMP0:%.*]] = load double, double* @a, align 8
// CHECK-NEXT:    [[TMP1:%.*]] = call double @llvm.ppc.frsqrte(double [[TMP0]])
// CHECK-NEXT:    ret double [[TMP1]]
//
double test_frsqrte() {
  return __frsqrte(a);
}

// CHECK-LABEL: @test_frsqrtes(
// CHECK:    [[TMP0:%.*]] = load float, float* @d, align 4
// CHECK-NEXT:    [[TMP1:%.*]] = call float @llvm.ppc.frsqrtes(float [[TMP0]])
// CHECK-NEXT:    ret float [[TMP1]]
//
float test_frsqrtes() {
  return __frsqrtes(d);
}

// CHECK-LABEL: @test_fsqrt(
// CHECK:    [[TMP0:%.*]] = load double, double* @a, align 8
// CHECK-NEXT:    [[TMP1:%.*]] = load double, double* @a, align 8
// CHECK-NEXT:    [[TMP2:%.*]] = call double @llvm.sqrt.f64(double [[TMP1]])
// CHECK-NEXT:    ret double [[TMP2]]
//
double test_fsqrt() {
  return __fsqrt(a);
}

// CHECK-LABEL: @test_fsqrts(
// CHECK:    [[TMP0:%.*]] = load float, float* @d, align 4
// CHECK-NEXT:    [[TMP1:%.*]] = load float, float* @d, align 4
// CHECK-NEXT:    [[TMP2:%.*]] = call float @llvm.sqrt.f32(float [[TMP1]])
// CHECK-NEXT:    ret float [[TMP2]]
//
float test_fsqrts() {
  return __fsqrts(d);
}

// CHECK-LABEL: @test_builtin_ppc_fric(
// CHECK:    [[TMP0:%.*]] = load double, double* @a, align 8
// CHECK-NEXT:    [[TMP1:%.*]] = load double, double* @a, align 8
// CHECK-NEXT:    [[TMP2:%.*]] = call double @llvm.rint.f64(double [[TMP1]])
// CHECK-NEXT:    ret double [[TMP2]]
//
double test_builtin_ppc_fric() {
  return __builtin_ppc_fric(a);
}

// CHECK-LABEL: @test_builtin_ppc_frim(
// CHECK:    [[TMP0:%.*]] = load double, double* @a, align 8
// CHECK-NEXT:    [[TMP1:%.*]] = load double, double* @a, align 8
// CHECK-NEXT:    [[TMP2:%.*]] = call double @llvm.floor.f64(double [[TMP1]])
// CHECK-NEXT:    ret double [[TMP2]]
//
double test_builtin_ppc_frim() {
  return __builtin_ppc_frim(a);
}

// CHECK-LABEL: @test_builtin_ppc_frims(
// CHECK:    [[TMP0:%.*]] = load float, float* @d, align 4
// CHECK-NEXT:    [[TMP1:%.*]] = load float, float* @d, align 4
// CHECK-NEXT:    [[TMP2:%.*]] = call float @llvm.floor.f32(float [[TMP1]])
// CHECK-NEXT:    ret float [[TMP2]]
//
float test_builtin_ppc_frims() {
  return __builtin_ppc_frims(d);
}

// CHECK-LABEL: @test_builtin_ppc_frin(
// CHECK:    [[TMP0:%.*]] = load double, double* @a, align 8
// CHECK-NEXT:    [[TMP1:%.*]] = load double, double* @a, align 8
// CHECK-NEXT:    [[TMP2:%.*]] = call double @llvm.round.f64(double [[TMP1]])
// CHECK-NEXT:    ret double [[TMP2]]
//
double test_builtin_ppc_frin() {
  return __builtin_ppc_frin(a);
}

// CHECK-LABEL: @test_builtin_ppc_frins(
// CHECK:    [[TMP0:%.*]] = load float, float* @d, align 4
// CHECK-NEXT:    [[TMP1:%.*]] = load float, float* @d, align 4
// CHECK-NEXT:    [[TMP2:%.*]] = call float @llvm.round.f32(float [[TMP1]])
// CHECK-NEXT:    ret float [[TMP2]]
//
float test_builtin_ppc_frins() {
  return __builtin_ppc_frins(d);
}

// CHECK-LABEL: @test_builtin_ppc_frip(
// CHECK:    [[TMP0:%.*]] = load double, double* @a, align 8
// CHECK-NEXT:    [[TMP1:%.*]] = load double, double* @a, align 8
// CHECK-NEXT:    [[TMP2:%.*]] = call double @llvm.ceil.f64(double [[TMP1]])
// CHECK-NEXT:    ret double [[TMP2]]
//
double test_builtin_ppc_frip() {
  return __builtin_ppc_frip(a);
}

// CHECK-LABEL: @test_builtin_ppc_frips(
// CHECK:    [[TMP0:%.*]] = load float, float* @d, align 4
// CHECK-NEXT:    [[TMP1:%.*]] = load float, float* @d, align 4
// CHECK-NEXT:    [[TMP2:%.*]] = call float @llvm.ceil.f32(float [[TMP1]])
// CHECK-NEXT:    ret float [[TMP2]]
//
float test_builtin_ppc_frips() {
  return __builtin_ppc_frips(d);
}

// CHECK-LABEL: @test_builtin_ppc_friz(
// CHECK:    [[TMP0:%.*]] = load double, double* @a, align 8
// CHECK-NEXT:    [[TMP1:%.*]] = load double, double* @a, align 8
// CHECK-NEXT:    [[TMP2:%.*]] = call double @llvm.trunc.f64(double [[TMP1]])
// CHECK-NEXT:    ret double [[TMP2]]
//
double test_builtin_ppc_friz() {
  return __builtin_ppc_friz(a);
}

// CHECK-LABEL: @test_builtin_ppc_frizs(
// CHECK:    [[TMP0:%.*]] = load float, float* @d, align 4
// CHECK-NEXT:    [[TMP1:%.*]] = load float, float* @d, align 4
// CHECK-NEXT:    [[TMP2:%.*]] = call float @llvm.trunc.f32(float [[TMP1]])
// CHECK-NEXT:    ret float [[TMP2]]
//
float test_builtin_ppc_frizs() {
  return __builtin_ppc_frizs(d);
}

// CHECK-LABEL: @test_builtin_ppc_fsel(
// CHECK:    [[TMP0:%.*]] = load double, double* @a, align 8
// CHECK-NEXT:    [[TMP1:%.*]] = load double, double* @b, align 8
// CHECK-NEXT:    [[TMP2:%.*]] = load double, double* @c, align 8
// CHECK-NEXT:    [[TMP3:%.*]] = call double @llvm.ppc.fsel(double [[TMP0]], double [[TMP1]], double [[TMP2]])
// CHECK-NEXT:    ret double [[TMP3]]
//
double test_builtin_ppc_fsel() {
  return __builtin_ppc_fsel(a, b, c);
}

// CHECK-LABEL: @test_builtin_ppc_fsels(
// CHECK:    [[TMP0:%.*]] = load float, float* @d, align 4
// CHECK-NEXT:    [[TMP1:%.*]] = load float, float* @e, align 4
// CHECK-NEXT:    [[TMP2:%.*]] = load float, float* @f, align 4
// CHECK-NEXT:    [[TMP3:%.*]] = call float @llvm.ppc.fsels(float [[TMP0]], float [[TMP1]], float [[TMP2]])
// CHECK-NEXT:    ret float [[TMP3]]
//
float test_builtin_ppc_fsels() {
  return __builtin_ppc_fsels(d, e, f);
}

// CHECK-LABEL: @test_builtin_ppc_frsqrte(
// CHECK:    [[TMP0:%.*]] = load double, double* @a, align 8
// CHECK-NEXT:    [[TMP1:%.*]] = call double @llvm.ppc.frsqrte(double [[TMP0]])
// CHECK-NEXT:    ret double [[TMP1]]
//
double test_builtin_ppc_frsqrte() {
  return __builtin_ppc_frsqrte(a);
}

// CHECK-LABEL: @test_builtin_ppc_frsqrtes(
// CHECK:    [[TMP0:%.*]] = load float, float* @d, align 4
// CHECK-NEXT:    [[TMP1:%.*]] = call float @llvm.ppc.frsqrtes(float [[TMP0]])
// CHECK-NEXT:    ret float [[TMP1]]
//
float test_builtin_ppc_frsqrtes() {
  return __builtin_ppc_frsqrtes(d);
}

// CHECK-LABEL: @test_builtin_ppc_fsqrt(
// CHECK:    [[TMP0:%.*]] = load double, double* @a, align 8
// CHECK-NEXT:    [[TMP1:%.*]] = load double, double* @a, align 8
// CHECK-NEXT:    [[TMP2:%.*]] = call double @llvm.sqrt.f64(double [[TMP1]])
// CHECK-NEXT:    ret double [[TMP2]]
//
double test_builtin_ppc_fsqrt() {
  return __builtin_ppc_fsqrt(a);
}

// CHECK-LABEL: @test_builtin_ppc_fsqrts(
// CHECK:    [[TMP0:%.*]] = load float, float* @d, align 4
// CHECK-NEXT:    [[TMP1:%.*]] = load float, float* @d, align 4
// CHECK-NEXT:    [[TMP2:%.*]] = call float @llvm.sqrt.f32(float [[TMP1]])
// CHECK-NEXT:    ret float [[TMP2]]
//
float test_builtin_ppc_fsqrts() {
  return __builtin_ppc_fsqrts(d);
}
