// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +neon -fopenmp -x c -std=c11 -emit-llvm %s -o - -femit-all-decls | FileCheck %s

// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +sve -fopenmp -x c -std=c11 -emit-llvm %s -o - -femit-all-decls | FileCheck %s --check-prefix=SVE

#pragma omp declare simd
#pragma omp declare simd simdlen(4) notinbranch
double _Complex double_complex(double _Complex);
// CHECK:  "_ZGVnM2v_double_complex" "_ZGVnN2v_double_complex" "_ZGVnN4v_double_complex"
// CHECK-NOT: double_complex
// SVE:   "_ZGVsM4v_double_complex" "_ZGVsMxv_double_complex"
// SVE-NOT: double_complex

#pragma omp declare simd
#pragma omp declare simd simdlen(8) notinbranch
float _Complex float_complex(float _Complex);
// CHECK:  "_ZGVnM2v_float_complex" "_ZGVnN2v_float_complex" "_ZGVnN8v_float_complex"
// CHECK-NOT: float_complex
// SVE: "_ZGVsM8v_float_complex" "_ZGVsMxv_float_complex"
// SVE-NOT: float_complex

static double _Complex *DC;
static float _Complex *DF;
void call_the_complex_functions() {
  double_complex(*DC);
  float_complex(*DF);
}
