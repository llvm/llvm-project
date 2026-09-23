// RUN: %clang_cc1 -fveclib=Darwin_libsystem_m -triple arm64-apple-darwin %s -target-cpu apple-a7 -vectorize-loops -emit-llvm -O3 -o - | FileCheck %s

// REQUIRES: aarch64-registered-target

// Make sure -fveclib=Darwin_libsystem_m gets passed through to LLVM as
// expected: a call to _simd_sin_f4 should be generated.

extern float sinf(float);

// CHECK-LABEL: define{{.*}}@apply_sin
// CHECK: call <4 x float> @_simd_sin_f4(
//
void apply_sin(float *A, float *B, float *C, unsigned N) {
  for (unsigned i = 0; i < N; i++)
    C[i] = sinf(A[i]) + sinf(B[i]);
}

// __exp10f is a Darwin-only spelling with no plain-named counterpart in
// math.h, so check that the mapping keyed on it is reachable from C.

extern float __exp10f(float);

// CHECK-LABEL: define{{.*}}@apply_exp10
// CHECK: call <4 x float> @_simd_exp10_f4(
//
void apply_exp10(float *A, float *B, float *C, unsigned N) {
  for (unsigned i = 0; i < N; i++)
    C[i] = __exp10f(A[i]) + __exp10f(B[i]);
}
