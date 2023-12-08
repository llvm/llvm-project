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
