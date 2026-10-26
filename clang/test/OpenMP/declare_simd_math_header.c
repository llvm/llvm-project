// REQUIRES: x86-registered-target

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -fopenmp -ffast-math \
// RUN:   -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -fopenmp -ffast-math \
// RUN:   -target-feature +avx -target-feature +avx2 -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=AVX2
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -fopenmp -ffast-math \
// RUN:   -target-feature +avx -target-feature +avx2 \
// RUN:   -target-feature +avx512f -mprefer-vector-width=256 -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=PREFER256

// Check declare-simd libm functions declared in a header: the vector variant
// mappings are attached to the scalar declaration, only for ISAs the target
// supports and no wider than -mprefer-vector-width, and the loop uses the
// variant instead of the intrinsic.

#include "Inputs/declare-simd-math.h"

double loop_acosh(double *restrict d, const double *restrict x, int n) {
#pragma omp simd
  for (int i = 0; i < n; i++)
    d[i] = acosh(x[i]);
}

double loop_sin(double *restrict d, const double *restrict x, int n) {
#pragma omp simd
  for (int i = 0; i < n; i++)
    d[i] = sin(x[i]);
}

// The vector functions are kept in llvm.compiler.used so that GlobalOpt does
// not drop their declarations before vectorization.
// CHECK: @llvm.compiler.used = appending global [2 x ptr] {{.*}}_ZGVbN2v_acosh{{.*}}_ZGVbN2v_sin
// Both loops use the SSE variants on the base x86-64 target, and the
// middle-end must not fold the calls back into intrinsics.
// CHECK: call{{.*}}<2 x double> @_ZGVbN2v_acosh(
// CHECK: call{{.*}}<2 x double> @_ZGVbN2v_sin(
// CHECK-NOT: llvm.acosh
// CHECK-NOT: llvm.sin
// CHECK: "vector-function-abi-variant"="_ZGVbN2v_acosh"
// CHECK: "vector-function-abi-variant"="_ZGVbN2v_sin"

// AVX2 additionally gets the 256-bit variants, the AVX2 one being preferred
// over the AVX one, but not the AVX-512 one.
// AVX2: @llvm.compiler.used = appending global [6 x ptr] {{.*}}_ZGVcN4v_acosh{{.*}}_ZGVdN4v_sin
// AVX2-NOT: ptr @_ZGVeN8v
// AVX2: call{{.*}}<4 x double> @_ZGVdN4v_acosh(
// AVX2: "vector-function-abi-variant"="_ZGVdN4v_acosh,_ZGVcN4v_acosh,_ZGVbN2v_acosh"

// -mprefer-vector-width=256 on an AVX-512 target must not offer or use the
// 512-bit variants.
// PREFER256: @llvm.compiler.used = appending global [6 x ptr] {{.*}}_ZGVdN4v_sin
// PREFER256-NOT: ptr @_ZGVeN8v
// PREFER256: call{{.*}}<4 x double> @_ZGVdN4v_acosh(
// PREFER256: "vector-function-abi-variant"="_ZGVdN4v_acosh,_ZGVcN4v_acosh,_ZGVbN2v_acosh"
