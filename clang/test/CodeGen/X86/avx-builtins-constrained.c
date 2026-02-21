// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -x c -ffreestanding %s -triple=x86_64-unknown-linux-gnu -target-feature +avx -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefix=UNCONSTRAINED --check-prefix=COMMON
// RUN: %clang_cc1 -x c -ffreestanding %s -triple=x86_64-unknown-linux-gnu -target-feature +avx -ffp-exception-behavior=maytrap -DSTRICT=1 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefix=CONSTRAINED --check-prefix=COMMON
// RUN: %clang_cc1 -x c -ffreestanding %s -triple=x86_64-unknown-linux-gnu -target-feature +avx -S -o - -Wall -Werror | FileCheck %s --check-prefix=CHECK-ASM --check-prefix=COMMON
// RUN: %clang_cc1 -x c -ffreestanding %s -triple=x86_64-unknown-linux-gnu -target-feature +avx -ffp-exception-behavior=maytrap -DSTRICT=1 -S -o - -Wall -Werror | FileCheck %s --check-prefix=CHECK-ASM --check-prefix=COMMON
// RUN: %clang_cc1 -x c++ -ffreestanding %s -triple=x86_64-unknown-linux-gnu -target-feature +avx -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefix=UNCONSTRAINED --check-prefix=COMMON
// RUN: %clang_cc1 -x c++ -ffreestanding %s -triple=x86_64-unknown-linux-gnu -target-feature +avx -ffp-exception-behavior=maytrap -DSTRICT=1 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefix=CONSTRAINED --check-prefix=COMMON
// RUN: %clang_cc1 -x c++ -ffreestanding %s -triple=x86_64-unknown-linux-gnu -target-feature +avx -S -o - -Wall -Werror | FileCheck %s --check-prefix=CHECK-ASM --check-prefix=COMMON
// RUN: %clang_cc1 -x c++ -ffreestanding %s -triple=x86_64-unknown-linux-gnu -target-feature +avx -ffp-exception-behavior=maytrap -DSTRICT=1 -S -o - -Wall -Werror | FileCheck %s --check-prefix=CHECK-ASM --check-prefix=COMMON

#ifdef STRICT
// Test that the constrained intrinsics are picking up the exception
// metadata from the AST instead of the global default from the command line.

#pragma float_control(except, on)
#endif


#include <immintrin.h>

__m256 test_mm256_sqrt_ps(__m256 x) {
  // COMMON-LABEL: test_mm256_sqrt_ps
  // UNCONSTRAINED: call {{.*}}<8 x float> @llvm.sqrt.v8f32(<8 x float> {{.*}})
  // CONSTRAINED: call {{.*}}<8 x float> @llvm.experimental.constrained.sqrt.v8f32(<8 x float> {{.*}}, metadata !{{.*}})
  // CHECK-ASM: vsqrtps %ymm{{.*}},
  return _mm256_sqrt_ps(x);
}

__m256d test_mm256_sqrt_pd(__m256d x) {
  // COMMON-LABEL: test_mm256_sqrt_pd
  // UNCONSTRAINED: call {{.*}}<4 x double> @llvm.sqrt.v4f64(<4 x double> {{.*}})
  // CONSTRAINED: call {{.*}}<4 x double> @llvm.experimental.constrained.sqrt.v4f64(<4 x double> {{.*}}, metadata !{{.*}})
  // CHECK-ASM: vsqrtpd %ymm{{.*}},
  return _mm256_sqrt_pd(x);
}

__m256d test_mm256_round_pd_mxcsr(__m256d x) {
  // CONSTRAINED-LABEL: test_mm256_round_pd_mxcsr
  // CONSTRAINED: %{{.*}} = call <4 x double> @llvm.x86.avx.round.pd.256(<4 x double> %{{.*}}, i32 12)
  return _mm256_round_pd(x, 0b1100);
}

__m256d test_mm256_round_pd_fround_no_exc(__m256d x) {
  // CONSTRAINED-LABEL: test_mm256_round_pd_fround_no_exc
  // CONSTRAINED: %{{.*}} = call <4 x double> @llvm.x86.avx.round.pd.256(<4 x double> %{{.*}}, i32 0)
  return _mm256_round_pd(x, 0b0000);
}

__m256d test_mm256_round_pd_trunc(__m256d x) {
  // CONSTRAINED-LABEL: test_mm256_round_pd_trunc
  // CONSTRAINED: %{{.*}} = call <4 x double> @llvm.experimental.constrained.trunc.v4f64(<4 x double> %{{.*}}, metadata !"fpexcept.ignore")
  return _mm256_round_pd(x, 0b1011);
}

__m256 test_mm256_round_ps_mxcsr(__m256 x) {
  // CONSTRAINED-LABEL: test_mm256_round_ps_mxcsr
  // CONSTRAINED: %{{.*}} = call <8 x float> @llvm.x86.avx.round.ps.256(<8 x float> %{{.*}}, i32 12)
  return _mm256_round_ps(x, 0b1100);
}

__m256 test_mm256_round_ps_fround_no_exc(__m256 x) {
  // CONSTRAINED-LABEL: test_mm256_round_ps_fround_no_exc
  // CONSTRAINED: %{{.*}} = call <8 x float> @llvm.x86.avx.round.ps.256(<8 x float> %{{.*}}, i32 0)
  return _mm256_round_ps(x, 0b0000);
}

__m256 test_mm256_round_ps_trunc(__m256 x) {
  // CONSTRAINED-LABEL: test_mm256_round_ps_trunc
  // CONSTRAINED: %{{.*}} = call <8 x float> @llvm.experimental.constrained.trunc.v8f32(<8 x float> %{{.*}}, metadata !"fpexcept.ignore")
  return _mm256_round_ps(x, 0b1011);
}
