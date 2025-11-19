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