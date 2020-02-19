// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-unknown-linux-gnu -target-feature +sse -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefix=UNCONSTRAINED --check-prefix=COMMON --check-prefix=COMMONIR
// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-unknown-linux-gnu -target-feature +sse -ffp-exception-behavior=strict -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefix=CONSTRAINED --check-prefix=COMMON --check-prefix=COMMONIR
// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-unknown-linux-gnu -target-feature +sse -S %s -o - -Wall -Werror | FileCheck %s --check-prefix=CHECK-ASM --check-prefix=COMMON
// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-unknown-linux-gnu -target-feature +sse -ffp-exception-behavior=strict -S %s -o - -Wall -Werror | FileCheck %s --check-prefix=CHECK-ASM --check-prefix=COMMON


#include <immintrin.h>

__m128 test_mm_sqrt_ps(__m128 x) {
  // COMMON-LABEL: test_mm_sqrt_ps
  // UNCONSTRAINED: call <4 x float> @llvm.sqrt.v4f32(<4 x float> {{.*}})
  // CONSTRAINED: call <4 x float> @llvm.experimental.constrained.sqrt.v4f32(<4 x float> {{.*}}, metadata !{{.*}})
  // CHECK-ASM: sqrtps
  return _mm_sqrt_ps(x);
}

__m128 test_sqrt_ss(__m128 x) {
  // COMMON-LABEL: test_sqrt_ss
  // COMMONIR: extractelement <4 x float> {{.*}}, i64 0
  // UNCONSTRAINED: call float @llvm.sqrt.f32(float {{.*}})
  // CONSTRAINED: call float @llvm.experimental.constrained.sqrt.f32(float {{.*}}, metadata !{{.*}})
  // CHECK-ASM: sqrtss
  // COMMONIR: insertelement <4 x float> {{.*}}, float {{.*}}, i64 0
  return _mm_sqrt_ss(x);
}

