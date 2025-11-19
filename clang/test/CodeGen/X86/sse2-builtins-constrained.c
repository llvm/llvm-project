// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -x c -ffreestanding %s -triple=x86_64-unknown-linux-gnu -target-feature +sse2 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefix=UNCONSTRAINED --check-prefix=COMMON --check-prefix=COMMONIR
// RUN: %clang_cc1 -x c -ffreestanding %s -triple=x86_64-unknown-linux-gnu -target-feature +sse2 -ffp-exception-behavior=maytrap -DSTRICT=1 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefix=CONSTRAINED --check-prefix=COMMON --check-prefix=COMMONIR
// RUN: %clang_cc1 -x c -ffreestanding %s -triple=x86_64-unknown-linux-gnu -target-feature +sse2 -S -o - -Wall -Werror | FileCheck %s --check-prefix=CHECK-ASM --check-prefix=COMMON
// RUN: %clang_cc1 -x c -ffreestanding %s -triple=x86_64-unknown-linux-gnu -target-feature +sse2 -ffp-exception-behavior=maytrap -DSTRICT=1 -S -o - -Wall -Werror | FileCheck %s --check-prefix=CHECK-ASM --check-prefix=COMMON
// RUN: %clang_cc1 -x c++ -ffreestanding %s -triple=x86_64-unknown-linux-gnu -target-feature +sse2 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefix=UNCONSTRAINED --check-prefix=COMMON --check-prefix=COMMONIR
// RUN: %clang_cc1 -x c++ -ffreestanding %s -triple=x86_64-unknown-linux-gnu -target-feature +sse2 -ffp-exception-behavior=maytrap -DSTRICT=1 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefix=CONSTRAINED --check-prefix=COMMON --check-prefix=COMMONIR
// RUN: %clang_cc1 -x c++ -ffreestanding %s -triple=x86_64-unknown-linux-gnu -target-feature +sse2 -S -o - -Wall -Werror | FileCheck %s --check-prefix=CHECK-ASM --check-prefix=COMMON
// RUN: %clang_cc1 -x c++ -ffreestanding %s -triple=x86_64-unknown-linux-gnu -target-feature +sse2 -ffp-exception-behavior=maytrap -DSTRICT=1 -S -o - -Wall -Werror | FileCheck %s --check-prefix=CHECK-ASM --check-prefix=COMMON

#ifdef STRICT
// Test that the constrained intrinsics are picking up the exception
// metadata from the AST instead of the global default from the command line.

#pragma float_control(except, on)
#endif


#include <immintrin.h>

__m128d test_mm_sqrt_pd(__m128d x) {
  // COMMON-LABEL: test_mm_sqrt_pd
  // UNCONSTRAINED: call {{.*}}<2 x double> @llvm.sqrt.v2f64(<2 x double> {{.*}})
  // CONSTRAINED: call {{.*}}<2 x double> @llvm.experimental.constrained.sqrt.v2f64(<2 x double> {{.*}}, metadata !{{.*}})
  // CHECK-ASM: sqrtpd
  return _mm_sqrt_pd(x);
}

__m128d test_sqrt_sd(__m128d x, __m128d y) {
  // COMMON-LABEL: test_sqrt_sd
  // COMMONIR: extractelement <2 x double> {{.*}}, i64 0
  // UNCONSTRAINED: call double @llvm.sqrt.f64(double {{.*}})
  // CONSTRAINED: call double @llvm.experimental.constrained.sqrt.f64(double {{.*}}, metadata !{{.*}})
  // CHECK-ASM: sqrtsd
  // COMMONIR: insertelement <2 x double> {{.*}}, double {{.*}}, i64 0
  return _mm_sqrt_sd(x, y);
}

