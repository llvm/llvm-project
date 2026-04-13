// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -x c -ffreestanding %s -triple=x86_64-unknown-linux-gnu -target-feature +avx512vl -target-feature +avx512fp16 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefix=UNCONSTRAINED --check-prefix=COMMON
// RUN: %clang_cc1 -x c -ffreestanding %s -triple=x86_64-unknown-linux-gnu -target-feature +avx512vl -target-feature +avx512fp16 -ffp-exception-behavior=maytrap -DSTRICT=1 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefix=CONSTRAINED --check-prefix=COMMON
// RUN: %clang_cc1 -x c -ffreestanding %s -triple=x86_64-unknown-linux-gnu -target-feature +avx512vl -target-feature +avx512fp16 -S -o - -Wall -Werror | FileCheck %s --check-prefix=CHECK-ASM --check-prefix=COMMON
// RUN: %clang_cc1 -x c -ffreestanding %s -triple=x86_64-unknown-linux-gnu -target-feature +avx512vl -target-feature +avx512fp16 -ffp-exception-behavior=maytrap -DSTRICT=1 -S -o - -Wall -Werror | FileCheck %s --check-prefix=CHECK-ASM --check-prefix=COMMON
// RUN: %clang_cc1 -x c++ -ffreestanding %s -triple=x86_64-unknown-linux-gnu -target-feature +avx512vl -target-feature +avx512fp16 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefix=UNCONSTRAINED --check-prefix=COMMON
// RUN: %clang_cc1 -x c++ -ffreestanding %s -triple=x86_64-unknown-linux-gnu -target-feature +avx512vl -target-feature +avx512fp16 -ffp-exception-behavior=maytrap -DSTRICT=1 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefix=CONSTRAINED --check-prefix=COMMON
// RUN: %clang_cc1 -x c++ -ffreestanding %s -triple=x86_64-unknown-linux-gnu -target-feature +avx512vl -target-feature +avx512fp16 -S -o - -Wall -Werror | FileCheck %s --check-prefix=CHECK-ASM --check-prefix=COMMON
// RUN: %clang_cc1 -x c++ -ffreestanding %s -triple=x86_64-unknown-linux-gnu -target-feature +avx512vl -target-feature +avx512fp16 -ffp-exception-behavior=maytrap -DSTRICT=1 -S -o - -Wall -Werror | FileCheck %s --check-prefix=CHECK-ASM --check-prefix=COMMON

#ifdef STRICT
// Test that the constrained intrinsics are picking up the exception
// metadata from the AST instead of the global default from the command line.

#pragma float_control(except, on)
#endif


#include <immintrin.h>

__m128h test_mm_sqrt_ph(__m128h x) {
  // COMMON-LABEL: test_mm_sqrt_ph
  // UNCONSTRAINED: call {{.*}}<8 x half> @llvm.sqrt.v8f16(<8 x half> {{.*}})
  // CONSTRAINED: call {{.*}}<8 x half> @llvm.experimental.constrained.sqrt.v8f16(<8 x half> {{.*}}, metadata !{{.*}})
  // CHECK-ASM: vsqrtph %xmm{{.*}},
  return _mm_sqrt_ph(x);
}


__m256h test_mm256_sqrt_ph(__m256h x) {
  // COMMON-LABEL: test_mm256_sqrt_ph
  // UNCONSTRAINED: call {{.*}}<16 x half> @llvm.sqrt.v16f16(<16 x half> {{.*}})
  // CONSTRAINED: call {{.*}}<16 x half> @llvm.experimental.constrained.sqrt.v16f16(<16 x half> {{.*}}, metadata !{{.*}})
  // CHECK-ASM: vsqrtph %ymm{{.*}},
  return _mm256_sqrt_ph(x);
}

