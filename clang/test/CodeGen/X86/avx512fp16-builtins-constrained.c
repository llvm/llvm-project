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

__m128h test_mm_sqrt_sh(__m128h x, __m128h y) {
  // COMMON-LABEL: test_mm_sqrt_sh
  // UNCONSTRAINED: call {{.*}}half @llvm.sqrt.f16(half {{.*}})
  // CONSTRAINED: call {{.*}}half @llvm.experimental.constrained.sqrt.f16(half {{.*}}, metadata !{{.*}})
  // CHECK-ASM: vsqrtsh %xmm{{.*}},
  return _mm_sqrt_sh(x, y);
}

__m128h test_mm_mask_sqrt_sh(__m128h __W, __mmask8 __U, __m128h __A, __m128h __B){
  // COMMON-LABEL: test_mm_mask_sqrt_sh
  // COMMONIR: extractelement <8 x half> %{{.*}}, i64 0
  // UNCONSTRAINED: call {{.*}}half @llvm.sqrt.f16(half %{{.*}})
  // CONSTRAINED: call {{.*}}half @llvm.experimental.constrained.sqrt.f16(half %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vsqrtsh %xmm{{.*}},
  // COMMONIR-NEXT: extractelement <8 x half> %{{.*}}, i64 0
  // COMMONIR-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // COMMONIR-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // COMMONIR-NEXT: select i1 {{.*}}, half {{.*}}, half {{.*}}
  // COMMONIR-NEXT: insertelement <8 x half> %{{.*}}, half {{.*}}, i64 0
  return _mm_mask_sqrt_sh(__W,__U,__A,__B);
}

__m128h test_mm_maskz_sqrt_sh(__mmask8 __U, __m128h __A, __m128h __B){
  // COMMON-LABEL: test_mm_maskz_sqrt_sh
  // COMMONIR: extractelement <2 x half> %{{.*}}, i64 0
  // UNCONSTRAINED: call {{.*}}half @llvm.sqrt.f16(half %{{.*}})
  // CONSTRAINED: call {{.*}}half @llvm.experimental.constrained.sqrt.f16(half %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vsqrtsh %xmm{{.*}},
  // COMMONIR-NEXT: extractelement <2 x half> %{{.*}}, i64 0
  // COMMONIR-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // COMMONIR-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // COMMONIR-NEXT: select i1 {{.*}}, half {{.*}}, half {{.*}}
  // COMMONIR-NEXT: insertelement <2 x half> %{{.*}}, half {{.*}}, i64 0
  return _mm_maskz_sqrt_sh(__U,__A,__B);
}

__m512h test_mm512_sqrt_ph(__m512h x) {
  // COMMON-LABEL: test_mm512_sqrt_ph
  // UNCONSTRAINED: call {{.*}}<32 x half> @llvm.sqrt.v32f16(<32 x half> {{.*}})
  // CONSTRAINED: call {{.*}}<32 x half> @llvm.experimental.constrained.sqrt.v32f16(<32 x half> {{.*}}, metadata !{{.*}})
  // CHECK-ASM: vsqrtph %zmm{{.*}},
  return _mm512_sqrt_ph(x);
}

__m512h test_mm512_mask_sqrt_ph (__m512h __W, __mmask32 __U, __m512h __A)
{
  // COMMON-LABEL: test_mm512_mask_sqrt_ph
  // UNCONSTRAINED: call {{.*}}<32 x half> @llvm.sqrt.v32f16(<32 x half> %{{.*}})
  // CONSTRAINED: call {{.*}}<32 x half> @llvm.experimental.constrained.sqrt.v32f16(<32 x half> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vsqrtph %zmm{{.*}},
  // COMMONIR: bitcast i32 %{{.*}} to <32 x i1>
  // COMMONIR: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}
  return _mm512_mask_sqrt_ph (__W,__U,__A);
}

__m512h test_mm512_maskz_sqrt_ph (__mmask32 __U, __m512h __A)
{
  // COMMON-LABEL: test_mm512_maskz_sqrt_ph
  // UNCONSTRAINED: call {{.*}}<32 x half> @llvm.sqrt.v32f16(<32 x half> %{{.*}})
  // CONSTRAINED: call {{.*}}<32 x half> @llvm.experimental.constrained.sqrt.v32f16(<32 x half> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vsqrtph %zmm{{.*}},
  // COMMONIR: bitcast i32 %{{.*}} to <32 x i1>
  // COMMONIR: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> {{.*}}
  return _mm512_maskz_sqrt_ph (__U,__A);
}
