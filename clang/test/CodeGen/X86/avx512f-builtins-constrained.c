// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -fexperimental-new-pass-manager -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux-gnu -target-feature +avx512f -emit-llvm -o - -Wall -Werror | FileCheck --check-prefix=COMMON --check-prefix=COMMONIR --check-prefix=UNCONSTRAINED %s
// RUN: %clang_cc1 -fexperimental-new-pass-manager -flax-vector-conversions=none -fms-extensions -fms-compatibility -ffreestanding %s -triple=x86_64-windows-msvc -target-feature +avx512f -emit-llvm -o - -Wall -Werror | FileCheck --check-prefix=COMMON --check-prefix=COMMONIR --check-prefix=UNCONSTRAINED %s
// RUN: %clang_cc1 -fexperimental-new-pass-manager -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux-gnu -target-feature +avx512f -ffp-exception-behavior=strict -emit-llvm -o - -Wall -Werror | FileCheck --check-prefix=COMMON --check-prefix=COMMONIR --check-prefix=CONSTRAINED %s
// RUN: %clang_cc1 -fexperimental-new-pass-manager -flax-vector-conversions=none -fms-compatibility -ffreestanding %s -triple=x86_64-unknown-linux-gnu -target-feature +avx512f -ffp-exception-behavior=strict -emit-llvm -o - -Wall -Werror | FileCheck --check-prefix=COMMON --check-prefix=COMMONIR --check-prefix=CONSTRAINED %s
// RUN: %clang_cc1 -fexperimental-new-pass-manager -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux-gnu -target-feature +avx512f -S -o - -Wall -Werror | FileCheck --check-prefix=COMMON --check-prefix=CHECK-ASM %s
// RUN: %clang_cc1 -fexperimental-new-pass-manager -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux-gnu -target-feature +avx512f -ffp-exception-behavior=strict -S -o - -Wall -Werror | FileCheck --check-prefix=COMMON --check-prefix=CHECK-ASM %s

#include <immintrin.h>

__m512d test_mm512_sqrt_pd(__m512d a)
{
  // COMMON-LABEL: test_mm512_sqrt_pd
  // UNCONSTRAINED: call <8 x double> @llvm.sqrt.v8f64(<8 x double> %{{.*}})
  // CONSTRAINED: call <8 x double> @llvm.experimental.constrained.sqrt.v8f64(<8 x double> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vsqrtpd
  return _mm512_sqrt_pd(a);
}

__m512d test_mm512_mask_sqrt_pd (__m512d __W, __mmask8 __U, __m512d __A)
{
  // COMMON-LABEL: test_mm512_mask_sqrt_pd
  // UNCONSTRAINED: call <8 x double> @llvm.sqrt.v8f64(<8 x double> %{{.*}})
  // CONSTRAINED: call <8 x double> @llvm.experimental.constrained.sqrt.v8f64(<8 x double> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vsqrtpd
  // COMMONIR: bitcast i8 %{{.*}} to <8 x i1>
  // COMMONIR: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask_sqrt_pd (__W,__U,__A);
}

__m512d test_mm512_maskz_sqrt_pd (__mmask8 __U, __m512d __A)
{
  // COMMON-LABEL: test_mm512_maskz_sqrt_pd
  // UNCONSTRAINED: call <8 x double> @llvm.sqrt.v8f64(<8 x double> %{{.*}})
  // CONSTRAINED: call <8 x double> @llvm.experimental.constrained.sqrt.v8f64(<8 x double> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vsqrtpd
  // COMMONIR: bitcast i8 %{{.*}} to <8 x i1>
  // COMMONIR: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> {{.*}}
  return _mm512_maskz_sqrt_pd (__U,__A);
}

__m512 test_mm512_sqrt_ps(__m512 a)
{
  // COMMON-LABEL: test_mm512_sqrt_ps
  // UNCONSTRAINED: call <16 x float> @llvm.sqrt.v16f32(<16 x float> %{{.*}})
  // CONSTRAINED: call <16 x float> @llvm.experimental.constrained.sqrt.v16f32(<16 x float> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vsqrtps
  return _mm512_sqrt_ps(a);
}

__m512 test_mm512_mask_sqrt_ps(__m512 __W, __mmask16 __U, __m512 __A)
{
  // COMMON-LABEL: test_mm512_mask_sqrt_ps
  // UNCONSTRAINED: call <16 x float> @llvm.sqrt.v16f32(<16 x float> %{{.*}})
  // CONSTRAINED: call <16 x float> @llvm.experimental.constrained.sqrt.v16f32(<16 x float> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vsqrtps
  // COMMONIR: bitcast i16 %{{.*}} to <16 x i1>
  // COMMONIR: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask_sqrt_ps( __W, __U, __A);
}

__m512 test_mm512_maskz_sqrt_ps( __mmask16 __U, __m512 __A)
{
  // COMMON-LABEL: test_mm512_maskz_sqrt_ps
  // UNCONSTRAINED: call <16 x float> @llvm.sqrt.v16f32(<16 x float> %{{.*}})
  // CONSTRAINED: call <16 x float> @llvm.experimental.constrained.sqrt.v16f32(<16 x float> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vsqrtps
  // COMMONIR: bitcast i16 %{{.*}} to <16 x i1>
  // COMMONIR: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> {{.*}}
  return _mm512_maskz_sqrt_ps(__U ,__A);
}

__m256i test_mm512_cvt_roundps_ph(__m512  __A)
{
    // COMMON-LABEL: test_mm512_cvt_roundps_ph
    // COMMONIR: @llvm.x86.avx512.mask.vcvtps2ph.512
    return _mm512_cvt_roundps_ph(__A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm512_mask_cvt_roundps_ph(__m256i __W , __mmask16 __U, __m512  __A)
{
    // COMMON-LABEL: test_mm512_mask_cvt_roundps_ph
    // COMMONIR: @llvm.x86.avx512.mask.vcvtps2ph.512
    return _mm512_mask_cvt_roundps_ph(__W, __U, __A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256i test_mm512_maskz_cvt_roundps_ph(__mmask16 __U, __m512  __A)
{
    // COMMON-LABEL: test_mm512_maskz_cvt_roundps_ph
    // COMMONIR: @llvm.x86.avx512.mask.vcvtps2ph.512
    return _mm512_maskz_cvt_roundps_ph(__U, __A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512 test_mm512_cvt_roundph_ps(__m256i __A)
{
    // COMMON-LABEL: test_mm512_cvt_roundph_ps
    // COMMONIR: @llvm.x86.avx512.mask.vcvtph2ps.512
    return _mm512_cvt_roundph_ps(__A, _MM_FROUND_NO_EXC);
}

__m512 test_mm512_mask_cvt_roundph_ps(__m512 __W, __mmask16 __U, __m256i __A)
{
    // COMMON-LABEL: test_mm512_mask_cvt_roundph_ps
    // COMMONIR: @llvm.x86.avx512.mask.vcvtph2ps.512
    return _mm512_mask_cvt_roundph_ps(__W, __U, __A, _MM_FROUND_NO_EXC);
}

__m512 test_mm512_maskz_cvt_roundph_ps(__mmask16 __U, __m256i __A)
{
    // COMMON-LABEL: test_mm512_maskz_cvt_roundph_ps
    // COMMONIR: @llvm.x86.avx512.mask.vcvtph2ps.512
    return _mm512_maskz_cvt_roundph_ps(__U, __A, _MM_FROUND_NO_EXC);
}

__m128d test_mm_mask_sqrt_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B){
  // COMMON-LABEL: test_mm_mask_sqrt_sd
  // COMMONIR: extractelement <2 x double> %{{.*}}, i64 0
  // UNCONSTRAINED-NEXT: call double @llvm.sqrt.f64(double %{{.*}})
  // CONSTRAINED-NEXT: call double @llvm.experimental.constrained.sqrt.f64(double %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vsqrtsd
  // COMMONIR-NEXT: extractelement <2 x double> %{{.*}}, i64 0
  // COMMONIR-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // COMMONIR-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // COMMONIR-NEXT: select i1 {{.*}}, double {{.*}}, double {{.*}}
  // COMMONIR-NEXT: insertelement <2 x double> %{{.*}}, double {{.*}}, i64 0
  return _mm_mask_sqrt_sd(__W,__U,__A,__B);
}

__m128d test_mm_maskz_sqrt_sd(__mmask8 __U, __m128d __A, __m128d __B){
  // COMMON-LABEL: test_mm_maskz_sqrt_sd
  // COMMONIR: extractelement <2 x double> %{{.*}}, i64 0
  // UNCONSTRAINED-NEXT: call double @llvm.sqrt.f64(double %{{.*}})
  // CONSTRAINED-NEXT: call double @llvm.experimental.constrained.sqrt.f64(double %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vsqrtsd
  // COMMONIR-NEXT: extractelement <2 x double> %{{.*}}, i64 0
  // COMMONIR-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // COMMONIR-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // COMMONIR-NEXT: select i1 {{.*}}, double {{.*}}, double {{.*}}
  // COMMONIR-NEXT: insertelement <2 x double> %{{.*}}, double {{.*}}, i64 0
  return _mm_maskz_sqrt_sd(__U,__A,__B);
}

__m128 test_mm_mask_sqrt_ss(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B){
  // COMMON-LABEL: test_mm_mask_sqrt_ss
  // COMMONIR: extractelement <4 x float> %{{.*}}, i64 0
  // UNCONSTRAINED-NEXT: call float @llvm.sqrt.f32(float %{{.*}})
  // CONSTRAINED-NEXT: call float @llvm.experimental.constrained.sqrt.f32(float %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vsqrtss
  // COMMONIR-NEXT: extractelement <4 x float> %{{.*}}, i64 0
  // COMMONIR-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // COMMONIR-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // COMMONIR-NEXT: select i1 {{.*}}, float {{.*}}, float {{.*}}
  // COMMONIR-NEXT: insertelement <4 x float> %{{.*}}, float {{.*}}, i64 0
  return _mm_mask_sqrt_ss(__W,__U,__A,__B);
}

__m128 test_mm_maskz_sqrt_ss(__mmask8 __U, __m128 __A, __m128 __B){
  // COMMON-LABEL: test_mm_maskz_sqrt_ss
  // COMMONIR: extractelement <4 x float> %{{.*}}, i64 0
  // UNCONSTRAINED-NEXT: call float @llvm.sqrt.f32(float %{{.*}})
  // CONSTRAINED-NEXT: call float @llvm.experimental.constrained.sqrt.f32(float %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vsqrtss
  // COMMONIR-NEXT: extractelement <4 x float> %{{.*}}, i64 0
  // COMMONIR-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // COMMONIR-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // COMMONIR-NEXT: select i1 {{.*}}, float {{.*}}, float {{.*}}
  // COMMONIR-NEXT: insertelement <4 x float> %{{.*}}, float {{.*}}, i64 0
  return _mm_maskz_sqrt_ss(__U,__A,__B);
}

__m512 test_mm512_cvtph_ps (__m256i __A)
{
  // COMMON-LABEL: test_mm512_cvtph_ps 
  // COMMONIR: bitcast <4 x i64> %{{.*}} to <16 x i16>
  // COMMONIR: bitcast <16 x i16> %{{.*}} to <16 x half>
  // UNCONSTRAINED: fpext <16 x half> %{{.*}} to <16 x float>
  // CONSTRAINED: call <16 x float> @llvm.experimental.constrained.fpext.v16f32.v16f16(<16 x half> %{{.*}}, metadata !"fpexcept.strict")
  return _mm512_cvtph_ps (__A);
}

__m512 test_mm512_mask_cvtph_ps (__m512 __W, __mmask16 __U, __m256i __A)
{
  // COMMON-LABEL: test_mm512_mask_cvtph_ps 
  // COMMONIR: bitcast <4 x i64> %{{.*}} to <16 x i16>
  // COMMONIR: bitcast <16 x i16> %{{.*}} to <16 x half>
  // UNCONSTRAINED: fpext <16 x half> %{{.*}} to <16 x float>
  // CONSTRAINED: call <16 x float> @llvm.experimental.constrained.fpext.v16f32.v16f16(<16 x half> %{{.*}}, metadata !"fpexcept.strict")
  // COMMONIR: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask_cvtph_ps (__W,__U,__A);
}

__m512 test_mm512_maskz_cvtph_ps (__mmask16 __U, __m256i __A)
{
  // COMMON-LABEL: test_mm512_maskz_cvtph_ps 
  // COMMONIR: bitcast <4 x i64> %{{.*}} to <16 x i16>
  // COMMONIR: bitcast <16 x i16> %{{.*}} to <16 x half>
  // UNCONSTRAINED: fpext <16 x half> %{{.*}} to <16 x float>
  // CONSTRAINED: call <16 x float> @llvm.experimental.constrained.fpext.v16f32.v16f16(<16 x half> %{{.*}}, metadata !"fpexcept.strict")
  // COMMONIR: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_maskz_cvtph_ps (__U,__A);
}

