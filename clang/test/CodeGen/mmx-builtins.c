// REQUIRES: x86-registered-target
// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin -target-feature +ssse3 -emit-llvm -o - -Werror | FileCheck %s
// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin -target-feature +ssse3 -fno-signed-char -emit-llvm -o - -Werror | FileCheck %s
// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin -target-feature +ssse3 -S -o - -Werror | FileCheck %s --check-prefix=CHECK-ASM
// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin -target-feature +ssse3 -fno-signed-char -S -o - -Werror | FileCheck %s --check-prefix=CHECK-ASM

// Don't include mm_malloc.h, it's system specific.
#define __MM_MALLOC_H

#include <x86intrin.h>

__m64 test_mm_abs_pi8(__m64 a) {
  // CHECK-LABEL: test_mm_abs_pi8
  // CHECK: call x86_mmx @llvm.x86.ssse3.pabs.b
  // CHECK-ASM: pabsb %mm{{.*}}, %mm{{.*}}
  return _mm_abs_pi8(a);
}

__m64 test_mm_abs_pi16(__m64 a) {
  // CHECK-LABEL: test_mm_abs_pi16
  // CHECK: call x86_mmx @llvm.x86.ssse3.pabs.w
  // CHECK-ASM: pabsw %mm{{.*}}, %mm{{.*}}
  return _mm_abs_pi16(a);
}

__m64 test_mm_abs_pi32(__m64 a) {
  // CHECK-LABEL: test_mm_abs_pi32
  // CHECK: call x86_mmx @llvm.x86.ssse3.pabs.d
  // CHECK-ASM: pabsd %mm{{.*}}, %mm{{.*}}
  return _mm_abs_pi32(a);
}

__m64 test_mm_add_pi8(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_add_pi8
  // CHECK: call x86_mmx @llvm.x86.mmx.padd.b
  // CHECK-ASM: paddb %mm{{.*}}, %mm{{.*}}
  return _mm_add_pi8(a, b);
}

__m64 test_mm_add_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_add_pi16
  // CHECK: call x86_mmx @llvm.x86.mmx.padd.w
  // CHECK-ASM: paddw %mm{{.*}}, %mm{{.*}}
  return _mm_add_pi16(a, b);
}

__m64 test_mm_add_pi32(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_add_pi32
  // CHECK: call x86_mmx @llvm.x86.mmx.padd.d
  // CHECK-ASM: paddd %mm{{.*}}, %mm{{.*}}
  return _mm_add_pi32(a, b);
}

__m64 test_mm_add_si64(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_add_si64
  // CHECK: call x86_mmx @llvm.x86.mmx.padd.q
  // CHECK-ASM: paddq %mm{{.*}}, %mm{{.*}}
  return __builtin_ia32_paddq(a, b);
}

__m64 test_mm_adds_pi8(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_adds_pi8
  // CHECK: call x86_mmx @llvm.x86.mmx.padds.b
  // CHECK-ASM: paddsb %mm{{.*}}, %mm{{.*}}
  return _mm_adds_pi8(a, b);
}

__m64 test_mm_adds_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_adds_pi16
  // CHECK: call x86_mmx @llvm.x86.mmx.padds.w
  // CHECK-ASM: paddsw %mm{{.*}}, %mm{{.*}}
  return _mm_adds_pi16(a, b);
}

__m64 test_mm_adds_pu8(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_adds_pu8
  // CHECK: call x86_mmx @llvm.x86.mmx.paddus.b
  // CHECK-ASM: paddusb %mm{{.*}}, %mm{{.*}}
  return _mm_adds_pu8(a, b);
}

__m64 test_mm_adds_pu16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_adds_pu16
  // CHECK: call x86_mmx @llvm.x86.mmx.paddus.w
  // CHECK-ASM: paddusw %mm{{.*}}, %mm{{.*}}
  return _mm_adds_pu16(a, b);
}

__m64 test_mm_alignr_pi8(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_alignr_pi8
  // CHECK: call x86_mmx @llvm.x86.mmx.palignr.b
  // CHECK-ASM: palignr $2, %mm{{.*}}, %mm{{.*}}
  return _mm_alignr_pi8(a, b, 2);
}

__m64 test_mm_and_si64(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_and_si64
  // CHECK: call x86_mmx @llvm.x86.mmx.pand
  // CHECK-ASM: pand %mm{{.*}}, %mm{{.*}}
  return _mm_and_si64(a, b);
}

__m64 test_mm_andnot_si64(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_andnot_si64
  // CHECK: call x86_mmx @llvm.x86.mmx.pandn
  // CHECK-ASM: pandn %mm{{.*}}, %mm{{.*}}
  return _mm_andnot_si64(a, b);
}

__m64 test_mm_avg_pu8(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_avg_pu8
  // CHECK: call x86_mmx @llvm.x86.mmx.pavg.b
  // CHECK-ASM: pavgb %mm{{.*}}, %mm{{.*}}
  return _mm_avg_pu8(a, b);
}

__m64 test_mm_avg_pu16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_avg_pu16
  // CHECK: call x86_mmx @llvm.x86.mmx.pavg.w
  // CHECK-ASM: pavgw %mm{{.*}}, %mm{{.*}}
  return _mm_avg_pu16(a, b);
}

__m64 test_mm_cmpeq_pi8(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_cmpeq_pi8
  // CHECK: call x86_mmx @llvm.x86.mmx.pcmpeq.b
  // CHECK-ASM: pcmpeqb %mm{{.*}}, %mm{{.*}}
  return _mm_cmpeq_pi8(a, b);
}

__m64 test_mm_cmpeq_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_cmpeq_pi16
  // CHECK: call x86_mmx @llvm.x86.mmx.pcmpeq.w
  // CHECK-ASM: pcmpeqw %mm{{.*}}, %mm{{.*}}
  return _mm_cmpeq_pi16(a, b);
}

__m64 test_mm_cmpeq_pi32(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_cmpeq_pi32
  // CHECK: call x86_mmx @llvm.x86.mmx.pcmpeq.d
  // CHECK-ASM: pcmpeqd %mm{{.*}}, %mm{{.*}}
  return _mm_cmpeq_pi32(a, b);
}

__m64 test_mm_cmpgt_pi8(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_cmpgt_pi8
  // CHECK: call x86_mmx @llvm.x86.mmx.pcmpgt.b
  // CHECK-ASM: pcmpgtb %mm{{.*}}, %mm{{.*}}
  return _mm_cmpgt_pi8(a, b);
}

__m64 test_mm_cmpgt_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_cmpgt_pi16
  // CHECK: call x86_mmx @llvm.x86.mmx.pcmpgt.w
  // CHECK-ASM: pcmpgtw %mm{{.*}}, %mm{{.*}}
  return _mm_cmpgt_pi16(a, b);
}

__m64 test_mm_cmpgt_pi32(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_cmpgt_pi32
  // CHECK: call x86_mmx @llvm.x86.mmx.pcmpgt.d
  // CHECK-ASM: pcmpgtd %mm{{.*}}, %mm{{.*}}
  return _mm_cmpgt_pi32(a, b);
}

__m128 test_mm_cvt_pi2ps(__m128 a, __m64 b) {
  // CHECK-LABEL: test_mm_cvt_pi2ps
  // CHECK: <4 x float> @llvm.x86.sse.cvtpi2ps
  // CHECK-ASM: cvtpi2ps %mm{{.*}}, %xmm{{.*}}
  return _mm_cvt_pi2ps(a, b);
}

__m64 test_mm_cvt_ps2pi(__m128 a) {
  // CHECK-LABEL: test_mm_cvt_ps2pi
  // CHECK: call x86_mmx @llvm.x86.sse.cvtps2pi
  // CHECK-ASM: cvtps2pi %xmm{{.*}}, %mm{{.*}}
  return _mm_cvt_ps2pi(a);
}

__m64 test_mm_cvtpd_pi32(__m128d a) {
  // CHECK-LABEL: test_mm_cvtpd_pi32
  // CHECK: call x86_mmx @llvm.x86.sse.cvtpd2pi
  // CHECK-ASM: cvtpd2pi %xmm{{.*}}, %mm{{.*}}
  return _mm_cvtpd_pi32(a);
}

__m128 test_mm_cvtpi16_ps(__m64 a) {
  // CHECK-LABEL: test_mm_cvtpi16_ps
  // CHECK: call <4 x float> @llvm.x86.sse.cvtpi2ps
  // CHECK-ASM: cvtpi2ps %mm{{.*}}, %xmm{{.*}}
  return _mm_cvtpi16_ps(a);
}

__m128d test_mm_cvtpi32_pd(__m64 a) {
  // CHECK-LABEL: test_mm_cvtpi32_pd
  // CHECK: call <2 x double> @llvm.x86.sse.cvtpi2pd
  // CHECK-ASM: cvtpi2pd %mm{{.*}}, %xmm{{.*}}
  return _mm_cvtpi32_pd(a);
}

__m128 test_mm_cvtpi32_ps(__m128 a, __m64 b) {
  // CHECK-LABEL: test_mm_cvtpi32_ps
  // CHECK: call <4 x float> @llvm.x86.sse.cvtpi2ps
  // CHECK-ASM: cvtpi2ps %mm{{.*}}, %xmm{{.*}}
  return _mm_cvtpi32_ps(a, b);
}

__m128 test_mm_cvtpi32x2_ps(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_cvtpi32x2_ps
  // CHECK: call <4 x float> @llvm.x86.sse.cvtpi2ps
  // CHECK: call <4 x float> @llvm.x86.sse.cvtpi2ps
  // CHECK-ASM: cvtpi2ps %mm{{.*}}, %xmm{{.*}}
  // CHECK-ASM: cvtpi2ps %mm{{.*}}, %xmm{{.*}}
  return _mm_cvtpi32x2_ps(a, b);
}

__m64 test_mm_cvtps_pi16(__m128 a) {
  // CHECK-LABEL: test_mm_cvtps_pi16
  // CHECK: call x86_mmx @llvm.x86.sse.cvtps2pi
  // CHECK-ASM: cvtps2pi %xmm{{.*}}, %mm{{.*}}
  // CHECK-ASM: cvtps2pi %xmm{{.*}}, %mm{{.*}}
  // CHECK-ASM: packssdw %mm{{.*}}, %mm{{.*}}
  return _mm_cvtps_pi16(a);
}

__m64 test_mm_cvtps_pi32(__m128 a) {
  // CHECK-LABEL: test_mm_cvtps_pi32
  // CHECK: call x86_mmx @llvm.x86.sse.cvtps2pi
  // CHECK-ASM: cvtps2pi %xmm{{.*}}, %mm{{.*}}
  return _mm_cvtps_pi32(a);
}

__m64 test_mm_cvtsi32_si64(int a) {
  // CHECK-LABEL: test_mm_cvtsi32_si64
  // CHECK: insertelement <2 x i32>
  // CHECK-ASM: movd
  return _mm_cvtsi32_si64(a);
}

int test_mm_cvtsi64_si32(__m64 a) {
  // CHECK-LABEL: test_mm_cvtsi64_si32
  // CHECK: extractelement <2 x i32>
  // CHECK-ASM: movd
  return _mm_cvtsi64_si32(a);
}

__m64 test_mm_cvttpd_pi32(__m128d a) {
  // CHECK-LABEL: test_mm_cvttpd_pi32
  // CHECK: call x86_mmx @llvm.x86.sse.cvttpd2pi
  // CHECK-ASM: cvttpd2pi %xmm{{.*}}, %mm{{.*}}
  return _mm_cvttpd_pi32(a);
}

__m64 test_mm_cvttps_pi32(__m128 a) {
  // CHECK-LABEL: test_mm_cvttps_pi32
  // CHECK: call x86_mmx @llvm.x86.sse.cvttps2pi
  // CHECK-ASM: cvttps2pi %xmm{{.*}}, %mm{{.*}}
  return _mm_cvttps_pi32(a);
}

__m64 test_m_from_int(int a) {
  // CHECK-LABEL: test_m_from_int
  // CHECK: insertelement <2 x i32>
  // CHECK-ASM: movd
  return _m_from_int(a);
}

__m64 test_mm_hadd_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_hadd_pi16
  // CHECK: call x86_mmx @llvm.x86.ssse3.phadd.w
  // CHECK-ASM: phaddw %mm{{.*}}, %mm{{.*}}
  return _mm_hadd_pi16(a, b);
}

__m64 test_mm_hadd_pi32(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_hadd_pi32
  // CHECK: call x86_mmx @llvm.x86.ssse3.phadd.d
  // CHECK-ASM: phaddd %mm{{.*}}, %mm{{.*}}
  return _mm_hadd_pi32(a, b);
}

__m64 test_mm_hadds_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_hadds_pi16
  // CHECK: call x86_mmx @llvm.x86.ssse3.phadd.sw
  // CHECK-ASM: phaddsw %mm{{.*}}, %mm{{.*}}
  return _mm_hadds_pi16(a, b);
}

__m64 test_mm_hsub_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_hsub_pi16
  // CHECK: call x86_mmx @llvm.x86.ssse3.phsub.w
  // CHECK-ASM: phsubw %mm{{.*}}, %mm{{.*}}
  return _mm_hsub_pi16(a, b);
}

__m64 test_mm_hsub_pi32(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_hsub_pi32
  // CHECK: call x86_mmx @llvm.x86.ssse3.phsub.d
  // CHECK-ASM: phsubd %mm{{.*}}, %mm{{.*}}
  return _mm_hsub_pi32(a, b);
}

__m64 test_mm_hsubs_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_hsubs_pi16
  // CHECK: call x86_mmx @llvm.x86.ssse3.phsub.sw
  // CHECK-ASM: phsubsw %mm{{.*}}, %mm{{.*}}
  return _mm_hsubs_pi16(a, b);
}

__m64 test_mm_madd_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_madd_pi16
  // CHECK: call x86_mmx @llvm.x86.mmx.pmadd.wd
  // CHECK-ASM: pmaddwd %mm{{.*}}, %mm{{.*}}
  return _mm_madd_pi16(a, b);
}

__m64 test_mm_maddubs_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_maddubs_pi16
  // CHECK: call x86_mmx @llvm.x86.ssse3.pmadd.ub.sw
  // CHECK-ASM: pmaddubsw %mm{{.*}}, %mm{{.*}}
  return _mm_maddubs_pi16(a, b);
}

void test_mm_maskmove_si64(__m64 d, __m64 n, char *p) {
  // CHECK-LABEL: test_mm_maskmove_si64
  // CHECK: call void @llvm.x86.mmx.maskmovq
  // CHECK-ASM: maskmovq
  _mm_maskmove_si64(d, n, p);
}

__m64 test_mm_max_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_max_pi16
  // CHECK: call x86_mmx @llvm.x86.mmx.pmaxs.w
  // CHECK-ASM: pmaxsw %mm{{.*}}, %mm{{.*}}
  return _mm_max_pi16(a, b);
}

__m64 test_mm_max_pu8(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_max_pu8
  // CHECK: call x86_mmx @llvm.x86.mmx.pmaxu.b
  // CHECK-ASM: pmaxub %mm{{.*}}, %mm{{.*}}
  return _mm_max_pu8(a, b);
}

__m64 test_mm_min_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_min_pi16
  // CHECK: call x86_mmx @llvm.x86.mmx.pmins.w
  // CHECK-ASM: pminsw %mm{{.*}}, %mm{{.*}}
  return _mm_min_pi16(a, b);
}

__m64 test_mm_min_pu8(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_min_pu8
  // CHECK: call x86_mmx @llvm.x86.mmx.pminu.b
  // CHECK-ASM: pminub %mm{{.*}}, %mm{{.*}}
  return _mm_min_pu8(a, b);
}

int test_mm_movemask_pi8(__m64 a) {
  // CHECK-LABEL: test_mm_movemask_pi8
  // CHECK: call i32 @llvm.x86.mmx.pmovmskb
  // CHECK-ASM: pmovmskb
  return _mm_movemask_pi8(a);
}

__m64 test_mm_mul_su32(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_mul_su32
  // CHECK: call x86_mmx @llvm.x86.mmx.pmulu.dq
  // CHECK-ASM: pmuludq %mm{{.*}}, %mm{{.*}}
  return _mm_mul_su32(a, b);
}

__m64 test_mm_mulhi_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_mulhi_pi16
  // CHECK: call x86_mmx @llvm.x86.mmx.pmulh.w
  // CHECK-ASM: pmulhw %mm{{.*}}, %mm{{.*}}
  return _mm_mulhi_pi16(a, b);
}

__m64 test_mm_mulhi_pu16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_mulhi_pu16
  // CHECK: call x86_mmx @llvm.x86.mmx.pmulhu.w
  // CHECK-ASM: pmulhuw %mm{{.*}}, %mm{{.*}}
  return _mm_mulhi_pu16(a, b);
}

__m64 test_mm_mulhrs_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_mulhrs_pi16
  // CHECK: call x86_mmx @llvm.x86.ssse3.pmul.hr.sw
  // CHECK-ASM: pmulhrsw %mm{{.*}}, %mm{{.*}}
  return _mm_mulhrs_pi16(a, b);
}

__m64 test_mm_mullo_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_mullo_pi16
  // CHECK: call x86_mmx @llvm.x86.mmx.pmull.w
  // CHECK-ASM: pmullw %mm{{.*}}, %mm{{.*}}
  return _mm_mullo_pi16(a, b);
}

__m64 test_mm_or_si64(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_or_si64
  // CHECK: call x86_mmx @llvm.x86.mmx.por
  // CHECK-ASM: por %mm{{.*}}, %mm{{.*}}
  return _mm_or_si64(a, b);
}

__m64 test_mm_packs_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_packs_pi16
  // CHECK: call x86_mmx @llvm.x86.mmx.packsswb
  // CHECK-ASM: packsswb %mm{{.*}}, %mm{{.*}}
  return _mm_packs_pi16(a, b);
}

__m64 test_mm_packs_pi32(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_packs_pi32
  // CHECK: call x86_mmx @llvm.x86.mmx.packssdw
  // CHECK-ASM: packssdw %mm{{.*}}, %mm{{.*}}
  return _mm_packs_pi32(a, b);
}

__m64 test_mm_packs_pu16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_packs_pu16
  // CHECK: call x86_mmx @llvm.x86.mmx.packuswb
  // CHECK-ASM: packuswb %mm{{.*}}, %mm{{.*}}
  return _mm_packs_pu16(a, b);
}

__m64 test_mm_sad_pu8(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_sad_pu8
  // CHECK: call x86_mmx @llvm.x86.mmx.psad.bw
  // CHECK-ASM: psadbw %mm{{.*}}, %mm{{.*}}
  return _mm_sad_pu8(a, b);
}

__m64 test_mm_shuffle_pi8(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_shuffle_pi8
  // CHECK: call x86_mmx @llvm.x86.ssse3.pshuf.b
  // CHECK-ASM: pshufb %mm{{.*}}, %mm{{.*}}
  return _mm_shuffle_pi8(a, b);
}

__m64 test_mm_shuffle_pi16(__m64 a) {
  // CHECK-LABEL: test_mm_shuffle_pi16
  // CHECK: call x86_mmx @llvm.x86.sse.pshuf.w
  // CHECK-ASM: pshufw $3, %mm{{.*}}, %mm{{.*}}
  return _mm_shuffle_pi16(a, 3);
}

__m64 test_mm_sign_pi8(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_sign_pi8
  // CHECK: call x86_mmx @llvm.x86.ssse3.psign.b
  // CHECK-ASM: psignb %mm{{.*}}, %mm{{.*}}
  return _mm_sign_pi8(a, b);
}

__m64 test_mm_sign_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_sign_pi16
  // CHECK: call x86_mmx @llvm.x86.ssse3.psign.w
  // CHECK-ASM: psignw %mm{{.*}}, %mm{{.*}}
  return _mm_sign_pi16(a, b);
}

__m64 test_mm_sign_pi32(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_sign_pi32
  // CHECK: call x86_mmx @llvm.x86.ssse3.psign.d
  // CHECK-ASM: psignd %mm{{.*}}, %mm{{.*}}
  return _mm_sign_pi32(a, b);
}

__m64 test_mm_sll_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_sll_pi16
  // CHECK: call x86_mmx @llvm.x86.mmx.psll.w
  // CHECK-ASM: psllw %mm{{.*}}, %mm{{.*}}
  return _mm_sll_pi16(a, b);
}

__m64 test_mm_sll_pi32(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_sll_pi32
  // CHECK: call x86_mmx @llvm.x86.mmx.psll.d
  // CHECK-ASM: pslld %mm{{.*}}, %mm{{.*}}
  return _mm_sll_pi32(a, b);
}

__m64 test_mm_sll_si64(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_sll_si64
  // CHECK: call x86_mmx @llvm.x86.mmx.psll.q
  // CHECK-ASM: psllq %mm{{.*}}, %mm{{.*}}
  return _mm_sll_si64(a, b);
}

__m64 test_mm_slli_pi16(__m64 a) {
  // CHECK-LABEL: test_mm_slli_pi16
  // CHECK: call x86_mmx @llvm.x86.mmx.pslli.w
  // CHECK-ASM: psllw
  return _mm_slli_pi16(a, 3);
}

__m64 test_mm_slli_pi32(__m64 a) {
  // CHECK-LABEL: test_mm_slli_pi32
  // CHECK: call x86_mmx @llvm.x86.mmx.pslli.d
  // CHECK-ASM: pslld
  return _mm_slli_pi32(a, 3);
}

__m64 test_mm_slli_si64(__m64 a) {
  // CHECK-LABEL: test_mm_slli_si64
  // CHECK: call x86_mmx @llvm.x86.mmx.pslli.q
  // CHECK-ASM: psllq
  return _mm_slli_si64(a, 3);
}

__m64 test_mm_sra_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_sra_pi16
  // CHECK: call x86_mmx @llvm.x86.mmx.psra.w
  // CHECK-ASM: psraw %mm{{.*}}, %mm{{.*}}
  return _mm_sra_pi16(a, b);
}

__m64 test_mm_sra_pi32(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_sra_pi32
  // CHECK: call x86_mmx @llvm.x86.mmx.psra.d
  // CHECK-ASM: psrad %mm{{.*}}, %mm{{.*}}
  return _mm_sra_pi32(a, b);
}

__m64 test_mm_srai_pi16(__m64 a) {
  // CHECK-LABEL: test_mm_srai_pi16
  // CHECK: call x86_mmx @llvm.x86.mmx.psrai.w
  // CHECK-ASM: psraw
  return _mm_srai_pi16(a, 3);
}

__m64 test_mm_srai_pi32(__m64 a) {
  // CHECK-LABEL: test_mm_srai_pi32
  // CHECK: call x86_mmx @llvm.x86.mmx.psrai.d
  // CHECK-ASM: psrad
  return _mm_srai_pi32(a, 3);
}

__m64 test_mm_srl_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_srl_pi16
  // CHECK: call x86_mmx @llvm.x86.mmx.psrl.w
  // CHECK-ASM: psrlw %mm{{.*}}, %mm{{.*}}
  return _mm_srl_pi16(a, b);
}

__m64 test_mm_srl_pi32(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_srl_pi32
  // CHECK: call x86_mmx @llvm.x86.mmx.psrl.d
  // CHECK-ASM: psrld %mm{{.*}}, %mm{{.*}}
  return _mm_srl_pi32(a, b);
}

__m64 test_mm_srl_si64(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_srl_si64
  // CHECK: call x86_mmx @llvm.x86.mmx.psrl.q
  // CHECK-ASM: psrlq %mm{{.*}}, %mm{{.*}}
  return _mm_srl_si64(a, b);
}

__m64 test_mm_srli_pi16(__m64 a) {
  // CHECK-LABEL: test_mm_srli_pi16
  // CHECK: call x86_mmx @llvm.x86.mmx.psrli.w
  // CHECK-ASM: psrlw
  return _mm_srli_pi16(a, 3);
}

__m64 test_mm_srli_pi32(__m64 a) {
  // CHECK-LABEL: test_mm_srli_pi32
  // CHECK: call x86_mmx @llvm.x86.mmx.psrli.d
  // CHECK-ASM: psrld
  return _mm_srli_pi32(a, 3);
}

__m64 test_mm_srli_si64(__m64 a) {
  // CHECK-LABEL: test_mm_srli_si64
  // CHECK: call x86_mmx @llvm.x86.mmx.psrli.q
  // CHECK-ASM: psrlq
  return _mm_srli_si64(a, 3);
}

void test_mm_stream_pi(__m64 *p, __m64 a) {
  // CHECK-LABEL: test_mm_stream_pi
  // CHECK: call void @llvm.x86.mmx.movnt.dq
  // CHECK-ASM: movntq
  _mm_stream_pi(p, a);
}

__m64 test_mm_sub_pi8(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_sub_pi8
  // CHECK: call x86_mmx @llvm.x86.mmx.psub.b
  // CHECK-ASM: psubb %mm{{.*}}, %mm{{.*}}
  return _mm_sub_pi8(a, b);
}

__m64 test_mm_sub_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_sub_pi16
  // CHECK: call x86_mmx @llvm.x86.mmx.psub.w
  // CHECK-ASM: psubw %mm{{.*}}, %mm{{.*}}
  return _mm_sub_pi16(a, b);
}

__m64 test_mm_sub_pi32(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_sub_pi32
  // CHECK: call x86_mmx @llvm.x86.mmx.psub.d
  // CHECK-ASM: psubd %mm{{.*}}, %mm{{.*}}
  return _mm_sub_pi32(a, b);
}

__m64 test_mm_sub_si64(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_sub_si64
  // CHECK: call x86_mmx @llvm.x86.mmx.psub.q
  // CHECK-ASM: psubq %mm{{.*}}, %mm{{.*}}
  return __builtin_ia32_psubq(a, b);
}

__m64 test_mm_subs_pi8(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_subs_pi8
  // CHECK: call x86_mmx @llvm.x86.mmx.psubs.b
  // CHECK-ASM: psubsb %mm{{.*}}, %mm{{.*}}
  return _mm_subs_pi8(a, b);
}

__m64 test_mm_subs_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_subs_pi16
  // CHECK: call x86_mmx @llvm.x86.mmx.psubs.w
  // CHECK-ASM: psubsw %mm{{.*}}, %mm{{.*}}
  return _mm_subs_pi16(a, b);
}

__m64 test_mm_subs_pu8(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_subs_pu8
  // CHECK: call x86_mmx @llvm.x86.mmx.psubus.b
  // CHECK-ASM: psubusb %mm{{.*}}, %mm{{.*}}
  return _mm_subs_pu8(a, b);
}

__m64 test_mm_subs_pu16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_subs_pu16
  // CHECK: call x86_mmx @llvm.x86.mmx.psubus.w
  // CHECK-ASM: psubusw %mm{{.*}}, %mm{{.*}}
  return _mm_subs_pu16(a, b);
}

int test_m_to_int(__m64 a) {
  // CHECK-LABEL: test_m_to_int
  // CHECK: extractelement <2 x i32>
  // CHECK-ASM: movd
  return _m_to_int(a);
}

__m64 test_mm_unpackhi_pi8(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_unpackhi_pi8
  // CHECK: call x86_mmx @llvm.x86.mmx.punpckhbw
  // CHECK-ASM: punpckhbw %mm{{.*}}, %mm{{.*}}
  return _mm_unpackhi_pi8(a, b);
}

__m64 test_mm_unpackhi_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_unpackhi_pi16
  // CHECK: call x86_mmx @llvm.x86.mmx.punpckhwd
  // CHECK-ASM: punpckhwd %mm{{.*}}, %mm{{.*}}
  return _mm_unpackhi_pi16(a, b);
}

__m64 test_mm_unpackhi_pi32(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_unpackhi_pi32
  // CHECK: call x86_mmx @llvm.x86.mmx.punpckhdq
  // CHECK-ASM: punpckhdq %mm{{.*}}, %mm{{.*}}
  return _mm_unpackhi_pi32(a, b);
}

__m64 test_mm_unpacklo_pi8(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_unpacklo_pi8
  // CHECK: call x86_mmx @llvm.x86.mmx.punpcklbw
  // CHECK-ASM: punpcklbw %mm{{.*}}, %mm{{.*}}
  return _mm_unpacklo_pi8(a, b);
}

__m64 test_mm_unpacklo_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_unpacklo_pi16
  // CHECK: call x86_mmx @llvm.x86.mmx.punpcklwd
  // CHECK-ASM: punpcklwd %mm{{.*}}, %mm{{.*}}
  return _mm_unpacklo_pi16(a, b);
}

__m64 test_mm_unpacklo_pi32(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_unpacklo_pi32
  // CHECK: call x86_mmx @llvm.x86.mmx.punpckldq
  // CHECK-ASM: punpckldq %mm{{.*}}, %mm{{.*}}
  return _mm_unpacklo_pi32(a, b);
}

__m64 test_mm_xor_si64(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_xor_si64
  // CHECK: call x86_mmx @llvm.x86.mmx.pxor
  // CHECK-ASM: pxor %mm{{.*}}, %mm{{.*}}
  return _mm_xor_si64(a, b);
}
