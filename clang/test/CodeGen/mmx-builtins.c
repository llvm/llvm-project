// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +ssse3 -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +ssse3 -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s


#include <x86intrin.h>

__m64 test_mm_abs_pi8(__m64 a) {
  // CHECK-LABEL: test_mm_abs_pi8
  // CHECK: call x86_mmx @llvm.x86.ssse3.pabs.b
  return _mm_abs_pi8(a);
}

__m64 test_mm_abs_pi16(__m64 a) {
  // CHECK-LABEL: test_mm_abs_pi16
  // CHECK: call x86_mmx @llvm.x86.ssse3.pabs.w
  return _mm_abs_pi16(a);
}

__m64 test_mm_abs_pi32(__m64 a) {
  // CHECK-LABEL: test_mm_abs_pi32
  // CHECK: call x86_mmx @llvm.x86.ssse3.pabs.d
  return _mm_abs_pi32(a);
}

__m64 test_mm_add_pi8(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_add_pi8
  // CHECK: call x86_mmx @llvm.x86.mmx.padd.b
  return _mm_add_pi8(a, b);
}

__m64 test_mm_add_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_add_pi16
  // CHECK: call x86_mmx @llvm.x86.mmx.padd.w
  return _mm_add_pi16(a, b);
}

__m64 test_mm_add_pi32(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_add_pi32
  // CHECK: call x86_mmx @llvm.x86.mmx.padd.d
  return _mm_add_pi32(a, b);
}

__m64 test_mm_add_si64(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_add_si64
  // CHECK: call x86_mmx @llvm.x86.mmx.padd.q(x86_mmx %{{.*}}, x86_mmx %{{.*}})
  return _mm_add_si64(a, b);
}

__m64 test_mm_adds_pi8(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_adds_pi8
  // CHECK: call x86_mmx @llvm.x86.mmx.padds.b
  return _mm_adds_pi8(a, b);
}

__m64 test_mm_adds_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_adds_pi16
  // CHECK: call x86_mmx @llvm.x86.mmx.padds.w
  return _mm_adds_pi16(a, b);
}

__m64 test_mm_adds_pu8(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_adds_pu8
  // CHECK: call x86_mmx @llvm.x86.mmx.paddus.b
  return _mm_adds_pu8(a, b);
}

__m64 test_mm_adds_pu16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_adds_pu16
  // CHECK: call x86_mmx @llvm.x86.mmx.paddus.w
  return _mm_adds_pu16(a, b);
}

__m64 test_mm_alignr_pi8(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_alignr_pi8
  // CHECK: call x86_mmx @llvm.x86.mmx.palignr.b
  return _mm_alignr_pi8(a, b, 2);
}

__m64 test_mm_and_si64(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_and_si64
  // CHECK: call x86_mmx @llvm.x86.mmx.pand
  return _mm_and_si64(a, b);
}

__m64 test_mm_andnot_si64(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_andnot_si64
  // CHECK: call x86_mmx @llvm.x86.mmx.pandn
  return _mm_andnot_si64(a, b);
}

__m64 test_mm_avg_pu8(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_avg_pu8
  // CHECK: call x86_mmx @llvm.x86.mmx.pavg.b
  return _mm_avg_pu8(a, b);
}

__m64 test_mm_avg_pu16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_avg_pu16
  // CHECK: call x86_mmx @llvm.x86.mmx.pavg.w
  return _mm_avg_pu16(a, b);
}

__m64 test_mm_cmpeq_pi8(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_cmpeq_pi8
  // CHECK: call x86_mmx @llvm.x86.mmx.pcmpeq.b
  return _mm_cmpeq_pi8(a, b);
}

__m64 test_mm_cmpeq_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_cmpeq_pi16
  // CHECK: call x86_mmx @llvm.x86.mmx.pcmpeq.w
  return _mm_cmpeq_pi16(a, b);
}

__m64 test_mm_cmpeq_pi32(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_cmpeq_pi32
  // CHECK: call x86_mmx @llvm.x86.mmx.pcmpeq.d
  return _mm_cmpeq_pi32(a, b);
}

__m64 test_mm_cmpgt_pi8(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_cmpgt_pi8
  // CHECK: call x86_mmx @llvm.x86.mmx.pcmpgt.b
  return _mm_cmpgt_pi8(a, b);
}

__m64 test_mm_cmpgt_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_cmpgt_pi16
  // CHECK: call x86_mmx @llvm.x86.mmx.pcmpgt.w
  return _mm_cmpgt_pi16(a, b);
}

__m64 test_mm_cmpgt_pi32(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_cmpgt_pi32
  // CHECK: call x86_mmx @llvm.x86.mmx.pcmpgt.d
  return _mm_cmpgt_pi32(a, b);
}

__m128 test_mm_cvt_pi2ps(__m128 a, __m64 b) {
  // CHECK-LABEL: test_mm_cvt_pi2ps
  // CHECK: <4 x float> @llvm.x86.sse.cvtpi2ps
  return _mm_cvt_pi2ps(a, b);
}

__m64 test_mm_cvt_ps2pi(__m128 a) {
  // CHECK-LABEL: test_mm_cvt_ps2pi
  // CHECK: call x86_mmx @llvm.x86.sse.cvtps2pi
  return _mm_cvt_ps2pi(a);
}

__m64 test_mm_cvtpd_pi32(__m128d a) {
  // CHECK-LABEL: test_mm_cvtpd_pi32
  // CHECK: call x86_mmx @llvm.x86.sse.cvtpd2pi
  return _mm_cvtpd_pi32(a);
}

__m128 test_mm_cvtpi16_ps(__m64 a) {
  // CHECK-LABEL: test_mm_cvtpi16_ps
  // CHECK: call <4 x float> @llvm.x86.sse.cvtpi2ps
  return _mm_cvtpi16_ps(a);
}

__m128d test_mm_cvtpi32_pd(__m64 a) {
  // CHECK-LABEL: test_mm_cvtpi32_pd
  // CHECK: call <2 x double> @llvm.x86.sse.cvtpi2pd
  return _mm_cvtpi32_pd(a);
}

__m128 test_mm_cvtpi32_ps(__m128 a, __m64 b) {
  // CHECK-LABEL: test_mm_cvtpi32_ps
  // CHECK: call <4 x float> @llvm.x86.sse.cvtpi2ps
  return _mm_cvtpi32_ps(a, b);
}

__m128 test_mm_cvtpi32x2_ps(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_cvtpi32x2_ps
  // CHECK: call <4 x float> @llvm.x86.sse.cvtpi2ps
  // CHECK: call <4 x float> @llvm.x86.sse.cvtpi2ps
  return _mm_cvtpi32x2_ps(a, b);
}

__m64 test_mm_cvtps_pi16(__m128 a) {
  // CHECK-LABEL: test_mm_cvtps_pi16
  // CHECK: call x86_mmx @llvm.x86.sse.cvtps2pi
  return _mm_cvtps_pi16(a);
}

__m64 test_mm_cvtps_pi32(__m128 a) {
  // CHECK-LABEL: test_mm_cvtps_pi32
  // CHECK: call x86_mmx @llvm.x86.sse.cvtps2pi
  return _mm_cvtps_pi32(a);
}

__m64 test_mm_cvtsi32_si64(int a) {
  // CHECK-LABEL: test_mm_cvtsi32_si64
  // CHECK: insertelement <2 x i32>
  return _mm_cvtsi32_si64(a);
}

int test_mm_cvtsi64_si32(__m64 a) {
  // CHECK-LABEL: test_mm_cvtsi64_si32
  // CHECK: extractelement <2 x i32>
  return _mm_cvtsi64_si32(a);
}

__m64 test_mm_cvttpd_pi32(__m128d a) {
  // CHECK-LABEL: test_mm_cvttpd_pi32
  // CHECK: call x86_mmx @llvm.x86.sse.cvttpd2pi
  return _mm_cvttpd_pi32(a);
}

__m64 test_mm_cvttps_pi32(__m128 a) {
  // CHECK-LABEL: test_mm_cvttps_pi32
  // CHECK: call x86_mmx @llvm.x86.sse.cvttps2pi
  return _mm_cvttps_pi32(a);
}

int test_mm_extract_pi16(__m64 a) {
  // CHECK-LABEL: test_mm_extract_pi16
  // CHECK: call i32 @llvm.x86.mmx.pextr.w
  return _mm_extract_pi16(a, 2);
}

__m64 test_m_from_int(int a) {
  // CHECK-LABEL: test_m_from_int
  // CHECK: insertelement <2 x i32>
  return _m_from_int(a);
}

__m64 test_m_from_int64(long long a) {
  // CHECK-LABEL: test_m_from_int64
  // CHECK: bitcast
  return _m_from_int64(a);
}

__m64 test_mm_hadd_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_hadd_pi16
  // CHECK: call x86_mmx @llvm.x86.ssse3.phadd.w
  return _mm_hadd_pi16(a, b);
}

__m64 test_mm_hadd_pi32(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_hadd_pi32
  // CHECK: call x86_mmx @llvm.x86.ssse3.phadd.d
  return _mm_hadd_pi32(a, b);
}

__m64 test_mm_hadds_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_hadds_pi16
  // CHECK: call x86_mmx @llvm.x86.ssse3.phadd.sw
  return _mm_hadds_pi16(a, b);
}

__m64 test_mm_hsub_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_hsub_pi16
  // CHECK: call x86_mmx @llvm.x86.ssse3.phsub.w
  return _mm_hsub_pi16(a, b);
}

__m64 test_mm_hsub_pi32(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_hsub_pi32
  // CHECK: call x86_mmx @llvm.x86.ssse3.phsub.d
  return _mm_hsub_pi32(a, b);
}

__m64 test_mm_hsubs_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_hsubs_pi16
  // CHECK: call x86_mmx @llvm.x86.ssse3.phsub.sw
  return _mm_hsubs_pi16(a, b);
}

__m64 test_mm_insert_pi16(__m64 a, int d) {
  // CHECK-LABEL: test_mm_insert_pi16
  // CHECK: call x86_mmx @llvm.x86.mmx.pinsr.w
  return _mm_insert_pi16(a, d, 2);
}

__m64 test_mm_madd_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_madd_pi16
  // CHECK: call x86_mmx @llvm.x86.mmx.pmadd.wd
  return _mm_madd_pi16(a, b);
}

__m64 test_mm_maddubs_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_maddubs_pi16
  // CHECK: call x86_mmx @llvm.x86.ssse3.pmadd.ub.sw
  return _mm_maddubs_pi16(a, b);
}

void test_mm_maskmove_si64(__m64 d, __m64 n, char *p) {
  // CHECK-LABEL: test_mm_maskmove_si64
  // CHECK: call void @llvm.x86.mmx.maskmovq
  _mm_maskmove_si64(d, n, p);
}

__m64 test_mm_max_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_max_pi16
  // CHECK: call x86_mmx @llvm.x86.mmx.pmaxs.w
  return _mm_max_pi16(a, b);
}

__m64 test_mm_max_pu8(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_max_pu8
  // CHECK: call x86_mmx @llvm.x86.mmx.pmaxu.b
  return _mm_max_pu8(a, b);
}

__m64 test_mm_min_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_min_pi16
  // CHECK: call x86_mmx @llvm.x86.mmx.pmins.w
  return _mm_min_pi16(a, b);
}

__m64 test_mm_min_pu8(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_min_pu8
  // CHECK: call x86_mmx @llvm.x86.mmx.pminu.b
  return _mm_min_pu8(a, b);
}

int test_mm_movemask_pi8(__m64 a) {
  // CHECK-LABEL: test_mm_movemask_pi8
  // CHECK: call i32 @llvm.x86.mmx.pmovmskb
  return _mm_movemask_pi8(a);
}

__m64 test_mm_mul_su32(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_mul_su32
  // CHECK: call x86_mmx @llvm.x86.mmx.pmulu.dq(x86_mmx %{{.*}}, x86_mmx %{{.*}})
  return _mm_mul_su32(a, b);
}

__m64 test_mm_mulhi_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_mulhi_pi16
  // CHECK: call x86_mmx @llvm.x86.mmx.pmulh.w
  return _mm_mulhi_pi16(a, b);
}

__m64 test_mm_mulhi_pu16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_mulhi_pu16
  // CHECK: call x86_mmx @llvm.x86.mmx.pmulhu.w
  return _mm_mulhi_pu16(a, b);
}

__m64 test_mm_mulhrs_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_mulhrs_pi16
  // CHECK: call x86_mmx @llvm.x86.ssse3.pmul.hr.sw
  return _mm_mulhrs_pi16(a, b);
}

__m64 test_mm_mullo_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_mullo_pi16
  // CHECK: call x86_mmx @llvm.x86.mmx.pmull.w
  return _mm_mullo_pi16(a, b);
}

__m64 test_mm_or_si64(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_or_si64
  // CHECK: call x86_mmx @llvm.x86.mmx.por
  return _mm_or_si64(a, b);
}

__m64 test_mm_packs_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_packs_pi16
  // CHECK: call x86_mmx @llvm.x86.mmx.packsswb
  return _mm_packs_pi16(a, b);
}

__m64 test_mm_packs_pi32(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_packs_pi32
  // CHECK: call x86_mmx @llvm.x86.mmx.packssdw
  return _mm_packs_pi32(a, b);
}

__m64 test_mm_packs_pu16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_packs_pu16
  // CHECK: call x86_mmx @llvm.x86.mmx.packuswb
  return _mm_packs_pu16(a, b);
}

__m64 test_mm_sad_pu8(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_sad_pu8
  // CHECK: call x86_mmx @llvm.x86.mmx.psad.bw
  return _mm_sad_pu8(a, b);
}

__m64 test_mm_set_pi8(char a, char b, char c, char d, char e, char f, char g, char h) {
  // CHECK-LABEL: test_mm_set_pi8
  // CHECK: insertelement <8 x i8>
  // CHECK: insertelement <8 x i8>
  // CHECK: insertelement <8 x i8>
  // CHECK: insertelement <8 x i8>
  // CHECK: insertelement <8 x i8>
  // CHECK: insertelement <8 x i8>
  // CHECK: insertelement <8 x i8>
  // CHECK: insertelement <8 x i8>
  return _mm_set_pi8(a, b, c, d, e, f, g, h);
}

__m64 test_mm_set_pi16(short a, short b, short c, short d) {
  // CHECK-LABEL: test_mm_set_pi16
  // CHECK: insertelement <4 x i16>
  // CHECK: insertelement <4 x i16>
  // CHECK: insertelement <4 x i16>
  // CHECK: insertelement <4 x i16>
  return _mm_set_pi16(a, b, c, d);
}

__m64 test_mm_set_pi32(int a, int b) {
  // CHECK-LABEL: test_mm_set_pi32
  // CHECK: insertelement <2 x i32>
  // CHECK: insertelement <2 x i32>
  return _mm_set_pi32(a, b);
}

__m64 test_mm_setr_pi8(char a, char b, char c, char d, char e, char f, char g, char h) {
  // CHECK-LABEL: test_mm_setr_pi8
  // CHECK: insertelement <8 x i8>
  // CHECK: insertelement <8 x i8>
  // CHECK: insertelement <8 x i8>
  // CHECK: insertelement <8 x i8>
  // CHECK: insertelement <8 x i8>
  // CHECK: insertelement <8 x i8>
  // CHECK: insertelement <8 x i8>
  // CHECK: insertelement <8 x i8>
  return _mm_setr_pi8(a, b, c, d, e, f, g, h);
}

__m64 test_mm_setr_pi16(short a, short b, short c, short d) {
  // CHECK-LABEL: test_mm_setr_pi16
  // CHECK: insertelement <4 x i16>
  // CHECK: insertelement <4 x i16>
  // CHECK: insertelement <4 x i16>
  // CHECK: insertelement <4 x i16>
  return _mm_setr_pi16(a, b, c, d);
}

__m64 test_mm_setr_pi32(int a, int b) {
  // CHECK-LABEL: test_mm_setr_pi32
  // CHECK: insertelement <2 x i32>
  // CHECK: insertelement <2 x i32>
  return _mm_setr_pi32(a, b);
}

__m64 test_mm_set1_pi8(char a) {
  // CHECK-LABEL: test_mm_set1_pi8
  // CHECK: insertelement <8 x i8>
  // CHECK: insertelement <8 x i8>
  // CHECK: insertelement <8 x i8>
  // CHECK: insertelement <8 x i8>
  // CHECK: insertelement <8 x i8>
  // CHECK: insertelement <8 x i8>
  // CHECK: insertelement <8 x i8>
  // CHECK: insertelement <8 x i8>
  return _mm_set1_pi8(a);
}

__m64 test_mm_set1_pi16(short a) {
  // CHECK-LABEL: test_mm_set1_pi16
  // CHECK: insertelement <4 x i16>
  // CHECK: insertelement <4 x i16>
  // CHECK: insertelement <4 x i16>
  // CHECK: insertelement <4 x i16>
  return _mm_set1_pi16(a);
}

__m64 test_mm_set1_pi32(int a) {
  // CHECK-LABEL: test_mm_set1_pi32
  // CHECK: insertelement <2 x i32>
  // CHECK: insertelement <2 x i32>
  return _mm_set1_pi32(a);
}

__m64 test_mm_shuffle_pi8(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_shuffle_pi8
  // CHECK: call x86_mmx @llvm.x86.ssse3.pshuf.b
  return _mm_shuffle_pi8(a, b);
}

__m64 test_mm_shuffle_pi16(__m64 a) {
  // CHECK-LABEL: test_mm_shuffle_pi16
  // CHECK: call x86_mmx @llvm.x86.sse.pshuf.w
  return _mm_shuffle_pi16(a, 3);
}

__m64 test_mm_sign_pi8(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_sign_pi8
  // CHECK: call x86_mmx @llvm.x86.ssse3.psign.b
  return _mm_sign_pi8(a, b);
}

__m64 test_mm_sign_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_sign_pi16
  // CHECK: call x86_mmx @llvm.x86.ssse3.psign.w
  return _mm_sign_pi16(a, b);
}

__m64 test_mm_sign_pi32(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_sign_pi32
  // CHECK: call x86_mmx @llvm.x86.ssse3.psign.d
  return _mm_sign_pi32(a, b);
}

__m64 test_mm_sll_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_sll_pi16
  // CHECK: call x86_mmx @llvm.x86.mmx.psll.w
  return _mm_sll_pi16(a, b);
}

__m64 test_mm_sll_pi32(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_sll_pi32
  // CHECK: call x86_mmx @llvm.x86.mmx.psll.d
  return _mm_sll_pi32(a, b);
}

__m64 test_mm_sll_si64(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_sll_si64
  // CHECK: call x86_mmx @llvm.x86.mmx.psll.q
  return _mm_sll_si64(a, b);
}

__m64 test_mm_slli_pi16(__m64 a) {
  // CHECK-LABEL: test_mm_slli_pi16
  // CHECK: call x86_mmx @llvm.x86.mmx.pslli.w
  return _mm_slli_pi16(a, 3);
}

__m64 test_mm_slli_pi32(__m64 a) {
  // CHECK-LABEL: test_mm_slli_pi32
  // CHECK: call x86_mmx @llvm.x86.mmx.pslli.d
  return _mm_slli_pi32(a, 3);
}

__m64 test_mm_slli_si64(__m64 a) {
  // CHECK-LABEL: test_mm_slli_si64
  // CHECK: call x86_mmx @llvm.x86.mmx.pslli.q
  return _mm_slli_si64(a, 3);
}

__m64 test_mm_sra_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_sra_pi16
  // CHECK: call x86_mmx @llvm.x86.mmx.psra.w
  return _mm_sra_pi16(a, b);
}

__m64 test_mm_sra_pi32(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_sra_pi32
  // CHECK: call x86_mmx @llvm.x86.mmx.psra.d
  return _mm_sra_pi32(a, b);
}

__m64 test_mm_srai_pi16(__m64 a) {
  // CHECK-LABEL: test_mm_srai_pi16
  // CHECK: call x86_mmx @llvm.x86.mmx.psrai.w
  return _mm_srai_pi16(a, 3);
}

__m64 test_mm_srai_pi32(__m64 a) {
  // CHECK-LABEL: test_mm_srai_pi32
  // CHECK: call x86_mmx @llvm.x86.mmx.psrai.d
  return _mm_srai_pi32(a, 3);
}

__m64 test_mm_srl_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_srl_pi16
  // CHECK: call x86_mmx @llvm.x86.mmx.psrl.w
  return _mm_srl_pi16(a, b);
}

__m64 test_mm_srl_pi32(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_srl_pi32
  // CHECK: call x86_mmx @llvm.x86.mmx.psrl.d
  return _mm_srl_pi32(a, b);
}

__m64 test_mm_srl_si64(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_srl_si64
  // CHECK: call x86_mmx @llvm.x86.mmx.psrl.q
  return _mm_srl_si64(a, b);
}

__m64 test_mm_srli_pi16(__m64 a) {
  // CHECK-LABEL: test_mm_srli_pi16
  // CHECK: call x86_mmx @llvm.x86.mmx.psrli.w
  return _mm_srli_pi16(a, 3);
}

__m64 test_mm_srli_pi32(__m64 a) {
  // CHECK-LABEL: test_mm_srli_pi32
  // CHECK: call x86_mmx @llvm.x86.mmx.psrli.d
  return _mm_srli_pi32(a, 3);
}

__m64 test_mm_srli_si64(__m64 a) {
  // CHECK-LABEL: test_mm_srli_si64
  // CHECK: call x86_mmx @llvm.x86.mmx.psrli.q
  return _mm_srli_si64(a, 3);
}

void test_mm_stream_pi(__m64 *p, __m64 a) {
  // CHECK-LABEL: test_mm_stream_pi
  // CHECK: call void @llvm.x86.mmx.movnt.dq
  _mm_stream_pi(p, a);
}

__m64 test_mm_sub_pi8(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_sub_pi8
  // CHECK: call x86_mmx @llvm.x86.mmx.psub.b
  return _mm_sub_pi8(a, b);
}

__m64 test_mm_sub_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_sub_pi16
  // CHECK: call x86_mmx @llvm.x86.mmx.psub.w
  return _mm_sub_pi16(a, b);
}

__m64 test_mm_sub_pi32(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_sub_pi32
  // CHECK: call x86_mmx @llvm.x86.mmx.psub.d
  return _mm_sub_pi32(a, b);
}

__m64 test_mm_sub_si64(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_sub_si64
  // CHECK: call x86_mmx @llvm.x86.mmx.psub.q(x86_mmx %{{.*}}, x86_mmx %{{.*}})
  return _mm_sub_si64(a, b);
}

__m64 test_mm_subs_pi8(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_subs_pi8
  // CHECK: call x86_mmx @llvm.x86.mmx.psubs.b
  return _mm_subs_pi8(a, b);
}

__m64 test_mm_subs_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_subs_pi16
  // CHECK: call x86_mmx @llvm.x86.mmx.psubs.w
  return _mm_subs_pi16(a, b);
}

__m64 test_mm_subs_pu8(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_subs_pu8
  // CHECK: call x86_mmx @llvm.x86.mmx.psubus.b
  return _mm_subs_pu8(a, b);
}

__m64 test_mm_subs_pu16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_subs_pu16
  // CHECK: call x86_mmx @llvm.x86.mmx.psubus.w
  return _mm_subs_pu16(a, b);
}

int test_m_to_int(__m64 a) {
  // CHECK-LABEL: test_m_to_int
  // CHECK: extractelement <2 x i32>
  return _m_to_int(a);
}

long long test_m_to_int64(__m64 a) {
  // CHECK-LABEL: test_m_to_int64
  // CHECK: bitcast
  return _m_to_int64(a);
}

__m64 test_mm_unpackhi_pi8(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_unpackhi_pi8
  // CHECK: call x86_mmx @llvm.x86.mmx.punpckhbw
  return _mm_unpackhi_pi8(a, b);
}

__m64 test_mm_unpackhi_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_unpackhi_pi16
  // CHECK: call x86_mmx @llvm.x86.mmx.punpckhwd
  return _mm_unpackhi_pi16(a, b);
}

__m64 test_mm_unpackhi_pi32(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_unpackhi_pi32
  // CHECK: call x86_mmx @llvm.x86.mmx.punpckhdq
  return _mm_unpackhi_pi32(a, b);
}

__m64 test_mm_unpacklo_pi8(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_unpacklo_pi8
  // CHECK: call x86_mmx @llvm.x86.mmx.punpcklbw
  return _mm_unpacklo_pi8(a, b);
}

__m64 test_mm_unpacklo_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_unpacklo_pi16
  // CHECK: call x86_mmx @llvm.x86.mmx.punpcklwd
  return _mm_unpacklo_pi16(a, b);
}

__m64 test_mm_unpacklo_pi32(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_unpacklo_pi32
  // CHECK: call x86_mmx @llvm.x86.mmx.punpckldq
  return _mm_unpacklo_pi32(a, b);
}

__m64 test_mm_xor_si64(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_xor_si64
  // CHECK: call x86_mmx @llvm.x86.mmx.pxor
  return _mm_xor_si64(a, b);
}
