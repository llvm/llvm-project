// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +ssse3 -emit-llvm -o - -Wall -Werror | FileCheck %s --implicit-check-not=x86mmx
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +ssse3 -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s --implicit-check-not=x86mmx


#include <immintrin.h>

__m64 test_mm_abs_pi8(__m64 a) {
  // CHECK-LABEL: test_mm_abs_pi8
  // CHECK: call <8 x i8> @llvm.abs.v8i8(
  return _mm_abs_pi8(a);
}

__m64 test_mm_abs_pi16(__m64 a) {
  // CHECK-LABEL: test_mm_abs_pi16
  // CHECK: call <4 x i16> @llvm.abs.v4i16(
  return _mm_abs_pi16(a);
}

__m64 test_mm_abs_pi32(__m64 a) {
  // CHECK-LABEL: test_mm_abs_pi32
  // CHECK: call <2 x i32> @llvm.abs.v2i32(
  return _mm_abs_pi32(a);
}

__m64 test_mm_add_pi8(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_add_pi8
  // CHECK: add <8 x i8> {{%.*}}, {{%.*}}
  return _mm_add_pi8(a, b);
}

__m64 test_mm_add_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_add_pi16
  // CHECK: add <4 x i16> {{%.*}}, {{%.*}}
  return _mm_add_pi16(a, b);
}

__m64 test_mm_add_pi32(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_add_pi32
  // CHECK: add <2 x i32> {{%.*}}, {{%.*}}
  return _mm_add_pi32(a, b);
}

__m64 test_mm_add_si64(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_add_si64
  // CHECK: add i64 {{%.*}}, {{%.*}}
  return _mm_add_si64(a, b);
}

__m64 test_mm_adds_pi8(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_adds_pi8
  // CHECK: call <8 x i8> @llvm.sadd.sat.v8i8(
  return _mm_adds_pi8(a, b);
}

__m64 test_mm_adds_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_adds_pi16
  // CHECK: call <4 x i16> @llvm.sadd.sat.v4i16(
  return _mm_adds_pi16(a, b);
}

__m64 test_mm_adds_pu8(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_adds_pu8
  // CHECK: call <8 x i8> @llvm.uadd.sat.v8i8(
  return _mm_adds_pu8(a, b);
}

__m64 test_mm_adds_pu16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_adds_pu16
  // CHECK: call <4 x i16> @llvm.uadd.sat.v4i16(
  return _mm_adds_pu16(a, b);
}

__m64 test_mm_alignr_pi8(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_alignr_pi8
  // CHECK: shufflevector <16 x i8> {{%.*}}, <16 x i8> zeroinitializer, <16 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17>
  return _mm_alignr_pi8(a, b, 2);
}

__m64 test_mm_and_si64(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_and_si64
  // CHECK: and <1 x i64> {{%.*}}, {{%.*}}
  return _mm_and_si64(a, b);
}

__m64 test_mm_andnot_si64(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_andnot_si64
  // CHECK: [[TMP:%.*]] = xor <1 x i64> {{%.*}}, <i64 -1>
  // CHECK: and <1 x i64> [[TMP]], {{%.*}}
  return _mm_andnot_si64(a, b);
}

__m64 test_mm_avg_pu8(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_avg_pu8
  // CHECK: call <16 x i8> @llvm.x86.sse2.pavg.b(
  return _mm_avg_pu8(a, b);
}

__m64 test_mm_avg_pu16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_avg_pu16
  // CHECK: call <8 x i16> @llvm.x86.sse2.pavg.w(
  return _mm_avg_pu16(a, b);
}

__m64 test_mm_cmpeq_pi8(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_cmpeq_pi8
  // CHECK:      [[CMP:%.*]] = icmp eq <8 x i8> {{%.*}}, {{%.*}}
  // CHECK-NEXT: {{%.*}} = sext <8 x i1> [[CMP]] to <8 x i8>
  return _mm_cmpeq_pi8(a, b);
}

__m64 test_mm_cmpeq_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_cmpeq_pi16
  // CHECK:      [[CMP:%.*]] = icmp eq <4 x i16> {{%.*}}, {{%.*}}
  // CHECK-NEXT: {{%.*}} = sext <4 x i1> [[CMP]] to <4 x i16>
  return _mm_cmpeq_pi16(a, b);
}

__m64 test_mm_cmpeq_pi32(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_cmpeq_pi32
  // CHECK:      [[CMP:%.*]] = icmp eq <2 x i32> {{%.*}}, {{%.*}}
  // CHECK-NEXT: {{%.*}} = sext <2 x i1> [[CMP]] to <2 x i32>
  return _mm_cmpeq_pi32(a, b);
}

__m64 test_mm_cmpgt_pi8(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_cmpgt_pi8
  // CHECK:      [[CMP:%.*]] = icmp sgt <8 x i8> {{%.*}}, {{%.*}}
  // CHECK-NEXT: {{%.*}} = sext <8 x i1> [[CMP]] to <8 x i8>
  return _mm_cmpgt_pi8(a, b);
}

__m64 test_mm_cmpgt_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_cmpgt_pi16
  // CHECK:      [[CMP:%.*]] = icmp sgt <4 x i16> {{%.*}}, {{%.*}}
  // CHECK-NEXT: {{%.*}} = sext <4 x i1> [[CMP]] to <4 x i16>
  return _mm_cmpgt_pi16(a, b);
}

__m64 test_mm_cmpgt_pi32(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_cmpgt_pi32
  // CHECK:      [[CMP:%.*]] = icmp sgt <2 x i32> {{%.*}}, {{%.*}}
  // CHECK-NEXT: {{%.*}} = sext <2 x i1> [[CMP]] to <2 x i32>
  return _mm_cmpgt_pi32(a, b);
}

__m128 test_mm_cvt_pi2ps(__m128 a, __m64 b) {
  // CHECK-LABEL: test_mm_cvt_pi2ps
  // CHECK: sitofp <4 x i32> {{%.*}} to <4 x float>
  return _mm_cvt_pi2ps(a, b);
}

__m64 test_mm_cvt_ps2pi(__m128 a) {
  // CHECK-LABEL: test_mm_cvt_ps2pi
  // CHECK: call <4 x i32> @llvm.x86.sse2.cvtps2dq(
  return _mm_cvt_ps2pi(a);
}

__m64 test_mm_cvtpd_pi32(__m128d a) {
  // CHECK-LABEL: test_mm_cvtpd_pi32
  // CHECK: call <4 x i32> @llvm.x86.sse2.cvtpd2dq(
  return _mm_cvtpd_pi32(a);
}

__m128 test_mm_cvtpi16_ps(__m64 a) {
  // CHECK-LABEL: test_mm_cvtpi16_ps
  // CHECK: sitofp <4 x i16> {{%.*}} to <4 x float>
  return _mm_cvtpi16_ps(a);
}

__m128d test_mm_cvtpi32_pd(__m64 a) {
  // CHECK-LABEL: test_mm_cvtpi32_pd
  // CHECK: sitofp <2 x i32> {{%.*}} to <2 x double>
  return _mm_cvtpi32_pd(a);
}

__m128 test_mm_cvtpi32_ps(__m128 a, __m64 b) {
  // CHECK-LABEL: test_mm_cvtpi32_ps
  // CHECK: sitofp <4 x i32> {{%.*}} to <4 x float>
  return _mm_cvtpi32_ps(a, b);
}

__m128 test_mm_cvtpi32x2_ps(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_cvtpi32x2_ps
  // CHECK: sitofp <4 x i32> {{%.*}} to <4 x float>
  return _mm_cvtpi32x2_ps(a, b);
}

__m64 test_mm_cvtps_pi16(__m128 a) {
  // CHECK-LABEL: test_mm_cvtps_pi16
  // CHECK: [[TMP0:%.*]] = call <4 x i32> @llvm.x86.sse2.cvtps2dq(<4 x float> {{%.*}})
  // CHECK: call <8 x i16> @llvm.x86.sse2.packssdw.128(<4 x i32> [[TMP0]],
  return _mm_cvtps_pi16(a);
}

__m64 test_mm_cvtps_pi32(__m128 a) {
  // CHECK-LABEL: test_mm_cvtps_pi32
  // CHECK: call <4 x i32> @llvm.x86.sse2.cvtps2dq(
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
  // CHECK: call <4 x i32> @llvm.x86.sse2.cvttpd2dq(
  return _mm_cvttpd_pi32(a);
}

__m64 test_mm_cvttps_pi32(__m128 a) {
  // CHECK-LABEL: test_mm_cvttps_pi32
  // CHECK: call <4 x i32> @llvm.x86.sse2.cvttps2dq(
  return _mm_cvttps_pi32(a);
}

int test_mm_extract_pi16(__m64 a) {
  // CHECK-LABEL: test_mm_extract_pi16
  // CHECK: extractelement <4 x i16> {{%.*}}, i64 2
  return _mm_extract_pi16(a, 2);
}

__m64 test_m_from_int(int a) {
  // CHECK-LABEL: test_m_from_int
  // CHECK: insertelement <2 x i32>
  return _m_from_int(a);
}

__m64 test_m_from_int64(long long a) {
  // CHECK-LABEL: test_m_from_int64
  return _m_from_int64(a);
}

__m64 test_mm_hadd_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_hadd_pi16
  // CHECK: call <8 x i16> @llvm.x86.ssse3.phadd.w.128(
  return _mm_hadd_pi16(a, b);
}

__m64 test_mm_hadd_pi32(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_hadd_pi32
  // CHECK: call <4 x i32> @llvm.x86.ssse3.phadd.d.128(
  return _mm_hadd_pi32(a, b);
}

__m64 test_mm_hadds_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_hadds_pi16
  // CHECK: call <8 x i16> @llvm.x86.ssse3.phadd.sw.128(
  return _mm_hadds_pi16(a, b);
}

__m64 test_mm_hsub_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_hsub_pi16
  // CHECK: call <8 x i16> @llvm.x86.ssse3.phsub.w.128(
  return _mm_hsub_pi16(a, b);
}

__m64 test_mm_hsub_pi32(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_hsub_pi32
  // CHECK: call <4 x i32> @llvm.x86.ssse3.phsub.d.128(
  return _mm_hsub_pi32(a, b);
}

__m64 test_mm_hsubs_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_hsubs_pi16
  // CHECK: call <8 x i16> @llvm.x86.ssse3.phsub.sw.128(
  return _mm_hsubs_pi16(a, b);
}

__m64 test_mm_insert_pi16(__m64 a, int d) {
  // CHECK-LABEL: test_mm_insert_pi16
  // CHECK: insertelement <4 x i16>
  return _mm_insert_pi16(a, d, 2);
}

__m64 test_mm_madd_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_madd_pi16
  // CHECK: call <4 x i32> @llvm.x86.sse2.pmadd.wd(
  return _mm_madd_pi16(a, b);
}

__m64 test_mm_maddubs_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_maddubs_pi16
  // CHECK: call <8 x i16> @llvm.x86.ssse3.pmadd.ub.sw.128(
  return _mm_maddubs_pi16(a, b);
}

void test_mm_maskmove_si64(__m64 d, __m64 n, char *p) {
  // CHECK-LABEL: test_mm_maskmove_si64
  // CHECK: call void @llvm.x86.sse2.maskmov.dqu(
  _mm_maskmove_si64(d, n, p);
}

__m64 test_mm_max_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_max_pi16
  // CHECK: call <4 x i16> @llvm.smax.v4i16(
  return _mm_max_pi16(a, b);
}

__m64 test_mm_max_pu8(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_max_pu8
  // CHECK: call <8 x i8> @llvm.umax.v8i8(
  return _mm_max_pu8(a, b);
}

__m64 test_mm_min_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_min_pi16
  // CHECK: call <4 x i16> @llvm.smin.v4i16(
  return _mm_min_pi16(a, b);
}

__m64 test_mm_min_pu8(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_min_pu8
  // CHECK: call <8 x i8> @llvm.umin.v8i8(
  return _mm_min_pu8(a, b);
}

int test_mm_movemask_pi8(__m64 a) {
  // CHECK-LABEL: test_mm_movemask_pi8
  // CHECK: call i32 @llvm.x86.sse2.pmovmskb.128(
  return _mm_movemask_pi8(a);
}

__m64 test_mm_mul_su32(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_mul_su32
  // CHECK: and <2 x i64> {{%.*}}, <i64 4294967295, i64 4294967295>
  // CHECK: and <2 x i64> {{%.*}}, <i64 4294967295, i64 4294967295>
  // CHECK: mul <2 x i64> %{{.*}}, %{{.*}}
  return _mm_mul_su32(a, b);
}

__m64 test_mm_mulhi_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_mulhi_pi16
  // CHECK: call <8 x i16> @llvm.x86.sse2.pmulh.w(
  return _mm_mulhi_pi16(a, b);
}

__m64 test_mm_mulhi_pu16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_mulhi_pu16
  // CHECK: call <8 x i16> @llvm.x86.sse2.pmulhu.w(
  return _mm_mulhi_pu16(a, b);
}

__m64 test_mm_mulhrs_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_mulhrs_pi16
  // CHECK: call <8 x i16> @llvm.x86.ssse3.pmul.hr.sw.128(
  return _mm_mulhrs_pi16(a, b);
}

__m64 test_mm_mullo_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_mullo_pi16
  // CHECK: mul <4 x i16> {{%.*}}, {{%.*}}
  return _mm_mullo_pi16(a, b);
}

__m64 test_mm_or_si64(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_or_si64
  // CHECK: or <1 x i64> {{%.*}}, {{%.*}}
  return _mm_or_si64(a, b);
}

__m64 test_mm_packs_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_packs_pi16
  // CHECK: call <16 x i8> @llvm.x86.sse2.packsswb.128(
  return _mm_packs_pi16(a, b);
}

__m64 test_mm_packs_pi32(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_packs_pi32
  // CHECK: call <8 x i16> @llvm.x86.sse2.packssdw.128(
  return _mm_packs_pi32(a, b);
}

__m64 test_mm_packs_pu16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_packs_pu16
  // CHECK: call <16 x i8> @llvm.x86.sse2.packuswb.128(
  return _mm_packs_pu16(a, b);
}

__m64 test_mm_sad_pu8(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_sad_pu8
  // CHECK: call <2 x i64> @llvm.x86.sse2.psad.bw(<16 x i8>
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
  // CHECK: call <16 x i8> @llvm.x86.ssse3.pshuf.b.128(
  return _mm_shuffle_pi8(a, b);
}

__m64 test_mm_shuffle_pi16(__m64 a) {
  // CHECK-LABEL: test_mm_shuffle_pi16
  // CHECK: shufflevector <4 x i16> {{%.*}}, <4 x i16> {{%.*}}, <4 x i32> <i32 3, i32 0, i32 0, i32 0>
  return _mm_shuffle_pi16(a, 3);
}

__m64 test_mm_sign_pi8(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_sign_pi8
  // CHECK: call <16 x i8> @llvm.x86.ssse3.psign.b.128(
  return _mm_sign_pi8(a, b);
}

__m64 test_mm_sign_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_sign_pi16
  // CHECK: call <8 x i16> @llvm.x86.ssse3.psign.w.128(
  return _mm_sign_pi16(a, b);
}

__m64 test_mm_sign_pi32(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_sign_pi32
  // CHECK: call <4 x i32> @llvm.x86.ssse3.psign.d.128(
  return _mm_sign_pi32(a, b);
}

__m64 test_mm_sll_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_sll_pi16
  // CHECK: call <8 x i16> @llvm.x86.sse2.psll.w(
  return _mm_sll_pi16(a, b);
}

__m64 test_mm_sll_pi32(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_sll_pi32
  // CHECK: call <4 x i32> @llvm.x86.sse2.psll.d(
  return _mm_sll_pi32(a, b);
}

__m64 test_mm_sll_si64(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_sll_si64
  // CHECK: call <2 x i64> @llvm.x86.sse2.psll.q(
  return _mm_sll_si64(a, b);
}

__m64 test_mm_slli_pi16(__m64 a) {
  // CHECK-LABEL: test_mm_slli_pi16
  // CHECK: call <8 x i16> @llvm.x86.sse2.pslli.w(
  return _mm_slli_pi16(a, 3);
}

__m64 test_mm_slli_pi32(__m64 a) {
  // CHECK-LABEL: test_mm_slli_pi32
  // CHECK: call <4 x i32> @llvm.x86.sse2.pslli.d(
  return _mm_slli_pi32(a, 3);
}

__m64 test_mm_slli_si64(__m64 a) {
  // CHECK-LABEL: test_mm_slli_si64
  // CHECK: call <2 x i64> @llvm.x86.sse2.pslli.q(
  return _mm_slli_si64(a, 3);
}

__m64 test_mm_sra_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_sra_pi16
  // CHECK: call <8 x i16> @llvm.x86.sse2.psra.w(
  return _mm_sra_pi16(a, b);
}

__m64 test_mm_sra_pi32(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_sra_pi32
  // CHECK: call <4 x i32> @llvm.x86.sse2.psra.d(
  return _mm_sra_pi32(a, b);
}

__m64 test_mm_srai_pi16(__m64 a) {
  // CHECK-LABEL: test_mm_srai_pi16
  // CHECK: call <8 x i16> @llvm.x86.sse2.psrai.w(
  return _mm_srai_pi16(a, 3);
}

__m64 test_mm_srai_pi32(__m64 a) {
  // CHECK-LABEL: test_mm_srai_pi32
  // CHECK: call <4 x i32> @llvm.x86.sse2.psrai.d(
  return _mm_srai_pi32(a, 3);
}

__m64 test_mm_srl_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_srl_pi16
  // CHECK: call <8 x i16> @llvm.x86.sse2.psrl.w(
  return _mm_srl_pi16(a, b);
}

__m64 test_mm_srl_pi32(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_srl_pi32
  // CHECK: call <4 x i32> @llvm.x86.sse2.psrl.d(
  return _mm_srl_pi32(a, b);
}

__m64 test_mm_srl_si64(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_srl_si64
  // CHECK: call <2 x i64> @llvm.x86.sse2.psrl.q(
  return _mm_srl_si64(a, b);
}

__m64 test_mm_srli_pi16(__m64 a) {
  // CHECK-LABEL: test_mm_srli_pi16
  // CHECK: call <8 x i16> @llvm.x86.sse2.psrli.w(
  return _mm_srli_pi16(a, 3);
}

__m64 test_mm_srli_pi32(__m64 a) {
  // CHECK-LABEL: test_mm_srli_pi32
  // CHECK: call <4 x i32> @llvm.x86.sse2.psrli.d(
  return _mm_srli_pi32(a, 3);
}

__m64 test_mm_srli_si64(__m64 a) {
  // CHECK-LABEL: test_mm_srli_si64
  // CHECK: call <2 x i64> @llvm.x86.sse2.psrli.q(
  return _mm_srli_si64(a, 3);
}

void test_mm_stream_pi(__m64 *p, __m64 a) {
  // CHECK-LABEL: test_mm_stream_pi
  // CHECK: store <1 x i64> {{%.*}}, ptr {{%.*}}, align 8, !nontemporal
  _mm_stream_pi(p, a);
}

void test_mm_stream_pi_void(void *p, __m64 a) {
  // CHECK-LABEL: test_mm_stream_pi_void
  // CHECK: store <1 x i64> {{%.*}}, ptr {{%.*}}, align 8, !nontemporal
  _mm_stream_pi(p, a);
}

__m64 test_mm_sub_pi8(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_sub_pi8
  // CHECK: sub <8 x i8> {{%.*}}, {{%.*}}
  return _mm_sub_pi8(a, b);
}

__m64 test_mm_sub_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_sub_pi16
  // CHECK: sub <4 x i16> {{%.*}}, {{%.*}}
  return _mm_sub_pi16(a, b);
}

__m64 test_mm_sub_pi32(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_sub_pi32
  // CHECK: sub <2 x i32> {{%.*}}, {{%.*}}
  return _mm_sub_pi32(a, b);
}

__m64 test_mm_sub_si64(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_sub_si64
  // CHECK: sub i64 {{%.*}}, {{%.*}}
  return _mm_sub_si64(a, b);
}

__m64 test_mm_subs_pi8(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_subs_pi8
  // CHECK: call <8 x i8> @llvm.ssub.sat.v8i8(
  return _mm_subs_pi8(a, b);
}

__m64 test_mm_subs_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_subs_pi16
  // CHECK: call <4 x i16> @llvm.ssub.sat.v4i16(
  return _mm_subs_pi16(a, b);
}

__m64 test_mm_subs_pu8(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_subs_pu8
  // CHECK: call <8 x i8> @llvm.usub.sat.v8i8(
  return _mm_subs_pu8(a, b);
}

__m64 test_mm_subs_pu16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_subs_pu16
  // CHECK: call <4 x i16> @llvm.usub.sat.v4i16(
  return _mm_subs_pu16(a, b);
}

int test_m_to_int(__m64 a) {
  // CHECK-LABEL: test_m_to_int
  // CHECK: extractelement <2 x i32>
  return _m_to_int(a);
}

long long test_m_to_int64(__m64 a) {
  // CHECK-LABEL: test_m_to_int64
  return _m_to_int64(a);
}

__m64 test_mm_unpackhi_pi8(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_unpackhi_pi8
  // CHECK: shufflevector <8 x i8> {{%.*}}, <8 x i8> {{%.*}}, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
  return _mm_unpackhi_pi8(a, b);
}

__m64 test_mm_unpackhi_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_unpackhi_pi16
  // CHECK: shufflevector <4 x i16> {{%.*}}, <4 x i16> {{%.*}}, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
  return _mm_unpackhi_pi16(a, b);
}

__m64 test_mm_unpackhi_pi32(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_unpackhi_pi32
  // CHECK: shufflevector <2 x i32> {{%.*}}, <2 x i32> {{%.*}}, <2 x i32> <i32 1, i32 3>
  return _mm_unpackhi_pi32(a, b);
}

__m64 test_mm_unpacklo_pi8(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_unpacklo_pi8
  // CHECK: shufflevector <8 x i8> {{%.*}}, <8 x i8> {{%.*}}, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
  return _mm_unpacklo_pi8(a, b);
}

__m64 test_mm_unpacklo_pi16(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_unpacklo_pi16
  // CHECK: shufflevector <4 x i16> {{%.*}}, <4 x i16> {{%.*}}, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
  return _mm_unpacklo_pi16(a, b);
}

__m64 test_mm_unpacklo_pi32(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_unpacklo_pi32
  // CHECK: shufflevector <2 x i32> {{%.*}}, <2 x i32> {{%.*}}, <2 x i32> <i32 0, i32 2>
  return _mm_unpacklo_pi32(a, b);
}

__m64 test_mm_xor_si64(__m64 a, __m64 b) {
  // CHECK-LABEL: test_mm_xor_si64
  // CHECK: xor <1 x i64> {{%.*}}, {{%.*}}
  return _mm_xor_si64(a, b);
}
