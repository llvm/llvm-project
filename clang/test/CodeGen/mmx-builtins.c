// REQUIRES: x86-registered-target
// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin -target-feature +ssse3 -S -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin -target-feature +ssse3 -fno-signed-char -S -o - | FileCheck %s

// FIXME: Disable inclusion of mm_malloc.h, our current implementation is broken
// on win32 since we don't generally know how to find errno.h.
#define __MM_MALLOC_H

#include <tmmintrin.h>

__m64 test1(__m64 a, __m64 b) {
  // CHECK: phaddw
  return _mm_hadd_pi16(a, b);
}

__m64 test2(__m64 a, __m64 b) {
  // CHECK: phaddd
  return _mm_hadd_pi32(a, b);
}

__m64 test3(__m64 a, __m64 b) {
  // CHECK: phaddsw
  return _mm_hadds_pi16(a, b);
}

__m64 test4(__m64 a, __m64 b) {
  // CHECK: phsubw
  return _mm_hsub_pi16(a, b);
}

__m64 test5(__m64 a, __m64 b) {
  // CHECK: phsubd
  return _mm_hsub_pi32(a, b);
}

__m64 test6(__m64 a, __m64 b) {
  // CHECK: phsubsw
  return _mm_hsubs_pi16(a, b);
}

__m64 test7(__m64 a, __m64 b) {
  // CHECK: pmaddubsw
  return _mm_maddubs_pi16(a, b);
}

__m64 test8(__m64 a, __m64 b) {
  // CHECK: pmulhrsw
  return _mm_mulhrs_pi16(a, b);
}

__m64 test9(__m64 a, __m64 b) {
  // CHECK: pshufb
  return _mm_shuffle_pi8(a, b);
}

__m64 test10(__m64 a, __m64 b) {
  // CHECK: psignb
  return _mm_sign_pi8(a, b);
}

__m64 test11(__m64 a, __m64 b) {
  // CHECK: psignw
  return _mm_sign_pi16(a, b);
}

__m64 test12(__m64 a, __m64 b) {
  // CHECK: psignd
  return _mm_sign_pi32(a, b);
}

__m64 test13(__m64 a) {
  // CHECK: pabsb
  return _mm_abs_pi8(a);
}

__m64 test14(__m64 a) {
  // CHECK: pabsw
  return _mm_abs_pi16(a);
}

__m64 test15(__m64 a) {
  // CHECK: pabsd
  return _mm_abs_pi32(a);
}

__m64 test16(__m64 a, __m64 b) {
  // CHECK: palignr
  return _mm_alignr_pi8(a, b, 2);
}

__m64 test17(__m128d a) {
  // CHECK: cvtpd2pi
  return _mm_cvtpd_pi32(a);
}

__m64 test18(__m128d a) {
  // CHECK: cvttpd2pi
  return _mm_cvttpd_pi32(a);
}

__m128d test19(__m64 a) {
  // CHECK: cvtpi2pd
  return _mm_cvtpi32_pd(a);
}

__m64 test20(__m64 a, __m64 b) {
  // CHECK: pmuludq
  return _mm_mul_su32(a, b);
}

__m64 test21(__m64 a) {
  // CHECK: pshufw
  return _mm_shuffle_pi16(a, 3);
}

__m64 test22(__m64 a, __m64 b) {
  // CHECK: pmulhuw
  return _mm_mulhi_pu16(a, b);
}

void test23(__m64 d, __m64 n, char *p) {
  // CHECK: maskmovq
  _mm_maskmove_si64(d, n, p);
}

int test24(__m64 a) {
  // CHECK: pmovmskb
  return _mm_movemask_pi8(a);
}

void test25(__m64 *p, __m64 a) {
  // CHECK: movntq
  _mm_stream_pi(p, a);
}

__m64 test26(__m64 a, __m64 b) {
  // CHECK: pavgb
  return _mm_avg_pu8(a, b);
}

__m64 test27(__m64 a, __m64 b) {
  // CHECK: pavgw
  return _mm_avg_pu16(a, b);
}

__m64 test28(__m64 a, __m64 b) {
  // CHECK: pmaxub
  return _mm_max_pu8(a, b);
}

__m64 test29(__m64 a, __m64 b) {
  // CHECK: pmaxsw
  return _mm_max_pi16(a, b);
}

__m64 test30(__m64 a, __m64 b) {
  // CHECK: pminub
  return _mm_min_pu8(a, b);
}

__m64 test31(__m64 a, __m64 b) {
  // CHECK: pminsw
  return _mm_min_pi16(a, b);
}

__m64 test32(__m64 a, __m64 b) {
  // CHECK: psadbw
  return _mm_sad_pu8(a, b);
}

__m64 test33(__m64 a, __m64 b) {
  // CHECK: paddb
  return _mm_add_pi8(a, b);
}

__m64 test34(__m64 a, __m64 b) {
  // CHECK: paddw
  return _mm_add_pi16(a, b);
}

__m64 test35(__m64 a, __m64 b) {
  // CHECK: paddd
  return _mm_add_pi32(a, b);
}

__m64 test36(__m64 a, __m64 b) {
  // CHECK: paddq
  return __builtin_ia32_paddq(a, b);
}

__m64 test37(__m64 a, __m64 b) {
  // CHECK: paddsb
  return _mm_adds_pi8(a, b);
}

__m64 test38(__m64 a, __m64 b) {
  // CHECK: paddsw
  return _mm_adds_pi16(a, b);
}

__m64 test39(__m64 a, __m64 b) {
  // CHECK: paddusb
  return _mm_adds_pu8(a, b);
}

__m64 test40(__m64 a, __m64 b) {
  // CHECK: paddusw
  return _mm_adds_pu16(a, b);
}

__m64 test41(__m64 a, __m64 b) {
  // CHECK: psubb
  return _mm_sub_pi8(a, b);
}

__m64 test42(__m64 a, __m64 b) {
  // CHECK: psubw
  return _mm_sub_pi16(a, b);
}

__m64 test43(__m64 a, __m64 b) {
  // CHECK: psubd
  return _mm_sub_pi32(a, b);
}

__m64 test44(__m64 a, __m64 b) {
  // CHECK: psubq
  return __builtin_ia32_psubq(a, b);
}

__m64 test45(__m64 a, __m64 b) {
  // CHECK: psubsb
  return _mm_subs_pi8(a, b);
}

__m64 test46(__m64 a, __m64 b) {
  // CHECK: psubsw
  return _mm_subs_pi16(a, b);
}

__m64 test47(__m64 a, __m64 b) {
  // CHECK: psubusb
  return _mm_subs_pu8(a, b);
}

__m64 test48(__m64 a, __m64 b) {
  // CHECK: psubusw
  return _mm_subs_pu16(a, b);
}

__m64 test49(__m64 a, __m64 b) {
  // CHECK: pmaddwd
  return _mm_madd_pi16(a, b);
}

__m64 test50(__m64 a, __m64 b) {
  // CHECK: pmulhw
  return _mm_mulhi_pi16(a, b);
}

__m64 test51(__m64 a, __m64 b) {
  // CHECK: pmullw
  return _mm_mullo_pi16(a, b);
}

__m64 test52(__m64 a, __m64 b) {
  // CHECK: pmullw
  return _mm_mullo_pi16(a, b);
}

__m64 test53(__m64 a, __m64 b) {
  // CHECK: pand
  return _mm_and_si64(a, b);
}

__m64 test54(__m64 a, __m64 b) {
  // CHECK: pandn
  return _mm_andnot_si64(a, b);
}

__m64 test55(__m64 a, __m64 b) {
  // CHECK: por
  return _mm_or_si64(a, b);
}

__m64 test56(__m64 a, __m64 b) {
  // CHECK: pxor
  return _mm_xor_si64(a, b);
}

__m64 test57(__m64 a, __m64 b) {
  // CHECK: pavgb
  return _mm_avg_pu8(a, b);
}

__m64 test58(__m64 a, __m64 b) {
  // CHECK: pavgw
  return _mm_avg_pu16(a, b);
}

__m64 test59(__m64 a, __m64 b) {
  // CHECK: psllw
  return _mm_sll_pi16(a, b);
}

__m64 test60(__m64 a, __m64 b) {
  // CHECK: pslld
  return _mm_sll_pi32(a, b);
}

__m64 test61(__m64 a, __m64 b) {
  // CHECK: psllq
  return _mm_sll_si64(a, b);
}

__m64 test62(__m64 a, __m64 b) {
  // CHECK: psrlw
  return _mm_srl_pi16(a, b);
}

__m64 test63(__m64 a, __m64 b) {
  // CHECK: psrld
  return _mm_srl_pi32(a, b);
}

__m64 test64(__m64 a, __m64 b) {
  // CHECK: psrlq
  return _mm_srl_si64(a, b);
}

__m64 test65(__m64 a, __m64 b) {
  // CHECK: psraw
  return _mm_sra_pi16(a, b);
}

__m64 test66(__m64 a, __m64 b) {
  // CHECK: psrad
  return _mm_sra_pi32(a, b);
}

__m64 test67(__m64 a) {
  // CHECK: psllw
  return _mm_slli_pi16(a, 3);
}

__m64 test68(__m64 a) {
  // CHECK: pslld
  return _mm_slli_pi32(a, 3);
}

__m64 test69(__m64 a) {
  // CHECK: psllq
  return _mm_slli_si64(a, 3);
}

__m64 test70(__m64 a) {
  // CHECK: psrlw
  return _mm_srli_pi16(a, 3);
}

__m64 test71(__m64 a) {
  // CHECK: psrld
  return _mm_srli_pi32(a, 3);
}

__m64 test72(__m64 a) {
  // CHECK: psrlq
  return _mm_srli_si64(a, 3);
}

__m64 test73(__m64 a) {
  // CHECK: psraw
  return _mm_srai_pi16(a, 3);
}

__m64 test74(__m64 a) {
  // CHECK: psrad
  return _mm_srai_pi32(a, 3);
}

__m64 test75(__m64 a, __m64 b) {
  // CHECK: packsswb
  return _mm_packs_pi16(a, b);
}

__m64 test76(__m64 a, __m64 b) {
  // CHECK: packssdw
  return _mm_packs_pi32(a, b);
}

__m64 test77(__m64 a, __m64 b) {
  // CHECK: packuswb
  return _mm_packs_pu16(a, b);
}

__m64 test78(__m64 a, __m64 b) {
  // CHECK: punpckhbw
  return _mm_unpackhi_pi8(a, b);
}

__m64 test79(__m64 a, __m64 b) {
  // CHECK: punpckhwd
  return _mm_unpackhi_pi16(a, b);
}

__m64 test80(__m64 a, __m64 b) {
  // CHECK: punpckhdq
  return _mm_unpackhi_pi32(a, b);
}

__m64 test81(__m64 a, __m64 b) {
  // CHECK: punpcklbw
  return _mm_unpacklo_pi8(a, b);
}

__m64 test82(__m64 a, __m64 b) {
  // CHECK: punpcklwd
  return _mm_unpacklo_pi16(a, b);
}

__m64 test83(__m64 a, __m64 b) {
  // CHECK: punpckldq
  return _mm_unpacklo_pi32(a, b);
}

__m64 test84(__m64 a, __m64 b) {
  // CHECK: pcmpeqb
  return _mm_cmpeq_pi8(a, b);
}

__m64 test85(__m64 a, __m64 b) {
  // CHECK: pcmpeqw
  return _mm_cmpeq_pi16(a, b);
}

__m64 test86(__m64 a, __m64 b) {
  // CHECK: pcmpeqd
  return _mm_cmpeq_pi32(a, b);
}

__m64 test87(__m64 a, __m64 b) {
  // CHECK: pcmpgtb
  return _mm_cmpgt_pi8(a, b);
}

__m64 test88(__m64 a, __m64 b) {
  // CHECK: pcmpgtw
  return _mm_cmpgt_pi16(a, b);
}

__m64 test89(__m64 a, __m64 b) {
  // CHECK: pcmpgtd
  return _mm_cmpgt_pi32(a, b);
}

__m64 test90(int a) {
  // CHECK: movd
  return _m_from_int(a);
}

int test91(__m64 a) {
  // CHECK: movd
  return _m_to_int(a);
}
