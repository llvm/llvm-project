// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fsyntax-only \
// RUN:  -target-feature +f16c -target-feature +avx -target-feature +sse4a \
// RUN:  -target-feature +aes -target-feature +xop -target-feature +avx2 \
// RUN:  -target-feature +tbm -verify %s

// Error test cases where a variable is passed to intrinsics but an
// immediate operand is required.

// Don't include mm_malloc.h, it's system specific.
#define __MM_MALLOC_H

#include <x86intrin.h>

unsigned short check__cvtss_sh(float val, const int I) {
  return _cvtss_sh(val, I); // expected-error {{argument to '__builtin_ia32_vcvtps2ph' must be a constant integer}}
}

__m128i check__mm_cvtps_ph(__m128 val, const int I) {
  return _mm_cvtps_ph(val, I); // expected-error {{argument to '__builtin_ia32_vcvtps2ph' must be a constant integer}}
}

__m128i check__mm256_cvtps_ph(__m256 val, const int I) {
  return _mm256_cvtps_ph(val, I); // expected-error  {{argument to '__builtin_ia32_vcvtps2ph256' must be a constant integer}}
}

void check__mm_slli_si128(__m128i a, const int count) {
  _mm_slli_si128(a, count); // expected-error {{argument to '__builtin_ia32_pslldqi128_byteshift' must be a constant integer}}
}

void check__mm_srli_si128(__m128i a, const int count) {
  _mm_srli_si128(a, count); // expected-error {{argument to '__builtin_ia32_psrldqi128_byteshift' must be a constant integer}}
}

void check__mm_shuffle_epi32(__m128i  a, const int imm) {
  _mm_shuffle_epi32(a, imm); // expected-error {{argument to '__builtin_ia32_pshufd' must be a constant integer}}
}

void check__mm_shufflelo_epi16(__m128i a, const int imm) {
  _mm_shufflelo_epi16(a, imm); // expected-error {{argument to '__builtin_ia32_pshuflw' must be a constant integer}}
}

void check__mm_shufflehi_epi16(__m128i a, const int imm) {
  _mm_shufflehi_epi16(a, imm); // expected-error {{argument to '__builtin_ia32_pshufhw' must be a constant integer}}
}

void check__mm_shuffle_pd(__m128d a, __m128d b, const int i) {
  _mm_shuffle_pd(a, b, i); // expected-error {{argument to '__builtin_ia32_shufpd' must be a constant integer}}
}

void check__mm256_round_pd(__m256d a, const int b) {
  _mm256_round_pd(a, b); // expected-error {{argument to '__builtin_ia32_roundpd256' must be a constant integer}}
}

void check__mm256_round_ps(__m256 a, const int b) {
  _mm256_round_ps(a, b); // expected-error {{argument to '__builtin_ia32_roundps256' must be a constant integer}}
}

void check__mm_permute_pd(__m128d a, const int b) {
  _mm_permute_pd(a, b); // expected-error {{argument to '__builtin_ia32_vpermilpd' must be a constant integer}}
}

void check__mm256_permute_pd(__m256d a, const int b) {
  _mm256_permute_pd(a, b); // expected-error {{argument to '__builtin_ia32_vpermilpd256' must be a constant integer}}
}

void check__mm_permute_ps(__m128 a, const int b) {
  _mm_permute_ps(a, b); // expected-error {{argument to '__builtin_ia32_vpermilps' must be a constant integer}}
}

void check__mm256_permute_ps(__m256 a, const int b) {
  _mm256_permute_ps(a, b); // expected-error {{argument to '__builtin_ia32_vpermilps256' must be a constant integer}}
}

void check__mm256_permute2f128_pd(__m256d v1, __m256d v2, const char m) {
  _mm256_permute2f128_pd(v1, v2, m); // expected-error {{argument to '__builtin_ia32_vperm2f128_pd256' must be a constant integer}}
}

void check__mm256_permute2f128_ps(__m256 v1, __m256 v2, const char m) {
  _mm256_permute2f128_ps(v1, v2, m); // expected-error {{argument to '__builtin_ia32_vperm2f128_ps256' must be a constant integer}}
}

void check__mm256_permute2f128_si256(__m256i v1, __m256i v2, const char m) {
  _mm256_permute2f128_si256(v1, v2, m); // expected-error {{argument to '__builtin_ia32_vperm2f128_si256' must be a constant integer}}
}

void check__m256_blend_pd(__m256d v1, __m256d v2, const char m) {
  _mm256_blend_pd(v1, v2, m); // expected-error {{argument to '__builtin_ia32_blendpd256' must be a constant integer}}
}

void check__m256_blend_ps(__m256 v1, __m256 v2, const char m) {
  _mm256_blend_ps(v1, v2, m); // expected-error {{argument to '__builtin_ia32_blendps256' must be a constant integer}}
}

void check__mm256_dp_ps(__m256 v1, __m256 v2, const char m) {
  _mm256_dp_ps(v1, v2, m); // expected-error {{argument to '__builtin_ia32_dpps256' must be a constant integer}}
}

void check__m256_shuffle_ps(__m256 a, __m256 b, const int m) {
  _mm256_shuffle_ps(a, b, m); // expected-error {{argument to '__builtin_ia32_shufps256' must be a constant integer}}
}

void check__m256_shuffle_pd(__m256d a, __m256d b, const int m) {
  _mm256_shuffle_pd(a, b, m); // expected-error {{argument to '__builtin_ia32_shufpd256' must be a constant integer}}
}

void check__mm_cmp_pd(__m128d a, __m128d b, const int c) {
  _mm_cmp_pd(a, b, c); // expected-error {{argument to '__builtin_ia32_cmppd' must be a constant integer}}
}

void check__mm_cmp_ps(__m128 a, __m128 b, const int c) {
  _mm_cmp_ps(a, b, c); // expected-error {{argument to '__builtin_ia32_cmpps' must be a constant integer}}
}

void check__mm256_cmp_pd(__m256d a, __m256d b, const int c) {
  _mm256_cmp_pd(a, b, c); // expected-error {{argument to '__builtin_ia32_cmppd256' must be a constant integer}}
}

void check__mm256_cmp_ps(__m256 a, __m256 b, const int c) {
  _mm256_cmp_ps(a, b, c); // expected-error {{argument to '__builtin_ia32_cmpps256' must be a constant integer}}
}

void check__mm_cmp_sd(__m128d a, __m128d b, const int c) {
  _mm_cmp_sd(a, b, c); // expected-error {{argument to '__builtin_ia32_cmpsd' must be a constant integer}}
}

void check__mm_cmp_ss(__m128 a, __m128 b, const int c) {
  _mm_cmp_ss(a, b, c); // expected-error {{argument to '__builtin_ia32_cmpss' must be a constant integer}}
}

void check__mm256_extractf128_pd(__m256d a, const int o) {
  _mm256_extractf128_pd(a, o); // expected-error {{argument to '__builtin_ia32_vextractf128_pd256' must be a constant integer}}
}

void check__mm256_extractf128_ps(__m256 a, const int o) {
  _mm256_extractf128_ps(a, o); // expected-error {{argument to '__builtin_ia32_vextractf128_ps256' must be a constant integer}}
}

void check__mm256_extractf128_si256(__m256i a, const int o) {
  _mm256_extractf128_si256(a, o); // expected-error {{argument to '__builtin_ia32_vextractf128_si256' must be a constant integer}}
}

void check__mm256_insertf128_pd(__m256d v1, __m128d v2, const int o) {
  _mm256_insertf128_pd(v1, v2, o); // expected-error {{argument to '__builtin_ia32_vinsertf128_pd256' must be a constant integer}}
}

void check__mm256_insertf128_ps(__m256 v1, __m128 v2, const int o) {
  _mm256_insertf128_ps(v1, v2, o); // expected-error {{argument to '__builtin_ia32_vinsertf128_ps256' must be a constant integer}}
}

void check__mm256_insertf128_si256(__m256i v1, __m128i v2, const int o) {
  _mm256_insertf128_si256(v1, v2, o); // expected-error {{argument to '__builtin_ia32_vinsertf128_si256' must be a constant integer}}
}

void check__mm_round_ps(__m128 x, const int m) {
  _mm_round_ps(x, m); // expected-error {{argument to '__builtin_ia32_roundps' must be a constant integer}}
}

void check__mm_round_ss(__m128 x, __m128 y, const int m) {
  _mm_round_ss(x, y, m); // expected-error {{argument to '__builtin_ia32_roundss' must be a constant integer}}
}

void check__mm_round_pd(__m128d x, const int m) {
  _mm_round_pd(x, m); // expected-error {{argument to '__builtin_ia32_roundpd' must be a constant integer}}
}

void check__mm_round_sd(__m128d x, __m128d y, const int m) {
  _mm_round_sd(x, y, m); // expected-error {{argument to '__builtin_ia32_roundsd' must be a constant integer}}
}

void check__mm_blend_pd(__m128d v1, __m128d v2, const int m) {
  _mm_blend_pd(v1, v2, m); // expected-error {{argument to '__builtin_ia32_blendpd' must be a constant integer}}
}

void check__mm_blend_ps(__m128 v1, __m128 v2, const int m) {
  _mm_blend_ps(v1, v2, m); // expected-error {{argument to '__builtin_ia32_blendps' must be a constant integer}}
}

void check__mm_blend_epi16(__m128i v1, __m128i v2, const int m) {
  _mm_blend_epi16(v1, v2, m); // expected-error {{argument to '__builtin_ia32_pblendw128' must be a constant integer}}
}

void check__mm_dp_ps(__m128 x, __m128 y, const int m) {
  _mm_dp_ps(x, y, m); // expected-error {{argument to '__builtin_ia32_dpps' must be a constant integer}}
}

void check__mm_dp_pd(__m128d x, __m128d y, const int m) {
  _mm_dp_pd(x, y, m); // expected-error {{argument to '__builtin_ia32_dppd' must be a constant integer}}
}

void check__mm_insert_ps(__m128 a, __m128 b, const int n) {
  _mm_insert_ps(a, b, n); // expected-error {{argument to '__builtin_ia32_insertps128' must be a constant integer}}
}

void check__mm_mpsadbw_epu8(__m128i x, __m128i y, const int m) {
  _mm_mpsadbw_epu8(x, y, m); // expected-error {{argument to '__builtin_ia32_mpsadbw128' must be a constant integer}}
}

void check__mm_cmpistrm(__m128 a, __m128 b, const int m) {
  _mm_cmpistrm(a, b, m); // expected-error {{argument to '__builtin_ia32_pcmpistrm128' must be a constant integer}}
}

void check__mm_cmpistri(__m128i a, __m128i b, const int m) {
  _mm_cmpistri(a, b, m); // expected-error {{argument to '__builtin_ia32_pcmpistri128' must be a constant integer}}
}

void check__mm_cmpestrm(__m128 a, int b, __m128 c, int d,  const int m) {
  _mm_cmpestrm(a, b, c, d, m); // expected-error {{argument to '__builtin_ia32_pcmpestrm128' must be a constant integer}}
}

void check__mm_cmpestri(__m128i a, int b, __m128i c, int d,  const int m) {
  _mm_cmpestri(a, b, c, d, m); // expected-error {{argument to '__builtin_ia32_pcmpestri128' must be a constant integer}}
}

void check__mm_alignr_epi8(__m128i a, __m128i b, const int n) {
  _mm_alignr_epi8(a, b, n); // expected-error {{argument to '__builtin_ia32_palignr128' must be a constant integer}}
}

void check__mm_alignr_pi8(__m64 a, __m64 b, const int n) {
  _mm_alignr_pi8(a, b, n); // expected-error {{argument to '__builtin_ia32_psrldqi128_byteshift' must be a constant integer}}
}

void check__mm_aeskeygenassist_si128(__m128 c, const int r) {
  _mm_aeskeygenassist_si128(c, r); // expected-error {{argument to '__builtin_ia32_aeskeygenassist128' must be a constant integer}}
}

__m64 check__mm_shuffle_pi16(__m64 a, const int n) {
  return _mm_shuffle_pi16(a, n); // expected-error {{index for __builtin_shufflevector must be a constant integer}}
}

void check__mm_shuffle_ps(__m128 a, __m128 b, const int m) {
  _mm_shuffle_ps(a, b, m); // expected-error {{argument to '__builtin_ia32_shufps' must be a constant integer}}
}

void check__mm_com_epi8(__m128 a, __m128 b, const char c) {
  _mm_com_epi8(a, b, c); // expected-error {{argument to '__builtin_ia32_vpcomb' must be a constant integer}}
}

void check__mm_com_epi16(__m128 a, __m128 b, const char c) {
  _mm_com_epi16(a, b, c); // expected-error {{argument to '__builtin_ia32_vpcomw' must be a constant integer}}
}

void check__mm_com_epi32(__m128 a, __m128 b, const char c) {
  _mm_com_epi32(a, b, c); // expected-error {{argument to '__builtin_ia32_vpcomd' must be a constant integer}}
}

void check__mm_com_epi64(__m128 a, __m128 b, const char c) {
  _mm_com_epi64(a, b, c); // expected-error {{argument to '__builtin_ia32_vpcomq' must be a constant integer}}
}

void check__mm_com_epu8(__m128 a, __m128 b, const char c) {
  _mm_com_epu8(a, b, c); // expected-error {{argument to '__builtin_ia32_vpcomub' must be a constant integer}}
}

void check__mm_com_epu16(__m128 a, __m128 b, const char c) {
  _mm_com_epu16(a, b, c); // expected-error {{argument to '__builtin_ia32_vpcomuw' must be a constant integer}}
}

void check__mm_com_epu32(__m128 a, __m128 b, const char c) {
  _mm_com_epu32(a, b, c); // expected-error {{argument to '__builtin_ia32_vpcomud' must be a constant integer}}
}

void check__mm_com_epu64(__m128 a, __m128 b, const char c) {
  _mm_com_epu64(a, b, c); // expected-error {{argument to '__builtin_ia32_vpcomuq' must be a constant integer}}
}

void check__mm_permute2_pd(__m128d a, __m128d b, __m128d c, const char d) {
  _mm_permute2_pd(a, b, c, d); // expected-error {{argument to '__builtin_ia32_vpermil2pd' must be a constant integer}}
}

void check__mm_permute2_ps(__m128 a, __m128 b, __m128 c, const char d) {
  _mm_permute2_ps(a, b, c, d); // expected-error {{argument to '__builtin_ia32_vpermil2ps' must be a constant integer}}
}

void check__mm256_permute2_pd(__m256d a, __m256d b, __m256d c, const char d) {
  _mm256_permute2_pd(a, b, c, d); // expected-error {{argument to '__builtin_ia32_vpermil2pd256' must be a constant integer}}
}

void check__mm256_permute2_ps(__m256 a, __m256 b, __m256 c, const char d) {
  _mm256_permute2_ps(a, b, c, d); // expected-error {{argument to '__builtin_ia32_vpermil2ps256' must be a constant integer}}
}

void check__mm_roti_epi8(__m128 a, const char b) {
  _mm_roti_epi8(a, b); // expected-error {{argument to '__builtin_ia32_vprotbi' must be a constant integer}}
}

void check__mm_roti_epi16(__m128 a, const char b) {
  _mm_roti_epi16(a, b); // expected-error {{argument to '__builtin_ia32_vprotwi' must be a constant integer}}
}

void check__mm_roti_epi32(__m128 a, const char b) {
  _mm_roti_epi32(a, b); // expected-error {{argument to '__builtin_ia32_vprotdi' must be a constant integer}}
}

void check__mm_roti_epi64(__m128 a, const char b) {
  _mm_roti_epi64(a, b); // expected-error {{argument to '__builtin_ia32_vprotqi' must be a constant integer}}
}

void check__mm256_mpsadbw_epu8(__m256i a, __m256i b, const int c) {
  _mm256_mpsadbw_epu8(a, b, c); // expected-error {{argument to '__builtin_ia32_mpsadbw256' must be a constant integer}}
}

void check__mm256_alignr_epi8(__m256i a, __m256i b, const int n) {
  _mm256_alignr_epi8(a, b, n); // expected-error {{argument to '__builtin_ia32_palignr256' must be a constant integer}}
}

void check__mm256_blend_epi16(__m256i a, __m256i b, const int m) {
  _mm256_blend_epi16(a, b, m); // expected-error {{argument to '__builtin_ia32_pblendw256' must be a constant integer}}
}

void check__mm256_slli_si256(__m256i a, const int count) {
  _mm256_slli_si256(a, count); // expected-error {{argument to '__builtin_ia32_pslldqi256_byteshift' must be a constant integer}}
}

void check__mm256_shuffle_epi32(__m256i  a, const int imm) {
  _mm256_shuffle_epi32(a, imm); // expected-error {{argument to '__builtin_ia32_pshufd256' must be a constant integer}}
}

void check__mm256_shufflehi_epi16(__m256i  a, const int imm) {
  _mm256_shufflehi_epi16(a, imm); // expected-error {{argument to '__builtin_ia32_pshufhw256' must be a constant integer}}
}

void check__mm256_shufflelo_epi16(__m256i  a, const int imm) {
  _mm256_shufflelo_epi16(a, imm); // expected-error {{argument to '__builtin_ia32_pshuflw256' must be a constant integer}}
}

void check__mm_blend_epi32(__m128i a, __m128i b, const int m) {
  _mm_blend_epi32(a, b, m); // expected-error {{argument to '__builtin_ia32_pblendd128' must be a constant integer}}
}

void check__mm256_blend_epi32(__m256i a, __m256i b, const int m) {
  _mm256_blend_epi32(a, b, m); // expected-error {{argument to '__builtin_ia32_pblendd256' must be a constant integer}}
}

void check__mm256_permute4x64_pd(__m256d v, const int m) {
  _mm256_permute4x64_pd(v, m); // expected-error {{argument to '__builtin_ia32_permdf256' must be a constant integer}}
}

void check__mm256_permute4x64_epi64(__m256i v, const int m) {
  _mm256_permute4x64_epi64(v, m); // expected-error {{argument to '__builtin_ia32_permdi256' must be a constant integer}}
}

void check__mm256_permute2x128_si256(__m256i v1, __m256i v2, const int m) {
  _mm256_permute2x128_si256(v1, v2, m); // expected-error {{argument to '__builtin_ia32_permti256' must be a constant integer}}
}

void check__mm256_extracti128_si256(__m256i v1, const int m) {
  _mm256_extracti128_si256(v1, m); // expected-error {{argument to '__builtin_ia32_extract128i256' must be a constant integer}}
}

void check__mm256_inserti128_si256(__m256i v1, __m128i v2, const int m) {
  _mm256_inserti128_si256(v1, v2, m); // expected-error {{argument to '__builtin_ia32_insert128i256' must be a constant integer}}
}

void check__mm256_srli_si256(__m256i a, int count) {
  _mm256_srli_si256(a, count); // expected-error {{argument to '__builtin_ia32_psrldqi256_byteshift' must be a constant integer}}
}

int check__bextri_u32(int a, int b) {
  return __bextri_u32(a, b); // expected-error {{argument to '__builtin_ia32_bextri_u32' must be a constant integer}}
}

int check__bextri_u64(long a, long b) {
  return __bextri_u64(a, b); // expected-error {{argument to '__builtin_ia32_bextri_u64' must be a constant integer}}
}

int check___builtin_eh_return_data_regno(int a) {
  return __builtin_eh_return_data_regno(a); // expected-error {{argument to '__builtin_eh_return_data_regno' must be a constant integer}}
}

void* check___builtin_frame_address(unsigned int a) {
  return __builtin_frame_address(a); // expected-error {{argument to '__builtin_frame_address' must be a constant integer}}
}

void* check___builtin_return_address(unsigned int a) {
  return __builtin_return_address(a); // expected-error {{argument to '__builtin_return_address' must be a constant integer}}
}

void check__mm_extracti_si64(__m128i a, const char len, const char id) {
    _mm_extracti_si64(a, len, id); // expected-error {{argument to '__builtin_ia32_extrqi' must be a constant integer}}
}

void check__insert_si64(__m128 a, __m128 b, const char len, const char id) {
    _mm_inserti_si64(a, b, len, id); // expected-error {{argument to '__builtin_ia32_insertqi' must be a constant integer}}
}

