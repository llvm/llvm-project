// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-unknown -target-feature +avx -target-feature +avx512f -target-feature +avx512vl -verify %s

// HACK: Prevent immintrin.h from pulling in standard library headers
// that don't exist in this test environment.
#define __MM_MALLOC_H

#include <immintrin.h>

namespace ExactFinite {
constexpr __m128d d2 = { -1.0, +2.0 };
constexpr __m128 r128 = _mm_cvtpd_ps(d2);
static_assert(r128[0] == -1.0f && r128[1] == +2.0f, "");
static_assert(r128[2] == 0.0f && r128[3] == 0.0f, "");

constexpr __m128 src128 = { 9.0f, 9.0f, 9.0f, 9.0f };
constexpr __m128 m128_full = _mm_mask_cvtpd_ps(src128, 0x3, d2);
static_assert(m128_full[0] == -1.0f && m128_full[1] == +2.0f, "");
static_assert(m128_full[2] == 9.0f && m128_full[3] == 9.0f, "");

constexpr __m128 m128_partial = _mm_mask_cvtpd_ps(src128, 0x1, d2);
static_assert(m128_partial[0] == -1.0f && m128_partial[1] == 9.0f, "");

constexpr __m128 m128_zero = _mm_maskz_cvtpd_ps(0x1, d2);
static_assert(m128_zero[0] == -1.0f && m128_zero[1] == 0.0f, "");
static_assert(m128_zero[2] == 0.0f && m128_zero[3] == 0.0f, "");

constexpr __m256d d4 = { 0.0, -1.0, +2.0, +3.5 };
constexpr __m128 r256 = _mm256_cvtpd_ps(d4);
static_assert(r256[0] == 0.0f && r256[1] == -1.0f, "");
static_assert(r256[2] == +2.0f && r256[3] == +3.5f, "");

constexpr __m512d d8 = { -1.0, +2.0, +4.0, +8.0, +16.0, +32.0, +64.0, +128.0 };
constexpr __m256 r512 = _mm512_cvtpd_ps(d8);
static_assert(r512[0] == -1.0f && r512[7] == +128.0f, "");

constexpr __m256 src256 = { 9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f };
constexpr __m256 r512_mask = _mm512_mask_cvtpd_ps(src256, 0x05, d8);
static_assert(r512_mask[0] == -1.0f && r512_mask[2] == +4.0f, "");
static_assert(r512_mask[1] == 9.0f && r512_mask[3] == 9.0f, "");

constexpr __m256 r512_maskz = _mm512_maskz_cvtpd_ps(0x81, d8);
static_assert(r512_maskz[0] == -1.0f && r512_maskz[7] == +128.0f, "");
static_assert(r512_maskz[1] == 0.0f && r512_maskz[6] == 0.0f, "");

constexpr __m512 r512lo = _mm512_cvtpd_pslo(d8);
static_assert(r512lo[0] == -1.0f && r512lo[7] == +128.0f, "");
static_assert(r512lo[8] == 0.0f && r512lo[15] == 0.0f, "");

constexpr __m512 ws = (__m512){ 9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,
                                9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f };
constexpr __m512 r512lo_mask = _mm512_mask_cvtpd_pslo(ws, 0x3, d8);
static_assert(r512lo_mask[0] == -1.0f, "");
static_assert(r512lo_mask[1] == +2.0f, "");
static_assert(r512lo_mask[2] == 9.0f && r512lo_mask[3] == 9.0f, "");

constexpr __m128 src_ss = { 9.0f, 5.0f, 6.0f, 7.0f };
constexpr __m128d b_ss = { -1.0, 42.0 };
constexpr __m128 r_ss = _mm_cvtsd_ss(src_ss, b_ss);
static_assert(r_ss[0] == -1.0f, "");
static_assert(r_ss[1] == 5.0f && r_ss[3] == 7.0f, "");

constexpr __m128 r_ss_mask_on = _mm_mask_cvtsd_ss(src_ss, 0x1, src_ss, b_ss);
static_assert(r_ss_mask_on[0] == -1.0f && r_ss_mask_on[1] == 5.0f, "");
constexpr __m128 r_ss_mask_off = _mm_mask_cvtsd_ss(src_ss, 0x0, src_ss, b_ss);
static_assert(r_ss_mask_off[0] == 9.0f, "");
constexpr __m128 r_ss_maskz_off = _mm_maskz_cvtsd_ss(0x0, src_ss, b_ss);
static_assert(r_ss_maskz_off[0] == 0.0f && r_ss_maskz_off[1] == 0.0f, "");
}

namespace InexactOrSpecialReject {
constexpr __m128d inexact = { 1.0000000000000002, 0.0 };
constexpr __m128 r_inexact = _mm_cvtpd_ps(inexact); // both-error {{not an integral constant expression}}
static_assert(r_inexact[0] == 1.0f, "");           // both-note {{subexpression not valid in a constant expression}}

constexpr __m128d dinf = { __builtin_huge_val(), 0.0 };
constexpr __m128 r_inf = _mm_cvtpd_ps(dinf); // both-error {{not an integral constant expression}}
static_assert(r_inf[0] == __builtin_inff(), ""); // both-note {{subexpression not valid in a constant expression}}

constexpr __m128d dnan = { __builtin_nan(""), 0.0 };
constexpr __m128 r_nan = _mm_cvtpd_ps(dnan); // both-error {{not an integral constant expression}}
static_assert(r_nan[0] != r_nan[0], "");  // both-note {{subexpression not valid in a constant expression}}

constexpr __m128d dsub = { 1e-310, 0.0 };
constexpr __m128 r_sub = _mm_cvtpd_ps(dsub); // both-error {{not an integral constant expression}}
static_assert(r_sub[0] == 0.0f, ""); // both-note {{subexpression not valid in a constant expression}}

constexpr __m128 src_ss2 = { 0.0f, 1.0f, 2.0f, 3.0f };
constexpr __m128d inexact_sd = { 1.0000000000000002, 0.0 };
constexpr __m128 r_ss_inexact = _mm_cvtsd_ss(src_ss2, inexact_sd); // both-error {{not an integral constant expression}}
static_assert(r_ss_inexact[0] == 1.0f, ""); // both-note {{subexpression not valid in a constant expression}}
}

namespace MaskedSpecialCasesAllowed {
constexpr __m128 src128a = { 9.0f, 9.0f, 9.0f, 9.0f };
constexpr __m128d d2_inexact = { -1.0, 1.0000000000000002 };
constexpr __m128 ok128 = _mm_mask_cvtpd_ps(src128a, 0x1, d2_inexact);
static_assert(ok128[0] == -1.0f && ok128[1] == 9.0f, "");

constexpr __m128 ok128z = _mm_maskz_cvtpd_ps(0x1, d2_inexact);
static_assert(ok128z[0] == -1.0f && ok128z[1] == 0.0f, "");

constexpr __m256d d4_inexact = { 0.0, 1.0000000000000002, 2.0, 3.0 };
constexpr __m128 src_m = { 9.0f, 9.0f, 9.0f, 9.0f };
constexpr __m128 ok256m = _mm256_mask_cvtpd_ps(src_m, 0b0101, d4_inexact);
static_assert(ok256m[0] == 0.0f && ok256m[1] == 9.0f && ok256m[2] == 2.0f && ok256m[3] == 9.0f, "");

constexpr __m128 ok256z = _mm256_maskz_cvtpd_ps(0b0101, d4_inexact);
static_assert(ok256z[0] == 0.0f && ok256z[1] == 0.0f && ok256z[2] == 2.0f && ok256z[3] == 0.0f, "");

constexpr __m512d d8_inexact = { -1.0, 2.0, 4.0, 8.0, 16.0, 1.0000000000000002, 64.0, 128.0 };
constexpr __m256 src256b = { 9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f };
constexpr __m256 ok512m = _mm512_mask_cvtpd_ps(src256b, 0b110111, d8_inexact);
static_assert(ok512m[0] == -1.0f && ok512m[5] == 9.0f && ok512m[7] == 128.0f, "");

constexpr __m256 ok512z = _mm512_maskz_cvtpd_ps(0b110111, d8_inexact);
static_assert(ok512z[5] == 0.0f && ok512z[0] == -1.0f && ok512z[7] == 128.0f, "");

constexpr __m128 bad128 = _mm_mask_cvtpd_ps(src128a, 0x2, d2_inexact); // both-error {{not an integral constant expression}}
static_assert(bad128[1] == 9.0f, ""); // both-note {{subexpression not valid in a constant expression}}
}
