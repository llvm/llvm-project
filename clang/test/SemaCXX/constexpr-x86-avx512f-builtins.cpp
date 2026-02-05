// RUN: %clang_cc1 -std=c++20 -ffreestanding -fexperimental-new-constant-interpreter -triple x86_64-unknown-unknown -target-feature +avx512f -verify %s

#include <immintrin.h>
#include "../CodeGen/X86/builtin_test_helpers.h"

namespace Test_mm_mask_cvtsd_ss {
namespace OK {
constexpr __m128 src = { 9.0f, 5.0f, 6.0f, 7.0f };
constexpr __m128 a = { 1.0f, 2.0f, 3.0f, 4.0f };
constexpr __m128d b = { -1.0, 42.0 };
TEST_CONSTEXPR(match_m128(_mm_mask_cvtsd_ss(src, 0x1, a, b), -1.0f, 2.0f, 3.0f, 4.0f));
}
namespace MaskOff {
constexpr __m128 src = { 9.0f, 5.0f, 6.0f, 7.0f };
constexpr __m128 a = { 1.0f, 2.0f, 3.0f, 4.0f };
constexpr __m128d b = { -1.0, 42.0 };
TEST_CONSTEXPR(match_m128(_mm_mask_cvtsd_ss(src, 0x0, a, b), 9.0f, 2.0f, 3.0f, 4.0f));
}
namespace MaskOffInexact {
constexpr __m128 src = { 9.0f, 5.0f, 6.0f, 7.0f };
constexpr __m128 a = { 1.0f, 2.0f, 3.0f, 4.0f };
constexpr __m128d b_inexact = { 1.0000000000000002, 0.0 };
constexpr __m128 r = _mm_mask_cvtsd_ss(src, 0x0, a, b_inexact);
TEST_CONSTEXPR(match_m128(r, 9.0f, 2.0f, 3.0f, 4.0f));
}
namespace MaskOnInexact {
constexpr __m128 src = { 9.0f, 5.0f, 6.0f, 7.0f };
constexpr __m128 a = { 1.0f, 2.0f, 3.0f, 4.0f };
constexpr __m128d b_inexact = { 1.0000000000000002, 0.0 };
constexpr __m128 r = _mm_mask_cvtsd_ss(src, 0x1, a, b_inexact);
// expected-error@-1 {{must be initialized by a constant expression}}
// expected-note@avx512fintrin.h:* {{compile time floating point arithmetic suppressed in strict evaluation modes}}
// expected-note@-3 {{in call to '_mm_mask_cvtsd_ss({9.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00}, 1, {1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00}, {1.000000e+00, 0.000000e+00})'}}
}
namespace MaskOnInf {
constexpr __m128 src = { 9.0f, 5.0f, 6.0f, 7.0f };
constexpr __m128 a = { 1.0f, 2.0f, 3.0f, 4.0f };
constexpr __m128d b_inf = { __builtin_huge_val(), 0.0 };
constexpr __m128 r = _mm_mask_cvtsd_ss(src, 0x1, a, b_inf);
// expected-error@-1 {{must be initialized by a constant expression}}
// expected-note@avx512fintrin.h:* {{floating point arithmetic produces an infinity}}
// expected-note@-3 {{in call to '_mm_mask_cvtsd_ss({9.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00}, 1, {1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00}, {INF, 0.000000e+00})'}}
}
namespace MaskOnNaN {
constexpr __m128 src = { 9.0f, 5.0f, 6.0f, 7.0f };
constexpr __m128 a = { 1.0f, 2.0f, 3.0f, 4.0f };
constexpr __m128d b_nan = { __builtin_nan(""), 0.0 };
constexpr __m128 r = _mm_mask_cvtsd_ss(src, 0x1, a, b_nan);
// expected-error@-1 {{must be initialized by a constant expression}}
// expected-note@avx512fintrin.h:* {{floating point arithmetic produces a NaN}}
// expected-note@-3 {{in call to '_mm_mask_cvtsd_ss({9.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00}, 1, {1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00}, {nan, 0.000000e+00})'}}
}
namespace MaskOnSubnormal {
constexpr __m128 src = { 9.0f, 5.0f, 6.0f, 7.0f };
constexpr __m128 a = { 1.0f, 2.0f, 3.0f, 4.0f };
constexpr __m128d b_sub = { 1e-310, 0.0 };
constexpr __m128 r = _mm_mask_cvtsd_ss(src, 0x1, a, b_sub);
// expected-error@-1 {{must be initialized by a constant expression}}
// expected-note@avx512fintrin.h:* {{compile time floating point arithmetic suppressed in strict evaluation modes}}
// expected-note@-3 {{in call to '_mm_mask_cvtsd_ss({9.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00}, 1, {1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00}, {1.000000e-310, 0.000000e+00})'}}
}
}

namespace Test_mm_maskz_cvtsd_ss {
namespace OK {
constexpr __m128 a = { 1.0f, 2.0f, 3.0f, 4.0f };
constexpr __m128d b = { -1.0, 42.0 };
TEST_CONSTEXPR(match_m128(_mm_maskz_cvtsd_ss(0x1, a, b), -1.0f, 2.0f, 3.0f, 4.0f));
}
namespace MaskOff {
constexpr __m128 a = { 1.0f, 2.0f, 3.0f, 4.0f };
constexpr __m128d b = { -1.0, 42.0 };
TEST_CONSTEXPR(match_m128(_mm_maskz_cvtsd_ss(0x0, a, b), 0.0f, 2.0f, 3.0f, 4.0f));
}
namespace MaskOffInexact {
constexpr __m128 a = { 1.0f, 2.0f, 3.0f, 4.0f };
constexpr __m128d b_inexact = { 1.0000000000000002, 0.0 };
TEST_CONSTEXPR(match_m128(_mm_maskz_cvtsd_ss(0x0, a, b_inexact), 0.0f, 2.0f, 3.0f, 4.0f));
}
namespace MaskOnInf {
constexpr __m128 a = { 1.0f, 2.0f, 3.0f, 4.0f };
constexpr __m128d b_inf = { __builtin_huge_val(), 0.0 };
constexpr __m128 r = _mm_maskz_cvtsd_ss(0x1, a, b_inf);
// expected-error@-1 {{must be initialized by a constant expression}}
// expected-note@avx512fintrin.h:* {{floating point arithmetic produces an infinity}}
// expected-note@-3 {{in call to '_mm_maskz_cvtsd_ss(1, {1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00}, {INF, 0.000000e+00})'}}
}
namespace MaskOnNaN {
constexpr __m128 a = { 1.0f, 2.0f, 3.0f, 4.0f };
constexpr __m128d b_nan = { __builtin_nan(""), 0.0 };
constexpr __m128 r = _mm_maskz_cvtsd_ss(0x1, a, b_nan);
// expected-error@-1 {{must be initialized by a constant expression}}
// expected-note@avx512fintrin.h:* {{floating point arithmetic produces a NaN}}
// expected-note@-3 {{in call to '_mm_maskz_cvtsd_ss(1, {1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00}, {nan, 0.000000e+00})'}}
}
}

namespace Test_mm512_cvtpd_ps {
namespace OK {
constexpr __m512d a = { -1.0, +2.0, +4.0, +8.0, +16.0, +32.0, +64.0, +128.0 };
TEST_CONSTEXPR(match_m256(_mm512_cvtpd_ps(a), -1.0f, +2.0f, +4.0f, +8.0f, +16.0f, +32.0f, +64.0f, +128.0f));
}
namespace Inexact {
constexpr __m512d a = { 1.0000000000000002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
constexpr __m256 r = _mm512_cvtpd_ps(a);
// expected-error@-1 {{must be initialized by a constant expression}}
// expected-note@avx512fintrin.h:* {{compile time floating point arithmetic suppressed in strict evaluation modes}}
// expected-note@-3 {{in call to '_mm512_cvtpd_ps({1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00})'}}
}
}

namespace Test_mm512_mask_cvtpd_ps {
namespace OK {
constexpr __m256 src = { 9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f };
constexpr __m512d a = { -1.0, +2.0, +4.0, +8.0, +16.0, +32.0, +64.0, +128.0 };
TEST_CONSTEXPR(match_m256(_mm512_mask_cvtpd_ps(src, 0x05, a), -1.0f, 9.0f, +4.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f));
}
namespace MaskOffInexact {
constexpr __m256 src = { 9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f };
constexpr __m512d a_inexact = { -1.0, +2.0, +4.0, +8.0, +16.0, 1.0000000000000002, +64.0, +128.0 };
TEST_CONSTEXPR(match_m256(_mm512_mask_cvtpd_ps(src, 0b11011111, a_inexact), -1.0f, +2.0f, +4.0f, +8.0f, +16.0f, 9.0f, +64.0f, +128.0f));
}
namespace MaskOffInf {
constexpr __m256 src = { 9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f };
constexpr __m512d a_inf = { -1.0, +2.0, +4.0, +8.0, +16.0, __builtin_huge_val(), +64.0, +128.0 };
TEST_CONSTEXPR(match_m256(_mm512_mask_cvtpd_ps(src, 0x1F, a_inf), -1.0f, +2.0f, +4.0f, +8.0f, +16.0f, 9.0f, 9.0f, 9.0f));
}
namespace MaskOffNaN {
constexpr __m256 src = { 9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f };
constexpr __m512d a_nan = { -1.0, +2.0, +4.0, +8.0, +16.0, __builtin_nan(""), +64.0, +128.0 };
TEST_CONSTEXPR(match_m256(_mm512_mask_cvtpd_ps(src, 0x1F, a_nan), -1.0f, +2.0f, +4.0f, +8.0f, +16.0f, 9.0f, 9.0f, 9.0f));
}
namespace MaskOnInf {
constexpr __m256 src = { 9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f };
constexpr __m512d a_inf = { -1.0, +2.0, +4.0, __builtin_huge_val(), +16.0, +32.0, +64.0, +128.0 };
constexpr __m256 r = _mm512_mask_cvtpd_ps(src, 0x08, a_inf);
// expected-error@-1 {{must be initialized by a constant expression}}
// expected-note@avx512fintrin.h:* {{floating point arithmetic produces an infinity}}
// expected-note@-3 {{in call to '_mm512_mask_cvtpd_ps({9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00}, 8, {-1.000000e+00, 2.000000e+00, 4.000000e+00, INF, 1.600000e+01, 3.200000e+01, 6.400000e+01, 1.280000e+02})'}}
}
namespace MaskOnNaN {
constexpr __m256 src = { 9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f };
constexpr __m512d a_nan = { -1.0, +2.0, +4.0, __builtin_nan(""), +16.0, +32.0, +64.0, +128.0 };
constexpr __m256 r = _mm512_mask_cvtpd_ps(src, 0x08, a_nan);
// expected-error@-1 {{must be initialized by a constant expression}}
// expected-note@avx512fintrin.h:* {{floating point arithmetic produces a NaN}}
// expected-note@-3 {{in call to '_mm512_mask_cvtpd_ps({9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00}, 8, {-1.000000e+00, 2.000000e+00, 4.000000e+00, nan, 1.600000e+01, 3.200000e+01, 6.400000e+01, 1.280000e+02})'}}
}
}

namespace Test_mm512_maskz_cvtpd_ps {
namespace OK {
constexpr __m512d a = { -1.0, +2.0, +4.0, +8.0, +16.0, +32.0, +64.0, +128.0 };
TEST_CONSTEXPR(match_m256(_mm512_maskz_cvtpd_ps(0x81, a), -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, +128.0f));
}
namespace MaskOffInexact {
constexpr __m512d a_inexact = { -1.0, +2.0, +4.0, +8.0, +16.0, 1.0000000000000002, +64.0, +128.0 };
TEST_CONSTEXPR(match_m256(_mm512_maskz_cvtpd_ps(0b11011111, a_inexact), -1.0f, +2.0f, +4.0f, +8.0f, +16.0f, 0.0f, +64.0f, +128.0f));
}
namespace MaskOffInf {
constexpr __m512d a_inf = { -1.0, +2.0, +4.0, +8.0, +16.0, __builtin_huge_val(), +64.0, +128.0 };
TEST_CONSTEXPR(match_m256(_mm512_maskz_cvtpd_ps(0x1F, a_inf), -1.0f, +2.0f, +4.0f, +8.0f, +16.0f, 0.0f, 0.0f, 0.0f));
}
namespace MaskOffNaN {
constexpr __m512d a_nan = { -1.0, +2.0, +4.0, +8.0, +16.0, __builtin_nan(""), +64.0, +128.0 };
TEST_CONSTEXPR(match_m256(_mm512_maskz_cvtpd_ps(0x1F, a_nan), -1.0f, +2.0f, +4.0f, +8.0f, +16.0f, 0.0f, 0.0f, 0.0f));
}
namespace MaskOnInf {
constexpr __m512d a_inf = { -1.0, +2.0, +4.0, __builtin_huge_val(), +16.0, +32.0, +64.0, +128.0 };
constexpr __m256 r = _mm512_maskz_cvtpd_ps(0x08, a_inf);
// expected-error@-1 {{must be initialized by a constant expression}}
// expected-note@avx512fintrin.h:* {{floating point arithmetic produces an infinity}}
// expected-note@-3 {{in call to '_mm512_maskz_cvtpd_ps(8, {-1.000000e+00, 2.000000e+00, 4.000000e+00, INF, 1.600000e+01, 3.200000e+01, 6.400000e+01, 1.280000e+02})'}}
}
namespace MaskOnNaN {
constexpr __m512d a_nan = { -1.0, +2.0, +4.0, __builtin_nan(""), +16.0, +32.0, +64.0, +128.0 };
constexpr __m256 r = _mm512_maskz_cvtpd_ps(0x08, a_nan);
// expected-error@-1 {{must be initialized by a constant expression}}
// expected-note@avx512fintrin.h:* {{floating point arithmetic produces a NaN}}
// expected-note@-3 {{in call to '_mm512_maskz_cvtpd_ps(8, {-1.000000e+00, 2.000000e+00, 4.000000e+00, nan, 1.600000e+01, 3.200000e+01, 6.400000e+01, 1.280000e+02})'}}
}
}

namespace Test_mm512_cvtpd_pslo {
namespace OK {
constexpr __m512d a = { -1.0, +2.0, +4.0, +8.0, +16.0, +32.0, +64.0, +128.0 };
TEST_CONSTEXPR(match_m512(_mm512_cvtpd_pslo(a), -1.0f, +2.0f, +4.0f, +8.0f, +16.0f, +32.0f, +64.0f, +128.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f));
}
}

namespace Test_mm512_mask_cvtpd_pslo {
namespace OK {
constexpr __m512 src = (__m512){ 9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,
                                9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f };
constexpr __m512d a = { -1.0, +2.0, +4.0, +8.0, +16.0, +32.0, +64.0, +128.0 };
TEST_CONSTEXPR(match_m512(_mm512_mask_cvtpd_pslo(src, 0x3, a), -1.0f, +2.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f));
}
namespace MaskOffInf {
constexpr __m512 src = (__m512){ 9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,
                                9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f };
constexpr __m512d a_inf = { -1.0, +2.0, __builtin_huge_val(), +8.0, +16.0, +32.0, +64.0, +128.0 };
TEST_CONSTEXPR(match_m512(_mm512_mask_cvtpd_pslo(src, 0x3, a_inf), -1.0f, +2.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f));
}
namespace MaskOffNaN {
constexpr __m512 src = (__m512){ 9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,
                                9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f };
constexpr __m512d a_nan = { -1.0, +2.0, +4.0, __builtin_nan(""), +16.0, +32.0, +64.0, +128.0 };
TEST_CONSTEXPR(match_m512(_mm512_mask_cvtpd_pslo(src, 0x7, a_nan), -1.0f, +2.0f, +4.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f));
}
namespace MaskOnInf {
constexpr __m512 src = (__m512){ 9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,
                                9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f };
constexpr __m512d a_inf = { -1.0, +2.0, __builtin_huge_val(), +8.0, +16.0, +32.0, +64.0, +128.0 };
constexpr __m512 r = _mm512_mask_cvtpd_pslo(src, 0x4, a_inf);
// expected-error@-1 {{must be initialized by a constant expression}}
// expected-note@avx512fintrin.h:* {{floating point arithmetic produces an infinity}}
// expected-note@avx512fintrin.h:* {{in call to '_mm512_mask_cvtpd_ps({9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00}, 4, {-1.000000e+00, 2.000000e+00, INF, 8.000000e+00, 1.600000e+01, 3.200000e+01, 6.400000e+01, 1.280000e+02})'}}
// expected-note@-4 {{in call to '_mm512_mask_cvtpd_pslo({9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00}, 4, {-1.000000e+00, 2.000000e+00, INF, 8.000000e+00, 1.600000e+01, 3.200000e+01, 6.400000e+01, 1.280000e+02})'}}
}
namespace MaskOnNaN {
constexpr __m512 src = (__m512){ 9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,
                                9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f };
constexpr __m512d a_nan = { -1.0, +2.0, __builtin_nan(""), +8.0, +16.0, +32.0, +64.0, +128.0 };
constexpr __m512 r = _mm512_mask_cvtpd_pslo(src, 0x4, a_nan);
// expected-error@-1 {{must be initialized by a constant expression}}
// expected-note@avx512fintrin.h:* {{floating point arithmetic produces a NaN}}
// expected-note@avx512fintrin.h:* {{in call to '_mm512_mask_cvtpd_ps({9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00}, 4, {-1.000000e+00, 2.000000e+00, nan, 8.000000e+00, 1.600000e+01, 3.200000e+01, 6.400000e+01, 1.280000e+02})'}}
// expected-note@-4 {{in call to '_mm512_mask_cvtpd_pslo({9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00}, 4, {-1.000000e+00, 2.000000e+00, nan, 8.000000e+00, 1.600000e+01, 3.200000e+01, 6.400000e+01, 1.280000e+02})'}}
}
}

constexpr int ROUND_CUR_DIRECTION = 4;
constexpr int ROUND_NO_EXC = 8;

namespace Test_mm_mask_min_ss_valid {
constexpr __m128 result = _mm_mask_min_ss((__m128)(__v4sf){1.0f, 2.0f, 3.0f, 4.0f}, 1, (__m128)(__v4sf){10.0f, 20.0f, 30.0f, 40.0f}, (__m128)(__v4sf){100.0f, 200.0f, 300.0f, 400.0f});
TEST_CONSTEXPR(match_m128(result, 10.0f, 20.0f, 30.0f, 40.0f));
}

namespace Test_mm_mask_max_ss_valid {
constexpr __m128 result = _mm_mask_max_ss((__m128)(__v4sf){1.0f, 2.0f, 3.0f, 4.0f}, 1, (__m128)(__v4sf){10.0f, 20.0f, 30.0f, 40.0f}, (__m128)(__v4sf){100.0f, 200.0f, 300.0f, 400.0f});
TEST_CONSTEXPR(match_m128(result, 100.0f, 20.0f, 30.0f, 40.0f));
}

namespace Test_mm_mask_min_sd_valid {
constexpr __m128d result = _mm_mask_min_sd((__m128d)(__v2df){1.0, 2.0}, 1, (__m128d)(__v2df){10.0, 20.0}, (__m128d)(__v2df){100.0, 200.0});
TEST_CONSTEXPR(match_m128d(result, 10.0, 20.0));
}

namespace Test_mm_mask_max_sd_valid {
constexpr __m128d result = _mm_mask_max_sd((__m128d)(__v2df){1.0, 2.0}, 1, (__m128d)(__v2df){10.0, 20.0}, (__m128d)(__v2df){100.0, 200.0});
TEST_CONSTEXPR(match_m128d(result, 100.0, 20.0));
}

namespace Test_mm_mask_min_ss_nan {
constexpr __m128 a = (__m128)(__v4sf){__builtin_nanf(""), 20.0f, 30.0f, 40.0f};
constexpr __m128 b = (__m128)(__v4sf){100.0f, 200.0f, 300.0f, 400.0f};
constexpr __m128 src = (__m128)(__v4sf){1.0f, 2.0f, 3.0f, 4.0f};
constexpr __m128 result = _mm_mask_min_ss(src, 1, a, b); // expected-error {{must be initialized by a constant expression}}
}

namespace Test_mm_mask_min_ss_pos_inf {
constexpr __m128 a = (__m128)(__v4sf){__builtin_inff(), 20.0f, 30.0f, 40.0f};
constexpr __m128 b = (__m128)(__v4sf){100.0f, 200.0f, 300.0f, 400.0f};
constexpr __m128 src = (__m128)(__v4sf){1.0f, 2.0f, 3.0f, 4.0f};
constexpr __m128 result = _mm_mask_min_ss(src, 1, a, b); // expected-error {{must be initialized by a constant expression}}
}

namespace Test_mm_mask_min_ss_neg_inf {
constexpr __m128 a = (__m128)(__v4sf){-__builtin_inff(), 20.0f, 30.0f, 40.0f};
constexpr __m128 b = (__m128)(__v4sf){100.0f, 200.0f, 300.0f, 400.0f};
constexpr __m128 src = (__m128)(__v4sf){1.0f, 2.0f, 3.0f, 4.0f};
constexpr __m128 result = _mm_mask_min_ss(src, 1, a, b); // expected-error {{must be initialized by a constant expression}}
}
namespace Test_mm_maskz_min_ss_valid {
constexpr __m128 result = _mm_maskz_min_ss(1, (__m128)(__v4sf){10.0f, 20.0f, 30.0f, 40.0f}, (__m128)(__v4sf){100.0f, 200.0f, 300.0f, 400.0f});
TEST_CONSTEXPR(match_m128(result, 10.0f, 20.0f, 30.0f, 40.0f));
}

namespace Test_mm_maskz_max_ss_valid {
constexpr __m128 result = _mm_maskz_max_ss(1, (__m128)(__v4sf){10.0f, 20.0f, 30.0f, 40.0f}, (__m128)(__v4sf){100.0f, 200.0f, 300.0f, 400.0f});
TEST_CONSTEXPR(match_m128(result, 100.0f, 20.0f, 30.0f, 40.0f));
}

namespace Test_mm_maskz_min_sd_valid {
constexpr __m128d result = _mm_maskz_min_sd(1, (__m128d)(__v2df){10.0, 20.0}, (__m128d)(__v2df){100.0, 200.0});
TEST_CONSTEXPR(match_m128d(result, 10.0, 20.0));
}

namespace Test_mm_maskz_max_sd_valid {
constexpr __m128d result = _mm_maskz_max_sd(1, (__m128d)(__v2df){10.0, 20.0}, (__m128d)(__v2df){100.0, 200.0});
TEST_CONSTEXPR(match_m128d(result, 100.0, 20.0));
}

namespace Test_mm_maskz_min_ss_mask_zero {
constexpr __m128 result = _mm_maskz_min_ss(0, (__m128)(__v4sf){10.0f, 20.0f, 30.0f, 40.0f}, (__m128)(__v4sf){100.0f, 200.0f, 300.0f, 400.0f});
TEST_CONSTEXPR(match_m128(result, 0.0f, 20.0f, 30.0f, 40.0f));
}

namespace Test_mm_mask_min_ss_mask_zero {
constexpr __m128 result = _mm_mask_min_ss((__m128)(__v4sf){1.0f, 2.0f, 3.0f, 4.0f}, 0, (__m128)(__v4sf){10.0f, 20.0f, 30.0f, 40.0f}, (__m128)(__v4sf){100.0f, 200.0f, 300.0f, 400.0f});
TEST_CONSTEXPR(match_m128(result, 1.0f, 20.0f, 30.0f, 40.0f));
}

namespace Test_mm_maskz_min_ss_nan {
constexpr __m128 a = (__m128)(__v4sf){__builtin_nanf(""), 20.0f, 30.0f, 40.0f};
constexpr __m128 b = (__m128)(__v4sf){100.0f, 200.0f, 300.0f, 400.0f};
constexpr __m128 result = _mm_maskz_min_ss(1, a, b); // expected-error {{must be initialized by a constant expression}}
}

namespace Test_mm_maskz_max_sd_nan {
constexpr __m128d a = (__m128d)(__v2df){__builtin_nan(""), 20.0};
constexpr __m128d b = (__m128d)(__v2df){100.0, 200.0};
constexpr __m128d result = _mm_maskz_max_sd(1, a, b); // expected-error {{must be initialized by a constant expression}}
}

namespace Test_mm_mask_max_ss_pos_inf {
constexpr __m128 a = (__m128)(__v4sf){__builtin_inff(), 20.0f, 30.0f, 40.0f};
constexpr __m128 b = (__m128)(__v4sf){100.0f, 200.0f, 300.0f, 400.0f};
constexpr __m128 src = (__m128)(__v4sf){1.0f, 2.0f, 3.0f, 4.0f};
constexpr __m128 result = _mm_mask_max_ss(src, 1, a, b); // expected-error {{must be initialized by a constant expression}}
}

namespace Test_mm_mask_max_sd_neg_inf {
constexpr __m128d a = (__m128d)(__v2df){-__builtin_inf(), 20.0};
constexpr __m128d b = (__m128d)(__v2df){100.0, 200.0};
constexpr __m128d src = (__m128d)(__v2df){1.0, 2.0};
constexpr __m128d result = _mm_mask_max_sd(src, 1, a, b); // expected-error {{must be initialized by a constant expression}}
}

namespace Test_mm_minss_round_mask_invalid_rounding_8 {
constexpr __m128 a = (__m128)(__v4sf){10.0f, 20.0f, 30.0f, 40.0f};
constexpr __m128 b = (__m128)(__v4sf){100.0f, 200.0f, 300.0f, 400.0f};
constexpr __m128 src = (__m128)(__v4sf){1.0f, 2.0f, 3.0f, 4.0f};
constexpr __m128 result = (__m128)__builtin_ia32_minss_round_mask((__v4sf)a, (__v4sf)b, (__v4sf)src, 1, ROUND_NO_EXC); // expected-error {{must be initialized by a constant expression}}
}

namespace Test_mm_maxss_round_mask_invalid_rounding_8 {
constexpr __m128 a = (__m128)(__v4sf){10.0f, 20.0f, 30.0f, 40.0f};
constexpr __m128 b = (__m128)(__v4sf){100.0f, 200.0f, 300.0f, 400.0f};
constexpr __m128 src = (__m128)(__v4sf){1.0f, 2.0f, 3.0f, 4.0f};
constexpr __m128 result = (__m128)__builtin_ia32_maxss_round_mask((__v4sf)a, (__v4sf)b, (__v4sf)src, 1, ROUND_NO_EXC); // expected-error {{must be initialized by a constant expression}}
}

namespace Test_mm_maxss_round_mask_invalid_rounding_12 {
constexpr int ROUND_CUR_DIRECTION_NO_EXC = 12;
constexpr __m128 a = (__m128)(__v4sf){10.0f, 20.0f, 30.0f, 40.0f};
constexpr __m128 b = (__m128)(__v4sf){100.0f, 200.0f, 300.0f, 400.0f};
constexpr __m128 src = (__m128)(__v4sf){1.0f, 2.0f, 3.0f, 4.0f};
constexpr __m128 result = (__m128)__builtin_ia32_maxss_round_mask((__v4sf)a, (__v4sf)b, (__v4sf)src, 1, ROUND_CUR_DIRECTION_NO_EXC); // expected-error {{must be initialized by a constant expression}}
}

namespace Test_mm_minss_round_mask_valid_rounding {
constexpr __m128 a = (__m128)(__v4sf){10.0f, 20.0f, 30.0f, 40.0f};
constexpr __m128 b = (__m128)(__v4sf){100.0f, 200.0f, 300.0f, 400.0f};
constexpr __m128 src = (__m128)(__v4sf){1.0f, 2.0f, 3.0f, 4.0f};
constexpr __m128 result = (__m128)__builtin_ia32_minss_round_mask((__v4sf)a, (__v4sf)b, (__v4sf)src, 1, ROUND_CUR_DIRECTION);
static_assert(match_m128(result, 10.0f, 20.0f, 30.0f, 40.0f));
}

namespace Test_mm_maxss_round_mask_valid_rounding {
constexpr __m128 a = (__m128)(__v4sf){10.0f, 20.0f, 30.0f, 40.0f};
constexpr __m128 b = (__m128)(__v4sf){100.0f, 200.0f, 300.0f, 400.0f};
constexpr __m128 src = (__m128)(__v4sf){1.0f, 2.0f, 3.0f, 4.0f};
constexpr __m128 result = (__m128)__builtin_ia32_maxss_round_mask((__v4sf)a, (__v4sf)b, (__v4sf)src, 1, ROUND_CUR_DIRECTION);
static_assert(match_m128(result, 100.0f, 20.0f, 30.0f, 40.0f));
}

namespace Test_mm_minss_round_mask_mask_zero {
constexpr __m128 a = (__m128)(__v4sf){10.0f, 20.0f, 30.0f, 40.0f};
constexpr __m128 b = (__m128)(__v4sf){100.0f, 200.0f, 300.0f, 400.0f};
constexpr __m128 src = (__m128)(__v4sf){1.0f, 2.0f, 3.0f, 4.0f};
constexpr __m128 result = (__m128)__builtin_ia32_minss_round_mask((__v4sf)a, (__v4sf)b, (__v4sf)src, 0, ROUND_CUR_DIRECTION);
static_assert(match_m128(result, 1.0f, 20.0f, 30.0f, 40.0f));
}

namespace Test_mm_maxss_round_mask_mask_zero {
constexpr __m128 a = (__m128)(__v4sf){10.0f, 20.0f, 30.0f, 40.0f};
constexpr __m128 b = (__m128)(__v4sf){100.0f, 200.0f, 300.0f, 400.0f};
constexpr __m128 src = (__m128)(__v4sf){1.0f, 2.0f, 3.0f, 4.0f};
constexpr __m128 result = (__m128)__builtin_ia32_maxss_round_mask((__v4sf)a, (__v4sf)b, (__v4sf)src, 0, ROUND_CUR_DIRECTION);
static_assert(match_m128(result, 1.0f, 20.0f, 30.0f, 40.0f));
}

namespace Test_mm_minss_round_mask_nan {
constexpr __m128 a = (__m128)(__v4sf){__builtin_nanf(""), 20.0f, 30.0f, 40.0f};
constexpr __m128 b = (__m128)(__v4sf){100.0f, 200.0f, 300.0f, 400.0f};
constexpr __m128 src = (__m128)(__v4sf){1.0f, 2.0f, 3.0f, 4.0f};
constexpr __m128 result = (__m128)__builtin_ia32_minss_round_mask((__v4sf)a, (__v4sf)b, (__v4sf)src, 1, ROUND_CUR_DIRECTION); // expected-error {{must be initialized by a constant expression}}
}

namespace Test_mm_maxss_round_mask_nan {
constexpr __m128 a = (__m128)(__v4sf){__builtin_nanf(""), 20.0f, 30.0f, 40.0f};
constexpr __m128 b = (__m128)(__v4sf){100.0f, 200.0f, 300.0f, 400.0f};
constexpr __m128 src = (__m128)(__v4sf){1.0f, 2.0f, 3.0f, 4.0f};
constexpr __m128 result = (__m128)__builtin_ia32_maxss_round_mask((__v4sf)a, (__v4sf)b, (__v4sf)src, 1, ROUND_CUR_DIRECTION); // expected-error {{must be initialized by a constant expression}}
}

namespace Test_mm_minss_round_mask_pos_infinity {
constexpr __m128 a = (__m128)(__v4sf){__builtin_inff(), 20.0f, 30.0f, 40.0f};
constexpr __m128 b = (__m128)(__v4sf){100.0f, 200.0f, 300.0f, 400.0f};
constexpr __m128 src = (__m128)(__v4sf){1.0f, 2.0f, 3.0f, 4.0f};
constexpr __m128 result = (__m128)__builtin_ia32_minss_round_mask((__v4sf)a, (__v4sf)b, (__v4sf)src, 1, ROUND_CUR_DIRECTION); // expected-error {{must be initialized by a constant expression}}
}

namespace Test_mm_maxss_round_mask_pos_infinity {
constexpr __m128 a = (__m128)(__v4sf){__builtin_inff(), 20.0f, 30.0f, 40.0f};
constexpr __m128 b = (__m128)(__v4sf){100.0f, 200.0f, 300.0f, 400.0f};
constexpr __m128 src = (__m128)(__v4sf){1.0f, 2.0f, 3.0f, 4.0f};
constexpr __m128 result = (__m128)__builtin_ia32_maxss_round_mask((__v4sf)a, (__v4sf)b, (__v4sf)src, 1, ROUND_CUR_DIRECTION); // expected-error {{must be initialized by a constant expression}}
}

namespace Test_mm_minss_round_mask_neg_infinity {
constexpr __m128 a = (__m128)(__v4sf){-__builtin_inff(), 20.0f, 30.0f, 40.0f};
constexpr __m128 b = (__m128)(__v4sf){100.0f, 200.0f, 300.0f, 400.0f};
constexpr __m128 src = (__m128)(__v4sf){1.0f, 2.0f, 3.0f, 4.0f};
constexpr __m128 result = (__m128)__builtin_ia32_minss_round_mask((__v4sf)a, (__v4sf)b, (__v4sf)src, 1, ROUND_CUR_DIRECTION); // expected-error {{must be initialized by a constant expression}}
}

namespace Test_mm_maxss_round_mask_neg_infinity {
constexpr __m128 a = (__m128)(__v4sf){-__builtin_inff(), 20.0f, 30.0f, 40.0f};
constexpr __m128 b = (__m128)(__v4sf){100.0f, 200.0f, 300.0f, 400.0f};
constexpr __m128 src = (__m128)(__v4sf){1.0f, 2.0f, 3.0f, 4.0f};
constexpr __m128 result = (__m128)__builtin_ia32_maxss_round_mask((__v4sf)a, (__v4sf)b, (__v4sf)src, 1, ROUND_CUR_DIRECTION); // expected-error {{must be initialized by a constant expression}}
}

namespace Test_mm_minsd_round_mask_invalid_rounding_8 {
constexpr __m128d a = (__m128d)(__v2df){10.0, 20.0};
constexpr __m128d b = (__m128d)(__v2df){100.0, 200.0};
constexpr __m128d src = (__m128d)(__v2df){1.0, 2.0};
constexpr __m128d result = (__m128d)__builtin_ia32_minsd_round_mask((__v2df)a, (__v2df)b, (__v2df)src, 1, ROUND_NO_EXC); // expected-error {{must be initialized by a constant expression}}
}

namespace Test_mm_maxsd_round_mask_invalid_rounding_8 {
constexpr __m128d a = (__m128d)(__v2df){10.0, 20.0};
constexpr __m128d b = (__m128d)(__v2df){100.0, 200.0};
constexpr __m128d src = (__m128d)(__v2df){1.0, 2.0};
constexpr __m128d result = (__m128d)__builtin_ia32_maxsd_round_mask((__v2df)a, (__v2df)b, (__v2df)src, 1, ROUND_NO_EXC); // expected-error {{must be initialized by a constant expression}}
}

namespace Test_mm_maxsd_round_mask_invalid_rounding_12 {
constexpr int ROUND_CUR_DIRECTION_NO_EXC = 12;
constexpr __m128d a = (__m128d)(__v2df){10.0, 20.0};
constexpr __m128d b = (__m128d)(__v2df){100.0, 200.0};
constexpr __m128d src = (__m128d)(__v2df){1.0, 2.0};
constexpr __m128d result = (__m128d)__builtin_ia32_maxsd_round_mask((__v2df)a, (__v2df)b, (__v2df)src, 1, ROUND_CUR_DIRECTION_NO_EXC); // expected-error {{must be initialized by a constant expression}}
}

namespace Test_mm_minsd_round_mask_valid_rounding {
constexpr __m128d a = (__m128d)(__v2df){10.0, 20.0};
constexpr __m128d b = (__m128d)(__v2df){100.0, 200.0};
constexpr __m128d src = (__m128d)(__v2df){1.0, 2.0};
constexpr __m128d result = (__m128d)__builtin_ia32_minsd_round_mask((__v2df)a, (__v2df)b, (__v2df)src, 1, ROUND_CUR_DIRECTION);
static_assert(match_m128d(result, 10.0, 20.0));
}

namespace Test_mm_maxsd_round_mask_valid_rounding {
constexpr __m128d a = (__m128d)(__v2df){10.0, 20.0};
constexpr __m128d b = (__m128d)(__v2df){100.0, 200.0};
constexpr __m128d src = (__m128d)(__v2df){1.0, 2.0};
constexpr __m128d result = (__m128d)__builtin_ia32_maxsd_round_mask((__v2df)a, (__v2df)b, (__v2df)src, 1, ROUND_CUR_DIRECTION);
static_assert(match_m128d(result, 100.0, 20.0));
}

namespace Test_mm_minsd_round_mask_mask_zero {
constexpr __m128d a = (__m128d)(__v2df){10.0, 20.0};
constexpr __m128d b = (__m128d)(__v2df){100.0, 200.0};
constexpr __m128d src = (__m128d)(__v2df){1.0, 2.0};
constexpr __m128d result = (__m128d)__builtin_ia32_minsd_round_mask((__v2df)a, (__v2df)b, (__v2df)src, 0, ROUND_CUR_DIRECTION);
static_assert(match_m128d(result, 1.0, 20.0));
}

namespace Test_mm_maxsd_round_mask_mask_zero {
constexpr __m128d a = (__m128d)(__v2df){10.0, 20.0};
constexpr __m128d b = (__m128d)(__v2df){100.0, 200.0};
constexpr __m128d src = (__m128d)(__v2df){1.0, 2.0};
constexpr __m128d result = (__m128d)__builtin_ia32_maxsd_round_mask((__v2df)a, (__v2df)b, (__v2df)src, 0, ROUND_CUR_DIRECTION);
static_assert(match_m128d(result, 1.0, 20.0));
}

namespace Test_mm_minsd_round_mask_nan {
constexpr __m128d a = (__m128d)(__v2df){__builtin_nan(""), 20.0};
constexpr __m128d b = (__m128d)(__v2df){100.0, 200.0};
constexpr __m128d src = (__m128d)(__v2df){1.0, 2.0};
constexpr __m128d result = (__m128d)__builtin_ia32_minsd_round_mask((__v2df)a, (__v2df)b, (__v2df)src, 1, ROUND_CUR_DIRECTION); // expected-error {{must be initialized by a constant expression}}
}

namespace Test_mm_maxsd_round_mask_nan {
constexpr __m128d a = (__m128d)(__v2df){__builtin_nan(""), 20.0};
constexpr __m128d b = (__m128d)(__v2df){100.0, 200.0};
constexpr __m128d src = (__m128d)(__v2df){1.0, 2.0};
constexpr __m128d result = (__m128d)__builtin_ia32_maxsd_round_mask((__v2df)a, (__v2df)b, (__v2df)src, 1, ROUND_CUR_DIRECTION); // expected-error {{must be initialized by a constant expression}}
}

namespace Test_mm_minsd_round_mask_pos_infinity {
constexpr __m128d a = (__m128d)(__v2df){__builtin_inf(), 20.0};
constexpr __m128d b = (__m128d)(__v2df){100.0, 200.0};
constexpr __m128d src = (__m128d)(__v2df){1.0, 2.0};
constexpr __m128d result = (__m128d)__builtin_ia32_minsd_round_mask((__v2df)a, (__v2df)b, (__v2df)src, 1, ROUND_CUR_DIRECTION); // expected-error {{must be initialized by a constant expression}}
}

namespace Test_mm_maxsd_round_mask_pos_infinity {
constexpr __m128d a = (__m128d)(__v2df){__builtin_inf(), 20.0};
constexpr __m128d b = (__m128d)(__v2df){100.0, 200.0};
constexpr __m128d src = (__m128d)(__v2df){1.0, 2.0};
constexpr __m128d result = (__m128d)__builtin_ia32_maxsd_round_mask((__v2df)a, (__v2df)b, (__v2df)src, 1, ROUND_CUR_DIRECTION); // expected-error {{must be initialized by a constant expression}}
}

namespace Test_mm_minsd_round_mask_neg_infinity {
constexpr __m128d a = (__m128d)(__v2df){-__builtin_inf(), 20.0};
constexpr __m128d b = (__m128d)(__v2df){100.0, 200.0};
constexpr __m128d src = (__m128d)(__v2df){1.0, 2.0};
constexpr __m128d result = (__m128d)__builtin_ia32_minsd_round_mask((__v2df)a, (__v2df)b, (__v2df)src, 1, ROUND_CUR_DIRECTION); // expected-error {{must be initialized by a constant expression}}
}

namespace Test_mm_maxsd_round_mask_neg_infinity {
constexpr __m128d a = (__m128d)(__v2df){-__builtin_inf(), 20.0};
constexpr __m128d b = (__m128d)(__v2df){100.0, 200.0};
constexpr __m128d src = (__m128d)(__v2df){1.0, 2.0};
constexpr __m128d result = (__m128d)__builtin_ia32_maxsd_round_mask((__v2df)a, (__v2df)b, (__v2df)src, 1, ROUND_CUR_DIRECTION); // expected-error {{must be initialized by a constant expression}}
}

namespace Test_mm_mask_min_round_ss_valid {
constexpr __m128 a = (__m128)(__v4sf){10.0f, 20.0f, 30.0f, 40.0f};
constexpr __m128 b = (__m128)(__v4sf){100.0f, 200.0f, 300.0f, 400.0f};
constexpr __m128 src = (__m128)(__v4sf){1.0f, 2.0f, 3.0f, 4.0f};
constexpr __m128 result = _mm_mask_min_round_ss(src, 1, a, b, ROUND_CUR_DIRECTION);
static_assert(match_m128(result, 10.0f, 20.0f, 30.0f, 40.0f));
}

namespace Test_mm_maskz_min_round_ss_valid {
constexpr __m128 a = (__m128)(__v4sf){10.0f, 20.0f, 30.0f, 40.0f};
constexpr __m128 b = (__m128)(__v4sf){100.0f, 200.0f, 300.0f, 400.0f};
constexpr __m128 result = _mm_maskz_min_round_ss(1, a, b, ROUND_CUR_DIRECTION);
static_assert(match_m128(result, 10.0f, 20.0f, 30.0f, 40.0f));
}

namespace Test_mm_mask_max_round_ss_valid {
constexpr __m128 a = (__m128)(__v4sf){10.0f, 20.0f, 30.0f, 40.0f};
constexpr __m128 b = (__m128)(__v4sf){100.0f, 200.0f, 300.0f, 400.0f};
constexpr __m128 src = (__m128)(__v4sf){1.0f, 2.0f, 3.0f, 4.0f};
constexpr __m128 result = _mm_mask_max_round_ss(src, 1, a, b, ROUND_CUR_DIRECTION);
static_assert(match_m128(result, 100.0f, 20.0f, 30.0f, 40.0f));
}

namespace Test_mm_maskz_max_round_ss_valid {
constexpr __m128 a = (__m128)(__v4sf){10.0f, 20.0f, 30.0f, 40.0f};
constexpr __m128 b = (__m128)(__v4sf){100.0f, 200.0f, 300.0f, 400.0f};
constexpr __m128 result = _mm_maskz_max_round_ss(1, a, b, ROUND_CUR_DIRECTION);
static_assert(match_m128(result, 100.0f, 20.0f, 30.0f, 40.0f));
}

namespace Test_mm_mask_min_round_sd_valid {
constexpr __m128d a = (__m128d)(__v2df){10.0, 20.0};
constexpr __m128d b = (__m128d)(__v2df){100.0, 200.0};
constexpr __m128d src = (__m128d)(__v2df){1.0, 2.0};
constexpr __m128d result = _mm_mask_min_round_sd(src, 1, a, b, ROUND_CUR_DIRECTION);
static_assert(match_m128d(result, 10.0, 20.0));
}

namespace Test_mm_maskz_min_round_sd_valid {
constexpr __m128d a = (__m128d)(__v2df){10.0, 20.0};
constexpr __m128d b = (__m128d)(__v2df){100.0, 200.0};
constexpr __m128d result = _mm_maskz_min_round_sd(1, a, b, ROUND_CUR_DIRECTION);
static_assert(match_m128d(result, 10.0, 20.0));
}

namespace Test_mm_mask_max_round_sd_valid {
constexpr __m128d a = (__m128d)(__v2df){10.0, 20.0};
constexpr __m128d b = (__m128d)(__v2df){100.0, 200.0};
constexpr __m128d src = (__m128d)(__v2df){1.0, 2.0};
constexpr __m128d result = _mm_mask_max_round_sd(src, 1, a, b, ROUND_CUR_DIRECTION);
static_assert(match_m128d(result, 100.0, 20.0));
}

namespace Test_mm_maskz_max_round_sd_valid {
constexpr __m128d a = (__m128d)(__v2df){10.0, 20.0};
constexpr __m128d b = (__m128d)(__v2df){100.0, 200.0};
constexpr __m128d result = _mm_maskz_max_round_sd(1, a, b, ROUND_CUR_DIRECTION);
static_assert(match_m128d(result, 100.0, 20.0));
}

namespace Test_mm_mask_min_round_ss_mask_zero {
constexpr __m128 a = (__m128)(__v4sf){10.0f, 20.0f, 30.0f, 40.0f};
constexpr __m128 b = (__m128)(__v4sf){100.0f, 200.0f, 300.0f, 400.0f};
constexpr __m128 src = (__m128)(__v4sf){1.0f, 2.0f, 3.0f, 4.0f};
constexpr __m128 result = _mm_mask_min_round_ss(src, 0, a, b, ROUND_CUR_DIRECTION);
static_assert(match_m128(result, 1.0f, 20.0f, 30.0f, 40.0f));
}

namespace Test_mm_maskz_min_round_ss_mask_zero {
constexpr __m128 a = (__m128)(__v4sf){10.0f, 20.0f, 30.0f, 40.0f};
constexpr __m128 b = (__m128)(__v4sf){100.0f, 200.0f, 300.0f, 400.0f};
constexpr __m128 result = _mm_maskz_min_round_ss(0, a, b, ROUND_CUR_DIRECTION);
static_assert(match_m128(result, 0.0f, 20.0f, 30.0f, 40.0f));
}

namespace Test_mm_mask_max_round_ss_mask_zero {
constexpr __m128 a = (__m128)(__v4sf){10.0f, 20.0f, 30.0f, 40.0f};
constexpr __m128 b = (__m128)(__v4sf){100.0f, 200.0f, 300.0f, 400.0f};
constexpr __m128 src = (__m128)(__v4sf){1.0f, 2.0f, 3.0f, 4.0f};
constexpr __m128 result = _mm_mask_max_round_ss(src, 0, a, b, ROUND_CUR_DIRECTION);
static_assert(match_m128(result, 1.0f, 20.0f, 30.0f, 40.0f));
}

namespace Test_mm_maskz_max_round_ss_mask_zero {
constexpr __m128 a = (__m128)(__v4sf){10.0f, 20.0f, 30.0f, 40.0f};
constexpr __m128 b = (__m128)(__v4sf){100.0f, 200.0f, 300.0f, 400.0f};
constexpr __m128 result = _mm_maskz_max_round_ss(0, a, b, ROUND_CUR_DIRECTION);
static_assert(match_m128(result, 0.0f, 20.0f, 30.0f, 40.0f));
}

namespace Test_mm_mask_min_round_sd_mask_zero {
constexpr __m128d a = (__m128d)(__v2df){10.0, 20.0};
constexpr __m128d b = (__m128d)(__v2df){100.0, 200.0};
constexpr __m128d src = (__m128d)(__v2df){1.0, 2.0};
constexpr __m128d result = _mm_mask_min_round_sd(src, 0, a, b, ROUND_CUR_DIRECTION);
static_assert(match_m128d(result, 1.0, 20.0));
}

namespace Test_mm_maskz_min_round_sd_mask_zero {
constexpr __m128d a = (__m128d)(__v2df){10.0, 20.0};
constexpr __m128d b = (__m128d)(__v2df){100.0, 200.0};
constexpr __m128d result = _mm_maskz_min_round_sd(0, a, b, ROUND_CUR_DIRECTION);
static_assert(match_m128d(result, 0.0, 20.0));
}

namespace Test_mm_mask_max_round_sd_mask_zero {
constexpr __m128d a = (__m128d)(__v2df){10.0, 20.0};
constexpr __m128d b = (__m128d)(__v2df){100.0, 200.0};
constexpr __m128d src = (__m128d)(__v2df){1.0, 2.0};
constexpr __m128d result = _mm_mask_max_round_sd(src, 0, a, b, ROUND_CUR_DIRECTION);
static_assert(match_m128d(result, 1.0, 20.0));
}

namespace Test_mm_maskz_max_round_sd_mask_zero {
constexpr __m128d a = (__m128d)(__v2df){10.0, 20.0};
constexpr __m128d b = (__m128d)(__v2df){100.0, 200.0};
constexpr __m128d result = _mm_maskz_max_round_sd(0, a, b, ROUND_CUR_DIRECTION);
static_assert(match_m128d(result, 0.0, 20.0));
}

namespace Test_mm_mask_min_round_ss_invalid_rounding {
constexpr __m128 a = (__m128)(__v4sf){10.0f, 20.0f, 30.0f, 40.0f};
constexpr __m128 b = (__m128)(__v4sf){100.0f, 200.0f, 300.0f, 400.0f};
constexpr __m128 src = (__m128)(__v4sf){1.0f, 2.0f, 3.0f, 4.0f};
constexpr __m128 result = _mm_mask_min_round_ss(src, 1, a, b, ROUND_NO_EXC); // expected-error {{must be initialized by a constant expression}}
}

namespace Test_mm_maskz_max_round_ss_invalid_rounding {
constexpr __m128 a = (__m128)(__v4sf){10.0f, 20.0f, 30.0f, 40.0f};
constexpr __m128 b = (__m128)(__v4sf){100.0f, 200.0f, 300.0f, 400.0f};
constexpr __m128 result = _mm_maskz_max_round_ss(1, a, b, ROUND_NO_EXC); // expected-error {{must be initialized by a constant expression}}
}

namespace Test_mm_mask_min_round_sd_invalid_rounding {
constexpr __m128d a = (__m128d)(__v2df){10.0, 20.0};
constexpr __m128d b = (__m128d)(__v2df){100.0, 200.0};
constexpr __m128d src = (__m128d)(__v2df){1.0, 2.0};
constexpr __m128d result = _mm_mask_min_round_sd(src, 1, a, b, ROUND_NO_EXC); // expected-error {{must be initialized by a constant expression}}
}

namespace Test_mm_maskz_max_round_sd_invalid_rounding {
constexpr __m128d a = (__m128d)(__v2df){10.0, 20.0};
constexpr __m128d b = (__m128d)(__v2df){100.0, 200.0};
constexpr __m128d result = _mm_maskz_max_round_sd(1, a, b, ROUND_NO_EXC); // expected-error {{must be initialized by a constant expression}}
}

namespace Test_mm_mask_min_round_ss_nan {
constexpr __m128 a = (__m128)(__v4sf){__builtin_nanf(""), 20.0f, 30.0f, 40.0f};
constexpr __m128 b = (__m128)(__v4sf){100.0f, 200.0f, 300.0f, 400.0f};
constexpr __m128 src = (__m128)(__v4sf){1.0f, 2.0f, 3.0f, 4.0f};
constexpr __m128 result = _mm_mask_min_round_ss(src, 1, a, b, ROUND_CUR_DIRECTION); // expected-error {{must be initialized by a constant expression}}
}

namespace Test_mm_maskz_max_round_ss_inf {
constexpr __m128 a = (__m128)(__v4sf){__builtin_inff(), 20.0f, 30.0f, 40.0f};
constexpr __m128 b = (__m128)(__v4sf){100.0f, 200.0f, 300.0f, 400.0f};
constexpr __m128 result = _mm_maskz_max_round_ss(1, a, b, ROUND_CUR_DIRECTION); // expected-error {{must be initialized by a constant expression}}
}

namespace Test_mm_mask_max_round_sd_nan {
constexpr __m128d a = (__m128d)(__v2df){__builtin_nan(""), 20.0};
constexpr __m128d b = (__m128d)(__v2df){100.0, 200.0};
constexpr __m128d src = (__m128d)(__v2df){1.0, 2.0};
constexpr __m128d result = _mm_mask_max_round_sd(src, 1, a, b, ROUND_CUR_DIRECTION); // expected-error {{must be initialized by a constant expression}}
}

namespace Test_mm_maskz_min_round_sd_inf {
constexpr __m128d a = (__m128d)(__v2df){-__builtin_inf(), 20.0};
constexpr __m128d b = (__m128d)(__v2df){100.0, 200.0};
constexpr __m128d result = _mm_maskz_min_round_sd(1, a, b, ROUND_CUR_DIRECTION); // expected-error {{must be initialized by a constant expression}}
}

namespace Test_mm_min_ss_valid {
constexpr __m128 a = (__m128)(__v4sf){10.0f, 20.0f, 30.0f, 40.0f};
constexpr __m128 b = (__m128)(__v4sf){100.0f, 200.0f, 300.0f, 400.0f};
constexpr __m128 result = _mm_min_ss(a, b);
static_assert(match_m128(result, 10.0f, 20.0f, 30.0f, 40.0f));
}

namespace Test_mm_max_ss_valid {
constexpr __m128 a = (__m128)(__v4sf){10.0f, 20.0f, 30.0f, 40.0f};
constexpr __m128 b = (__m128)(__v4sf){100.0f, 200.0f, 300.0f, 400.0f};
constexpr __m128 result = _mm_max_ss(a, b);
static_assert(match_m128(result, 100.0f, 20.0f, 30.0f, 40.0f));
}

namespace Test_mm_min_sd_valid {
constexpr __m128d a = (__m128d)(__v2df){10.0, 20.0};
constexpr __m128d b = (__m128d)(__v2df){100.0, 200.0};
constexpr __m128d result = _mm_min_sd(a, b);
static_assert(match_m128d(result, 10.0, 20.0));
}

namespace Test_mm_max_sd_valid {
constexpr __m128d a = (__m128d)(__v2df){10.0, 20.0};
constexpr __m128d b = (__m128d)(__v2df){100.0, 200.0};
constexpr __m128d result = _mm_max_sd(a, b);
static_assert(match_m128d(result, 100.0, 20.0));
}

namespace Test_mm_min_ss_nan {
constexpr __m128 a = (__m128)(__v4sf){__builtin_nanf(""), 20.0f, 30.0f, 40.0f};
constexpr __m128 b = (__m128)(__v4sf){100.0f, 200.0f, 300.0f, 400.0f};
constexpr __m128 result = _mm_min_ss(a, b); // expected-error {{must be initialized by a constant expression}}
}

namespace Test_mm_max_ss_inf {
constexpr __m128 a = (__m128)(__v4sf){__builtin_inff(), 20.0f, 30.0f, 40.0f};
constexpr __m128 b = (__m128)(__v4sf){100.0f, 200.0f, 300.0f, 400.0f};
constexpr __m128 result = _mm_max_ss(a, b); // expected-error {{must be initialized by a constant expression}}
}

namespace Test_mm_min_sd_nan {
constexpr __m128d a = (__m128d)(__v2df){__builtin_nan(""), 20.0};
constexpr __m128d b = (__m128d)(__v2df){100.0, 200.0};
constexpr __m128d result = _mm_min_sd(a, b); // expected-error {{must be initialized by a constant expression}}
}

namespace Test_mm_max_sd_inf {
constexpr __m128d a = (__m128d)(__v2df){-__builtin_inf(), 20.0};
constexpr __m128d b = (__m128d)(__v2df){100.0, 200.0};
constexpr __m128d result = _mm_max_sd(a, b); // expected-error {{must be initialized by a constant expression}}
}
