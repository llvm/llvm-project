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

namespace Test_mm512_min_ps {
namespace OK {
constexpr __m512 a = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f };
constexpr __m512 b = { 16.0f, 15.0f, 14.0f, 13.0f, 12.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f };
TEST_CONSTEXPR(match_m512(_mm512_min_ps(a, b), 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f));
}
namespace NaN_A {
constexpr __m512 a = { __builtin_nanf(""), 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f };
constexpr __m512 b = { 16.0f, 15.0f, 14.0f, 13.0f, 12.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f };
constexpr __m512 r = _mm512_min_ps(a, b);
// expected-error@-1 {{must be initialized by a constant expression}}
}
namespace Inf_B {
constexpr __m512 a = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f };
constexpr __m512 b = { 16.0f, -__builtin_huge_valf(), 14.0f, 13.0f, 12.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f };
constexpr __m512 r = _mm512_min_ps(a, b);
// expected-error@-1 {{must be initialized by a constant expression}}
}
namespace Denormal_A {
constexpr __m512 a = { 1e-40f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f };
constexpr __m512 b = { 16.0f, 15.0f, 14.0f, 13.0f, 12.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f };
constexpr __m512 r = _mm512_min_ps(a, b);
// expected-error@-1 {{must be initialized by a constant expression}}
}
}

namespace Test_mm512_max_ps {
namespace OK {
constexpr __m512 a = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f };
constexpr __m512 b = { 16.0f, 15.0f, 14.0f, 13.0f, 12.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f };
TEST_CONSTEXPR(match_m512(_mm512_max_ps(a, b), 16.0f, 15.0f, 14.0f, 13.0f, 12.0f, 11.0f, 10.0f, 9.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f));
}
namespace NaN_B {
constexpr __m512 a = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f };
constexpr __m512 b = { 16.0f, 15.0f, __builtin_nanf(""), 13.0f, 12.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f };
constexpr __m512 r = _mm512_max_ps(a, b);
// expected-error@-1 {{must be initialized by a constant expression}}
}
namespace Inf_A {
constexpr __m512 a = { __builtin_huge_valf(), 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f };
constexpr __m512 b = { 16.0f, 15.0f, 14.0f, 13.0f, 12.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f };
constexpr __m512 r = _mm512_max_ps(a, b);
// expected-error@-1 {{must be initialized by a constant expression}}
}
}

namespace Test_mm512_max_round_ps {
namespace InvalidRounding {
constexpr __m512 a = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f };
constexpr __m512 b = { 16.0f, 15.0f, 14.0f, 13.0f, 12.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f };
constexpr __m512 r = _mm512_max_round_ps(a, b, _MM_FROUND_TO_ZERO);
// expected-error@-1 {{invalid rounding argument}}
}
}

namespace Test_mm512_min_pd {
namespace OK {
constexpr __m512d a = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
constexpr __m512d b = { 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0 };
TEST_CONSTEXPR(match_m512d(_mm512_min_pd(a, b), 1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0));
}
namespace NaN_A {
constexpr __m512d a = { __builtin_nan(""), 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
constexpr __m512d b = { 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0 };
constexpr __m512d r = _mm512_min_pd(a, b);
// expected-error@-1 {{must be initialized by a constant expression}}
}
namespace Inf_B {
constexpr __m512d a = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
constexpr __m512d b = { 8.0, 7.0, -__builtin_huge_val(), 5.0, 4.0, 3.0, 2.0, 1.0 };
constexpr __m512d r = _mm512_min_pd(a, b);
// expected-error@-1 {{must be initialized by a constant expression}}
}
namespace Denormal_B {
constexpr __m512d a = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
constexpr __m512d b = { 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, -1e-310 };
constexpr __m512d r = _mm512_min_pd(a, b);
// expected-error@-1 {{must be initialized by a constant expression}}
}
}

namespace Test_mm512_min_round_pd {
namespace InvalidRounding {
constexpr __m512d a = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
constexpr __m512d b = { 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0 };
constexpr __m512d r = _mm512_min_round_pd(a, b, _MM_FROUND_TO_ZERO);
// expected-error@-1 {{invalid rounding argument}}
}
}

namespace Test_mm512_min_round_ps {
namespace InvalidRounding {
constexpr __m512 a = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f };
constexpr __m512 b = { 16.0f, 15.0f, 14.0f, 13.0f, 12.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f };
constexpr __m512 r = _mm512_min_round_ps(a, b, _MM_FROUND_TO_NEG_INF);
// expected-error@-1 {{invalid rounding argument}}
}
}

namespace Test_mm512_max_pd {
namespace OK {
constexpr __m512d a = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
constexpr __m512d b = { 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0 };
TEST_CONSTEXPR(match_m512d(_mm512_max_pd(a, b), 8.0, 7.0, 6.0, 5.0, 5.0, 6.0, 7.0, 8.0));
}
namespace NaN_B {
constexpr __m512d a = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
constexpr __m512d b = { 8.0, __builtin_nan(""), 6.0, 5.0, 4.0, 3.0, 2.0, 1.0 };
constexpr __m512d r = _mm512_max_pd(a, b);
// expected-error@-1 {{must be initialized by a constant expression}}
}
namespace Inf_A {
constexpr __m512d a = { 1.0, __builtin_huge_val(), 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
constexpr __m512d b = { 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0 };
constexpr __m512d r = _mm512_max_pd(a, b);
// expected-error@-1 {{must be initialized by a constant expression}}
}
}

namespace Test_mm512_max_round_pd {
namespace InvalidRounding {
constexpr __m512d a = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
constexpr __m512d b = { 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0 };
constexpr __m512d r = _mm512_max_round_pd(a, b, _MM_FROUND_TO_POS_INF);
// expected-error@-1 {{invalid rounding argument}}
}
}
