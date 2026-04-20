// RUN: %clang_cc1 -std=c++20 -ffreestanding -fexperimental-new-constant-interpreter -triple x86_64-unknown-unknown -target-feature +avx512f -target-feature +avx512vl -verify %s

#include <immintrin.h>
#include "../CodeGen/X86/builtin_test_helpers.h"

namespace Test_mm_mask_cvtpd_ps {
namespace OK {
constexpr __m128 src = { 9.0f, 9.0f, 9.0f, 9.0f };
constexpr __m128d a = { -1.0, +2.0 };
TEST_CONSTEXPR(match_m128(_mm_mask_cvtpd_ps(src, 0x3, a), -1.0f, +2.0f, 9.0f, 9.0f));
}
namespace Partial {
constexpr __m128 src = { 9.0f, 9.0f, 9.0f, 9.0f };
constexpr __m128d a = { -1.0, +2.0 };
TEST_CONSTEXPR(match_m128(_mm_mask_cvtpd_ps(src, 0x1, a), -1.0f, 9.0f, 9.0f, 9.0f));
}
namespace MaskOffInexact {
constexpr __m128 src = { 9.0f, 9.0f, 9.0f, 9.0f };
constexpr __m128d a_inexact = { -1.0, 1.0000000000000002 };
TEST_CONSTEXPR(match_m128(_mm_mask_cvtpd_ps(src, 0x1, a_inexact), -1.0f, 9.0f, 9.0f, 9.0f));
}
namespace MaskOnInexact {
constexpr __m128 src = { 9.0f, 9.0f, 9.0f, 9.0f };
constexpr __m128d a_inexact = { -1.0, 1.0000000000000002 };
constexpr __m128 r = _mm_mask_cvtpd_ps(src, 0x2, a_inexact);
// expected-error@-1 {{must be initialized by a constant expression}}
// expected-note@avx512vlintrin.h:* {{compile time floating point arithmetic suppressed in strict evaluation modes}}
// expected-note@-3 {{in call to '_mm_mask_cvtpd_ps({9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00}, 2, {-1.000000e+00, 1.000000e+00})'}}
}
namespace MaskOnInf {
constexpr __m128 src = { 9.0f, 9.0f, 9.0f, 9.0f };
constexpr __m128d a_inf = { -1.0, __builtin_huge_val() };
constexpr __m128 r = _mm_mask_cvtpd_ps(src, 0x2, a_inf);
// expected-error@-1 {{must be initialized by a constant expression}}
// expected-note@avx512vlintrin.h:* {{floating point arithmetic produces an infinity}}
// expected-note@-3 {{in call to '_mm_mask_cvtpd_ps({9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00}, 2, {-1.000000e+00, INF})'}}
}
namespace MaskOnNaN {
constexpr __m128 src = { 9.0f, 9.0f, 9.0f, 9.0f };
constexpr __m128d a_nan = { -1.0, __builtin_nan("") };
constexpr __m128 r = _mm_mask_cvtpd_ps(src, 0x2, a_nan);
// expected-error@-1 {{must be initialized by a constant expression}}
// expected-note@avx512vlintrin.h:* {{floating point arithmetic produces a NaN}}
// expected-note@-3 {{in call to '_mm_mask_cvtpd_ps({9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00}, 2, {-1.000000e+00, nan})'}}
}
}

namespace Test_mm_maskz_cvtpd_ps {
namespace OK {
constexpr __m128d a = { -1.0, +2.0 };
TEST_CONSTEXPR(match_m128(_mm_maskz_cvtpd_ps(0x1, a), -1.0f, 0.0f, 0.0f, 0.0f));
}
namespace MaskOffInexact {
constexpr __m128d a_inexact = { -1.0, 1.0000000000000002 };
TEST_CONSTEXPR(match_m128(_mm_maskz_cvtpd_ps(0x1, a_inexact), -1.0f, 0.0f, 0.0f, 0.0f));
}
namespace MaskOnInf {
constexpr __m128d a_inf = { -1.0, __builtin_huge_val() };
constexpr __m128 r = _mm_maskz_cvtpd_ps(0x2, a_inf);
// expected-error@-1 {{must be initialized by a constant expression}}
// expected-note@avx512vlintrin.h:* {{floating point arithmetic produces an infinity}}
// expected-note@-3 {{in call to '_mm_maskz_cvtpd_ps(2, {-1.000000e+00, INF})'}}
}
namespace MaskOnNaN {
constexpr __m128d a_nan = { -1.0, __builtin_nan("") };
constexpr __m128 r = _mm_maskz_cvtpd_ps(0x2, a_nan);
// expected-error@-1 {{must be initialized by a constant expression}}
// expected-note@avx512vlintrin.h:* {{floating point arithmetic produces a NaN}}
// expected-note@-3 {{in call to '_mm_maskz_cvtpd_ps(2, {-1.000000e+00, nan})'}}
}
}

namespace Test_mm256_mask_cvtpd_ps {
namespace OK {
constexpr __m128 src = { 9.0f, 9.0f, 9.0f, 9.0f };
constexpr __m256d a = { 0.0, -1.0, +2.0, +3.5 };
TEST_CONSTEXPR(match_m128(_mm256_mask_cvtpd_ps(src, 0xF, a), 0.0f, -1.0f, +2.0f, +3.5f));
}
namespace MaskOffInf {
constexpr __m128 src = { 9.0f, 9.0f, 9.0f, 9.0f };
constexpr __m256d a_inf = { -1.0, +2.0, __builtin_huge_val(), +8.0 };
constexpr __m128 r = _mm256_mask_cvtpd_ps(src, 0x3, a_inf);
// expected-error@-1 {{must be initialized by a constant expression}}
// expected-note@avxintrin.h:* {{floating point arithmetic produces an infinity}}
// expected-note@avx512vlintrin.h:* {{in call to '_mm256_cvtpd_ps({-1.000000e+00, 2.000000e+00, INF, 8.000000e+00})'}}
// expected-note@-4 {{in call to '_mm256_mask_cvtpd_ps({9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00}, 3, {-1.000000e+00, 2.000000e+00, INF, 8.000000e+00})'}}
}
namespace MaskOffNaN {
constexpr __m128 src = { 9.0f, 9.0f, 9.0f, 9.0f };
constexpr __m256d a_nan = { -1.0, +2.0, +4.0, __builtin_nan("") };
constexpr __m128 r = _mm256_mask_cvtpd_ps(src, 0x7, a_nan);
// expected-error@-1 {{must be initialized by a constant expression}}
// expected-note@avxintrin.h:* {{floating point arithmetic produces a NaN}}
// expected-note@avx512vlintrin.h:* {{in call to '_mm256_cvtpd_ps({-1.000000e+00, 2.000000e+00, 4.000000e+00, nan})'}}
// expected-note@-4 {{in call to '_mm256_mask_cvtpd_ps({9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00}, 7, {-1.000000e+00, 2.000000e+00, 4.000000e+00, nan})'}}
}
}

namespace Test_mm256_maskz_cvtpd_ps {
namespace OK {
constexpr __m256d a = { 0.0, -1.0, +2.0, +3.5 };
TEST_CONSTEXPR(match_m128(_mm256_maskz_cvtpd_ps(0x5, a), 0.0f, 0.0f, +2.0f, 0.0f));
}
namespace MaskOffInf {
constexpr __m256d a_inf = { -1.0, +2.0, __builtin_huge_val(), +8.0 };
constexpr __m128 r = _mm256_maskz_cvtpd_ps(0x3, a_inf);
// expected-error@-1 {{must be initialized by a constant expression}}
// expected-note@avxintrin.h:* {{floating point arithmetic produces an infinity}}
// expected-note@avx512vlintrin.h:* {{in call to '_mm256_cvtpd_ps({-1.000000e+00, 2.000000e+00, INF, 8.000000e+00})'}}
// expected-note@-4 {{in call to '_mm256_maskz_cvtpd_ps(3, {-1.000000e+00, 2.000000e+00, INF, 8.000000e+00})'}}
}
namespace MaskOffNaN {
constexpr __m256d a_nan = { -1.0, +2.0, +4.0, __builtin_nan("") };
constexpr __m128 r = _mm256_maskz_cvtpd_ps(0x7, a_nan);
// expected-error@-1 {{must be initialized by a constant expression}}
// expected-note@avxintrin.h:* {{floating point arithmetic produces a NaN}}
// expected-note@avx512vlintrin.h:* {{in call to '_mm256_cvtpd_ps({-1.000000e+00, 2.000000e+00, 4.000000e+00, nan})'}}
// expected-note@-4 {{in call to '_mm256_maskz_cvtpd_ps(7, {-1.000000e+00, 2.000000e+00, 4.000000e+00, nan})'}}
}
}
