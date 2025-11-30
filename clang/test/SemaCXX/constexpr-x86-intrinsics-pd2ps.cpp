// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-unknown -target-feature +avx -target-feature +avx512f -target-feature +avx512vl -verify %s

#define __MM_MALLOC_H 
#include <immintrin.h>

namespace Test_mm_cvtsd_ss {
namespace OK {
constexpr __m128 a = { 9.0f, 5.0f, 6.0f, 7.0f };
constexpr __m128d b = { -1.0, 42.0 };
constexpr __m128 r = _mm_cvtsd_ss(a, b);
static_assert(r[0] == -1.0f && r[1] == 5.0f && r[2] == 6.0f && r[3] == 7.0f, "");
}
namespace Inexact {
constexpr __m128 a = { 0.0f, 1.0f, 2.0f, 3.0f };
constexpr __m128d b = { 1.0000000000000002, 0.0 };
constexpr __m128 r = _mm_cvtsd_ss(a, b);
// expected-error@-1 {{must be initialized by a constant expression}}
// expected-note@emmintrin.h:* {{compile time floating point arithmetic suppressed in strict evaluation modes}}
// expected-note@-3 {{in call to '_mm_cvtsd_ss({0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00}, {1.000000e+00, 0.000000e+00})'}}
}
namespace Inf {
constexpr __m128 a = { 0.0f, 1.0f, 2.0f, 3.0f };
constexpr __m128d b = { __builtin_huge_val(), 0.0 };
constexpr __m128 r = _mm_cvtsd_ss(a, b);
// expected-error@-1 {{must be initialized by a constant expression}}
// expected-note@emmintrin.h:* {{floating point arithmetic produces an infinity}}
// expected-note@-3 {{in call to '_mm_cvtsd_ss({0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00}, {INF, 0.000000e+00})'}}
}
namespace NaN {
constexpr __m128 a = { 0.0f, 1.0f, 2.0f, 3.0f };
constexpr __m128d b = { __builtin_nan(""), 0.0 };
constexpr __m128 r = _mm_cvtsd_ss(a, b);
// expected-error@-1 {{must be initialized by a constant expression}}
// expected-note@emmintrin.h:* {{floating point arithmetic produces a NaN}}
// expected-note@-3 {{in call to '_mm_cvtsd_ss({0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00}, {nan, 0.000000e+00})'}}
}
namespace Subnormal {
constexpr __m128 a = { 0.0f, 1.0f, 2.0f, 3.0f };
constexpr __m128d b = { 1e-310, 0.0 };
constexpr __m128 r = _mm_cvtsd_ss(a, b);
// expected-error@-1 {{must be initialized by a constant expression}}
// expected-note@emmintrin.h:* {{compile time floating point arithmetic suppressed in strict evaluation modes}}
// expected-note@-3 {{in call to '_mm_cvtsd_ss({0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00}, {1.000000e-310, 0.000000e+00})'}}
}
}

namespace Test_mm_mask_cvtsd_ss {
namespace OK {
constexpr __m128 src = { 9.0f, 5.0f, 6.0f, 7.0f };
constexpr __m128 a = { 1.0f, 2.0f, 3.0f, 4.0f };
constexpr __m128d b = { -1.0, 42.0 };
constexpr __m128 r = _mm_mask_cvtsd_ss(src, 0x1, a, b);
static_assert(r[0] == -1.0f && r[1] == 2.0f && r[2] == 3.0f && r[3] == 4.0f, "");
}
namespace MaskOff {
constexpr __m128 src = { 9.0f, 5.0f, 6.0f, 7.0f };
constexpr __m128 a = { 1.0f, 2.0f, 3.0f, 4.0f };
constexpr __m128d b = { -1.0, 42.0 };
constexpr __m128 r = _mm_mask_cvtsd_ss(src, 0x0, a, b);
static_assert(r[0] == 9.0f && r[1] == 2.0f, "");
}
namespace MaskOffInexact {
constexpr __m128 src = { 9.0f, 5.0f, 6.0f, 7.0f };
constexpr __m128 a = { 1.0f, 2.0f, 3.0f, 4.0f };
constexpr __m128d b_inexact = { 1.0000000000000002, 0.0 };
constexpr __m128 r = _mm_mask_cvtsd_ss(src, 0x0, a, b_inexact);
static_assert(r[0] == 9.0f, "");
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
constexpr __m128 r = _mm_maskz_cvtsd_ss(0x1, a, b);
static_assert(r[0] == -1.0f && r[1] == 2.0f, "");
}
namespace MaskOff {
constexpr __m128 a = { 1.0f, 2.0f, 3.0f, 4.0f };
constexpr __m128d b = { -1.0, 42.0 };
constexpr __m128 r = _mm_maskz_cvtsd_ss(0x0, a, b);
static_assert(r[0] == 0.0f && r[1] == 2.0f, "");
}
namespace MaskOffInexact {
constexpr __m128 a = { 1.0f, 2.0f, 3.0f, 4.0f };
constexpr __m128d b_inexact = { 1.0000000000000002, 0.0 };
constexpr __m128 r = _mm_maskz_cvtsd_ss(0x0, a, b_inexact);
static_assert(r[0] == 0.0f, "");
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

namespace Test_mm_cvtpd_ps {
namespace OK {
constexpr __m128d a = { -1.0, +2.0 };
constexpr __m128 r = _mm_cvtpd_ps(a);
static_assert(r[0] == -1.0f && r[1] == +2.0f, "");
static_assert(r[2] == 0.0f && r[3] == 0.0f, "");
}
namespace Inexact {
constexpr __m128d a = { 1.0000000000000002, 0.0 };
constexpr __m128 r = _mm_cvtpd_ps(a);
// expected-error@-1 {{must be initialized by a constant expression}}
// expected-note@emmintrin.h:* {{compile time floating point arithmetic suppressed in strict evaluation modes}}
// expected-note@-3 {{in call to '_mm_cvtpd_ps({1.000000e+00, 0.000000e+00})'}}
}
namespace Inf {
constexpr __m128d a = { __builtin_huge_val(), 0.0 };
constexpr __m128 r = _mm_cvtpd_ps(a);
// expected-error@-1 {{must be initialized by a constant expression}}
// expected-note@emmintrin.h:* {{floating point arithmetic produces an infinity}}
// expected-note@-3 {{in call to '_mm_cvtpd_ps({INF, 0.000000e+00})'}}
}
namespace NaN {
constexpr __m128d a = { __builtin_nan(""), 0.0 };
constexpr __m128 r = _mm_cvtpd_ps(a);
// expected-error@-1 {{must be initialized by a constant expression}}
// expected-note@emmintrin.h:* {{floating point arithmetic produces a NaN}}
// expected-note@-3 {{in call to '_mm_cvtpd_ps({nan, 0.000000e+00})'}}
}
namespace Subnormal {
constexpr __m128d a = { 1e-310, 0.0 };
constexpr __m128 r = _mm_cvtpd_ps(a);
// expected-error@-1 {{must be initialized by a constant expression}}
// expected-note@emmintrin.h:* {{compile time floating point arithmetic suppressed in strict evaluation modes}}
// expected-note@-3 {{in call to '_mm_cvtpd_ps({1.000000e-310, 0.000000e+00})'}}
}
}

namespace Test_mm_mask_cvtpd_ps {
namespace OK {
constexpr __m128 src = { 9.0f, 9.0f, 9.0f, 9.0f };
constexpr __m128d a = { -1.0, +2.0 };
constexpr __m128 r = _mm_mask_cvtpd_ps(src, 0x3, a);
static_assert(r[0] == -1.0f && r[1] == +2.0f, "");
static_assert(r[2] == 9.0f && r[3] == 9.0f, "");
}
namespace Partial {
constexpr __m128 src = { 9.0f, 9.0f, 9.0f, 9.0f };
constexpr __m128d a = { -1.0, +2.0 };
constexpr __m128 r = _mm_mask_cvtpd_ps(src, 0x1, a);
static_assert(r[0] == -1.0f && r[1] == 9.0f, "");
}
namespace MaskOffInexact {
constexpr __m128 src = { 9.0f, 9.0f, 9.0f, 9.0f };
constexpr __m128d a_inexact = { -1.0, 1.0000000000000002 };
constexpr __m128 r = _mm_mask_cvtpd_ps(src, 0x1, a_inexact);
static_assert(r[0] == -1.0f && r[1] == 9.0f, "");
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
constexpr __m128 r = _mm_maskz_cvtpd_ps(0x1, a);
static_assert(r[0] == -1.0f && r[1] == 0.0f, "");
static_assert(r[2] == 0.0f && r[3] == 0.0f, "");
}
namespace MaskOffInexact {
constexpr __m128d a_inexact = { -1.0, 1.0000000000000002 };
constexpr __m128 r = _mm_maskz_cvtpd_ps(0x1, a_inexact);
static_assert(r[0] == -1.0f && r[1] == 0.0f, "");
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

namespace Test_mm256_cvtpd_ps {
namespace OK {
constexpr __m256d a = { 0.0, -1.0, +2.0, +3.5 };
constexpr __m128 r = _mm256_cvtpd_ps(a);
static_assert(r[0] == 0.0f && r[1] == -1.0f, "");
static_assert(r[2] == +2.0f && r[3] == +3.5f, "");
}
namespace Inexact {
constexpr __m256d a = { 1.0000000000000002, 0.0, 0.0, 0.0 };
constexpr __m128 r = _mm256_cvtpd_ps(a);
// expected-error@-1 {{must be initialized by a constant expression}}
// expected-note@avxintrin.h:* {{compile time floating point arithmetic suppressed in strict evaluation modes}}
// expected-note@-3 {{in call to '_mm256_cvtpd_ps({1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00})'}}
}
}

namespace Test_mm256_mask_cvtpd_ps {
namespace OK {
constexpr __m128 src = { 9.0f, 9.0f, 9.0f, 9.0f };
constexpr __m256d a = { 0.0, -1.0, +2.0, +3.5 };
constexpr __m128 r = _mm256_mask_cvtpd_ps(src, 0xF, a);
static_assert(r[0] == 0.0f && r[1] == -1.0f && r[2] == +2.0f && r[3] == +3.5f, "");
}
namespace MaskOffInf {
// Note: 256-bit masked operations use selectps, which evaluates ALL lanes before masking
// So even masked-off Inf/NaN values cause errors (architectural limitation)
constexpr __m128 src = { 9.0f, 9.0f, 9.0f, 9.0f };
constexpr __m256d a_inf = { -1.0, +2.0, __builtin_huge_val(), +8.0 };
constexpr __m128 r = _mm256_mask_cvtpd_ps(src, 0x3, a_inf);
// expected-error@-1 {{must be initialized by a constant expression}}
// expected-note@avxintrin.h:* {{floating point arithmetic produces an infinity}}
// expected-note@avx512vlintrin.h:* {{in call to '_mm256_cvtpd_ps({-1.000000e+00, 2.000000e+00, INF, 8.000000e+00})'}}
// expected-note@-4 {{in call to '_mm256_mask_cvtpd_ps({9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00}, 3, {-1.000000e+00, 2.000000e+00, INF, 8.000000e+00})'}}
}
namespace MaskOffNaN {
// Note: 256-bit masked operations use selectps, which evaluates ALL lanes before masking
// So even masked-off Inf/NaN values cause errors (architectural limitation)
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
constexpr __m128 r = _mm256_maskz_cvtpd_ps(0x5, a);
static_assert(r[0] == 0.0f && r[1] == 0.0f && r[2] == +2.0f && r[3] == 0.0f, "");
}
namespace MaskOffInf {
// Note: 256-bit masked operations use selectps, which evaluates ALL lanes before masking
// So even masked-off Inf/NaN values cause errors (architectural limitation)
constexpr __m256d a_inf = { -1.0, +2.0, __builtin_huge_val(), +8.0 };
constexpr __m128 r = _mm256_maskz_cvtpd_ps(0x3, a_inf);
// expected-error@-1 {{must be initialized by a constant expression}}
// expected-note@avxintrin.h:* {{floating point arithmetic produces an infinity}}
// expected-note@avx512vlintrin.h:* {{in call to '_mm256_cvtpd_ps({-1.000000e+00, 2.000000e+00, INF, 8.000000e+00})'}}
// expected-note@-4 {{in call to '_mm256_maskz_cvtpd_ps(3, {-1.000000e+00, 2.000000e+00, INF, 8.000000e+00})'}}
}
namespace MaskOffNaN {
// Note: 256-bit masked operations use selectps, which evaluates ALL lanes before masking
// So even masked-off Inf/NaN values cause errors (architectural limitation)
constexpr __m256d a_nan = { -1.0, +2.0, +4.0, __builtin_nan("") };
constexpr __m128 r = _mm256_maskz_cvtpd_ps(0x7, a_nan);
// expected-error@-1 {{must be initialized by a constant expression}}
// expected-note@avxintrin.h:* {{floating point arithmetic produces a NaN}}
// expected-note@avx512vlintrin.h:* {{in call to '_mm256_cvtpd_ps({-1.000000e+00, 2.000000e+00, 4.000000e+00, nan})'}}
// expected-note@-4 {{in call to '_mm256_maskz_cvtpd_ps(7, {-1.000000e+00, 2.000000e+00, 4.000000e+00, nan})'}}
}
}

namespace Test_mm512_cvtpd_ps {
namespace OK {
constexpr __m512d a = { -1.0, +2.0, +4.0, +8.0, +16.0, +32.0, +64.0, +128.0 };
constexpr __m256 r = _mm512_cvtpd_ps(a);
static_assert(r[0] == -1.0f && r[7] == +128.0f, "");
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
constexpr __m256 r = _mm512_mask_cvtpd_ps(src, 0x05, a);
static_assert(r[0] == -1.0f && r[2] == +4.0f, "");
static_assert(r[1] == 9.0f && r[3] == 9.0f, "");
}
namespace MaskOffInexact {
constexpr __m256 src = { 9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f };
constexpr __m512d a_inexact = { -1.0, +2.0, +4.0, +8.0, +16.0, 1.0000000000000002, +64.0, +128.0 };
constexpr __m256 r = _mm512_mask_cvtpd_ps(src, 0b11011111, a_inexact);
static_assert(r[0] == -1.0f && r[5] == 9.0f && r[6] == 64.0f && r[7] == 128.0f, "");
}
namespace MaskOffInf {
constexpr __m256 src = { 9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f };
constexpr __m512d a_inf = { -1.0, +2.0, +4.0, +8.0, +16.0, __builtin_huge_val(), +64.0, +128.0 };
constexpr __m256 r = _mm512_mask_cvtpd_ps(src, 0x1F, a_inf);
static_assert(r[0] == -1.0f && r[4] == 16.0f && r[5] == 9.0f, "");
}
namespace MaskOffNaN {
constexpr __m256 src = { 9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f };
constexpr __m512d a_nan = { -1.0, +2.0, +4.0, +8.0, +16.0, __builtin_nan(""), +64.0, +128.0 };
constexpr __m256 r = _mm512_mask_cvtpd_ps(src, 0x1F, a_nan);
static_assert(r[0] == -1.0f && r[4] == 16.0f && r[5] == 9.0f, "");
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
constexpr __m256 r = _mm512_maskz_cvtpd_ps(0x81, a);
static_assert(r[0] == -1.0f && r[7] == +128.0f, "");
static_assert(r[1] == 0.0f && r[6] == 0.0f, "");
}
namespace MaskOffInexact {
constexpr __m512d a_inexact = { -1.0, +2.0, +4.0, +8.0, +16.0, 1.0000000000000002, +64.0, +128.0 };
constexpr __m256 r = _mm512_maskz_cvtpd_ps(0b11011111, a_inexact);
static_assert(r[0] == -1.0f && r[5] == 0.0f && r[6] == 64.0f && r[7] == 128.0f, "");
}
namespace MaskOffInf {
constexpr __m512d a_inf = { -1.0, +2.0, +4.0, +8.0, +16.0, __builtin_huge_val(), +64.0, +128.0 };
constexpr __m256 r = _mm512_maskz_cvtpd_ps(0x1F, a_inf);
static_assert(r[0] == -1.0f && r[4] == 16.0f && r[5] == 0.0f, "");
}
namespace MaskOffNaN {
constexpr __m512d a_nan = { -1.0, +2.0, +4.0, +8.0, +16.0, __builtin_nan(""), +64.0, +128.0 };
constexpr __m256 r = _mm512_maskz_cvtpd_ps(0x1F, a_nan);
static_assert(r[0] == -1.0f && r[4] == 16.0f && r[5] == 0.0f, "");
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
constexpr __m512 r = _mm512_cvtpd_pslo(a);
static_assert(r[0] == -1.0f && r[7] == +128.0f, "");
static_assert(r[8] == 0.0f && r[15] == 0.0f, "");
}
}

namespace Test_mm512_mask_cvtpd_pslo {
namespace OK {
constexpr __m512 src = (__m512){ 9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,
                                9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f };
constexpr __m512d a = { -1.0, +2.0, +4.0, +8.0, +16.0, +32.0, +64.0, +128.0 };
constexpr __m512 r = _mm512_mask_cvtpd_pslo(src, 0x3, a);
static_assert(r[0] == -1.0f && r[1] == +2.0f, "");
static_assert(r[2] == 9.0f && r[3] == 9.0f, "");
}
namespace MaskOffInf {
constexpr __m512 src = (__m512){ 9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,
                                9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f };
constexpr __m512d a_inf = { -1.0, +2.0, __builtin_huge_val(), +8.0, +16.0, +32.0, +64.0, +128.0 };
constexpr __m512 r = _mm512_mask_cvtpd_pslo(src, 0x3, a_inf);
static_assert(r[0] == -1.0f && r[1] == +2.0f && r[2] == 9.0f, "");
}
namespace MaskOffNaN {
constexpr __m512 src = (__m512){ 9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,
                                9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f,9.0f };
constexpr __m512d a_nan = { -1.0, +2.0, +4.0, __builtin_nan(""), +16.0, +32.0, +64.0, +128.0 };
constexpr __m512 r = _mm512_mask_cvtpd_pslo(src, 0x7, a_nan);
static_assert(r[0] == -1.0f && r[1] == +2.0f && r[2] == 4.0f && r[3] == 9.0f, "");
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
