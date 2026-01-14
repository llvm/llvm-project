// RUN: %clang_cc1 -std=c++20 -ffreestanding -fexperimental-new-constant-interpreter -triple x86_64-unknown-unknown -target-feature +sse2 -verify %s

#include <immintrin.h>
#include "../CodeGen/X86/builtin_test_helpers.h"

namespace Test_mm_cvtsd_ss {
namespace OK {
constexpr __m128 a = { 9.0f, 5.0f, 6.0f, 7.0f };
constexpr __m128d b = { -1.0, 42.0 };
TEST_CONSTEXPR(match_m128(_mm_cvtsd_ss(a, b), -1.0f, 5.0f, 6.0f, 7.0f));
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

namespace Test_mm_cvtpd_ps {
namespace OK {
constexpr __m128d a = { -1.0, +2.0 };
TEST_CONSTEXPR(match_m128(_mm_cvtpd_ps(a), -1.0f, +2.0f, 0.0f, 0.0f));
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
