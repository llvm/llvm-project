// RUN: %clang_cc1 -std=c++20 -ffreestanding -fexperimental-new-constant-interpreter -triple x86_64-unknown-unknown -target-feature +avx -verify %s

#include <immintrin.h>
#include "../CodeGen/X86/builtin_test_helpers.h"

namespace Test_mm256_cvtpd_ps {
namespace OK {
constexpr __m256d a = { 0.0, -1.0, +2.0, +3.5 };
TEST_CONSTEXPR(match_m128(_mm256_cvtpd_ps(a), 0.0f, -1.0f, +2.0f, +3.5f));
}
namespace Inexact {
constexpr __m256d a = { 1.0000000000000002, 0.0, 0.0, 0.0 };
constexpr __m128 r = _mm256_cvtpd_ps(a);
// expected-error@-1 {{must be initialized by a constant expression}}
// expected-note@avxintrin.h:* {{compile time floating point arithmetic suppressed in strict evaluation modes}}
// expected-note@-3 {{in call to '_mm256_cvtpd_ps({1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00})'}}
}
}

namespace Test_mm256_min_ps {
namespace OK {
constexpr __m256 a = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
constexpr __m256 b = { 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f };
TEST_CONSTEXPR(match_m256(_mm256_min_ps(a, b), 1.0f, 2.0f, 3.0f, 4.0f, 4.0f, 3.0f, 2.0f, 1.0f));
}
namespace NaN_A {
constexpr __m256 a = { __builtin_nanf(""), 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
constexpr __m256 b = { 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f };
constexpr __m256 r = _mm256_min_ps(a, b);
// expected-error@-1 {{must be initialized by a constant expression}}
}
namespace Inf_B {
constexpr __m256 a = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
constexpr __m256 b = { 8.0f, -__builtin_huge_valf(), 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f };
constexpr __m256 r = _mm256_min_ps(a, b);
// expected-error@-1 {{must be initialized by a constant expression}}
}
namespace Denormal_A {
constexpr __m256 a = { 1e-40f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
constexpr __m256 b = { 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f };
constexpr __m256 r = _mm256_min_ps(a, b);
// expected-error@-1 {{must be initialized by a constant expression}}
}
}

namespace Test_mm256_max_ps {
namespace OK {
constexpr __m256 a = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
constexpr __m256 b = { 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f };
TEST_CONSTEXPR(match_m256(_mm256_max_ps(a, b), 8.0f, 7.0f, 6.0f, 5.0f, 5.0f, 6.0f, 7.0f, 8.0f));
}
namespace NaN_B {
constexpr __m256 a = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
constexpr __m256 b = { 8.0f, 7.0f, __builtin_nanf(""), 5.0f, 4.0f, 3.0f, 2.0f, 1.0f };
constexpr __m256 r = _mm256_max_ps(a, b);
// expected-error@-1 {{must be initialized by a constant expression}}
}
namespace Inf_A {
constexpr __m256 a = { __builtin_huge_valf(), 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
constexpr __m256 b = { 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f };
constexpr __m256 r = _mm256_max_ps(a, b);
// expected-error@-1 {{must be initialized by a constant expression}}
}
}

namespace Test_mm256_min_pd {
namespace OK {
constexpr __m256d a = { 1.0, 2.0, 3.0, 4.0 };
constexpr __m256d b = { 4.0, 3.0, 2.0, 1.0 };
TEST_CONSTEXPR(match_m256d(_mm256_min_pd(a, b), 1.0, 2.0, 2.0, 1.0));
}
namespace NaN_A {
constexpr __m256d a = { __builtin_nan(""), 2.0, 3.0, 4.0 };
constexpr __m256d b = { 4.0, 3.0, 2.0, 1.0 };
constexpr __m256d r = _mm256_min_pd(a, b);
// expected-error@-1 {{must be initialized by a constant expression}}
}
namespace Inf_B {
constexpr __m256d a = { 1.0, 2.0, 3.0, 4.0 };
constexpr __m256d b = { 4.0, 3.0, -__builtin_huge_val(), 1.0 };
constexpr __m256d r = _mm256_min_pd(a, b);
// expected-error@-1 {{must be initialized by a constant expression}}
}
namespace Denormal_B {
constexpr __m256d a = { 1.0, 2.0, 3.0, 4.0 };
constexpr __m256d b = { 4.0, 3.0, 2.0, -1e-310 };
constexpr __m256d r = _mm256_min_pd(a, b);
// expected-error@-1 {{must be initialized by a constant expression}}
}
}

namespace Test_mm256_max_pd {
namespace OK {
constexpr __m256d a = { 1.0, 2.0, 3.0, 4.0 };
constexpr __m256d b = { 4.0, 3.0, 2.0, 1.0 };
TEST_CONSTEXPR(match_m256d(_mm256_max_pd(a, b), 4.0, 3.0, 3.0, 4.0));
}
namespace NaN_B {
constexpr __m256d a = { 1.0, 2.0, 3.0, 4.0 };
constexpr __m256d b = { 4.0, __builtin_nan(""), 2.0, 1.0 };
constexpr __m256d r = _mm256_max_pd(a, b);
// expected-error@-1 {{must be initialized by a constant expression}}
}
namespace Inf_A {
constexpr __m256d a = { 1.0, __builtin_huge_val(), 3.0, 4.0 };
constexpr __m256d b = { 4.0, 3.0, 2.0, 1.0 };
constexpr __m256d r = _mm256_max_pd(a, b);
// expected-error@-1 {{must be initialized by a constant expression}}
}
}
