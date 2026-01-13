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
