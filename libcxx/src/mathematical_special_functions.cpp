//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__cmath/special_functions.h>
#include <__config>
#include <cerrno>
#include <cfenv>
#include <cmath>
#include <limits>

// GCC defines __STDCPP_FLOATnn_T__ whenever the _Floatnn extended types exist at the
// language level, independent of the standard library. libc++ currently ships no
// <stdfloat>, so std::floatnn_t is never declared -- but Boost.Math keys its
// std::floatnn_t overloads off these macros and would reference the missing types.
// Suppress those overloads while <stdfloat> is unavailable. The __has_include guard is
// the same condition Boost uses to include <stdfloat>, so this workaround disables itself
// automatically once libc++ provides the header (the overloads then light up on their own
// -- no manual re-enable needed here).
#if !__has_include(<stdfloat>)
#  undef __STDCPP_FLOAT16_T__
#  undef __STDCPP_FLOAT32_T__
#  undef __STDCPP_FLOAT64_T__
#  undef __STDCPP_FLOAT128_T__
#  undef __STDCPP_BFLOAT16_T__
#endif

// Boost.Math detects thread support via __has_include(<thread>/<mutex>/...), but libc++
// ships those headers even when threads are disabled (_LIBCPP_HAS_THREADS == 0), so the
// detection wrongly enables std::mutex use and breaks on no-thread targets (e.g. picolibc).
// Tell Boost there are no threads; lazy-init tables don't need locking without threads.
#if !_LIBCPP_HAS_THREADS
#  define BOOST_MATH_DISABLE_THREADS
#endif

#define BOOST_MATH_NO_EXCEPTIONS
#include <boost/math/policies/policy.hpp>
#include <boost/math/special_functions/laguerre.hpp>

_LIBCPP_BEGIN_NAMESPACE_STD
_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS
#if _LIBCPP_STD_VER >= 17

namespace __math {

// Boost.Math computes; the per-function wrappers below add the standard's error rules.
// Notes that apply to all of them:
//  - underflow is left alone: C 7.12.1/5 makes both the ERANGE and the flag
//    implementation-defined there, and Boost's underflow policy defaults to ignore.
//  - C 7.12.1/1 ("as if a single operation") is not implemented: intermediate flags from
//    Boost's recurrences leak into the caller's environment.
//  - promotion: Boost's default promote_float=true computes float inputs in double and
//    rounds once -- more accurate and overflow-resistant, matching the existing
//    std::hermite(float) approach. We keep it.
namespace {
// Error policy for all Boost.Math calls: report domain/pole/overflow/evaluation
// errors via errno (errno_on_error) instead of throwing. Boost sets errno to
// EDOM (domain/pole/evaluation) or ERANGE (overflow) and returns NaN/inf. The
// remaining categories (underflow/denorm/indeterminate) default to ignore.
namespace __bmp = boost::math::policies;
using __policy =
    __bmp::policy<__bmp::domain_error<__bmp::errno_on_error>,
                  __bmp::pole_error<__bmp::errno_on_error>,
                  __bmp::overflow_error<__bmp::errno_on_error>,
                  __bmp::evaluation_error<__bmp::errno_on_error>>;

// Reports a domain error and returns the value libc++ hands back for one.
//
// C 7.12.1/2, imported by [cmath.syn]/1: errno acquires EDOM when math_errhandling &
// MATH_ERRNO, "invalid" is raised when math_errhandling & MATH_ERREXCEPT, and the
// returned value is implementation-defined. Both channels are used unconditionally
// because math_errhandling describes the caller's translation unit, which this
// out-of-line definition cannot see. FE_INVALID is optional -- picolibc without
// hardware floating point does not define it.
template <class _Ret>
_Ret __report_domain_error() {
  errno = EDOM;
#  ifdef FE_INVALID
  std::feraiseexcept(FE_INVALID);
#  endif
  return std::numeric_limits<_Ret>::quiet_NaN();
}

// Reports a range error from overflow and returns __value, which the caller supplies as
// the correctly signed HUGE_VAL (an infinity on IEEE 754 targets), per C 7.12.1/4. No
// Boost policy raises floating-point exceptions, so the flag is ours to raise; Boost's
// own overflow detection sets ERANGE too, and the two channels have to agree. A pole would
// want FE_DIVBYZERO instead, and Boost's policies cannot tell the two apart (both map to
// ERANGE) -- none of the functions implemented so far has one.
template <class _Ret>
_Ret __report_overflow(_Ret __value) {
  errno = ERANGE;
#  ifdef FE_OVERFLOW
  std::feraiseexcept(FE_OVERFLOW);
#  endif
  return __value;
}
} // namespace

// assoc_laguerre
namespace {
template <class _Real>
_Real __assoc_laguerre_impl(unsigned __n, unsigned __m, _Real __x) {
  // [sf.cmath.general]/1: a NaN argument returns a NaN and reports no domain error. It is
  // returned unchanged, so a signaling NaN stays signaling. IEEE 754 would raise "invalid"
  // for it -- and the comparisons below can too, since the relational operators may signal
  // for unordered operands (C 7.12.14) -- which [sf.cmath] neither requires nor forbids
  // for a NaN argument. Quiet NaNs, the case that matters, reach neither.
  if (std::isnan(__x))
    return __x;

  // [sf.cmath.assoc.laguerre] Returns: states the domain as x >= 0, so a negative
  // argument -- -inf included ([sf.cmath.general]/2) -- is a domain error
  // ([sf.cmath.general]/1.1). Boost evaluates such an argument happily, so the check has
  // to live here.
  if (__x < 0)
    return __report_domain_error<_Real>();

  // x == +inf is in the domain. The leading term of L^m_n is (-1)^n x^n / n!, so the value
  // there is 1 for n == 0 and (-1)^n * inf otherwise. That is not a range error: the
  // mathematical result is itself infinite. Answered here because Boost's recurrence is
  // not infinity-safe (it forms inf - inf for n >= 3) and its narrowing cast reports a
  // spurious ERANGE for an infinite value.
  if (std::isinf(__x))
    return __n == 0 ? _Real(1) : (__n % 2 == 0 ? __x : -__x);

  _Real __result = boost::math::laguerre(__n, __m, __x, __policy{});

  // A non-finite result from finite arguments is a range error from overflow (C 7.12.1/4).
  // Boost runs the recurrence in the result type, so a value on the way to L^m_n that does
  // not fit becomes an infinity and, two steps further, inf - inf == NaN. Past its largest
  // root L^m_n has the sign of its leading term, so both cases are overflowed values with
  // a known sign rather than undefined ones.
  if (!std::isfinite(__result)) {
    _Real __inf = std::numeric_limits<_Real>::infinity();
    return __report_overflow(__n % 2 == 0 ? __inf : -__inf);
  }

  return __result;
}
} // namespace

float __assoc_laguerre(unsigned __n, unsigned __m, float __x) noexcept { return __assoc_laguerre_impl(__n, __m, __x); }

double __assoc_laguerre(unsigned __n, unsigned __m, double __x) noexcept {
  return __assoc_laguerre_impl(__n, __m, __x);
}

long double __assoc_laguerre(unsigned __n, unsigned __m, long double __x) noexcept {
  return __assoc_laguerre_impl(__n, __m, __x);
}

} // namespace __math

#endif
_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_END_NAMESPACE_STD
