//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__config>
#include <__math/special_functions.h>
#include <cmath>
#include <optional>
#include <type_traits>

#define BOOST_MATH_NO_EXCEPTIONS
#include <boost/math/special_functions.hpp>

_LIBCPP_BEGIN_NAMESPACE_STD
_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS
#if _LIBCPP_STD_VER >= 17

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

template <class _Ret>
std::optional<_Ret> __check_nan() {
  return std::nullopt;
}

template <class _Ret, class _Arg, class... _Args>
std::optional<_Ret> __check_nan(_Arg __arg, _Args... __args) {
  if constexpr (std::is_floating_point_v<_Arg>)
    if (std::isnan(__arg))
      return __arg;
  return __check_nan<_Ret>(__args...);
}

// Shared back-end for the C++17 mathematical special functions ([sf.cmath]).
// Boost.Math is the compute kernel; this wrapper enforces the standard's
// error-reporting rules ([sf.cmath.general]):
//   1. NaN argument -> return NaN, do NOT report a domain error (the
//      __check_nan pre-filter below).
//   2. domain/range error -> reported via errno only: Boost's errno_on_error
//      policy sets errno = EDOM (domain/pole/evaluation) or ERANGE (overflow).
//      The <cfenv> floating-point-exception side of math_errhandling
//      (MATH_ERREXCEPT) is intentionally not mirrored, matching the shipped
//      std::hermite and libstdc++'s special-function implementations.
// Promotion: Boost's default promote_float=true computes float inputs in double
// and rounds once -- more accurate and overflow-resistant, matching the
// existing std::hermite(float) approach. We keep it.
template <class _Func, class... _Args, class _Ret = std::invoke_result_t<_Func, _Args...>>
_Ret __invoke_boost_math(_Func __f, _Args... __args) {
  if (auto __maybe_nan = __check_nan<_Ret>(__args...); __maybe_nan.has_value())
    return *__maybe_nan;

  return __f(__args..., __policy{});
}
} // namespace

float assoc_laguerref(unsigned int __n, unsigned int __m, float __x) noexcept {
  return __invoke_boost_math([](auto... __args) { return boost::math::laguerre(__args...); }, __n, __m, __x);
}

#endif
_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_END_NAMESPACE_STD
