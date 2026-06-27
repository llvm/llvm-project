//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__config>
#include <__math/special_functions.h>
#include <boost/math/special_functions.hpp>
#include <cmath>
#include <limits>
#include <optional>
#include <type_traits>

_LIBCPP_BEGIN_NAMESPACE_STD
_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS
#if _LIBCPP_STD_VER >= 17

namespace __math {
namespace {
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

template <class _Func, class... _Args, class _Ret = std::invoke_result_t<_Func, _Args...>>
__sf_result<_Ret> __invoke_boost_math(_Func __f, _Args... __args) {
  if (auto __maybe_nan = __check_nan<_Ret>(__args...); __maybe_nan.has_value())
    return {.__domain_error = false, .__ret = *__maybe_nan};

  try {
    return {.__domain_error = false, .__ret = __f(__args...)};
  } catch (...) {
    return {.__domain_error = true, .__ret = std::numeric_limits<_Ret>::quiet_NaN()};
  }
}
} // namespace

__sf_result<float> __assoc_laguerre(unsigned int __n, unsigned int __m, float __x) noexcept {
  return __invoke_boost_math([&](auto... __args) { return boost::math::laguerre(__args...); }, __n, __m, __x);
}
} // namespace __math

#endif
_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_END_NAMESPACE_STD
