//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_RANGES_FIND_LAST_IF_NOT_H
#define _LIBCPP___ALGORITHM_RANGES_FIND_LAST_IF_NOT_H

#include <__algorithm/ranges_find_last_if.h>
#include <__config>
#include <__functional/identity.h>
#include <__functional/invoke.h>
#include <__iterator/concepts.h>
#include <__iterator/projected.h>
#include <__ranges/access.h>
#include <__ranges/concepts.h>
#include <__ranges/dangling.h>
#include <__utility/forward.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER >= 23

_LIBCPP_BEGIN_NAMESPACE_STD

namespace ranges {
namespace __find_last_if_not {
struct __fn {
  static constexpr auto __make_negate_pred = [](auto&& __pred) {
    return [&__pred](auto&& __e) { return !std::invoke(__pred, std::forward<decltype(__e)>(__e)); };
  };

  template <forward_iterator _Ip,
            sentinel_for<_Ip> _Sp,
            class _Proj = identity,
            indirect_unary_predicate<projected<_Ip, _Proj>> _Pred>
  _LIBCPP_NODISCARD_EXT _LIBCPP_HIDE_FROM_ABI constexpr subrange<_Ip>
  operator()(_Ip __first, _Sp __last, _Pred __pred, _Proj __proj = {}) const {
    auto __negate_pred = __make_negate_pred(__pred);
    return ranges::__find_last_if_impl(std::move(__first), std::move(__last), __negate_pred, __proj);
  }

  template <forward_range _Rp,
            class _Proj = identity,
            indirect_unary_predicate<projected<iterator_t<_Rp>, _Proj>> _Pred>
  _LIBCPP_NODISCARD_EXT _LIBCPP_HIDE_FROM_ABI constexpr borrowed_subrange_t<_Rp>
  operator()(_Rp&& __r, _Pred __pred, _Proj __proj = {}) const {
    auto __negate_pred = __make_negate_pred(__pred);
    return ranges::__find_last_if_impl(ranges::begin(__r), ranges::end(__r), __negate_pred, __proj);
  }
};
} // namespace __find_last_if_not

inline namespace __cpo {
inline constexpr auto find_last_if_not = __find_last_if_not::__fn{};
} // namespace __cpo
} // namespace ranges

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 23

#endif // _LIBCPP___ALGORITHM_RANGES_FIND_LAST_IF_NOT_H
