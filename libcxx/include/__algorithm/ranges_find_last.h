//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_RANGES_FIND_LAST_H
#define _LIBCPP___ALGORITHM_RANGES_FIND_LAST_H

#include <__config>
#include <__functional/identity.h>
#include <__functional/invoke.h>
#include <__functional/ranges_operations.h>
#include <__iterator/concepts.h>
#include <__iterator/projected.h>
#include <__ranges/access.h>
#include <__ranges/concepts.h>
#include <__ranges/dangling.h>
#include <__utility/forward.h>
#include <__utility/move.h>
#include <optional>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 20

_LIBCPP_BEGIN_NAMESPACE_STD

namespace ranges {

namespace __find_last {

struct __fn {
  template <forward_iterator _Ip, sentinel_for<_Ip> _Sp, class _Tp, class _Proj = identity>
    requires indirect_binary_predicate<ranges::equal_to, projected<_Ip, _Proj>, const _Tp*>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr subrange<_Ip>
  operator()(_Ip __first, _Sp __last, const _Tp& __value, _Proj __proj = {}) const {
    std::optional<_Ip> __found;
    for (; __first != __last; ++__first) {
      if (std::invoke(__proj, *__first) == __value) {
        __found = __first;
      }
    }
    if (!__found)
      return {__first, __first};
    return {*__found, std::ranges::next(*__found, __last)};
  }

  template <forward_range _Rp, class _Tp, class _Proj = identity>
    requires indirect_binary_predicate<ranges::equal_to, projected<iterator_t<_Rp>, _Proj>, const _Tp*>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr borrowed_subrange_t<_Rp>
  operator()(_Rp&& __r, const _Tp& __value, _Proj __proj = {}) const {
    return this->operator()(ranges::begin(__r), ranges::end(__r), __value, std::ref(__proj));
  }
};

} // namespace __find_last

inline namespace __cpo {
inline constexpr auto find_last        = __find_last::__fn{};
inline constexpr auto find_last_if     = __find_last::__fn{};
inline constexpr auto find_last_if_not = __find_last::__fn{};
} // namespace __cpo

} // namespace ranges

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 20

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ALGORITHM_RANGES_FIND_LAST_H
