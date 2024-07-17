//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_RANGES_FIND_LAST_IF_H
#define _LIBCPP___ALGORITHM_RANGES_FIND_LAST_IF_H

#include <__config>
#include <__functional/identity.h>
#include <__functional/invoke.h>
#include <__functional/ranges_operations.h>
#include <__iterator/concepts.h>
#include <__iterator/next.h>
#include <__iterator/prev.h>
#include <__iterator/projected.h>
#include <__ranges/access.h>
#include <__ranges/concepts.h>
#include <__ranges/subrange.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 23

_LIBCPP_BEGIN_NAMESPACE_STD

namespace ranges {

template <class _Iter, class _Sent, class _Pred, class _Proj>
_LIBCPP_HIDE_FROM_ABI constexpr static subrange<_Iter>
__find_last_if_impl(_Iter __first, _Sent __last, _Pred& __pred, _Proj& __proj) {
  if (__first == __last) {
    return subrange<_Iter>(__first, __first);
  }

  if constexpr (bidirectional_iterator<_Iter>) {
    auto __last_it = ranges::next(__first, __last);
    for (auto __it = ranges::prev(__last_it); __it != __first; --__it) {
      if (std::invoke(__pred, std::invoke(__proj, *__it))) {
        return subrange<_Iter>(std::move(__it), std::move(__last_it));
      }
    }
    if (std::invoke(__pred, std::invoke(__proj, *__first))) {
      return subrange<_Iter>(std::move(__first), std::move(__last_it));
    }
    return subrange<_Iter>(__last_it, __last_it);
  } else {
    bool __found = false;
    _Iter __found_it;
    for (; __first != __last; ++__first) {
      if (std::invoke(__pred, std::invoke(__proj, *__first))) {
        __found    = true;
        __found_it = __first;
      }
    }

    if (__found) {
      return subrange<_Iter>(std::move(__found_it), std::move(__first));
    } else {
      return subrange<_Iter>(__first, __first);
    }
  }
}

namespace __find_last_if {
struct __fn {
  template <forward_iterator _Iter,
            sentinel_for<_Iter> _Sent,
            class _Proj = identity,
            indirect_unary_predicate<projected<_Iter, _Proj>> _Pred>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr static subrange<_Iter>
  operator()(_Iter __first, _Sent __last, _Pred __pred, _Proj __proj = {}) {
    return __find_last_if_impl(std::move(__first), std::move(__last), __pred, __proj);
  }

  template <forward_range _Range,
            class _Proj = identity,
            indirect_unary_predicate<projected<iterator_t<_Range>, _Proj>> _Pred>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr static borrowed_subrange_t<_Range>
  operator()(_Range&& __range, _Pred __pred, _Proj __proj = {}) {
    return __find_last_if_impl(ranges::begin(__range), ranges::end(__range), __pred, __proj);
  }
};
} // namespace __find_last_if

inline namespace __cpo {
inline constexpr auto find_last_if = __find_last_if::__fn{};
} // namespace __cpo
} // namespace ranges

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 23

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ALGORITHM_RANGES_FIND_LAST_IF_H
