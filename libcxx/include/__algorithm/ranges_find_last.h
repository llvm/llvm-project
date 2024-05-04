//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_RANGES_FIND_LAST_H
#define _LIBCPP___ALGORITHM_RANGES_FIND_LAST_H

#include <__algorithm/ranges_find.h>
#include <__config>
#include <__functional/identity.h>
#include <__functional/invoke.h>
#include <__functional/ranges_operations.h>
#include <__iterator/concepts.h>
#include <__iterator/projected.h>
#include <__iterator/reverse_iterator.h>
#include <__ranges/access.h>
#include <__ranges/concepts.h>
#include <__ranges/dangling.h>
#include <__ranges/subrange.h>
#include <__utility/forward.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 23

_LIBCPP_BEGIN_NAMESPACE_STD

namespace ranges {

namespace __find_last {
struct __fn {
  template <forward_iterator _It, sentinel_for<_It> _Sent, typename _Tp, typename _Proj = identity>
    requires indirect_binary_predicate<equal_to, projected<_It, _Proj>, const _Tp*>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr subrange<_It>
  operator()(_It __first, _Sent __last, const _Tp& __value, _Proj __proj = {}) const {
    if constexpr (same_as<_It, _Sent> && bidirectional_iterator<_It>) {
      const auto __found{find(reverse_iterator{__last}, reverse_iterator{__first}, __value, std::move(__proj)).base()};
      if (__found == __first)
        return {__last, __last};
      return {prev(__found), __last};
    } else {
      auto __found{find(__first, __last, __value, __proj)};
      if (__found == __last)
        return {__last, __last};

      for (__first = __found;; __found = __first++)
        if ((__first == find(__first, __last, __value, __proj)) == __last)
          return {__found, __last};
    }
  }

  template <forward_range _Range, typename _Tp, typename _Proj = identity>
    requires indirect_binary_predicate<equal_to, projected<iterator_t<_Range>, _Proj>, const _Tp*>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr borrowed_subrange_t<_Range>
  operator()(_Range&& __r, const _Tp& __value, _Proj __proj = {}) const {
    return this->operator()(begin(__r), end(__r), __value, std::move(__proj));
  }
};

} // namespace __find_last

inline namespace __cpo {
inline constexpr __find_last::__fn find_last{};
} // namespace __cpo
} // namespace ranges

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 23

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ALGORITHM_RANGES_FIND_LAST_H
