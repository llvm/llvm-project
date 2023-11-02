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
#include <__concepts/assignable.h>
#include <__functional/identity.h>
#include <__functional/invoke.h>
#include <__functional/ranges_operations.h>
#include <__iterator/concepts.h>
#include <__iterator/next.h>
#include <__iterator/projected.h>
#include <__ranges/access.h>
#include <__ranges/concepts.h>
#include <__ranges/dangling.h>
#include <__ranges/range_adaptor.h>
#include <__ranges/subrange.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER >= 23

_LIBCPP_BEGIN_NAMESPACE_STD

namespace ranges {
namespace __find_last {
struct __fn {
  template <forward_iterator _Ip, sentinel_for<_Ip> _Sp, class _Tp, class _Proj = identity>
    requires indirect_binary_predicate<ranges::equal_to, projected<_Ip, _Proj>, const _Tp*>
  _LIBCPP_NODISCARD_EXT _LIBCPP_HIDE_FROM_ABI constexpr subrange<_Ip>
  operator()(_Ip __first, _Sp __last, const _Tp& __value, _Proj __proj = {}) const {
    if constexpr ((bidirectional_range<_Ip> && common_range<_Ip>) ||
                  (bidirectional_iterator<_Ip> && assignable_from<_Ip&, _Sp&>)) {
      // Implement optimized bidirectional range and common range version.
      // Perform a reverse search from the end.
      _Ip __original_last = __last;                        // Save the original value of __last
      _Ip __result        = ranges::next(__first, __last); // Set __result to the end of the range
      while (__first != __last) {
        --__last;
        if (std::invoke(__proj, *__last) == __value) {
          __result = __last;
          break;
        }
      }
      return {__result, __original_last};
    } else {
      _Ip __original_first = __first;
      _Ip __result         = __first;
      while (__first != __last) {
        if (std::invoke(__proj, *__first) == __value) {
          __result = __first;
        }
        ++__first;
      }
      return {__result, __original_first};
    }
  }

  template <forward_range _Rp, class _Tp, class _Proj = identity>
    requires indirect_binary_predicate<ranges::equal_to, projected<iterator_t<_Rp>, _Proj>, const _Tp*>
  _LIBCPP_NODISCARD_EXT _LIBCPP_HIDE_FROM_ABI constexpr borrowed_subrange_t<_Rp>
  operator()(_Rp&& __r, const _Tp& __value, _Proj __proj = {}) const {
    return operator()(ranges::begin(__r), ranges::end(__r), __value, __proj);
  }
};
} // namespace __find_last

inline namespace __cpo {
inline constexpr auto find_last = __find_last::__fn{};
} // namespace __cpo
} // namespace ranges

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 23

#endif // _LIBCPP___ALGORITHM_RANGES_FIND_LAST_H
