//===----------------------------------------------------------------------===//
//
// Part of the LLVM __project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_RANGES_EQUAL_RANGE_H
#define _LIBCPP___ALGORITHM_RANGES_EQUAL_RANGE_H

#include <__algorithm/equal_range.h>
#include <__algorithm/make_projected.h>
#include <__config>
#include <__functional/identity.h>
#include <__functional/invoke.h>
#include <__functional/ranges_operations.h>
#include <__iterator/concepts.h>
#include <__iterator/iterator_traits.h>
#include <__iterator/projected.h>
#include <__ranges/access.h>
#include <__ranges/concepts.h>
#include <__ranges/dangling.h>
#include <__ranges/subrange.h>
#include <__utility/forward.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER > 17 && !defined(_LIBCPP_HAS_NO_INCOMPLETE_RANGES)

_LIBCPP_BEGIN_NAMESPACE_STD

namespace ranges {
namespace __equal_range {

struct __fn {

  template <forward_iterator _Iter, sentinel_for<_Iter> _Sent, class _Tp, class _Proj = identity,
            indirect_strict_weak_order<const _Tp*, projected<_Iter, _Proj>> _Comp = ranges::less>
  _LIBCPP_HIDE_FROM_ABI constexpr
  subrange<_Iter> operator()(_Iter __first, _Sent __last, const _Tp& __value, _Comp __comp = {},
                             _Proj __proj = {}) const {
    // TODO: implement
    (void)__first; (void)__last; (void)__value; (void)__comp; (void)__proj;
    return {};
  }

  template <forward_range _Range, class _Tp, class _Proj = identity,
            indirect_strict_weak_order<const _Tp*, projected<iterator_t<_Range>, _Proj>> _Comp = ranges::less>
  _LIBCPP_HIDE_FROM_ABI constexpr
  borrowed_subrange_t<_Range> operator()(_Range&& __range, const _Tp& __value, _Comp __comp = {},
                                         _Proj __proj = {}) const {
    // TODO: implement
    (void)__range; (void)__value; (void)__comp; (void)__proj;
    return {};
  }

};

} // namespace __equal_range

inline namespace __cpo {
  inline constexpr auto equal_range = __equal_range::__fn{};
} // namespace __cpo
} // namespace ranges

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER > 17 && !defined(_LIBCPP_HAS_NO_INCOMPLETE_RANGES)

#endif // _LIBCPP___ALGORITHM_RANGES_EQUAL_RANGE_H
