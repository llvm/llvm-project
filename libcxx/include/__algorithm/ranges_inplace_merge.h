//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_RANGES_INPLACE_MERGE_H
#define _LIBCPP___ALGORITHM_RANGES_INPLACE_MERGE_H

#include <__algorithm/inplace_merge.h>
#include <__algorithm/make_projected.h>
#include <__config>
#include <__functional/identity.h>
#include <__functional/invoke.h>
#include <__functional/ranges_operations.h>
#include <__iterator/concepts.h>
#include <__iterator/iterator_traits.h>
#include <__iterator/projected.h>
#include <__iterator/sortable.h>
#include <__ranges/access.h>
#include <__ranges/concepts.h>
#include <__ranges/dangling.h>
#include <__utility/forward.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER > 17 && !defined(_LIBCPP_HAS_NO_INCOMPLETE_RANGES)

_LIBCPP_BEGIN_NAMESPACE_STD

namespace ranges {
namespace __inplace_merge {

struct __fn {

  template <bidirectional_iterator _Iter, sentinel_for<_Iter> _Sent, class _Comp = ranges::less, class _Proj = identity>
  requires sortable<_Iter, _Comp, _Proj>
  _LIBCPP_HIDE_FROM_ABI
  _Iter operator()(_Iter __first, _Iter __middle, _Sent __last, _Comp __comp = {}, _Proj __proj = {}) const {
    // TODO: implement
    (void)__first; (void)__middle; (void)__last; (void)__comp; (void)__proj;
    return {};
  }

  template <bidirectional_range _Range, class _Comp = ranges::less, class _Proj = identity>
  requires sortable<iterator_t<_Range>, _Comp, _Proj>
  _LIBCPP_HIDE_FROM_ABI
  borrowed_iterator_t<_Range> operator()(_Range&& __range, iterator_t<_Range> __middle,
                                            _Comp __comp = {}, _Proj __proj = {}) const {
    // TODO: implement
    (void)__range; (void)__middle; (void)__comp; (void)__proj;
    return {};
  }

};

} // namespace __inplace_merge

inline namespace __cpo {
  inline constexpr auto inplace_merge = __inplace_merge::__fn{};
} // namespace __cpo
} // namespace ranges

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER > 17 && !defined(_LIBCPP_HAS_NO_INCOMPLETE_RANGES)

#endif // _LIBCPP___ALGORITHM_RANGES_INPLACE_MERGE_H
