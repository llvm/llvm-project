//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_RANGES_UNIQUE_COPY_H
#define _LIBCPP___ALGORITHM_RANGES_UNIQUE_COPY_H

#include <__algorithm/in_out_result.h>
#include <__algorithm/make_projected.h>
#include <__algorithm/unique_copy.h>
#include <__concepts/same_as.h>
#include <__config>
#include <__functional/identity.h>
#include <__functional/invoke.h>
#include <__functional/ranges_operations.h>
#include <__iterator/concepts.h>
#include <__iterator/iterator_traits.h>
#include <__iterator/readable_traits.h>
#include <__iterator/projected.h>
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

template <class _InIter, class _OutIter>
using unique_copy_result = in_out_result<_InIter, _OutIter>;

namespace __unique_copy {

struct __fn {

  template <input_iterator _InIter, sentinel_for<_InIter> _Sent, weakly_incrementable _OutIter, class _Proj = identity,
            indirect_equivalence_relation<projected<_InIter, _Proj>> _Comp = ranges::equal_to>
  requires indirectly_copyable<_InIter, _OutIter> &&
            (forward_iterator<_InIter> ||
            (input_iterator<_OutIter> && same_as<iter_value_t<_InIter>, iter_value_t<_OutIter>>) ||
            indirectly_copyable_storable<_InIter, _OutIter>)
  _LIBCPP_HIDE_FROM_ABI constexpr
  unique_copy_result<_InIter, _OutIter>
  operator()(_InIter __first, _Sent __last, _OutIter __result, _Comp __comp = {}, _Proj __proj = {}) const {
    // TODO: implement
    (void)__first; (void)__last; (void)__result; (void)__comp; (void)__proj;
    return {};
  }

  template <input_range _Range, weakly_incrementable _OutIter, class _Proj = identity,
            indirect_equivalence_relation<projected<iterator_t<_Range>, _Proj>> _Comp = ranges::equal_to>
  requires indirectly_copyable<iterator_t<_Range>, _OutIter> &&
            (forward_iterator<iterator_t<_Range>> ||
            (input_iterator<_OutIter> && same_as<range_value_t<_Range>, iter_value_t<_OutIter>>) ||
            indirectly_copyable_storable<iterator_t<_Range>, _OutIter>)
  _LIBCPP_HIDE_FROM_ABI constexpr
  unique_copy_result<borrowed_iterator_t<_Range>, _OutIter>
  operator()(_Range&& __range, _OutIter __result, _Comp __comp = {}, _Proj __proj = {}) const {
    // TODO: implement
    (void)__range; (void)__result; (void)__comp; (void)__proj;
    return {};
  }

};

} // namespace __unique_copy

inline namespace __cpo {
  inline constexpr auto unique_copy = __unique_copy::__fn{};
} // namespace __cpo
} // namespace ranges

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER > 17 && !defined(_LIBCPP_HAS_NO_INCOMPLETE_RANGES)

#endif // _LIBCPP___ALGORITHM_RANGES_UNIQUE_COPY_H
