//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_RANGES_PARTITION_COPY_H
#define _LIBCPP___ALGORITHM_RANGES_PARTITION_COPY_H

#include <__algorithm/in_out_out_result.h>
#include <__algorithm/make_projected.h>
#include <__algorithm/partition_copy.h>
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
#include <__utility/forward.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER > 17 && !defined(_LIBCPP_HAS_NO_INCOMPLETE_RANGES)

_LIBCPP_BEGIN_NAMESPACE_STD

namespace ranges {

template <class _InIter, class _OutIter1, class _OutIter2>
using partition_copy_result = in_out_out_result<_InIter, _OutIter1, _OutIter2>;

namespace __partition_copy {

struct __fn {

  template <input_iterator _InIter, sentinel_for<_InIter> _Sent,
            weakly_incrementable _OutIter1, weakly_incrementable _OutIter2,
            class _Proj = identity, indirect_unary_predicate<projected<_InIter, _Proj>> _Pred>
  requires indirectly_copyable<_InIter, _OutIter1> && indirectly_copyable<_InIter, _OutIter2>
  _LIBCPP_HIDE_FROM_ABI constexpr
  partition_copy_result<_InIter, _OutIter1, _OutIter2>
  operator()(_InIter __first, _Sent __last, _OutIter1 __out_true, _OutIter2 __out_false,
             _Pred __pred, _Proj __proj = {}) const {
    // TODO: implement
    (void)__first; (void)__last; (void)__out_true; (void)__out_false; (void)__pred; (void)__proj;
    return {};
  }

  template <input_range _Range, weakly_incrementable _OutIter1, weakly_incrementable _OutIter2,
            class _Proj = identity, indirect_unary_predicate<projected<iterator_t<_Range>, _Proj>> _Pred>
  requires indirectly_copyable<iterator_t<_Range>, _OutIter1> && indirectly_copyable<iterator_t<_Range>, _OutIter2>
  _LIBCPP_HIDE_FROM_ABI constexpr
  partition_copy_result<borrowed_iterator_t<_Range>, _OutIter1, _OutIter2>
  operator()(_Range&& __range, _OutIter1 __out_true, _OutIter2 __out_false, _Pred __pred, _Proj __proj = {}) const {
    // TODO: implement
    (void)__range; (void)__out_true; (void)__out_false; (void)__pred; (void)__proj;
    return {};
  }

};

} // namespace __partition_copy

inline namespace __cpo {
  inline constexpr auto partition_copy = __partition_copy::__fn{};
} // namespace __cpo
} // namespace ranges

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER > 17 && !defined(_LIBCPP_HAS_NO_INCOMPLETE_RANGES)

#endif // _LIBCPP___ALGORITHM_RANGES_PARTITION_COPY_H
