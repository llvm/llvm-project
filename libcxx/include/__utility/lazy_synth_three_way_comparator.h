//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___UTILITY_LAZY_SYNTH_THREE_WAY_COMPARATOR_H
#define _LIBCPP___UTILITY_LAZY_SYNTH_THREE_WAY_COMPARATOR_H

#include <__config>
#include <__type_traits/conjunction.h>
#include <__type_traits/desugars_to.h>
#include <__type_traits/enable_if.h>
#include <__utility/default_three_way_comparator.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

// This file implements a __lazy_synth_three_way_comparator, which tries to build an efficient three way comparison from
// a binary comparator. That is done in multiple steps:
// 1) Check whether the comparator desugars to a less-than operator
//    If that is the case, check whether there exists a specialization of `__default_three_way_comparator`, which
//    can be specialized to implement a three way comparator for the specific types.
// 2) Fall back to doing a lazy less than/greater than comparison

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Comparator, class _LHS, class _RHS>
struct __lazy_compare_result {
  const _Comparator& __comp_;
  const _LHS& __lhs_;
  const _RHS& __rhs_;

  _LIBCPP_HIDE_FROM_ABI
  __lazy_compare_result(_LIBCPP_CTOR_LIFETIMEBOUND const _Comparator& __comp,
                        _LIBCPP_CTOR_LIFETIMEBOUND const _LHS& __lhs,
                        _LIBCPP_CTOR_LIFETIMEBOUND const _RHS& __rhs)
      : __comp_(__comp), __lhs_(__lhs), __rhs_(__rhs) {}

  _LIBCPP_HIDE_FROM_ABI bool __less() const { return __comp_(__lhs_, __rhs_); }
  _LIBCPP_HIDE_FROM_ABI bool __greater() const { return __comp_(__rhs_, __lhs_); }
};

// This class provides three way comparison between _LHS and _RHS as efficiently as possible. This can be specialized if
// a comparator only compares part of the object, potentially allowing an efficient three way comparison between the
// subobjects. The specialization should use the __lazy_synth_three_way_comparator for the subobjects to achieve this.
template <class _Comparator, class _LHS, class _RHS, class = void>
struct __lazy_synth_three_way_comparator {
  const _Comparator& __comp_;

  _LIBCPP_HIDE_FROM_ABI __lazy_synth_three_way_comparator(_LIBCPP_CTOR_LIFETIMEBOUND const _Comparator& __comp)
      : __comp_(__comp) {}

  _LIBCPP_HIDE_FROM_ABI __lazy_compare_result<_Comparator, _LHS, _RHS>
  operator()(_LIBCPP_LIFETIMEBOUND const _LHS& __lhs, _LIBCPP_LIFETIMEBOUND const _RHS& __rhs) const {
    return __lazy_compare_result<_Comparator, _LHS, _RHS>(__comp_, __lhs, __rhs);
  }
};

struct __eager_compare_result {
  int __res_;

  _LIBCPP_HIDE_FROM_ABI explicit __eager_compare_result(int __res) : __res_(__res) {}

  _LIBCPP_HIDE_FROM_ABI bool __less() const { return __res_ < 0; }
  _LIBCPP_HIDE_FROM_ABI bool __greater() const { return __res_ > 0; }
};

template <class _Comparator, class _LHS, class _RHS>
struct __lazy_synth_three_way_comparator<_Comparator,
                                         _LHS,
                                         _RHS,
                                         __enable_if_t<_And<__desugars_to<__less_tag, _Comparator, _LHS, _RHS>,
                                                            __has_default_three_way_comparator<_LHS, _RHS> >::value> > {
  // This lifetimebound annotation is technically incorrect, but other specializations actually capture the lifetime of
  // the comparator.
  _LIBCPP_HIDE_FROM_ABI __lazy_synth_three_way_comparator(_LIBCPP_CTOR_LIFETIMEBOUND const _Comparator&) {}

  // Same comment as above.
  _LIBCPP_HIDE_FROM_ABI static __eager_compare_result
  operator()(_LIBCPP_LIFETIMEBOUND const _LHS& __lhs, _LIBCPP_LIFETIMEBOUND const _RHS& __rhs) {
    return __eager_compare_result(__default_three_way_comparator<_LHS, _RHS>()(__lhs, __rhs));
  }
};

template <class _Comparator, class _LHS, class _RHS>
struct __lazy_synth_three_way_comparator<_Comparator,
                                         _LHS,
                                         _RHS,
                                         __enable_if_t<_And<__desugars_to<__greater_tag, _Comparator, _LHS, _RHS>,
                                                            __has_default_three_way_comparator<_LHS, _RHS> >::value> > {
  // This lifetimebound annotation is technically incorrect, but other specializations actually capture the lifetime of
  // the comparator.
  _LIBCPP_HIDE_FROM_ABI __lazy_synth_three_way_comparator(_LIBCPP_CTOR_LIFETIMEBOUND const _Comparator&) {}

  // Same comment as above.
  _LIBCPP_HIDE_FROM_ABI static __eager_compare_result
  operator()(_LIBCPP_LIFETIMEBOUND const _LHS& __lhs, _LIBCPP_LIFETIMEBOUND const _RHS& __rhs) {
    return __eager_compare_result(-__default_three_way_comparator<_LHS, _RHS>()(__lhs, __rhs));
  }
};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___UTILITY_LAZY_SYNTH_THREE_WAY_COMPARATOR_H
