//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_COPY_MOVE_COMMON_H
#define _LIBCPP___ALGORITHM_COPY_MOVE_COMMON_H

#include <__algorithm/in_out_result.h>
#include <__algorithm/unwrap_iter.h>
#include <__algorithm/unwrap_range.h>
#include <__assert>
#include <__config>
#include <__cstddef/size_t.h>
#include <__iterator/concepts.h>
#include <__iterator/incrementable_traits.h>
#include <__iterator/iterator_traits.h>
#include <__string/constexpr_c_functions.h>
#include <__type_traits/enable_if.h>
#include <__type_traits/integral_constant.h>
#include <__type_traits/is_always_bitcastable.h>
#include <__type_traits/is_constructible.h>
#include <__type_traits/is_same.h>
#include <__type_traits/is_trivially_assignable.h>
#include <__type_traits/is_volatile.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

// Type traits.

template <class _From, class _To>
struct __can_lower_copy_assignment_to_memmove {
  static const bool value =
      // If the types are always bitcastable, it's valid to do a bitwise copy between them.
      __is_always_bitcastable<_From, _To>::value &&
      // Reject conversions that wouldn't be performed by the regular built-in assignment (e.g. between arrays).
      is_trivially_assignable<_To&, const _From&>::value &&
      // `memmove` doesn't accept `volatile` pointers, make sure the optimization SFINAEs away in that case.
      !is_volatile<_From>::value && !is_volatile<_To>::value;
};

template <class _From, class _To>
struct __can_lower_move_assignment_to_memmove {
  static const bool value =
      __is_always_bitcastable<_From, _To>::value && is_trivially_assignable<_To&, _From&&>::value &&
      !is_volatile<_From>::value && !is_volatile<_To>::value;
};

// `memmove` algorithms implementation.

template <class _In, class _Out>
_LIBCPP_CONSTEXPR_SINCE_CXX14 __in_out_result<_In*, _Out*>
__copy_trivial_impl(_In* __first, _In* __last, _Out* __result) {
  const size_t __n = static_cast<size_t>(__last - __first);

  std::__constexpr_memmove(__result, __first, __element_count(__n));

  return {__last, __result + __n};
}

template <class _In, class _Out>
_LIBCPP_CONSTEXPR_SINCE_CXX14 __in_out_result<_In*, _Out*>
__copy_backward_trivial_impl(_In* __first, _In* __last, _Out* __result) {
  const size_t __n = static_cast<size_t>(__last - __first);
  __result -= __n;

  std::__constexpr_memmove(__result, __first, __element_count(__n));

  return {__last, __result};
}

// Iterator unwrapping and dispatching to the correct overload.

template <class _InIter, class _OutIter>
struct __can_rewrap
    : integral_constant<bool, is_copy_constructible<_InIter>::value && is_copy_constructible<_OutIter>::value> {};

// When _InIter and _Sent have the same type, check random-access category/concept rather than sized_sentinel_for:
// classic std::copy only requires Cpp17InputIterator, which allows pre-C++20 iterator wrappers with an unconstrained
// operator- that do not specialize disable_sized_sentinel_for (mirroring std::distance).
template <class _InIter, class _Sent>
struct __has_random_access_distance
    : integral_constant<bool,
                        is_same<_InIter, _Sent>::value ? (__has_random_access_iterator_category<_InIter>::value
#if _LIBCPP_STD_VER >= 20
                                                          || random_access_iterator<_InIter>
#endif
                                                          )
#if _LIBCPP_STD_VER >= 20
                                                       : sized_sentinel_for<_Sent, _InIter>
#else
                                                       : false
#endif
                        > {
};

// If _OutIter is a bounded or user-defined contiguous iterator, we can only unwrap it up front when the input range's
// length is known in O(1); otherwise keep _OutIter wrapped so the element-by-element loop checks bounds before each
// write.
template <class _InIter, class _Sent, class _OutIter>
struct __can_unwrap_copy_move
    : integral_constant< bool,
                         __can_rewrap<_InIter, _OutIter>::value &&
                             (!__unwrap_iter_may_strip_checks<_OutIter>::value ||
                              __has_random_access_distance<_InIter, _Sent>::value)> {};

template <class _Algorithm,
          bool _IsBackward = false,
          class _InIter,
          class _Sent,
          class _OutIter,
          __enable_if_t<__can_unwrap_copy_move<_InIter, _Sent, _OutIter>::value &&
                            !__unwrap_iter_may_strip_checks<_OutIter>::value,
                        int> = 0>
_LIBCPP_CONSTEXPR_SINCE_CXX17 __in_out_result<_InIter, _OutIter>
__copy_move_unwrap_iters(_InIter __first, _Sent __last, _OutIter __out_first) {
  auto __range  = std::__unwrap_range(__first, std::move(__last));
  auto __result = _Algorithm()(std::move(__range.first), std::move(__range.second), std::__unwrap_iter(__out_first));
  return {std::__rewrap_range<_Sent>(std::move(__first), std::move(__result.__in_)),
          std::__rewrap_iter(std::move(__out_first), std::move(__result.__out_))};
}

// Unwrapping _OutIter would strip its bounds checks, so advance the still-wrapped iterator over the whole output range
// first: that lets the iterator itself reject an undersized destination before anything is written.
template <class _Algorithm,
          bool _IsBackward = false,
          class _InIter,
          class _Sent,
          class _OutIter,
          __enable_if_t<__can_unwrap_copy_move<_InIter, _Sent, _OutIter>::value &&
                            __unwrap_iter_may_strip_checks<_OutIter>::value,
                        int> = 0>
_LIBCPP_CONSTEXPR_SINCE_CXX17 __in_out_result<_InIter, _OutIter>
__copy_move_unwrap_iters(_InIter __first, _Sent __last, _OutIter __out_first) {
#if _LIBCPP_STD_VER >= 20
  using _OutDiff _LIBCPP_NODEBUG = iter_difference_t<_OutIter>;
#else
  using _OutDiff _LIBCPP_NODEBUG = typename iterator_traits<_OutIter>::difference_type;
#endif
  auto __n         = __last - __first;
  auto __range     = std::__unwrap_range(__first, std::move(__last));
  _OutDiff __out_n = static_cast<_OutDiff>(__n);
  _LIBCPP_ASSERT_VALID_INPUT_RANGE(
      __n >= 0 && static_cast<decltype(__n)>(__out_n) == __n, "std::copy/move: [first, last) must be a valid range");
  _OutIter __out_final = _IsBackward ? __out_first - __out_n : __out_first + __out_n;
  auto __result =
      _Algorithm()(std::move(__range.first), std::move(__range.second), std::__unwrap_iter(std::move(__out_first)));
  return {std::__rewrap_range<_Sent>(std::move(__first), std::move(__result.__in_)), std::move(__out_final)};
}

template <class _Algorithm,
          bool _IsBackward = false,
          class _InIter,
          class _Sent,
          class _OutIter,
          __enable_if_t<!__can_unwrap_copy_move<_InIter, _Sent, _OutIter>::value, int> = 0>
_LIBCPP_CONSTEXPR_SINCE_CXX17 __in_out_result<_InIter, _OutIter>
__copy_move_unwrap_iters(_InIter __first, _Sent __last, _OutIter __out_first) {
  return _Algorithm()(std::move(__first), std::move(__last), std::move(__out_first));
}

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ALGORITHM_COPY_MOVE_COMMON_H
