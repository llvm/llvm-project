//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___PSTL_CPU_ALGOS_MIN_ELEMENT_H
#define _LIBCPP___PSTL_CPU_ALGOS_MIN_ELEMENT_H

#include <__algorithm/min_element.h>
#include <__config>
#include <__iterator/concepts.h>
#include <__iterator/iterator_traits.h>
#include <__pstl/backend_fwd.h>
#include <__pstl/cpu_algos/cpu_traits.h>
#include <__type_traits/desugars_to.h>
#include <__type_traits/is_execution_policy.h>
#include <__type_traits/is_trivially_copyable.h>
#include <__utility/move.h>
#include <__utility/unreachable.h>
#include <optional>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __pstl {

// Check if the comparator is totally ordered, and if it is,
// we can use == instead of the double comparison !(comp(a,b) && !comp(b,a)).
template <class _Compare, class _Tp>
inline constexpr bool __desugars_to_totally_ordered_v =
    __desugars_to_v<__totally_ordered_less_tag, _Compare, _Tp, _Tp> ||
    __desugars_to_v<__totally_ordered_greater_tag, _Compare, _Tp, _Tp>;

template <class _Backend, class _Index, class _DifferenceType, class _Compare>
_LIBCPP_HIDE_FROM_ABI _Index __simd_min_element(_Index __first, _DifferenceType __n, _Compare __comp) noexcept {
  if (__n == 0)
    return __first;

  using _ValueType                   = __iterator_value_type<_Index>;
  constexpr size_t __lane_size       = __cpu_traits<_Backend>::__lane_size;
  const _DifferenceType __block_size = __lane_size / sizeof(_ValueType);

  if (__n < 2 * __block_size || __block_size < 2) {
    _Index __result = __first;
    for (_DifferenceType __i = 1; __i < __n; ++__i) {
      if (__comp(__first[__i], *__result)) {
        __result = __first + __i;
      }
    }
    return __result;
  }

  // Pass 1: find minimum value
  alignas(__lane_size) char __lane_buffer[__lane_size];
  _ValueType* __lane = reinterpret_cast<_ValueType*>(__lane_buffer);

  // initializer
  _PSTL_PRAGMA_SIMD
  for (_DifferenceType __i = 0; __i < __block_size; ++__i) {
    _ValueType __a = __first[__i];
    _ValueType __b = __first[__block_size + __i];
    ::new (__lane + __i) _ValueType(__comp(__a, __b) ? __a : __b);
  }

  // main loop
  _DifferenceType __i                    = 2 * __block_size;
  const _DifferenceType __last_iteration = __block_size * (__n / __block_size);
  for (; __i < __last_iteration; __i += __block_size) {
    _PSTL_PRAGMA_SIMD
    for (_DifferenceType __j = 0; __j < __block_size; ++__j) {
      if (__comp(__first[__i + __j], __lane[__j])) {
        __lane[__j] = __first[__i + __j];
      }
    }
  }

  // remainder
  for (_DifferenceType __j = 0; __j < __n - __last_iteration; ++__j) {
    if (__comp(__first[__last_iteration + __j], __lane[__j])) {
      __lane[__j] = __first[__last_iteration + __j];
    }
  }

  // combiner
  _ValueType __min_val = __lane[0];
  for (_DifferenceType __j = 1; __j < __block_size; ++__j) {
    if (__comp(__lane[__j], __min_val)) {
      __min_val = __lane[__j];
    }
  }

  // destroyer
  _PSTL_PRAGMA_SIMD
  for (_DifferenceType __j = 0; __j < __block_size; ++__j) {
    __lane[__j].~_ValueType();
  }

  // Pass 2: find first index with minimum value
  constexpr _DifferenceType __find_block_size                          = __lane_size / sizeof(_DifferenceType);
  alignas(__lane_size) _DifferenceType __found_lane[__find_block_size] = {0};
  _DifferenceType __begin                                              = 0;

  while (__n - __begin >= __find_block_size) {
    _DifferenceType __found = 0;
    _PSTL_PRAGMA_SIMD_REDUCTION(| : __found)
    for (_DifferenceType __k = 0; __k < __find_block_size; ++__k) {
      _DifferenceType __t;
      if constexpr (__desugars_to_totally_ordered_v<_Compare, _ValueType>) {
        __t = __first[__begin + __k] == __min_val;
      } else {
        __t = !__comp(__first[__begin + __k], __min_val) && !__comp(__min_val, __first[__begin + __k]);
      }
      __found_lane[__k] = __t;
      __found |= __t;
    }
    if (__found) {
      for (_DifferenceType __k = 0; __k < __find_block_size; ++__k) {
        if (__found_lane[__k]) {
          return __first + __begin + __k;
        }
      }
    }
    __begin += __find_block_size;
  }

  // remainder
  while (__begin < __n) {
    bool __is_equal;
    if constexpr (__desugars_to_totally_ordered_v<_Compare, _ValueType>) {
      __is_equal = __first[__begin] == __min_val;
    } else {
      __is_equal = !__comp(__first[__begin], __min_val) && !__comp(__min_val, __first[__begin]);
    }
    if (__is_equal) {
      return __first + __begin;
    }
    ++__begin;
  }

  __libcpp_unreachable();
}

template <class _Backend, class _RawExecutionPolicy>
struct __cpu_parallel_min_element {
  template <class _Policy, class _ForwardIterator, class _Compare>
  _LIBCPP_HIDE_FROM_ABI optional<_ForwardIterator>
  operator()(_Policy&&, _ForwardIterator __first, _ForwardIterator __last, _Compare __comp) const noexcept {
    if constexpr (__is_parallel_execution_policy_v<_RawExecutionPolicy> &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator>::value) {
      if (__first == __last) {
        return __last;
      }

      return __cpu_traits<_Backend>::__transform_reduce(
          std::move(__first),
          __last,
          [](_ForwardIterator __iter) { return __iter; },
          __last,
          [__comp, __last](_ForwardIterator __lhs_min, _ForwardIterator __rhs_min) {
            if (__lhs_min == __last)
              return __rhs_min;
            if (__rhs_min == __last)
              return __lhs_min;
            if (__comp(*__lhs_min, *__rhs_min))
              return __lhs_min;
            if (__comp(*__rhs_min, *__lhs_min))
              return __rhs_min;
            return __lhs_min < __rhs_min ? __lhs_min : __rhs_min;
          },
          [__comp, __last](_ForwardIterator __brick_first, _ForwardIterator __brick_last, _ForwardIterator __acc) {
            _ForwardIterator __local_min;
            if constexpr (__is_unsequenced_execution_policy_v<__remove_parallel_policy_t<_RawExecutionPolicy>> &&
                          is_trivially_copyable_v<__iterator_value_type<_ForwardIterator>>) {
              __local_min =
                  __pstl::__simd_min_element<_Backend>(std::move(__brick_first), __brick_last - __brick_first, __comp);
            } else {
              __local_min = std::min_element(std::move(__brick_first), __brick_last, __comp);
            }
            if (__local_min == __brick_last)
              return __acc;
            if (__acc == __last)
              return __local_min;
            if (__comp(*__local_min, *__acc))
              return __local_min;
            if (__comp(*__acc, *__local_min))
              return __acc;
            return __local_min < __acc ? __local_min : __acc;
          });
    } else if constexpr (__is_unsequenced_execution_policy_v<_RawExecutionPolicy> &&
                         __has_random_access_iterator_category_or_concept<_ForwardIterator>::value &&
                         is_trivially_copyable_v<__iterator_value_type<_ForwardIterator>>) {
      return __pstl::__simd_min_element<_Backend>(__first, __last - __first, std::move(__comp));
    } else {
      return std::min_element(std::move(__first), std::move(__last), std::move(__comp));
    }
  }
};

} // namespace __pstl
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 17

_LIBCPP_POP_MACROS

#endif // _LIBCPP___PSTL_CPU_ALGOS_MIN_ELEMENT_H
