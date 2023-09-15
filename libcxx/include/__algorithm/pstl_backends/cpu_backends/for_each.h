//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_PSTL_BACKENDS_CPU_BACKNEDS_FOR_EACH_H
#define _LIBCPP___ALGORITHM_PSTL_BACKENDS_CPU_BACKNEDS_FOR_EACH_H

#include <__algorithm/for_each.h>
#include <__algorithm/pstl_backends/cpu_backends/backend.h>
#include <__config>
#include <__iterator/concepts.h>
#include <__type_traits/is_execution_policy.h>
#include <__utility/terminate_on_exception.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Iterator, class _DifferenceType, class _Function>
_LIBCPP_HIDE_FROM_ABI _Iterator __simd_walk_1(_Iterator __first, _DifferenceType __n, _Function __f) noexcept {
  _PSTL_OMP_MAP_TO(__first,__n);
  _PSTL_PRAGMA_SIMD(__n)
  for (_DifferenceType __i = 0; __i < __n; ++__i)
    __f(__first[__i]);
  _PSTL_OMP_MAP_FROM(__first,__n);
  return __first + __n;
}

/**
 * Specialization for std::vector where the base pointer must be extrated to map
 * the data to and from the GPU.
*/

template <typename T, class _DifferenceType, class _Function>
_LIBCPP_HIDE_FROM_ABI std::__wrap_iter<T*> __simd_walk_1(std::__wrap_iter<T*> __first, _DifferenceType __n, _Function __f) noexcept {
  _PSTL_OMP_MAP_TO(__first,__n);
  // For std::vector the base pointer of the data buffer needs to be extracted 
  std::pointer_traits<std::__wrap_iter<T*>> PT;
  T* data = PT.to_address(__first);
  _PSTL_PRAGMA_SIMD(__n)
  for (_DifferenceType __i = 0; __i < __n; ++__i)
    __f(data[__i]);
  _PSTL_OMP_MAP_FROM(__first,__n);
  return __first + __n;
}

template <class _ExecutionPolicy, class _ForwardIterator, class _Functor>
_LIBCPP_HIDE_FROM_ABI void
__pstl_for_each(__cpu_backend_tag, _ForwardIterator __first, _ForwardIterator __last, _Functor __func) {
  if constexpr (__is_parallel_execution_policy_v<_ExecutionPolicy> &&
                __has_random_access_iterator_category_or_concept<_ForwardIterator>::value) {
    std::__terminate_on_exception([&] {
      std::__par_backend::__parallel_for(
          __first, __last, [__func](_ForwardIterator __brick_first, _ForwardIterator __brick_last) {
            std::__pstl_for_each<__remove_parallel_policy_t<_ExecutionPolicy>>(
                __cpu_backend_tag{}, __brick_first, __brick_last, __func);
          });
    });
  } else if constexpr (__is_unsequenced_execution_policy_v<_ExecutionPolicy> &&
                       __has_random_access_iterator_category_or_concept<_ForwardIterator>::value) {
    std::__simd_walk_1(__first, __last - __first, __func);
  } else {
    std::for_each(__first, __last, __func);
  }
}

_LIBCPP_END_NAMESPACE_STD

#endif // !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

#endif // _LIBCPP___ALGORITHM_PSTL_BACKENDS_CPU_BACKNEDS_FOR_EACH_H
