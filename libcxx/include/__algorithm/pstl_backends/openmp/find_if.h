//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_PSTL_BACKENDS_OPENMP_BACKEND_FIND_IF_H
#define _LIBCPP___ALGORITHM_PSTL_BACKENDS_OPENMP_BACKEND_FIND_IF_H

#include <__algorithm/find_if.h>
#include <__algorithm/pstl_backends/cpu_backends/backend.h>
#include <__algorithm/pstl_backends/openmp/backend.h>
#include <__algorithm/unwrap_iter.h>
#include <__config>
#include <__iterator/wrap_iter.h>
#include <__type_traits/is_execution_policy.h>
#include <__type_traits/is_trivially_copyable.h>
#include <__type_traits/remove_pointer.h>
#include <optional>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Tp, class _DifferenceType, class _Predicate>
_LIBCPP_HIDE_FROM_ABI _Tp* __omp_find_if(_Tp* __first, _DifferenceType __n, _Predicate __pred) noexcept {
  __par_backend::__omp_map_to(__first, __n);
  _DifferenceType __idx = __n;
#  pragma omp target teams distribute parallel for reduction(min : __idx)
  for (_DifferenceType __i = 0; __i < __n; ++__i) {
    if (__pred(*(__first + __i))) {
      __idx = (__i < __idx) ? __i : __idx;
    }
  }
  __par_backend::__omp_map_release(__first, __n);
  return __first + __idx;
}

template <class _ExecutionPolicy, class _ForwardIterator, class _Predicate>
_LIBCPP_HIDE_FROM_ABI optional<_ForwardIterator>
__pstl_find_if(__omp_backend_tag, _ForwardIterator __first, _ForwardIterator __last, _Predicate __pred) {
  // If it is safe to offload the computations to the GPU, we call the OpenMP
  // implementation of find_if
  if constexpr (__is_unsequenced_execution_policy_v<_ExecutionPolicy> &&
                __is_parallel_execution_policy_v<_ExecutionPolicy> &&
                __libcpp_is_contiguous_iterator<_ForwardIterator>::value &&
                is_trivially_copyable_v<remove_pointer_t<decltype(std::__unwrap_iter(__first))> >) {
    return std::__rewrap_iter(__first, std::__omp_find_if(std::__unwrap_iter(__first), __last - __first, __pred));
    // Else we rey on the CPU PSTL backend
  } else {
    return std::__pstl_find_if<_ExecutionPolicy>(__cpu_backend_tag{}, __first, __last, __pred);
  }
}

_LIBCPP_END_NAMESPACE_STD

#endif // !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ALGORITHM_PSTL_BACKENDS_OPENMP_BACKEND_FIND_IF_H
