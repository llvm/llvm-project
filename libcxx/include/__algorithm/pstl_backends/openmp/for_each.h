//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_PSTL_BACKENDS_OPENMP_BACKEND_FOR_EACH_H
#define _LIBCPP___ALGORITHM_PSTL_BACKENDS_OPENMP_BACKEND_FOR_EACH_H

#include <__algorithm/for_each.h>
#include <__algorithm/pstl_backends/cpu_backends/backend.h>
#include <__algorithm/pstl_backends/openmp/backend.h>
#include <__algorithm/unwrap_iter.h>
#include <__config>
#include <__iterator/wrap_iter.h>
#include <__type_traits/is_execution_policy.h>
#include <__type_traits/is_trivially_copyable.h>
#include <__type_traits/remove_pointer.h>
#include <__utility/empty.h>
#include <optional>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Tp, class _DifferenceType, class _Function>
_LIBCPP_HIDE_FROM_ABI _Tp* __omp_for_each(_Tp* __inout1, _DifferenceType __n, _Function __f) noexcept {
  __par_backend::__omp_map_to(__inout1, __n);
#  pragma omp target teams distribute parallel for
  for (_DifferenceType __i = 0; __i < __n; ++__i)
    __f(*(__inout1 + __i));
  __par_backend::__omp_map_from(__inout1, __n);
  return __inout1 + __n;
}

template <class _ExecutionPolicy, class _ForwardIterator, class _Functor>
_LIBCPP_HIDE_FROM_ABI optional<__empty>
__pstl_for_each(__omp_backend_tag, _ForwardIterator __first, _ForwardIterator __last, _Functor __func) {
  // If it is safe to offload the computations to the GPU, we call the OpenMP
  // implementation of for_each
  if constexpr (__is_unsequenced_execution_policy_v<_ExecutionPolicy> &&
                __libcpp_is_contiguous_iterator<_ForwardIterator>::value &&
                __libcpp_is_contiguous_iterator<_ForwardIterator>::value &&
                is_trivially_copyable_v<remove_pointer_t<decltype(std::__unwrap_iter(__first))> >) {
    std::__omp_for_each(std::__unwrap_iter(__first), __last - __first, __func);
    return __empty{};
  }
  // Else we fall back to the serial backend
  else {
    return std::__pstl_for_each<_ExecutionPolicy>(__cpu_backend_tag{}, __first, __last, __func);
  }
}

_LIBCPP_END_NAMESPACE_STD

#endif // !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

#endif // _LIBCPP___ALGORITHM_PSTL_BACKENDS_OPENMP_BACKEND_FOR_EACH_H
