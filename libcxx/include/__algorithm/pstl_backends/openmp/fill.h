//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_PSTL_BACKENDS_OPENMP_BACKEND_FILL_H
#define _LIBCPP___ALGORITHM_PSTL_BACKENDS_OPENMP_BACKEND_FILL_H

#include <__algorithm/fill.h>
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

template <class _Tp, class _DifferenceType, class _Up>
_LIBCPP_HIDE_FROM_ABI _Tp* __omp_fill(_Tp* __out1, _DifferenceType __n, const _Up& __value) noexcept {
  __par_backend::__omp_map_alloc(__out1, __n);
#  pragma omp target teams distribute parallel for
  for (_DifferenceType __i = 0; __i < __n; ++__i)
    *(__out1 + __i) = __value;
  __par_backend::__omp_map_from(__out1, __n);
  return __out1 + __n;
}

template <class _ExecutionPolicy, class _ForwardIterator, class _Tp>
_LIBCPP_HIDE_FROM_ABI optional<__empty>
__pstl_fill(__omp_backend_tag, _ForwardIterator __first, _ForwardIterator __last, const _Tp& __value) {
  // If it is safe to offload the computations to the GPU, we call the OpenMP
  // implementation of for_each. In the case of fill, we provide for_Each with a
  // lambda returning a constant.
  if constexpr (__is_unsequenced_execution_policy_v<_ExecutionPolicy> &&
                __is_parallel_execution_policy_v<_ExecutionPolicy> &&
                __libcpp_is_contiguous_iterator<_ForwardIterator>::value &&
                is_trivially_copyable_v<remove_pointer_t<decltype(std::__unwrap_iter(__first))> > &&
                is_trivially_copyable_v<_Tp>) {
    std::__rewrap_iter(__first, std::__omp_fill(std::__unwrap_iter(__first), __last - __first, __value));
    return __empty{};
  }
  // Otherwise, we execute fill on the CPU instead
  else {
    return std::__pstl_fill<_ExecutionPolicy>(__cpu_backend_tag{}, __first, __last, __value);
  }
}

_LIBCPP_END_NAMESPACE_STD

#endif // !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

#endif // _LIBCPP___ALGORITHM_PSTL_BACKENDS_OPENMP_BACKEND_FILL_H
