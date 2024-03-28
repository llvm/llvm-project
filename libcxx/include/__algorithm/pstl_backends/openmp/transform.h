//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_PSTL_BACKENDS_OPENMP_BACKEND_TRANSFORM_H
#define _LIBCPP___ALGORITHM_PSTL_BACKENDS_OPENMP_BACKEND_TRANSFORM_H

#include <__algorithm/pstl_backends/cpu_backends/backend.h>
#include <__algorithm/pstl_backends/openmp/backend.h>
#include <__algorithm/transform.h>
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

#if !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD

//===----------------------------------------------------------------------===//
// OpenMP implementations of transform for one and two input iterators and one
// output iterator
//===----------------------------------------------------------------------===//

template <class _Tp, class _DifferenceType, class _Up, class _Function>
_LIBCPP_HIDE_FROM_ABI _Tp* __omp_transform(_Tp* __in1, _DifferenceType __n, _Up* __out1, _Function __f) noexcept {
  // The order of the following maps matter, as we wish to move the data. If
  // they were placed in the reverse order, and __in equals __out, then we would
  // allocate the buffer on the device without copying the data.
  __par_backend::__omp_map_to(__in1, __n);
  __par_backend::__omp_map_alloc(__out1, __n);
#  pragma omp target teams distribute parallel for
  for (_DifferenceType __i = 0; __i < __n; ++__i)
    *(__out1 + __i) = __f(*(__in1 + __i));
  // The order of the following two maps matters, since the user could legally
  // overwrite __in The "release" map modifier decreases the reference counter
  // by one, and "from" only moves the data to the host, when the reference
  // count is decremented to zero.
  __par_backend::__omp_map_release(__in1, __n);
  __par_backend::__omp_map_from(__out1, __n);
  return __out1 + __n;
}

template <class _Tp, class _DifferenceType, class _Up, class _Vp, class _Function>
_LIBCPP_HIDE_FROM_ABI _Tp*
__omp_transform(_Tp* __in1, _DifferenceType __n, _Up* __in2, _Vp* __out1, _Function __f) noexcept {
  // The order of the following maps matter, as we wish to move the data. If
  // they were placed in the reverse order, and __out equals __in1 or __in2,
  // then we would allocate one of the buffer on the device without copying the
  // data.
  __par_backend::__omp_map_to(__in1, __n);
  __par_backend::__omp_map_to(__in2, __n);
  __par_backend::__omp_map_alloc(__out1, __n);
#  pragma omp target teams distribute parallel for
  for (_DifferenceType __i = 0; __i < __n; ++__i)
    *(__out1 + __i) = __f(*(__in1 + __i), *(__in2 + __i));
  // The order of the following three maps matters, since the user could legally
  // overwrite either of the inputs if __out equals __in1 or __in2. The
  // "release" map modifier decreases the reference counter by one, and "from"
  // only moves the data from the device, when the reference count is
  // decremented to zero.
  __par_backend::__omp_map_release(__in1, __n);
  __par_backend::__omp_map_release(__in2, __n);
  __par_backend::__omp_map_from(__out1, __n);
  return __out1 + __n;
}

template <class _ExecutionPolicy, class _ForwardIterator, class _ForwardOutIterator, class _UnaryOperation>
_LIBCPP_HIDE_FROM_ABI optional<_ForwardOutIterator> __pstl_transform(
    __omp_backend_tag,
    _ForwardIterator __first,
    _ForwardIterator __last,
    _ForwardOutIterator __result,
    _UnaryOperation __op) {
  if constexpr (__is_unsequenced_execution_policy_v<_ExecutionPolicy> &&
                __is_parallel_execution_policy_v<_ExecutionPolicy> &&
                __libcpp_is_contiguous_iterator<_ForwardIterator>::value &&
                __libcpp_is_contiguous_iterator<_ForwardOutIterator>::value &&
                is_trivially_copyable_v<remove_pointer_t<decltype(std::__unwrap_iter(__first))> > &&
                is_trivially_copyable_v<remove_pointer_t<decltype(std::__unwrap_iter(__result))> >) {
    std::__rewrap_iter(
        __result,
        std::__omp_transform(std::__unwrap_iter(__first), __last - __first, std::__unwrap_iter(__result), __op));
  }
  // If it is not safe to offload to the GPU, we rely on the CPU backend.
  return std::__pstl_transform<_ExecutionPolicy>(__cpu_backend_tag{}, __first, __last, __result, __op);
}

template <class _ExecutionPolicy,
          class _ForwardIterator1,
          class _ForwardIterator2,
          class _ForwardOutIterator,
          class _BinaryOperation,
          enable_if_t<is_execution_policy_v<__remove_cvref_t<_ExecutionPolicy>>, int> = 0>
_LIBCPP_HIDE_FROM_ABI optional<_ForwardOutIterator> __pstl_transform(
    __omp_backend_tag,
    _ForwardIterator1 __first1,
    _ForwardIterator1 __last1,
    _ForwardIterator2 __first2,
    _ForwardOutIterator __result,
    _BinaryOperation __op) {
  if constexpr (__is_unsequenced_execution_policy_v<_ExecutionPolicy> &&
                __is_parallel_execution_policy_v<_ExecutionPolicy> &&
                __libcpp_is_contiguous_iterator<_ForwardIterator1>::value &&
                __libcpp_is_contiguous_iterator<_ForwardIterator2>::value &&
                __libcpp_is_contiguous_iterator<_ForwardOutIterator>::value &&
                is_trivially_copyable_v<remove_pointer_t<decltype(std::__unwrap_iter(__first1))> > &&
                is_trivially_copyable_v<remove_pointer_t<decltype(std::__unwrap_iter(__first2))> > &&
                is_trivially_copyable_v<remove_pointer_t<decltype(std::__unwrap_iter(__result))> >) {
    return std::__rewrap_iter(
        __result,
        std::__omp_transform(
            std::__unwrap_iter(__first1),
            __last1 - __first1,
            std::__unwrap_iter(__first2),
            std::__unwrap_iter(__result),
            __op));
  }
  // If it is not safe to offload to the GPU, we rely on the CPU backend.
  return std::__pstl_transform<_ExecutionPolicy>(__cpu_backend_tag{}, __first1, __last1, __first2, __result, __op);
}

_LIBCPP_END_NAMESPACE_STD

#endif // !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

#endif // _LIBCPP___ALGORITHM_PSTL_BACKENDS_OPENMP_BACKEND_TRANSFORM_H
