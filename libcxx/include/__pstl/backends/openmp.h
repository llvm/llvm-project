//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___PSTL_BACKENDS_OPENMP_H
#define _LIBCPP___PSTL_BACKENDS_OPENMP_H

// Combined OpenMP CPU and GPU Backend
// ===================================
// Contrary to the CPU backends found in ./cpu_backends/, the OpenMP backend can
// target both CPUs and GPUs. The OpenMP standard defines that when offloading
// code to an accelerator, the compiler must generate a fallback code for
// execution on the host. Thereby, the backend works as a CPU backend if no
// targeted accelerator is available at execution time. The target regions can
// also be compiled directly for a CPU architecture, for instance by adding the
// command-line option `-fopenmp-targets=x86_64-pc-linux-gnu` in Clang.
//
// When is an Algorithm Offloaded?
// -------------------------------
// Only parallel algorithms with the parallel unsequenced execution policy are
// offloaded to the device. We cannot offload parallel algorithms with a
// parallel execution policy to GPUs because invocations executing in the same
// thread "are indeterminately sequenced with respect to each other" which we
// cannot guarantee on a GPU.
//
// The standard draft states that "the semantics [...] allow the implementation
// to fall back to sequential execution if the system cannot parallelize an
// algorithm invocation". If it is not deemed safe to offload the parallel
// algorithm to the device, we first fall back to a parallel unsequenced
// implementation from ./cpu_backends. The CPU implementation may then fall back
// to sequential execution. In that way we strive to achieve the best possible
// performance.
//
// Further, "it is the caller's responsibility to ensure that the invocation
// does not introduce data races or deadlocks."
//
// Implicit Assumptions
// --------------------
// If the user provides a function pointer as an argument to a parallel
// algorithm, it is assumed that it is the device pointer as there is currently
// no way to check whether a host or device pointer was passed.
//
// Mapping Clauses
// ---------------
// In some of the parallel algorithms, the user is allowed to provide the same
// iterator as input and output. The order of the maps matters because OpenMP
// keeps a reference counter of which variables have been mapped to the device.
// Thereby, a varible is only copied to the device if its reference counter is
// incremented from zero, and it is only copied back to the host when the
// reference counter is decremented to zero again.
// This allows nesting mapped regions, for instance in recursive functions,
// without enforcing a lot of unnecessary data movement.
// Therefore, `pragma omp target data map(to:...)` must be used before
// `pragma omp target data map(alloc:...)`. Conversely, the maps with map
// modifier `release` must be placed before the maps with map modifier `from`
// when transferring the result from the device to the host.
//
// Example: Assume `a` and `b` are pointers to the same array.
// ``` C++
// #pragma omp target enter data map(alloc:a[0:n])
// // The reference counter is incremented from 0 to 1. a is not copied to the
// // device because of the `alloc` map modifier.
// #pragma omp target enter data map(to:b[0:n])
// // The reference counter is incremented from 1 to 2. b is not copied because
// // the reference counter is positive. Therefore b, and a, are uninitialized
// // on the device.
// ```
//
// Exceptions
// ----------
// Currently, GPU architectures do not handle exceptions. OpenMP target regions
// are allowed to contain try/catch statements and throw expressions in Clang,
// but if a throw expression is reached, it will terminate the program. That
// does not conform to the C++ standard.
//
// [This document](https://eel.is/c++draft/algorithms.parallel) has been used as
// reference for these considerations.

#include <__algorithm/unwrap_iter.h>
#include <__config>
#include <__functional/operations.h>
#include <__iterator/iterator_traits.h>
#include <__iterator/wrap_iter.h>
#include <__pstl/backend_fwd.h>
#include <__pstl/dispatch.h>
#include <__type_traits/desugars_to.h>
#include <__type_traits/is_arithmetic.h>
#include <__type_traits/is_trivially_copyable.h>
#include <__type_traits/remove_cvref.h>
#include <__utility/empty.h>
#include <__utility/forward.h>
#include <__utility/move.h>
#include <execution>
#include <optional>

#if !defined(_OPENMP)
#  error "Trying to use the OpenMP PSTL backend, but OpenMP is not enabled. Did you compile with -fopenmp?"
#elif (defined(_OPENMP) && _OPENMP < 201511)
#  error                                                                                                               \
      "OpenMP target offloading has been supported since OpenMP version 4.5 (201511). Please use a more recent version of OpenMP."
#endif

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __pstl {

// The following functions can be used to map contiguous array sections to and from the device.
// For now, they are simple overlays of the OpenMP pragmas, but they should be updated when adding
// support for other iterator types.
template <class _Iterator, class _DifferenceType>
_LIBCPP_HIDE_FROM_ABI void
__omp_map_to([[maybe_unused]] const _Iterator __p, [[maybe_unused]] const _DifferenceType __len) noexcept {
  static_assert(__libcpp_is_contiguous_iterator<_Iterator>::value);
#pragma omp target enter data map(to : __p[0 : __len])
}

template <class _Iterator, class _DifferenceType>
_LIBCPP_HIDE_FROM_ABI void
__omp_map_from([[maybe_unused]] const _Iterator __p, [[maybe_unused]] const _DifferenceType __len) noexcept {
  static_assert(__libcpp_is_contiguous_iterator<_Iterator>::value);
#pragma omp target exit data map(from : __p[0 : __len])
}

template <class _Iterator, class _DifferenceType>
_LIBCPP_HIDE_FROM_ABI void
__omp_map_alloc([[maybe_unused]] const _Iterator __p, [[maybe_unused]] const _DifferenceType __len) noexcept {
  static_assert(__libcpp_is_contiguous_iterator<_Iterator>::value);
#pragma omp target enter data map(alloc : __p[0 : __len])
}

template <class _Iterator, class _DifferenceType>
_LIBCPP_HIDE_FROM_ABI void
__omp_map_release([[maybe_unused]] const _Iterator __p, [[maybe_unused]] const _DifferenceType __len) noexcept {
  static_assert(__libcpp_is_contiguous_iterator<_Iterator>::value);
#pragma omp target exit data map(release : __p[0 : __len])
}

//
// fill
//
template <class _Tp, class _DifferenceType, class _Up>
_LIBCPP_HIDE_FROM_ABI _Tp* __omp_fill(_Tp* __out1, _DifferenceType __n, const _Up& __value) noexcept {
  __pstl::__omp_map_alloc(__out1, __n);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wopenmp-mapping"
#pragma omp target teams distribute parallel for
  for (_DifferenceType __i = 0; __i < __n; ++__i)
    *(__out1 + __i) = __value;
#pragma clang diagnostic pop
  __pstl::__omp_map_from(__out1, __n);
  return __out1 + __n;
}

template <>
struct __fill<__openmp_backend_tag, execution::parallel_unsequenced_policy> {
  template <class _Policy, class _ForwardIterator, class _Tp>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<__empty>
  operator()(_Policy&& __policy, _ForwardIterator __first, _ForwardIterator __last, _Tp const& __value) const noexcept {
    using _ValueType = typename iterator_traits<_ForwardIterator>::value_type;
    if constexpr (__libcpp_is_contiguous_iterator<_ForwardIterator>::value && is_trivially_copyable_v<_ValueType> &&
                  is_trivially_copyable_v<_Tp>) {
      __pstl::__omp_fill(std::__unwrap_iter(__first), __last - __first, __value);
      return __empty{};
    } else {
      using _Backends = __backends_after<__current_configuration, __openmp_backend_tag>;
      using _Fallback = __dispatch<__pstl::__fill, _Backends, __remove_cvref_t<_Policy>>;
      return _Fallback{}(std::forward<_Policy>(__policy), std::move(__first), std::move(__last), __value);
    }
  }
};

//
// find_if
//
template <class _Tp, class _DifferenceType, class _Predicate>
_LIBCPP_HIDE_FROM_ABI _Tp* __omp_find_if(_Tp* __first, _DifferenceType __n, _Predicate __pred) noexcept {
  __pstl::__omp_map_to(__first, __n);
  _DifferenceType __idx = __n;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wopenmp-mapping"
#pragma omp target teams distribute parallel for reduction(min : __idx)
  for (_DifferenceType __i = 0; __i < __n; ++__i) {
    if (__pred(*(__first + __i))) {
      __idx = (__i < __idx) ? __i : __idx;
    }
  }
#pragma clang diagnostic pop
  __pstl::__omp_map_release(__first, __n);
  return __first + __idx;
}

template <>
struct __find_if<__openmp_backend_tag, execution::parallel_unsequenced_policy> {
  template <class _Policy, class _ForwardIterator, class _Predicate>
  _LIBCPP_HIDE_FROM_ABI optional<_ForwardIterator>
  operator()(_Policy&& __policy, _ForwardIterator __first, _ForwardIterator __last, _Predicate __pred) const noexcept {
    using _ValueType = typename iterator_traits<_ForwardIterator>::value_type;
    if constexpr (__libcpp_is_contiguous_iterator<_ForwardIterator>::value && is_trivially_copyable_v<_ValueType>) {
      return std::__rewrap_iter(__first, __pstl::__omp_find_if(std::__unwrap_iter(__first), __last - __first, __pred));
    } else {
      using _Backends = __backends_after<__current_configuration, __openmp_backend_tag>;
      using _Fallback = __dispatch<__pstl::__find_if, _Backends, __remove_cvref_t<_Policy>>;
      return _Fallback{}(std::forward<_Policy>(__policy), std::move(__first), std::move(__last), std::move(__pred));
    }
  }
};

//
// for_each
//
template <class _Tp, class _DifferenceType, class _Function>
_LIBCPP_HIDE_FROM_ABI _Tp* __omp_for_each(_Tp* __inout1, _DifferenceType __n, _Function __f) noexcept {
  __pstl::__omp_map_to(__inout1, __n);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wopenmp-mapping"
#pragma omp target teams distribute parallel for
  for (_DifferenceType __i = 0; __i < __n; ++__i)
    __f(*(__inout1 + __i));
#pragma clang diagnostic pop
  __pstl::__omp_map_from(__inout1, __n);
  return __inout1 + __n;
}

template <>
struct __for_each<__openmp_backend_tag, execution::parallel_unsequenced_policy> {
  template <class _Policy, class _ForwardIterator, class _Functor>
  _LIBCPP_HIDE_FROM_ABI optional<__empty>
  operator()(_Policy&& __policy, _ForwardIterator __first, _ForwardIterator __last, _Functor __func) const noexcept {
    using _ValueType = typename iterator_traits<_ForwardIterator>::value_type;
    if constexpr (__libcpp_is_contiguous_iterator<_ForwardIterator>::value &&
                  __libcpp_is_contiguous_iterator<_ForwardIterator>::value && is_trivially_copyable_v<_ValueType>) {
      __pstl::__omp_for_each(std::__unwrap_iter(__first), __last - __first, std::move(__func));
      return __empty{};
    } else {
      using _Backends = __backends_after<__current_configuration, __openmp_backend_tag>;
      using _Fallback = __dispatch<__pstl::__for_each, _Backends, __remove_cvref_t<_Policy>>;
      return _Fallback{}(std::forward<_Policy>(__policy), std::move(__first), std::move(__last), std::move(__func));
    }
  }
};

//
// transform
//
template <class _Tp, class _DifferenceType, class _Up, class _Function>
_LIBCPP_HIDE_FROM_ABI _Tp* __omp_transform(_Tp* __in1, _DifferenceType __n, _Up* __out1, _Function __f) noexcept {
  // The order of the following maps matter, as we wish to move the data. If
  // they were placed in the reverse order, and __in equals __out, then we would
  // allocate the buffer on the device without copying the data.
  __pstl::__omp_map_to(__in1, __n);
  __pstl::__omp_map_alloc(__out1, __n);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wopenmp-mapping"
#pragma omp target teams distribute parallel for
  for (_DifferenceType __i = 0; __i < __n; ++__i)
    *(__out1 + __i) = __f(*(__in1 + __i));
#pragma clang diagnostic pop
  // The order of the following two maps matters, since the user could legally
  // overwrite __in The "release" map modifier decreases the reference counter
  // by one, and "from" only moves the data to the host, when the reference
  // count is decremented to zero.
  __pstl::__omp_map_release(__in1, __n);
  __pstl::__omp_map_from(__out1, __n);
  return __out1 + __n;
}

template <>
struct __transform<__openmp_backend_tag, execution::parallel_unsequenced_policy> {
  template <class _Policy, class _ForwardIterator, class _ForwardOutIterator, class _UnaryOperation>
  _LIBCPP_HIDE_FROM_ABI optional<_ForwardOutIterator>
  operator()(_Policy&& __policy,
             _ForwardIterator __first,
             _ForwardIterator __last,
             _ForwardOutIterator __outit,
             _UnaryOperation __op) const noexcept {
    using _ValueType = typename iterator_traits<_ForwardIterator>::value_type;
    if constexpr (__libcpp_is_contiguous_iterator<_ForwardIterator>::value &&
                  __libcpp_is_contiguous_iterator<_ForwardOutIterator>::value && is_trivially_copyable_v<_ValueType>) {
      return std::__rewrap_iter(
          __outit,
          __omp_transform(std::__unwrap_iter(__first), __last - __first, std::__unwrap_iter(__outit), std::move(__op)));
    } else {
      using _Backends = __backends_after<__current_configuration, __openmp_backend_tag>;
      using _Fallback = __dispatch<__pstl::__transform, _Backends, __remove_cvref_t<_Policy>>;
      return _Fallback{}(
          std::forward<_Policy>(__policy), std::move(__first), std::move(__last), std::move(__outit), std::move(__op));
    }
  }
};

//
// transform_binary
//
template <class _Tp, class _DifferenceType, class _Up, class _Vp, class _Function>
_LIBCPP_HIDE_FROM_ABI _Tp*
__omp_transform(_Tp* __in1, _DifferenceType __n, _Up* __in2, _Vp* __out1, _Function __f) noexcept {
  // The order of the following maps matter, as we wish to move the data. If
  // they were placed in the reverse order, and __out equals __in1 or __in2,
  // then we would allocate one of the buffer on the device without copying the
  // data.
  __pstl::__omp_map_to(__in1, __n);
  __pstl::__omp_map_to(__in2, __n);
  __pstl::__omp_map_alloc(__out1, __n);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wopenmp-mapping"
#pragma omp target teams distribute parallel for
  for (_DifferenceType __i = 0; __i < __n; ++__i)
    *(__out1 + __i) = __f(*(__in1 + __i), *(__in2 + __i));
#pragma clang diagnostic pop
  // The order of the following three maps matters, since the user could legally
  // overwrite either of the inputs if __out equals __in1 or __in2. The
  // "release" map modifier decreases the reference counter by one, and "from"
  // only moves the data from the device, when the reference count is
  // decremented to zero.
  __pstl::__omp_map_release(__in1, __n);
  __pstl::__omp_map_release(__in2, __n);
  __pstl::__omp_map_from(__out1, __n);
  return __out1 + __n;
}

template <>
struct __transform_binary<__openmp_backend_tag, execution::parallel_unsequenced_policy> {
  template <class _Policy,
            class _ForwardIterator1,
            class _ForwardIterator2,
            class _ForwardOutIterator,
            class _BinaryOperation>
  _LIBCPP_HIDE_FROM_ABI optional<_ForwardOutIterator>
  operator()(_Policy&& __policy,
             _ForwardIterator1 __first1,
             _ForwardIterator1 __last1,
             _ForwardIterator2 __first2,
             _ForwardOutIterator __outit,
             _BinaryOperation __op) const noexcept {
    using _ValueType1 = typename iterator_traits<_ForwardIterator1>::value_type;
    using _ValueType2 = typename iterator_traits<_ForwardIterator2>::value_type;
    if constexpr (__libcpp_is_contiguous_iterator<_ForwardIterator1>::value &&
                  __libcpp_is_contiguous_iterator<_ForwardIterator2>::value &&
                  __libcpp_is_contiguous_iterator<_ForwardOutIterator>::value && is_trivially_copyable_v<_ValueType1> &&
                  is_trivially_copyable_v<_ValueType2>) {
      return std::__rewrap_iter(
          __outit,
          __pstl::__omp_transform(
              std::__unwrap_iter(__first1),
              __last1 - __first1,
              std::__unwrap_iter(__first2),
              std::__unwrap_iter(__outit),
              std::move(__op)));
    } else {
      using _Backends = __backends_after<__current_configuration, __openmp_backend_tag>;
      using _Fallback = __dispatch<__pstl::__transform_binary, _Backends, __remove_cvref_t<_Policy>>;
      return _Fallback{}(
          std::forward<_Policy>(__policy),
          std::move(__first1),
          std::move(__last1),
          std::move(__first2),
          std::move(__outit),
          std::move(__op));
    }
  }
};

//
// transform_reduce
//
#define _LIBCPP_PSTL_OMP_SIMD_1_REDUCTION(omp_op, std_op)                                                              \
  template <class _Iterator,                                                                                           \
            class _DifferenceType,                                                                                     \
            typename _Tp,                                                                                              \
            typename _BinaryOperationType,                                                                             \
            typename _UnaryOperation>                                                                                  \
  _LIBCPP_HIDE_FROM_ABI _Tp __omp_transform_reduce(                                                                    \
      _Iterator __first,                                                                                               \
      _DifferenceType __n,                                                                                             \
      _Tp __init,                                                                                                      \
      std_op<_BinaryOperationType> __reduce,                                                                           \
      _UnaryOperation __transform) noexcept {                                                                          \
    __pstl::__omp_map_to(__first, __n);                                                                                \
    _PSTL_PRAGMA(clang diagnostic push)                                                                                \
    _PSTL_PRAGMA(clang diagnostic ignored "-Wopenmp-mapping")                                                          \
_PSTL_PRAGMA(omp target teams distribute parallel for reduction(omp_op:__init))                                        \
    for (_DifferenceType __i = 0; __i < __n; ++__i)                                                                    \
      __init = __reduce(__init, __transform(*(__first + __i)));                                                        \
    _PSTL_PRAGMA(clang diagnostic pop)                                                                                 \
    __pstl::__omp_map_release(__first, __n);                                                                           \
    return __init;                                                                                                     \
  }

#define _LIBCPP_PSTL_OMP_SIMD_2_REDUCTION(omp_op, std_op)                                                              \
  template <class _Iterator1,                                                                                          \
            class _Iterator2,                                                                                          \
            class _DifferenceType,                                                                                     \
            typename _Tp,                                                                                              \
            typename _BinaryOperationType,                                                                             \
            typename _UnaryOperation >                                                                                 \
  _LIBCPP_HIDE_FROM_ABI _Tp __omp_transform_reduce(                                                                    \
      _Iterator1 __first1,                                                                                             \
      _Iterator2 __first2,                                                                                             \
      _DifferenceType __n,                                                                                             \
      _Tp __init,                                                                                                      \
      std_op<_BinaryOperationType> __reduce,                                                                           \
      _UnaryOperation __transform) noexcept {                                                                          \
    __pstl::__omp_map_to(__first1, __n);                                                                               \
    __pstl::__omp_map_to(__first2, __n);                                                                               \
    _PSTL_PRAGMA(clang diagnostic push)                                                                                \
    _PSTL_PRAGMA(clang diagnostic ignored "-Wopenmp-mapping")                                                          \
_PSTL_PRAGMA(omp target teams distribute parallel for reduction(omp_op:__init))                                        \
    for (_DifferenceType __i = 0; __i < __n; ++__i)                                                                    \
      __init = __reduce(__init, __transform(*(__first1 + __i), *(__first2 + __i)));                                    \
    _PSTL_PRAGMA(clang diagnostic pop)                                                                                 \
    __pstl::__omp_map_release(__first1, __n);                                                                          \
    __pstl::__omp_map_release(__first2, __n);                                                                          \
    return __init;                                                                                                     \
  }

#define _LIBCPP_PSTL_OMP_SIMD_REDUCTION(omp_op, std_op)                                                                \
  _LIBCPP_PSTL_OMP_SIMD_1_REDUCTION(omp_op, std_op)                                                                    \
  _LIBCPP_PSTL_OMP_SIMD_2_REDUCTION(omp_op, std_op)

_LIBCPP_PSTL_OMP_SIMD_REDUCTION(+, std::plus)
_LIBCPP_PSTL_OMP_SIMD_REDUCTION(-, std::minus)
_LIBCPP_PSTL_OMP_SIMD_REDUCTION(*, std::multiplies)
_LIBCPP_PSTL_OMP_SIMD_REDUCTION(&&, std::logical_and)
_LIBCPP_PSTL_OMP_SIMD_REDUCTION(||, std::logical_or)
_LIBCPP_PSTL_OMP_SIMD_REDUCTION(&, std::bit_and)
_LIBCPP_PSTL_OMP_SIMD_REDUCTION(|, std::bit_or)
_LIBCPP_PSTL_OMP_SIMD_REDUCTION(^, std::bit_xor)

// Determine whether a reduction is supported by the OpenMP backend
template <class _T1, class _T2, class _T3>
struct __is_supported_reduction : std::false_type {};

#define _LIBCPP_PSTL_IS_SUPPORTED_REDUCTION(func)                                                                      \
  template <class _Tp>                                                                                                 \
  struct __is_supported_reduction<func<_Tp>, _Tp, _Tp> : true_type {};                                                 \
  template <class _Tp, class _Up>                                                                                      \
  struct __is_supported_reduction<func<>, _Tp, _Up> : true_type {};

// __is_trivial_plus_operation already exists
_LIBCPP_PSTL_IS_SUPPORTED_REDUCTION(std::plus)
_LIBCPP_PSTL_IS_SUPPORTED_REDUCTION(std::minus)
_LIBCPP_PSTL_IS_SUPPORTED_REDUCTION(std::multiplies)
_LIBCPP_PSTL_IS_SUPPORTED_REDUCTION(std::logical_and)
_LIBCPP_PSTL_IS_SUPPORTED_REDUCTION(std::logical_or)
_LIBCPP_PSTL_IS_SUPPORTED_REDUCTION(std::bit_and)
_LIBCPP_PSTL_IS_SUPPORTED_REDUCTION(std::bit_or)
_LIBCPP_PSTL_IS_SUPPORTED_REDUCTION(std::bit_xor)

template <>
struct __transform_reduce<__openmp_backend_tag, execution::parallel_unsequenced_policy> {
  template <class _Policy, class _ForwardIterator, class _Tp, class _Reduction, class _Transform>
  _LIBCPP_HIDE_FROM_ABI optional<_Tp>
  operator()(_Policy&& __policy,
             _ForwardIterator __first,
             _ForwardIterator __last,
             _Tp __init,
             _Reduction __reduce,
             _Transform __transform) const noexcept {
    using _ValueType = typename iterator_traits<_ForwardIterator>::value_type;
    if constexpr (__libcpp_is_contiguous_iterator<_ForwardIterator>::value && is_arithmetic_v<_Tp> &&
                  __is_supported_reduction<_Reduction, _Tp, _Tp>::value && is_trivially_copyable_v<_ValueType>) {
      return __pstl::__omp_transform_reduce(
          std::__unwrap_iter(__first), __last - __first, __init, std::move(__reduce), std::move(__transform));
    } else {
      using _Backends = __backends_after<__current_configuration, __openmp_backend_tag>;
      using _Fallback = __dispatch<__pstl::__transform_reduce, _Backends, __remove_cvref_t<_Policy>>;
      return _Fallback{}(
          std::forward<_Policy>(__policy),
          std::move(__first),
          std::move(__last),
          std::move(__init),
          std::move(__reduce),
          std::move(__transform));
    }
  }
};

//
// transform_reduce_binary
//
template <>
struct __transform_reduce_binary<__openmp_backend_tag, execution::parallel_unsequenced_policy> {
  template <class _Policy,
            class _ForwardIterator1,
            class _ForwardIterator2,
            class _Tp,
            class _Reduction,
            class _Transform>
  _LIBCPP_HIDE_FROM_ABI optional<_Tp> operator()(
      _Policy&& __policy,
      _ForwardIterator1 __first1,
      _ForwardIterator1 __last1,
      _ForwardIterator2 __first2,
      _Tp __init,
      _Reduction __reduce,
      _Transform __transform) const noexcept {
    using _ValueType1 = typename iterator_traits<_ForwardIterator1>::value_type;
    using _ValueType2 = typename iterator_traits<_ForwardIterator2>::value_type;
    if constexpr (__libcpp_is_contiguous_iterator<_ForwardIterator1>::value &&
                  __libcpp_is_contiguous_iterator<_ForwardIterator2>::value && is_arithmetic_v<_Tp> &&
                  __is_supported_reduction<_Reduction, _Tp, _Tp>::value && is_trivially_copyable_v<_ValueType1> &&
                  is_trivially_copyable_v<_ValueType2>) {
      return __pstl::__omp_transform_reduce(
          std::__unwrap_iter(__first1),
          std::__unwrap_iter(__first2),
          __last1 - __first1,
          std::move(__init),
          std::move(__reduce),
          std::move(__transform));
    } else {
      using _Backends = __backends_after<__current_configuration, __openmp_backend_tag>;
      using _Fallback = __dispatch<__pstl::__transform_reduce_binary, _Backends, __remove_cvref_t<_Policy>>;
      return _Fallback{}(
          std::forward<_Policy>(__policy),
          std::move(__first1),
          std::move(__last1),
          std::move(__first2),
          std::move(__init),
          std::move(__reduce),
          std::move(__transform));
    }
  }
};

} // namespace __pstl
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___PSTL_BACKENDS_OPENMP_H
