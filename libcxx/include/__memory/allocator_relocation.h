//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___MEMORY_ALLOCATOR_RELOCATION_H
#define _LIBCPP___MEMORY_ALLOCATOR_RELOCATION_H

#include <__assert>
#include <__config>
#include <__iterator/iterator_traits.h>
#include <__memory/addressof.h>
#include <__memory/allocator_traits.h>
#include <__memory/destroy.h>
#include <__memory/pointer_traits.h>
#include <__memory/relocate_at.h>
#include <__memory/uninitialized_relocate.h>
#include <__type_traits/enable_if.h>
#include <__type_traits/integral_constant.h>
#include <__type_traits/is_constant_evaluated.h>
#include <__type_traits/is_nothrow_constructible.h>
#include <__type_traits/is_nothrow_destructible.h>
#include <__type_traits/is_nothrow_relocatable.h>
#include <__type_traits/is_trivially_relocatable.h>
#include <__type_traits/negation.h>
#include <__utility/move.h>
#include <__utility/scope_guard.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

//
// This file implements allocator-aware companions to the various relocation facilities. Those are
// necessary to use from allocator-aware containers like std::vector.
//

template <class _Alloc, class _Type>
struct __allocator_has_trivial_move_construct : _Not<__has_construct<_Alloc, _Type*, _Type&&> > {};

template <class _Type>
struct __allocator_has_trivial_move_construct<allocator<_Type>, _Type> : true_type {};

template <class _Alloc, class _Tp>
struct __allocator_has_trivial_destroy : _Not<__has_destroy<_Alloc, _Tp*> > {};

template <class _Tp, class _Up>
struct __allocator_has_trivial_destroy<allocator<_Tp>, _Up> : true_type {};

// __is_trivially_allocator_relocatable:
//
// A type is trivially allocator-relocatable if the allocator's move construction and destruction
// don't do anything beyond calling the type's move constructor and destructor, and if the type
// itself is trivially relocatable.
template <class _Alloc, class _Tp>
struct __is_trivially_allocator_relocatable
    : integral_constant<bool,
                        __allocator_has_trivial_move_construct<_Alloc, _Tp>::value &&
                            __allocator_has_trivial_destroy<_Alloc, _Tp>::value &&
                            __libcpp_is_trivially_relocatable<_Tp>::value> {};

// __is_nothrow_allocator_relocatable:
//
// A type is nothrow allocator-relocatable if the allocator operations are trivial and the type is
// nothrow relocatable, or if it is trivially allocator-relocatable.
template <class _Alloc, class _Tp>
struct __is_nothrow_allocator_relocatable
    : _BoolConstant<(__allocator_has_trivial_move_construct<_Alloc, _Tp>::value &&
                     __allocator_has_trivial_destroy<_Alloc, _Tp>::value && __is_nothrow_relocatable<_Tp>::value) ||
                    __is_trivially_allocator_relocatable<_Alloc, _Tp>::value> {};

// __allocator_relocate_at:
//
// Either perform relocation using the allocator's non-trivial operations, or forward to allocator
// unaware relocation (which may then be trivial or not).
template <class _Alloc,
          class _Tp,
          __enable_if_t<__allocator_has_trivial_move_construct<_Alloc, _Tp>::value &&
                            __allocator_has_trivial_destroy<_Alloc, _Tp>::value,
                        int> = 0>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _Tp*
__allocator_relocate_at(_Alloc& __alloc, _Tp* __dest, _Tp* __source)
    _NOEXCEPT_(__is_nothrow_allocator_relocatable<_Alloc, _Tp>::value) {
  (void)__alloc; // ignore the allocator since its operations are trivial anyway
  return std::__relocate_at(__dest, __source);
}

template <class _Alloc, class _Tp>
struct __allocator_destroy_object {
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI __allocator_destroy_object(_Alloc& __alloc, _Tp* __obj)
      : __alloc_(__alloc), __obj_(__obj) {}
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void operator()() const {
    allocator_traits<_Alloc>::destroy(__alloc_, __obj_);
  }
  _Alloc& __alloc_;
  _Tp* __obj_;
};

template <class _Alloc,
          class _Tp,
          __enable_if_t<!(__allocator_has_trivial_move_construct<_Alloc, _Tp>::value &&
                          __allocator_has_trivial_destroy<_Alloc, _Tp>::value),
                        int> = 0>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _Tp*
__allocator_relocate_at(_Alloc& __alloc, _Tp* __dest, _Tp* __source)
    _NOEXCEPT_(__is_nothrow_allocator_relocatable<_Alloc, _Tp>::value) {
  auto __guard = std::__make_scope_guard(__allocator_destroy_object<_Alloc, _Tp>(__alloc, __source));
  allocator_traits<_Alloc>::construct(__alloc, __dest, std::move(*__source));
  return __dest;
}

// __uninitialized_allocator_relocate:
//
// Equivalent to __uninitialized_relocate, but uses the provided allocator's construct() and destroy() methods.
template <class _Alloc, class _ForwardIter, class _NothrowForwardIter>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _NothrowForwardIter __uninitialized_allocator_relocate(
    _Alloc& __alloc, _ForwardIter __first, _ForwardIter __last, _NothrowForwardIter __result) {
#ifndef _LIBCPP_CXX03_LANG
  if constexpr (__libcpp_is_contiguous_iterator<_ForwardIter>::value &&
                __libcpp_is_contiguous_iterator<_NothrowForwardIter>::value) {
    _LIBCPP_ASSERT_NON_OVERLAPPING_RANGES(
        !std::__is_pointer_in_range(std::__to_address(__first), std::__to_address(__last), std::__to_address(__last)),
        "uninitialized_allocator_relocate requires the result not to overlap with the input range");
  }
#endif

  using _ValueType = typename iterator_traits<_ForwardIter>::value_type;
  if _LIBCPP_CONSTEXPR (__allocator_has_trivial_move_construct<_Alloc, _ValueType>::value &&
                        __allocator_has_trivial_destroy<_Alloc, _ValueType>::value) {
    (void)__alloc; // ignore the allocator
    return std::__uninitialized_relocate(std::move(__first), std::move(__last), std::move(__result));
  } else {
    auto const __first_result = __result;
    try {
      while (__first != __last) {
        std::__allocator_relocate_at(__alloc, std::addressof(*__result), std::addressof(*__first));
        ++__first;
        ++__result;
      }
    } catch (...) {
      std::__allocator_destroy(__alloc, ++__first, __last);
      std::__allocator_destroy(__alloc, __first_result, __result);
      throw;
    }
    return __result;
  }
}

// __uninitialized_allocator_relocate_backward:
//
// Equivalent to __uninitialized_relocate_backward, but uses the provided allocator's construct() and destroy() methods.
template <class _Alloc, class _BidirectionalIter, class _NothrowBidirectionalIter>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _NothrowBidirectionalIter
__uninitialized_allocator_relocate_backward(
    _Alloc& __alloc, _BidirectionalIter __first, _BidirectionalIter __last, _NothrowBidirectionalIter __result_last) {
#ifndef _LIBCPP_CXX03_LANG
  if constexpr (__libcpp_is_contiguous_iterator<_BidirectionalIter>::value &&
                __libcpp_is_contiguous_iterator<_NothrowBidirectionalIter>::value) {
    _LIBCPP_ASSERT_NON_OVERLAPPING_RANGES(
        !std::__is_pointer_in_range(
            std::__to_address(__first), std::__to_address(__last), std::__to_address(__result_last) - 1),
        "uninitialized_allocator_relocate_backward requires the end of the result not to overlap with the input range");
  }
#endif

  using _ValueType = typename iterator_traits<_BidirectionalIter>::value_type;
  if _LIBCPP_CONSTEXPR (__allocator_has_trivial_move_construct<_Alloc, _ValueType>::value &&
                        __allocator_has_trivial_destroy<_Alloc, _ValueType>::value) {
    (void)__alloc; // ignore the allocator
    return std::__uninitialized_relocate_backward(std::move(__first), std::move(__last), std::move(__result_last));
  } else {
    auto __result = __result_last;
    try {
      while (__last != __first) {
        --__last;
        --__result;
        std::__allocator_relocate_at(__alloc, std::addressof(*__result), std::addressof(*__last));
      }
    } catch (...) {
      std::__allocator_destroy(__alloc, __first, __last);
      std::__allocator_destroy(__alloc, __result, __result_last);
      throw;
    }

    return __result;
  }
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___MEMORY_ALLOCATOR_RELOCATION_H
