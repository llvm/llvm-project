//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___MEMORY_UNINITIALIZED_RELOCATE_H
#define _LIBCPP___MEMORY_UNINITIALIZED_RELOCATE_H

#include <__config>
#include <__iterator/iterator_traits.h>
#include <__memory/addressof.h>
#include <__memory/allocator_traits.h>
#include <__memory/destroy.h>
#include <__memory/is_trivially_allocator_relocatable.h>
#include <__memory/pointer_traits.h>
#include <__memory/relocate_at.h>
#include <__type_traits/is_constant_evaluated.h>
#include <__type_traits/is_nothrow_constructible.h>
#include <__type_traits/is_trivially_relocatable.h>
#include <__utility/is_pointer_in_range.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

// __uninitialized_relocate relocates the objects in [__first, __last) into __result.
//
// Relocation means that the objects in [__first, __last) are placed into __result as-if by move-construct and destroy,
// except that the move constructor and destructor may never be called if they are known to be trivially relocatable.
//
// This algorithm works even if part of the resulting range overlaps with [__first, __last), as long as __result itself
// is not in [__first, last).
//
// If an exception is thrown, all the elements in the input range and in the output range are destroyed.
//
// Preconditions:
//  - __result doesn't contain any objects and [__first, __last) contains objects
//  - __result is not in [__first, __last)
// Postconditions:
//  - __result contains the objects from [__first, __last)
//  - [__first, __last) doesn't contain any objects
template <class _InputIter, class _NothrowForwardIter>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _NothrowForwardIter
__uninitialized_relocate(_InputIter __first, _InputIter __last, _NothrowForwardIter __result) {
  if constexpr (__libcpp_is_contiguous_iterator<_InputIter>::value &&
                __libcpp_is_contiguous_iterator<_NothrowForwardIter>::value) {
    _LIBCPP_ASSERT_NON_OVERLAPPING_RANGES(
        !std::__is_pointer_in_range(std::__to_address(__first), std::__to_address(__last), std::__to_address(__result)),
        "uninitialized_relocate requires the start of the result not to overlap with the input range");
  }
  using _ValueType = typename iterator_traits<_InputIter>::value_type;

  // If we have contiguous iterators over a trivially relocatable type, use the builtin that is
  // roughly equivalent to memmove.
  if constexpr (__libcpp_is_contiguous_iterator<_InputIter>::value &&
                __libcpp_is_contiguous_iterator<_NothrowForwardIter>::value &&
                __libcpp_is_trivially_relocatable<_ValueType>::value) {
    if (!__libcpp_is_constant_evaluated()) {
      // TODO: We might be able to memcpy if we don't overlap at all?
      std::__libcpp_builtin_trivially_relocate_at(
          std::__to_address(__first), std::__to_address(__last), std::__to_address(__result));
      return __result + (__last - __first);
    }
  }

  // Otherwise, relocate elements one by one.
  //
  // If this throws an exception, we destroy the rest of the range we were relocating, and
  // we also destroy everything we had relocated up to now.
  auto const __first_result = __result;
  try {
    while (__first != __last) {
      std::__relocate_at(std::addressof(*__result), std::addressof(*__first));
      ++__first;
      ++__result;
    }
  } catch (...) {
    std::__destroy(++__first, __last);
    std::__destroy(__first_result, __result);
    throw;
  }
  return __result;
}

// __uninitialized_relocate_backward is like __uninitialized_relocate, but it relocates the elements in
// the range [first, last) to another range ending at __result_last. The elements are relocated in reverse
// order, but their relative order is preserved.
template <class _BidirectionalIter, class _NothrowBidirectionalIter>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _NothrowBidirectionalIter __uninitialized_relocate_backward(
    _BidirectionalIter __first, _BidirectionalIter __last, _NothrowBidirectionalIter __result_last) {
  if constexpr (__libcpp_is_contiguous_iterator<_BidirectionalIter>::value &&
                __libcpp_is_contiguous_iterator<_NothrowBidirectionalIter>::value) {
    _LIBCPP_ASSERT_NON_OVERLAPPING_RANGES(
        !std::__is_pointer_in_range(
            std::__to_address(__first), std::__to_address(__last), std::__to_address(__result_last) - 1),
        "uninitialized_relocate_backward requires the end of the result not to overlap with the input range");
  }
  using _ValueType = typename iterator_traits<_BidirectionalIter>::value_type;

  // If we have contiguous iterators over a trivially relocatable type, use the builtin that is
  // roughly equivalent to memmove.
  if constexpr (__libcpp_is_contiguous_iterator<_BidirectionalIter>::value &&
                __libcpp_is_contiguous_iterator<_NothrowBidirectionalIter>::value &&
                __libcpp_is_trivially_relocatable<_ValueType>::value) {
    if (!__libcpp_is_constant_evaluated()) {
      auto __result = __result_last - (__last - __first);
      // TODO: We might be able to memcpy if we don't overlap at all?
      std::__libcpp_builtin_trivially_relocate_at(
          std::__to_address(__first), std::__to_address(__last), std::__to_address(__result));
      return __result;
    }
  }

  // Otherwise, relocate elements one by one, starting from the end.
  //
  // If this throws an exception, we destroy the rest of the range we were relocating, and
  // we also destroy everything we had relocated up to now.
  auto __result = __result_last;
  try {
    while (__last != __first) {
      --__last;
      --__result;
      std::__relocate_at(std::addressof(*__result), std::addressof(*__last));
    }
  } catch (...) {
    std::__destroy(__first, __last);
    std::__destroy(__result, __result_last);
    throw;
  }
  return __result;
}

// Equivalent to __uninitialized_relocate, but uses the provided allocator's construct() and destroy() methods.
template <class _Alloc, class _InputIter, class _NothrowForwardIter>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _NothrowForwardIter __uninitialized_allocator_relocate(
    _Alloc& __alloc, _InputIter __first, _InputIter __last, _NothrowForwardIter __result) {
  if constexpr (__libcpp_is_contiguous_iterator<_InputIter>::value &&
                __libcpp_is_contiguous_iterator<_NothrowForwardIter>::value) {
    _LIBCPP_ASSERT_NON_OVERLAPPING_RANGES(
        !std::__is_pointer_in_range(std::__to_address(__first), std::__to_address(__last), std::__to_address(__last)),
        "uninitialized_allocator_relocate requires the result not to overlap with the input range");
  }

  using _ValueType = typename iterator_traits<_InputIter>::value_type;
  if (__allocator_has_trivial_move_construct<_Alloc, _ValueType>::value &&
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

// Equivalent to __uninitialized_relocate_backward, but uses the provided allocator's construct() and destroy() methods.
template <class _Alloc, class _BidirectionalIter, class _NothrowBidirectionalIter>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _NothrowBidirectionalIter
__uninitialized_allocator_relocate_backward(
    _Alloc& __alloc, _BidirectionalIter __first, _BidirectionalIter __last, _NothrowBidirectionalIter __result_last) {
  if constexpr (__libcpp_is_contiguous_iterator<_BidirectionalIter>::value &&
                __libcpp_is_contiguous_iterator<_NothrowBidirectionalIter>::value) {
    _LIBCPP_ASSERT_NON_OVERLAPPING_RANGES(
        !std::__is_pointer_in_range(
            std::__to_address(__first), std::__to_address(__last), std::__to_address(__result_last) - 1),
        "uninitialized_allocator_relocate_backward requires the end of the result not to overlap with the input range");
  }

  using _ValueType = typename iterator_traits<_BidirectionalIter>::value_type;
  if (__allocator_has_trivial_move_construct<_Alloc, _ValueType>::value &&
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

_LIBCPP_POP_MACROS

#endif // _LIBCPP___MEMORY_UNINITIALIZED_RELOCATE_H
