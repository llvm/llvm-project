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
#include <__memory/allocator_traits.h>
#include <__memory/is_trivially_allocator_relocatable.h>
#include <__memory/pointer_traits.h>
#include <__memory/relocate_at.h>
#include <__type_traits/is_constant_evaluated.h>
#include <__type_traits/is_nothrow_constructible.h>
#include <__type_traits/is_trivially_relocatable.h>
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
// except that the move constructor and destructor may never be called if they are known to be equivalent to a memcpy.
//
// This algorithm works even if part of the resulting range overlaps with [__first, __last), as long as __result itself
// is not in [__first, last).
//
// This algorithm doesn't throw any exceptions. However, it requires the types in the range to be nothrow
// move-constructible and the iterator operations not to throw any exceptions.
//
// Preconditions:
//  - __result doesn't contain any objects and [__first, __last) contains objects
//  - __result is not in [__first, __last)
// Postconditions: __result contains the objects from [__first, __last) and
//                 [__first, __last) doesn't contain any objects
template <class _ContiguousIterator>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 _ContiguousIterator __uninitialized_relocate(
    _ContiguousIterator __first, _ContiguousIterator __last, _ContiguousIterator __result) _NOEXCEPT {
  using _ValueType = typename iterator_traits<_ContiguousIterator>::value_type;
  static_assert(__libcpp_is_contiguous_iterator<_ContiguousIterator>::value, "");
  static_assert(is_nothrow_move_constructible<_ValueType>::value, "");
  if (!__libcpp_is_constant_evaluated() && __libcpp_is_trivially_relocatable<_ValueType>::value) {
    auto const __n = __last - __first;
    // Casting to void* to suppress clang complaining that this is technically UB.
    __builtin_memmove(
        static_cast<void*>(std::__to_address(__result)), std::__to_address(__first), sizeof(_ValueType) * __n);
    return __result + __n;
  } else {
    while (__first != __last) {
      std::__relocate_at(std::__to_address(__first), std::__to_address(__result));
      ++__first;
      ++__result;
    }
    return __result;
  }
}

// Equivalent to __uninitialized_relocate, but uses the provided allocator's construct() and destroy() methods.
template <class _Alloc, class _ContiguousIterator>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 _ContiguousIterator __uninitialized_allocator_relocate(
    _Alloc& __alloc, _ContiguousIterator __first, _ContiguousIterator __last, _ContiguousIterator __result) _NOEXCEPT {
  using _ValueType = typename iterator_traits<_ContiguousIterator>::value_type;
  static_assert(__libcpp_is_contiguous_iterator<_ContiguousIterator>::value, "");
  static_assert(is_nothrow_move_constructible<_ValueType>::value, "");
  if (!__libcpp_is_constant_evaluated() && __is_trivially_allocator_relocatable<_Alloc, _ValueType>::value) {
    (void)__alloc; // ignore the allocator
    return std::__uninitialized_relocate(std::move(__first), std::move(__last), std::move(__result));
  } else {
    while (__first != __last) {
      std::__allocator_relocate_at(__alloc, std::__to_address(__first), std::__to_address(__result));
      ++__first;
      ++__result;
    }
    return __result;
  }
}

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___MEMORY_UNINITIALIZED_RELOCATE_H
