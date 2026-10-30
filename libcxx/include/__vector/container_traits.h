//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___VECTOR_CONTAINER_TRAITS_H
#define _LIBCPP___VECTOR_CONTAINER_TRAITS_H

#include <__config>
#include <__fwd/vector.h>
#include <__type_traits/container_traits.h>
#include <__type_traits/is_nothrow_assignable.h>
#include <__type_traits/is_nothrow_constructible.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Tp, class _Allocator>
struct __container_traits<vector<_Tp, _Allocator> > {
  // http://eel.is/c++draft/vector.modifiers#2
  //  If an exception is thrown other than by the copy constructor, move constructor, assignment operator, or move
  //  assignment operator of T or by any InputIterator operation, there are no effects. If an exception is thrown while
  //  inserting a single element at the end and T is Cpp17CopyInsertable or is_nothrow_move_constructible_v<T> is true,
  //  there are no effects. Otherwise, if an exception is thrown by the move constructor of a non-Cpp17CopyInsertable T,
  //  the effects are unspecified.
  //
  // The "no effects" guarantee for an insertion at the end does not cover an insertion in the middle, which
  // move-constructs and move-assigns existing elements. Emplacement at an arbitrary position therefore has the
  // strong guarantee only when neither of those operations can throw.
  static _LIBCPP_CONSTEXPR const bool __emplacement_has_strong_exception_safety_guarantee =
      is_nothrow_move_constructible<_Tp>::value && is_nothrow_move_assignable<_Tp>::value;

  static _LIBCPP_CONSTEXPR const bool __reservable = true;
};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___VECTOR_CONTAINER_TRAITS_H
