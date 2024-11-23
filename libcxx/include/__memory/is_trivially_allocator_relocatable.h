//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___MEMORY_IS_TRIVIALLY_ALLOCATOR_RELOCATABLE_H
#define _LIBCPP___MEMORY_IS_TRIVIALLY_ALLOCATOR_RELOCATABLE_H

#include <__config>
#include <__memory/allocator_traits.h>
#include <__type_traits/conjunction.h>
#include <__type_traits/disjunction.h>
#include <__type_traits/integral_constant.h>
#include <__type_traits/is_nothrow_constructible.h>
#include <__type_traits/is_trivially_relocatable.h>
#include <__type_traits/negation.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

// A type is trivially allocator relocatable if the allocator's move construction and destruction
// don't do anything beyond calling the type's move constructor and destructor, and if the type
// itself is trivially relocatable.

template <class _Alloc, class _Type>
struct __allocator_has_trivial_move_construct : _Not<__has_construct<_Alloc, _Type*, _Type&&> > {};

template <class _Type>
struct __allocator_has_trivial_move_construct<allocator<_Type>, _Type> : true_type {};

template <class _Alloc, class _Tp>
struct __allocator_has_trivial_destroy : _Not<__has_destroy<_Alloc, _Tp*> > {};

template <class _Tp, class _Up>
struct __allocator_has_trivial_destroy<allocator<_Tp>, _Up> : true_type {};

template <class _Alloc, class _Tp>
struct __is_trivially_allocator_relocatable
    : integral_constant<bool,
                        __allocator_has_trivial_move_construct<_Alloc, _Tp>::value &&
                            __allocator_has_trivial_destroy<_Alloc, _Tp>::value &&
                            __libcpp_is_trivially_relocatable<_Tp>::value> {};

template <class _Alloc, class _Tp>
struct __is_nothrow_allocator_relocatable
    : _Or<_And<__allocator_has_trivial_move_construct<_Alloc, _Tp>, is_nothrow_move_constructible<_Tp>>,
          __is_trivially_allocator_relocatable<_Alloc, _Tp>> {};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___MEMORY_IS_TRIVIALLY_ALLOCATOR_RELOCATABLE_H
