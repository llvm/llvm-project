// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CXX03___MEMORY_CONSTRUCT_AT_H
#define _LIBCPP___CXX03___MEMORY_CONSTRUCT_AT_H

#include <__cxx03/__assert>
#include <__cxx03/__config>
#include <__cxx03/__iterator/access.h>
#include <__cxx03/__memory/addressof.h>
#include <__cxx03/__memory/voidify.h>
#include <__cxx03/__type_traits/enable_if.h>
#include <__cxx03/__type_traits/is_array.h>
#include <__cxx03/__utility/declval.h>
#include <__cxx03/__utility/forward.h>
#include <__cxx03/__utility/move.h>
#include <__cxx03/new>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__cxx03/__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

// construct_at

template <class _Tp, class... _Args, class = decltype(::new(std::declval<void*>()) _Tp(std::declval<_Args>()...))>
_LIBCPP_HIDE_FROM_ABI _Tp* __construct_at(_Tp* __location, _Args&&... __args) {
  return _LIBCPP_ASSERT_NON_NULL(__location != nullptr, "null pointer given to construct_at"),
         ::new (std::__voidify(*__location)) _Tp(std::forward<_Args>(__args)...);
}

// destroy_at

// The internal functions are available regardless of the language version (with the exception of the `__destroy_at`
// taking an array).

template <class _ForwardIterator>
_LIBCPP_HIDE_FROM_ABI _ForwardIterator __destroy(_ForwardIterator, _ForwardIterator);

template <class _Tp, __enable_if_t<!is_array<_Tp>::value, int> = 0>
_LIBCPP_HIDE_FROM_ABI void __destroy_at(_Tp* __loc) {
  _LIBCPP_ASSERT_NON_NULL(__loc != nullptr, "null pointer given to destroy_at");
  __loc->~_Tp();
}

template <class _ForwardIterator>
_LIBCPP_HIDE_FROM_ABI _ForwardIterator __destroy(_ForwardIterator __first, _ForwardIterator __last) {
  for (; __first != __last; ++__first)
    std::__destroy_at(std::addressof(*__first));
  return __first;
}

template <class _BidirectionalIterator>
_LIBCPP_HIDE_FROM_ABI _BidirectionalIterator
__reverse_destroy(_BidirectionalIterator __first, _BidirectionalIterator __last) {
  while (__last != __first) {
    --__last;
    std::__destroy_at(std::addressof(*__last));
  }
  return __last;
}

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___CXX03___MEMORY_CONSTRUCT_AT_H
