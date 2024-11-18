//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___MEMORY_RELOCATE_AT_H
#define _LIBCPP___MEMORY_RELOCATE_AT_H

#include <__memory/allocator_traits.h>
#include <__memory/construct_at.h>
#include <__memory/is_trivially_allocator_relocatable.h>
#include <__type_traits/enable_if.h>
#include <__type_traits/is_constant_evaluated.h>
#include <__type_traits/is_trivially_relocatable.h>
#include <__utility/move.h>
#include <__utility/scope_guard.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Tp>
struct __destroy_object {
  _LIBCPP_CONSTEXPR_SINCE_CXX14 void operator()() const { std::__destroy_at(__obj_); }
  _Tp* __obj_;
};

template <class _Tp, __enable_if_t<!__libcpp_is_trivially_relocatable<_Tp>::value, int> = 0>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 _Tp* __relocate_at(_Tp* __source, _Tp* __dest) {
  __scope_guard<__destroy_object<_Tp> > __guard(__destroy_object<_Tp>{__source});
  return std::__construct_at(__dest, std::move(*__source));
}

template <class _Tp, __enable_if_t<__libcpp_is_trivially_relocatable<_Tp>::value, int> = 0>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 _Tp* __relocate_at(_Tp* __source, _Tp* __dest) {
  if (__libcpp_is_constant_evaluated()) {
    std::__construct_at(__dest, std::move(*__source));
  } else {
    // Casting to void* to suppress clang complaining that this is technically UB.
    __builtin_memcpy(static_cast<void*>(__dest), __source, sizeof(_Tp));
  }
  return __dest;
}

template <class _Alloc, class _Tp, __enable_if_t<!__is_trivially_allocator_relocatable<_Alloc, _Tp>::value, int> = 0>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 _Tp*
__allocator_relocate_at(_Alloc& __alloc, _Tp* __source, _Tp* __dest) {
  allocator_traits<_Alloc>::construct(__alloc, __dest, std::move(*__source));
  allocator_traits<_Alloc>::destroy(__alloc, __source);
  return __dest;
}

template <class _Alloc, class _Tp, __enable_if_t<__is_trivially_allocator_relocatable<_Alloc, _Tp>::value, int> = 0>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 _Tp* __allocator_relocate_at(_Alloc&, _Tp* __source, _Tp* __dest) {
  return std::__relocate_at(__source, __dest);
}

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___MEMORY_RELOCATE_AT_H
