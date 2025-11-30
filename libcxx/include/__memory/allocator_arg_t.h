// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___MEMORY_ALLOCATOR_ARG_T_H
#define _LIBCPP___MEMORY_ALLOCATOR_ARG_T_H

#include <__config>
#include <__memory/uses_allocator.h>
#include <__type_traits/integral_constant.h>
#include <__type_traits/is_constructible.h>
#include <__type_traits/remove_cv.h>
#include <__type_traits/remove_cvref.h>
#include <__utility/forward.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

struct allocator_arg_t {
  explicit allocator_arg_t() = default;
};

#if _LIBCPP_STD_VER >= 17
inline constexpr allocator_arg_t allocator_arg = allocator_arg_t();
#elif !defined(_LIBCPP_CXX03_LANG)
constexpr allocator_arg_t allocator_arg = allocator_arg_t();
#endif

#ifndef _LIBCPP_CXX03_LANG

// allocator construction

template <class _Tp, class _Alloc, class... _Args>
struct __uses_alloc_ctor_imp {
  using _RawAlloc _LIBCPP_NODEBUG = __remove_cvref_t<_Alloc>;
  static constexpr bool __ua      = uses_allocator<__remove_cv_t<_Tp>, _RawAlloc>::value;
  static constexpr bool __ic_head = is_constructible<_Tp, allocator_arg_t, const _RawAlloc&, _Args...>::value;
  static constexpr bool __ic_tail = is_constructible<_Tp, _Args..., const _RawAlloc&>::value;
  static constexpr int value      = __ua ? (__ic_head ? 1 : __ic_tail ? 2 : -1) : 0;
};

template <class _Tp, class _Alloc, class... _Args>
struct __uses_alloc_ctor : integral_constant<int, __uses_alloc_ctor_imp<_Tp, _Alloc, _Args...>::value> {};

#endif // _LIBCPP_CXX03_LANG

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___MEMORY_ALLOCATOR_ARG_T_H
