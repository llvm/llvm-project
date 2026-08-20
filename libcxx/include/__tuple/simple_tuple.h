//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___TUPLE_SIMPLE_TUPLE_H
#define _LIBCPP___TUPLE_SIMPLE_TUPLE_H

#include <__config>
#include <__cstddef/size_t.h>
#include <__type_traits/conditional.h>
#include <__type_traits/copy_cvref.h>
#include <__type_traits/disjunction.h>
#include <__type_traits/enable_if.h>
#include <__type_traits/integral_constant.h>
#include <__type_traits/invoke.h>
#include <__type_traits/is_empty.h>
#include <__type_traits/is_final.h>
#include <__type_traits/is_reference.h>
#include <__type_traits/is_same.h>
#include <__type_traits/remove_cvref.h>
#include <__utility/forward.h>
#include <__utility/integer_sequence.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#ifndef _LIBCPP_CXX03_LANG

_LIBCPP_BEGIN_NAMESPACE_STD

template <size_t, class _Tp, bool = is_empty<_Tp>::value && !__is_final_v<_Tp>>
struct __simple_tuple_base {
  _Tp __val_;

  __simple_tuple_base() = default;

  template <class _Arg, __enable_if_t<_IsNotSame<__remove_cvref_t<_Arg>, __simple_tuple_base>::value, int> = 0>
  constexpr __simple_tuple_base(_Arg&& __arg) : __val_(std::forward<_Arg>(__arg)) {}
};

template <size_t _Index, class _Tp>
struct __simple_tuple_base<_Index, _Tp, true> {
  _LIBCPP_NO_UNIQUE_ADDRESS _Tp __val_;

  __simple_tuple_base() = default;

  template <class _Arg, __enable_if_t<_IsNotSame<__remove_cvref_t<_Arg>, __simple_tuple_base>::value, int> = 0>
  constexpr __simple_tuple_base(_Arg&& __arg) : __val_(std::forward<_Arg>(__arg)) {}
};

template <class _IndexSequence, class... _Bases>
struct __simple_tuple_impl;

template <class _Ap, class _Bp>
using __tuple_forward _LIBCPP_NODEBUG = _If<is_reference<_Bp>::value, _Bp, __copy_cvref_t<_Ap, _Bp>>;

template <size_t... _Indices, class... _Types>
struct __simple_tuple_impl<__index_sequence<_Indices...>, _Types...> : __simple_tuple_base<_Indices, _Types>... {
  __simple_tuple_impl() = default;

  template <class... _Args,
            __enable_if_t<_Or<integral_constant<bool, (sizeof...(_Args) != 1)>,
                              _IsNotSame<__remove_cvref_t<_Args>, __simple_tuple_impl>...>::value,
                          int> = 0>
  constexpr __simple_tuple_impl(_Args&&... __args)
      : __simple_tuple_base<_Indices, _Types>(std::forward<_Args>(__args))... {}

  template <class _Self, class _Func, class... _Args>
  static constexpr auto __apply(_Self&& __self, _Func&& __func, _Args&&... __args) noexcept(
      __is_nothrow_invocable_v<_Func, __tuple_forward<_Self, _Types>..., _Args...>)
      -> __invoke_result_t<_Func, __tuple_forward<_Self, _Types>..., _Args...> {
    return std::__invoke(
        std::forward<_Func>(__func),
        static_cast<__tuple_forward<_Self, _Types>&&>(
            static_cast<__copy_cvref_t<_Self, __simple_tuple_base<_Indices, _Types>>&&>(__self).__val_)...,
        std::forward<_Args>(__args)...);
  }

  template <class _Self, class _Func, class... _Args>
  static constexpr auto __apply_back(_Self&& __self, _Func&& __func, _Args&&... __args) noexcept(
      __is_nothrow_invocable_v<_Func, _Args..., __tuple_forward<_Self, _Types>...>)
      -> __invoke_result_t<_Func, _Args..., __tuple_forward<_Self, _Types>...> {
    return std::__invoke(
        std::forward<_Func>(__func),
        std::forward<_Args>(__args)...,
        static_cast<__tuple_forward<_Self, _Types>&&>(
            static_cast<__copy_cvref_t<_Self, __simple_tuple_base<_Indices, _Types>>&&>(__self).__val_)...);
  }
};

template <class... _Args>
using __simple_tuple _LIBCPP_NODEBUG = __simple_tuple_impl<__index_sequence_for<_Args...>, _Args...>;

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_CXX03_LANG

_LIBCPP_POP_MACROS

#endif // _LIBCPP___TUPLE_SIMPLE_TUPLE_H
