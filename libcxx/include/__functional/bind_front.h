// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___FUNCTIONAL_BIND_FRONT_H
#define _LIBCPP___FUNCTIONAL_BIND_FRONT_H

#include <__config>
#include <__functional/invoke.h>
#include <__functional/perfect_forward.h>
#include <__type_traits/conjunction.h>
#include <__type_traits/decay.h>
#include <__type_traits/enable_if.h>
#include <__type_traits/is_constructible.h>
#include <__type_traits/is_member_pointer.h>
#include <__type_traits/is_pointer.h>
#include <__utility/forward.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 20

struct __bind_front_op {
  template <class... _Args>
  _LIBCPP_HIDE_FROM_ABI constexpr auto operator()(_Args&&... __args) const noexcept(
      noexcept(std::invoke(std::forward<_Args>(__args)...))) -> decltype(std::invoke(std::forward<_Args>(__args)...)) {
    return std::invoke(std::forward<_Args>(__args)...);
  }
};

template <class _Fn, class... _BoundArgs>
struct __bind_front_t : __perfect_forward<__bind_front_op, _Fn, _BoundArgs...> {
  using __perfect_forward<__bind_front_op, _Fn, _BoundArgs...>::__perfect_forward;
};

template <class _Fn, class... _Args>
  requires is_constructible_v<decay_t<_Fn>, _Fn> && is_move_constructible_v<decay_t<_Fn>> &&
           (is_constructible_v<decay_t<_Args>, _Args> && ...) && (is_move_constructible_v<decay_t<_Args>> && ...)
_LIBCPP_HIDE_FROM_ABI constexpr auto bind_front(_Fn&& __f, _Args&&... __args) {
  return __bind_front_t<decay_t<_Fn>, decay_t<_Args>...>(std::forward<_Fn>(__f), std::forward<_Args>(__args)...);
}

#endif // _LIBCPP_STD_VER >= 20

#if _LIBCPP_STD_VER >= 26

template <auto _Fn, class _Indices, class... _BoundArgs>
struct __nttp_bind_front_t;

template <auto _Fn, size_t... _Indices, class... _BoundArgs>
struct __nttp_bind_front_t<_Fn, index_sequence<_Indices...>, _BoundArgs...> {
  tuple<_BoundArgs...> __bound_args_;

  template <class _Self, class... _Args>
  _LIBCPP_HIDE_FROM_ABI constexpr auto operator()(this _Self&& __self, _Args&&... __args) noexcept(noexcept(std::invoke(
      _Fn, std::get<_Indices>(std::forward<_Self>(__self).__bound_args_)..., std::forward<_Args>(__args)...)))
      -> decltype(std::invoke(
          _Fn, std::get<_Indices>(std::forward<_Self>(__self).__bound_args_)..., std::forward<_Args>(__args)...)) {
    return std::invoke(
        _Fn, std::get<_Indices>(std::forward<_Self>(__self).__bound_args_)..., std::forward<_Args>(__args)...);
  }
};

template <auto _Fn>
struct __nttp_bind_without_bound_args_t {
  template <class... _Args>
  _LIBCPP_HIDE_FROM_ABI static constexpr auto
  operator()(_Args&&... __args) noexcept(noexcept(std::invoke(_Fn, std::forward<_Args>(__args)...)))
      -> decltype(std::invoke(_Fn, std::forward<_Args>(__args)...)) {
    return std::invoke(_Fn, std::forward<_Args>(__args)...);
  }
};

template <auto _Fn, class... _Args>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto bind_front(_Args&&... __args) {
  static_assert((is_constructible_v<decay_t<_Args>, _Args> && ...),
                "bind_front requires all decay_t<Args> to be constructible from respective Args");
  static_assert((is_move_constructible_v<decay_t<_Args>> && ...),
                "bind_front requires all decay_t<Args> to be move constructible");
  if constexpr (using _Ty = decltype(_Fn); is_pointer_v<_Ty> || is_member_pointer_v<_Ty>)
    static_assert(_Fn != nullptr, "bind_front: f cannot be equal to nullptr");

  if constexpr (sizeof...(_Args) == 0)
    return __nttp_bind_without_bound_args_t<_Fn>{};
  else
    return __nttp_bind_front_t<_Fn, index_sequence_for<_Args...>, decay_t<_Args>...>{
        .__bound_args_{std::forward<_Args>(__args)...}};
}

#endif // _LIBCPP_STD_VER >= 26

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___FUNCTIONAL_BIND_FRONT_H
