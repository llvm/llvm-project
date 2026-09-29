//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___OPTIONAL_MAKE_OPTIONAL_H
#define _LIBCPP___OPTIONAL_MAKE_OPTIONAL_H

#include <__config>
#include <__fwd/optional.h>
#include <__type_traits/decay.h>
#include <__type_traits/enable_if.h>
#include <__type_traits/integral_constant.h>
#include <__type_traits/is_constructible.h>
#include <__type_traits/reference_constructs_from_temporary.h>
#include <__utility/forward.h>
#include <__utility/in_place.h>

#include <initializer_list>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD

struct __make_optional_barrier_tag {
  explicit __make_optional_barrier_tag() = default;
};

template <class _Tp, class... _Args>
inline constexpr bool __is_constructible_for_optional_v = is_constructible_v<_Tp, _Args...>;

template <class _Tp, class... _Args>
struct __is_constructible_for_optional : bool_constant<__is_constructible_for_optional_v<_Tp, _Args...>> {};

template <class _Tp, class _Up, class... _Args>
inline constexpr bool __is_constructible_for_optional_initializer_list_v =
    is_constructible_v<_Tp, initializer_list<_Up>&, _Args...>;

#  if _LIBCPP_STD_VER >= 26
template <class _Tp, class... _Args>
inline constexpr bool __is_constructible_for_optional_v<_Tp&, _Args...> = false;

template <class _Tp, class _Arg>
inline constexpr bool __is_constructible_for_optional_v<_Tp&, _Arg> =
    is_constructible_v<_Tp&, _Arg> && !reference_constructs_from_temporary_v<_Tp&, _Arg>;

template <class _Tp, class _Up, class... _Args>
inline constexpr bool __is_constructible_for_optional_initializer_list_v<_Tp&, _Up, _Args...> = false;
#  endif

template <
#  if _LIBCPP_STD_VER >= 26
    __make_optional_barrier_tag = __make_optional_barrier_tag{},
#  endif
    class _Tp,
    enable_if_t<is_constructible_v<decay_t<_Tp>, _Tp>, int> = 0>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr optional<decay_t<_Tp>> make_optional(_Tp&& __v) {
  return optional<decay_t<_Tp>>(std::forward<_Tp>(__v));
}

template <class _Tp, class... _Args, enable_if_t<__is_constructible_for_optional_v<_Tp, _Args...>, int> = 0>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr optional<_Tp> make_optional(_Args&&... __args) {
  return optional<_Tp>(in_place, std::forward<_Args>(__args)...);
}

template <class _Tp,
          class _Up,
          class... _Args,
          enable_if_t<__is_constructible_for_optional_initializer_list_v<_Tp, _Up, _Args...>, int> = 0>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr optional<_Tp>
make_optional(initializer_list<_Up> __il, _Args&&... __args) {
  return optional<_Tp>(in_place, __il, std::forward<_Args>(__args)...);
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 17

_LIBCPP_POP_MACROS

#endif // _LIBCPP___OPTIONAL_MAKE_OPTIONAL_H
