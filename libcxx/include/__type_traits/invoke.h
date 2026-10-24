// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___TYPE_TRAITS_INVOKE_H
#define _LIBCPP___TYPE_TRAITS_INVOKE_H

#include <__config>
#include <__type_traits/conditional.h>
#include <__type_traits/decay.h>
#include <__type_traits/enable_if.h>
#include <__type_traits/integral_constant.h>
#include <__type_traits/is_base_of.h>
#include <__type_traits/is_core_convertible.h>
#include <__type_traits/is_member_pointer.h>
#include <__type_traits/is_reference_wrapper.h>
#include <__type_traits/is_same.h>
#include <__type_traits/is_void.h>
#include <__type_traits/nat.h>
#include <__type_traits/remove_cvref.h>
#include <__type_traits/void_t.h>
#include <__utility/declval.h>
#include <__utility/forward.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

// This file defines the following libc++-internal API (back-ported to C++03):
//
// template <class... Args>
// decltype(auto) __invoke(Args&&... args) noexcept(noexcept(std::invoke(std::forward<Args>(args...)))) {
//   return std::invoke(std::forward<Args>(args)...);
// }
//
// template <class Ret, class... Args>
// Ret __invoke_r(Args&&... args) {
//   return std::invoke_r(std::forward<Args>(args)...);
// }
//
// template <class Func, class... Args>
// struct __is_invocable : is_invocable<Func, Args...> {};
//
// template <class Func, class... Args>
// inline const bool __is_invocable_v = is_invocable_v<Func, Args...>;
//
// template <class Ret, class Func, class... Args>
// inline const bool __is_invocable_r_v = is_invocable_r_v<Ret, Func, Args...>;
//
// template <class Func, class... Args>
// inline const bool __is_nothrow_invocable_v = is_nothrow_invocable_v<Func, Args...>;
//
// template <class Func, class... Args>
// inline const bool __is_nothrow_invocable_r_v = is_nothrow_invocable_r_v<Func, Args...>;
//
// template <class Func, class... Args>
// struct __invoke_result : invoke_result {};
//
// template <class Func, class... Args>
// using __invoke_result_t = invoke_result_t<Func, Args...>;
//
// template <class Ret, class Func, class... Args>
// struct __is_invocable_r : is_invocable_r<Ret, Func, Args...> {};

_LIBCPP_BEGIN_NAMESPACE_STD

#if __has_builtin(__builtin_invoke)

template <class, class... _Args>
struct __invoke_result_impl {};

template <class... _Args>
struct __invoke_result_impl<__void_t<decltype(__builtin_invoke(std::declval<_Args>()...))>, _Args...> {
  using type _LIBCPP_NODEBUG = decltype(__builtin_invoke(std::declval<_Args>()...));
};

template <class... _Args>
using __invoke_result _LIBCPP_NODEBUG = __invoke_result_impl<void, _Args...>;

template <class... _Args>
using __invoke_result_t _LIBCPP_NODEBUG = typename __invoke_result<_Args...>::type;

template <class... _Args>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR __invoke_result_t<_Args...> __invoke(_Args&&... __args)
    _NOEXCEPT_(noexcept(__builtin_invoke(std::forward<_Args>(__args)...))) {
  return __builtin_invoke(std::forward<_Args>(__args)...);
}

template <class _Void, class... _Args>
inline const bool __is_invocable_impl = false;

template <class... _Args>
inline const bool __is_invocable_impl<__void_t<__invoke_result_t<_Args...> >, _Args...> = true;

template <class... _Args>
inline const bool __is_invocable_v = __is_invocable_impl<void, _Args...>;

template <class... _Args>
struct __is_invocable : integral_constant<bool, __is_invocable_v<_Args...> > {};

template <bool __is_invocable, class... _Args>
inline const bool __is_nothrow_invocable_impl = false;

#  ifndef _LIBCPP_CXX03_LANG
template <class... _Args>
inline const bool __is_nothrow_invocable_impl<true, _Args...> = noexcept(__builtin_invoke(std::declval<_Args>()...));
#  endif

template <class... _Args>
inline const bool __is_nothrow_invocable_v = __is_nothrow_invocable_impl<__is_invocable_v<_Args...>, _Args...>;

#else // __has_builtin(__builtin_invoke)

template <class _DecayedFp>
struct __member_pointer_class_type {};

template <class _Ret, class _ClassType>
struct __member_pointer_class_type<_Ret _ClassType::*> {
  typedef _ClassType type;
};

template <class _Func, class... _Args>
inline const bool __is_invocable_v = __is_invocable(_Func, _Args...);

template <class _Func, class... _Args>
inline const bool __is_nothrow_invocable_v = __is_nothrow_invocable(_Func, _Args...);

template <class _Func, class... _Args, class = __enable_if_t<__is_invocable_v<_Func, _Args...>>>
constexpr decltype(auto)
__invoke(_Func&& __func, _Args&&... __args) noexcept(__is_nothrow_invocable_v<_Func, _Args...>) {
  using _RawFunc = __remove_cvref_t<_Func>;
  if constexpr (is_member_pointer<_RawFunc>::value) {
    using _Tp    = typename __member_pointer_class_type<_RawFunc>::type;
    using _T1    = _Args...[0];
    using _RawT1 = __remove_cvref_t<_T1>;
    if constexpr (is_member_function_pointer<_RawFunc>::value) {
      if constexpr (is_same<_Tp, __remove_cvref_t<_T1>>::value || is_base_of<_Tp, __remove_cvref_t<_T1>>::value) {
        return []<class _Func2, class _Arg0, class... _Args2>(
                   _Func2&& __func2, _Arg0&& __arg0, _Args2&&... __args2) -> decltype(auto) {
          return (std::forward<_Arg0>(__arg0).*__func2)(std::forward<_Args2>(__args2)...);
        }(std::forward<_Func>(__func), std::forward<_Args>(__args)...);
      } else if constexpr (__is_reference_wrapper<_RawT1>::value) {
        return []<class _Func2, class _Arg0, class... _Args2>(
                   _Func2&& __func2, _Arg0&& __arg0, _Args2&&... __args2) -> decltype(auto) {
          return ((std::forward<_Arg0>(__arg0).get()).*__func2)(std::forward<_Args2>(__args2)...);
        }(std::forward<_Func>(__func), std::forward<_Args>(__args)...);
      } else {
        return []<class _Func2, class _Arg0, class... _Args2>(
                   _Func2&& __func2, _Arg0&& __arg0, _Args2&&... __args2) -> decltype(auto) {
          return ((*std::forward<_Arg0>(__arg0)).*__func2)(std::forward<_Args2>(__args2)...);
        }(std::forward<_Func>(__func), std::forward<_Args>(__args)...);
      }
    } else {
      if constexpr (is_same<_Tp, __remove_cvref_t<_T1>>::value || is_base_of<_Tp, __remove_cvref_t<_T1>>::value) {
        return std::forward<_Args...[0]>(__args...[0]).*__func;
      } else if constexpr (__is_reference_wrapper<_RawT1>::value) {
        return std::forward<_Args...[0]>(__args...[0]).get().*__func;
      } else {
        return (*std::forward<_Args...[0]>(__args...[0])).*__func;
      }
    }
  } else {
    return std::forward<_Func>(__func)(std::forward<_Args>(__args)...);
  }
}

template <class... _Args>
using __invoke_result_t = decltype(std::__invoke(std::declval<_Args>()...));

template <bool _Invocable, class _Func, class... _Args>
struct __invoke_result_impl {};

template <class _Func, class... _Args>
struct __invoke_result_impl<true, _Func, _Args...> {
  using type = __invoke_result_t<_Func, _Args...>;
};

template <class _Func, class... _Args>
using __invoke_result = __invoke_result_impl<__is_invocable_v<_Func, _Args...>, _Func, _Args...>;

#endif // __has_builtin(__builtin_invoke_r)

template <class _Ret, bool, class... _Args>
inline const bool __is_invocable_r_impl = false;

template <class _Ret, class... _Args>
inline const bool __is_invocable_r_impl<_Ret, true, _Args...> =
    __is_core_convertible_v<__invoke_result_t<_Args...>, _Ret> || is_void<_Ret>::value;

template <class _Ret, class... _Args>
inline const bool __is_invocable_r_v = __is_invocable_r_impl<_Ret, __is_invocable_v<_Args...>, _Args...>;

template <bool __is_invocable, class _Ret, class... _Args>
inline const bool __is_nothrow_invocable_r_impl = false;

template <class _Ret, class... _Args>
inline const bool __is_nothrow_invocable_r_impl<true, _Ret, _Args...> =
    __is_nothrow_core_convertible_v<__invoke_result_t<_Args...>, _Ret> || is_void<_Ret>::value;

template <class _Ret, class... _Args>
inline const bool __is_nothrow_invocable_r_v =
    __is_nothrow_invocable_r_impl<__is_nothrow_invocable_v<_Args...>, _Ret, _Args...>;

template <class _Ret, class _Func, class... _Args>
struct __is_invocable_r : integral_constant<bool, __is_invocable_r_v<_Ret, _Func, _Args...> > {};

template <class _Ret, bool = is_void<_Ret>::value>
struct __invoke_void_return_wrapper {
  template <class... _Args>
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 static _Ret __call(_Args&&... __args) {
    return std::__invoke(std::forward<_Args>(__args)...);
  }
};

template <class _Ret>
struct __invoke_void_return_wrapper<_Ret, true> {
  template <class... _Args>
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 static void __call(_Args&&... __args) {
    std::__invoke(std::forward<_Args>(__args)...);
  }
};

template <class _Ret, class... _Args>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _Ret __invoke_r(_Args&&... __args) {
  return __invoke_void_return_wrapper<_Ret>::__call(std::forward<_Args>(__args)...);
}

#if _LIBCPP_STD_VER >= 17

// is_invocable

template <class _Fn, class... _Args>
struct _LIBCPP_NO_SPECIALIZATIONS is_invocable : bool_constant<__is_invocable_v<_Fn, _Args...> > {};

template <class _Ret, class _Fn, class... _Args>
struct _LIBCPP_NO_SPECIALIZATIONS is_invocable_r : bool_constant<__is_invocable_r_v<_Ret, _Fn, _Args...>> {};

template <class _Fn, class... _Args>
_LIBCPP_NO_SPECIALIZATIONS inline constexpr bool is_invocable_v = __is_invocable_v<_Fn, _Args...>;

template <class _Ret, class _Fn, class... _Args>
_LIBCPP_NO_SPECIALIZATIONS inline constexpr bool is_invocable_r_v = __is_invocable_r_v<_Ret, _Fn, _Args...>;

// is_nothrow_invocable

template <class _Fn, class... _Args>
struct _LIBCPP_NO_SPECIALIZATIONS is_nothrow_invocable : bool_constant<__is_nothrow_invocable_v<_Fn, _Args...> > {};

template <class _Ret, class _Fn, class... _Args>
struct _LIBCPP_NO_SPECIALIZATIONS is_nothrow_invocable_r
    : bool_constant<__is_nothrow_invocable_r_v<_Ret, _Fn, _Args...>> {};

template <class _Fn, class... _Args>
_LIBCPP_NO_SPECIALIZATIONS inline constexpr bool is_nothrow_invocable_v = __is_nothrow_invocable_v<_Fn, _Args...>;

template <class _Ret, class _Fn, class... _Args>
_LIBCPP_NO_SPECIALIZATIONS inline constexpr bool is_nothrow_invocable_r_v =
    __is_nothrow_invocable_r_v<_Ret, _Fn, _Args...>;

template <class _Fn, class... _Args>
struct _LIBCPP_NO_SPECIALIZATIONS invoke_result : __invoke_result<_Fn, _Args...> {};

template <class _Fn, class... _Args>
using invoke_result_t = __invoke_result_t<_Fn, _Args...>;

#endif

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___TYPE_TRAITS_INVOKE_H
