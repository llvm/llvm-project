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
#include <__type_traits/is_function.h>
#include <__type_traits/is_reference_wrapper.h>
#include <__type_traits/is_same.h>
#include <__type_traits/is_void.h>
#include <__type_traits/nat.h>
#include <__type_traits/remove_cvref.h>
#include <__type_traits/remove_reference.h>
#include <__type_traits/type_identity.h>
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
// struct __is_nothrow_invocable_r : is_nothrow_invocable_r<Func, Args...> {};
//
// template <class Func, class... Args>
// struct __invoke_result : invoke_result<Func, Args...> {};
//
// template <class Func, class... Args>
// using __invoke_result_t = invoke_result_t<Func, Args...>;
//
// template <class Ret, class Func, class... Args>
// struct __is_invocable_r : is_invocable_r<Ret, Func, Args...> {};

_LIBCPP_BEGIN_NAMESPACE_STD

#if __has_builtin(__builtin_invoke)

template <class... _Args>
using __invoke_result_t _LIBCPP_NODEBUG =
    decltype(__builtin_invoke(reinterpret_cast<_Args&&>(static_cast<int&&>(0))...));

template <class _Void, class... _Args>
inline const bool __is_invocable_impl = false;

template <class... _Args>
inline const bool __is_invocable_impl<__void_t<__invoke_result_t<_Args...> >, _Args...> = true;

template <class... _Args>
inline const bool __is_invocable_v = __is_invocable_impl<void, _Args...>;

template <class... _Args>
using __is_invocable _LIBCPP_NODEBUG = _BoolConstant<__is_invocable_v<_Args...> >;

template <class _Ret, bool, class... _Args>
inline const bool __is_invocable_r_impl = false;

template <class _Ret, class... _Args>
inline const bool __is_invocable_r_impl<_Ret, true, _Args...> =
    __is_core_convertible<__invoke_result_t<_Args...>, _Ret>::value || is_void<_Ret>::value;

template <class _Ret, class... _Args>
_LIBCPP_NO_SPECIALIZATIONS inline const bool __is_invocable_r_v =
    __is_invocable_r_impl<_Ret, __is_invocable_v<_Args...>, _Args...>;

template <bool _Expected, class... _Args>
inline const bool __nothrow_invocability_matches_v = false;

#  ifndef _LIBCPP_CXX03_LANG
template <class... _Args>
inline const bool
    __nothrow_invocability_matches_v<noexcept(__builtin_invoke(reinterpret_cast<_Args&&>(static_cast<int&&>(0))...)),
                                     _Args...> = true;
#  endif

template <class... _Args>
_LIBCPP_NO_SPECIALIZATIONS inline const bool __is_nothrow_invocable_v =
    __nothrow_invocability_matches_v<true, _Args...>;

template <bool __is_invocable, class _Ret, class... _Args>
inline const bool __is_nothrow_invocable_r_impl = false;

template <class _Ret, class... _Args>
inline const bool __is_nothrow_invocable_r_impl<true, _Ret, _Args...> =
    __is_nothrow_core_convertible_v<__invoke_result_t<_Args...>, _Ret> || is_void<_Ret>::value;

template <class _Ret, class... _Args>
using __is_nothrow_invocable_r _LIBCPP_NODEBUG =
    _BoolConstant<__is_nothrow_invocable_r_impl<__is_nothrow_invocable_v<_Args...>, _Ret, _Args...> >;

template <class... _Args>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR __invoke_result_t<_Args...> __invoke(_Args&&... __args)
    _NOEXCEPT_(__is_nothrow_invocable_v<_Args...>) {
  return __builtin_invoke(static_cast<_Args&&>(__args)...);
}

#else // __has_builtin(__builtin_invoke)

// The _Kind is the bullet number in the standard
template <unsigned _Kind>
struct _Invoker;

template <class _Member,
          class _Class,
          class _A0,
          class _DecayA0      = __decay_t<_A0>,
          class _IsRefWrapper = __is_reference_wrapper<_DecayA0> >
using __invoker_t _LIBCPP_NODEBUG = _Invoker<
    (!is_function<_Member>::value * 3) +
    (!_If< _IsRefWrapper::value,
           false_type,
           _If<is_same<_Class, _DecayA0>::value, is_same<_Class, _DecayA0>, is_base_of<_Class, _DecayA0> > >::value
     << 1) +
    !_IsRefWrapper::value>;

// bullets 1, 2 and 3

template <>
struct _Invoker<1> {
  template <class _Fp, class _A0, class... _Args>
  static inline _LIBCPP_HIDE_FROM_ABI
  _LIBCPP_CONSTEXPR decltype((std::declval<_A0>().*std::declval<_Fp>())(std::declval<_Args>()...))
  _Fn(_Fp __f, _A0&& __a0, _Args&&... __args)
      _NOEXCEPT_(noexcept((static_cast<_A0&&>(__a0).*static_cast<_Fp&&>(__f))(static_cast<_Args&&>(__args)...))) {
    return (static_cast<_A0&&>(__a0).*__f)(static_cast<_Args&&>(__args)...);
  }
};

template <>
struct _Invoker<2> {
  template <class _Fp, class _A0, class... _Args>
  static inline _LIBCPP_HIDE_FROM_ABI
  _LIBCPP_CONSTEXPR decltype((std::declval<_A0&>().*std::declval<_Fp>())(std::declval<_Args>()...))
  _Fn(_Fp __f, reference_wrapper<_A0> __a0, _Args&&... __args)
      _NOEXCEPT_(noexcept((std::declval<_A0&>().*static_cast<_Fp&&>(__f))(static_cast<_Args&&>(__args)...))) {
    return (__a0.get().*__f)(static_cast<_Args&&>(__args)...);
  }
};

template <>
struct _Invoker<3> {
  template <class _Fp, class _A0, class... _Args>
  static inline _LIBCPP_HIDE_FROM_ABI
  _LIBCPP_CONSTEXPR decltype(((*std::declval<_A0>()).*std::declval<_Fp>())(std::declval<_Args>()...))
  _Fn(_Fp __f, _A0&& __a0, _Args&&... __args)
      _NOEXCEPT_(noexcept(((*static_cast<_A0&&>(__a0)).*static_cast<_Fp&&>(__f))(static_cast<_Args&&>(__args)...))) {
    return ((*static_cast<_A0&&>(__a0)).*__f)(static_cast<_Args&&>(__args)...);
  }
};

// bullets 4, 5 and 6

template <>
struct _Invoker<4> {
  template <class _Fp, class _A0>
  static inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR decltype(std::declval<_A0>().*std::declval<_Fp>())
  _Fn(_Fp __f, _A0&& __a0) _NOEXCEPT_(noexcept(static_cast<_A0&&>(__a0).*static_cast<_Fp&&>(__f))) {
    return static_cast<_A0&&>(__a0).*__f;
  }
};

template <>
struct _Invoker<5> {
  template <class _Fp, class _A0>
  static inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR decltype(std::declval<_A0&>().*std::declval<_Fp>())
  _Fn(_Fp __f, reference_wrapper<_A0> __a0) _NOEXCEPT_(noexcept(std::declval<_A0&>().*static_cast<_Fp&&>(__f))) {
    return __a0.get().*__f;
  }
};

template <>
struct _Invoker<6> {
  template <class _Fp, class _A0>
  static inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR decltype((*std::declval<_A0>()).*std::declval<_Fp>())
  _Fn(_Fp __f, _A0&& __a0) _NOEXCEPT_(noexcept((*static_cast<_A0&&>(__a0)).*static_cast<_Fp&&>(__f))) {
    return (*static_cast<_A0&&>(__a0)).*__f;
  }
};

template <class _Fp, class... _Args>
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR decltype(std::declval<_Fp>()(std::declval<_Args>()...))
__invoke(_Fp&& __f, _Args&&... __args) _NOEXCEPT_(noexcept(static_cast<_Fp&&>(__f)(static_cast<_Args&&>(__args)...))) {
  return static_cast<_Fp&&>(__f)(static_cast<_Args&&>(__args)...);
}

template <class _Member, class _Class, class _A0, class... _Args>
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR decltype(std::__invoker_t<_Member, _Class, _A0>::_Fn(
    std::declval<_Member _Class::*>(), std::declval<_A0>(), std::declval<_Args>()...))
__invoke(_Member _Class::* __f, _A0&& __a0, _Args&&... __args) _NOEXCEPT_(noexcept(
    std::__invoker_t<_Member, _Class, _A0>::_Fn(__f, static_cast<_A0&&>(__a0), static_cast<_Args&&>(__args)...))) {
  return std::__invoker_t<_Member, _Class, _A0>::_Fn(__f, static_cast<_A0&&>(__a0), static_cast<_Args&&>(__args)...);
}

template <class... _Args>
using __invoke_result_t _LIBCPP_NODEBUG = decltype(std::__invoke(reinterpret_cast<_Args&&>(static_cast<int&&>(0))...));

template <class _Void, class _Ret, class... _Args>
inline const bool __is_invocable_r_impl = false;

template <class _Ret, class... _Args>
inline const bool __is_invocable_r_impl<decltype(void(std::__invoke(std::declval<_Args>()...))), _Ret, _Args...> =
    is_void<_Ret>::value || __is_core_convertible<__invoke_result_t<_Args...>, _Ret>::value;

template <class... _Args>
_LIBCPP_NO_SPECIALIZATIONS inline const bool __is_invocable_r_v = __is_invocable_r_impl<void, _Args...>;

template <class... _Args>
_LIBCPP_NO_SPECIALIZATIONS inline const bool __is_invocable_v = __is_invocable_r_v<void, _Args...>;

template <class... _Args>
using __is_invocable _LIBCPP_NODEBUG = _BoolConstant<__is_invocable_v<_Args...> >;

template <class _Void, class... _Args>
inline const bool __nothrow_invocability_matches_v = false;

#  ifndef _LIBCPP_CXX03_LANG
template <class... _Args>
inline const bool
    __nothrow_invocability_matches_v<__enable_if_t<noexcept(std::__invoke(std::declval<_Args>()...))>, _Args...> = true;
#  endif

template <class... _Args>
_LIBCPP_NO_SPECIALIZATIONS inline const bool __is_nothrow_invocable_v =
    __nothrow_invocability_matches_v<void, _Args...>;

template <class _Void, class _Ret, class... _Args>
inline const bool __nothrow_invocable_r_imp = false;

#  ifndef _LIBCPP_CXX03_LANG
template <class _Ret, class _Fp, class... _Args>
inline const bool __nothrow_invocable_r_imp<
    __enable_if_t<noexcept(static_cast<void (*)(_Ret) _NOEXCEPT>(nullptr)(
        std::__invoke(std::declval<__enable_if_t<!is_void<_Ret>::value, _Fp> >(), std::declval<_Args>()...)))>,
    _Ret,
    _Fp,
    _Args...> = true;
#  endif

template <class... _Args>
inline const bool __nothrow_invocable_r_imp<__enable_if_t<__is_nothrow_invocable_v<_Args...> >, void, _Args...> = true;

template <class _Ret, class... _Args>
using __is_nothrow_invocable_r _LIBCPP_NODEBUG =
    _BoolConstant<__nothrow_invocable_r_imp<void, _If<is_void<_Ret>::value, void, _Ret>, _Args...> >;

#endif // __has_builtin(__builtin_invoke)

template <class _Void, class... _Args>
struct __invoke_result_impl {};

template <class... _Args>
struct __invoke_result_impl<__void_t<__invoke_result_t<_Args...> >, _Args...> {
  using type _LIBCPP_NODEBUG = __invoke_result_t<_Args...>;
};

template <class... _Args>
using __invoke_result _LIBCPP_NODEBUG = __invoke_result_impl<void, _Args...>;

template <class... _Args>
using __is_invocable_r _LIBCPP_NODEBUG = _BoolConstant<__is_invocable_r_v<_Args...> >;

template <class _Ret, class... _Args>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 __enable_if_t<!is_void<_Ret>::value, _Ret>
__invoke_r(_Args&&... __args) {
  return std::__invoke(static_cast<_Args&&>(__args)...);
}

template <class _Ret, class... _Args>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 __enable_if_t<is_void<_Ret>::value, _Ret>
__invoke_r(_Args&&... __args) {
  std::__invoke(static_cast<_Args&&>(__args)...);
}

#if _LIBCPP_STD_VER >= 17

// is_invocable

template <class _Fn, class... _Args>
struct _LIBCPP_NO_SPECIALIZATIONS is_invocable : bool_constant<__is_invocable_v<_Fn, _Args...> > {};

template <class _Ret, class _Fn, class... _Args>
struct _LIBCPP_NO_SPECIALIZATIONS is_invocable_r : bool_constant<__is_invocable_r_v<_Ret, _Fn, _Args...> > {};

template <class _Fn, class... _Args>
_LIBCPP_NO_SPECIALIZATIONS inline constexpr bool is_invocable_v = __is_invocable_v<_Fn, _Args...>;

template <class _Ret, class _Fn, class... _Args>
_LIBCPP_NO_SPECIALIZATIONS inline constexpr bool is_invocable_r_v = __is_invocable_r_v<_Ret, _Fn, _Args...>;

// is_nothrow_invocable

template <class _Fn, class... _Args>
struct _LIBCPP_NO_SPECIALIZATIONS is_nothrow_invocable : bool_constant<__is_nothrow_invocable_v<_Fn, _Args...> > {};

template <class _Ret, class _Fn, class... _Args>
struct _LIBCPP_NO_SPECIALIZATIONS is_nothrow_invocable_r
    : bool_constant<__is_nothrow_invocable_r<_Ret, _Fn, _Args...>::value> {};

template <class _Fn, class... _Args>
_LIBCPP_NO_SPECIALIZATIONS inline constexpr bool is_nothrow_invocable_v = __is_nothrow_invocable_v<_Fn, _Args...>;

template <class _Ret, class _Fn, class... _Args>
_LIBCPP_NO_SPECIALIZATIONS inline constexpr bool is_nothrow_invocable_r_v =
    __is_nothrow_invocable_r<_Ret, _Fn, _Args...>::value;

template <class _Fn, class... _Args>
struct _LIBCPP_NO_SPECIALIZATIONS invoke_result : __invoke_result<_Fn, _Args...> {};

template <class... _Args>
using invoke_result_t = typename invoke_result<_Args...>::type;

#endif

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___TYPE_TRAITS_INVOKE_H
