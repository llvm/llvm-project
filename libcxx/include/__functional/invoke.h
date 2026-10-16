// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___FUNCTIONAL_INVOKE_H
#define _LIBCPP___FUNCTIONAL_INVOKE_H

#include <__config>
#include <__type_traits/invoke.h>
#include <__type_traits/is_void.h>
#include <__utility/forward.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD

#  if __has_builtin(__builtin_invoke)

template <class... _Args>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 __invoke_result_t<_Args...> invoke(_Args&&... __args)
    _NOEXCEPT_(noexcept(__builtin_invoke(static_cast<_Args&&>(__args)...))) {
  return __builtin_invoke(static_cast<_Args&&>(__args)...);
}

#  else

template <class _Fp, class... _Args>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR auto invoke(_Fp&& __f, _Args&&... __args)
    _NOEXCEPT_(noexcept(static_cast<_Fp&&>(__f)(static_cast<_Args&&>(__args)...)))
        -> decltype(static_cast<_Fp&&>(__f)(static_cast<_Args&&>(__args)...)) {
  return static_cast<_Fp&&>(__f)(static_cast<_Args&&>(__args)...);
}

template <class _Member, class _Class, class _A0, class... _Args>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR auto invoke(_Member _Class::* __f, _A0&& __a0, _Args&&... __args)
    _NOEXCEPT_(noexcept(
        std::__invoker_t<_Member, _Class, _A0>::_Fn(__f, static_cast<_A0&&>(__a0), static_cast<_Args&&>(__args)...)))
        -> decltype(std::__invoker_t<_Member, _Class, _A0>::_Fn(
            __f, static_cast<_A0&&>(__a0), static_cast<_Args&&>(__args)...)) {
  return std::__invoker_t<_Member, _Class, _A0>::_Fn(__f, static_cast<_A0&&>(__a0), static_cast<_Args&&>(__args)...);
}

#  endif

#  if _LIBCPP_STD_VER >= 23
template <class _Result, class _Fn, class... _Args>
  requires is_invocable_r_v<_Result, _Fn, _Args...>
_LIBCPP_HIDE_FROM_ABI constexpr _Result
invoke_r(_Fn&& __f, _Args&&... __args) noexcept(is_nothrow_invocable_r_v<_Result, _Fn, _Args...>) {
  if constexpr (is_void_v<_Result>) {
    static_cast<void>(std::invoke(std::forward<_Fn>(__f), std::forward<_Args>(__args)...));
  } else {
    // TODO: Use reference_converts_from_temporary_v once implemented
    // using _ImplicitInvokeResult = invoke_result_t<_Fn, _Args...>;
    // static_assert(!reference_converts_from_temporary_v<_Result, _ImplicitInvokeResult>,
    static_assert(true,
                  "Returning from invoke_r would bind a temporary object to the reference return type, "
                  "which would result in a dangling reference.");
    return std::invoke(std::forward<_Fn>(__f), std::forward<_Args>(__args)...);
  }
}
#  endif

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 17

#endif // _LIBCPP___FUNCTIONAL_INVOKE_H
