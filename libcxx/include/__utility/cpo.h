//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___UTILITY_CPO_H
#define _LIBCPP___UTILITY_CPO_H

#include <__config>
#include <__utility/forward.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER >= 20

_LIBCPP_BEGIN_NAMESPACE_STD

struct __string {
  consteval __string(const char*) {}
};

template <__string>
struct __diagnostic {};

// This is a utility to do overload resolution via a resolver function. That resolver function returns either a
// `__diagnostic` if overload resolution fails or a callable object (usually a stateless lambda).
// For expression-equivalence the callable object has to be marked `noexcept` if the expression inside is `noexcept`.
// The resolver function should be marked `noexcept` unconditionally and be `consteval`.
template <auto __resolver>
struct _CPO {
  template <class... _Args>
  [[nodiscard]] static constexpr auto operator()(_Args&&... __args) noexcept(
      noexcept(__resolver.template operator()<_Args&&...>()(std::forward<_Args>(__args)...)))
      -> decltype(__resolver.template operator()<_Args&&...>()(std::forward<_Args>(__args)...)) {
    return __resolver.template operator()<_Args&&...>()(std::forward<_Args>(__args)...);
  }
};

_LIBCPP_END_NAMESPACE_STD

#endif

#endif // _LIBCPP___UTILITY_CPO_H
