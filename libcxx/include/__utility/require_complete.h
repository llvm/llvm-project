//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___UTILITY_REQUIRE_COMPLETE_H
#define _LIBCPP___UTILITY_REQUIRE_COMPLETE_H

#include <__config>
#include <__cstddef/size_t.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Tp, size_t = sizeof(_Tp)>
_LIBCPP_CONSTEXPR void __require_complete_impl(int) {}

template <class _Tp, bool _False = false>
_LIBCPP_CONSTEXPR void __require_complete_impl(long) {
  static_assert(_False, "Type is required to be complete");
}

// Produce a compiler error if the given type is not complete.
template <class _Tp>
_LIBCPP_CONSTEXPR void __require_complete() {
  std::__require_complete_impl<_Tp>(0);
}

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___UTILITY_REQUIRE_COMPLETE_H
