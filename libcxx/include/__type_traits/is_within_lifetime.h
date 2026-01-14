//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___TYPE_TRAITS_IS_WITHIN_LIFETIME_H
#define _LIBCPP___TYPE_TRAITS_IS_WITHIN_LIFETIME_H

#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 26 && __has_builtin(__builtin_is_within_lifetime)
template <class _Tp>
_LIBCPP_HIDE_FROM_ABI consteval bool is_within_lifetime(const _Tp* __p) noexcept {
  return __builtin_is_within_lifetime(__p);
}
#endif

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___TYPE_TRAITS_IS_WITHIN_LIFETIME_H
