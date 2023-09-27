//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___TYPE_TRAITS_IS_CONSTANT_EVALUATED_H
#define _LIBCPP___TYPE_TRAITS_IS_CONSTANT_EVALUATED_H

#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 20
_LIBCPP_INLINE_VISIBILITY inline constexpr bool is_constant_evaluated() noexcept {
  return __builtin_is_constant_evaluated();
}
#endif

_LIBCPP_HIDE_FROM_ABI inline _LIBCPP_CONSTEXPR bool __libcpp_is_constant_evaluated() _NOEXCEPT {
// __builtin_is_constant_evaluated() in this function always evaluates to false in pre-C++11 mode
// because this function is not constexpr-qualified.
// The following macro use clarifies this and avoids warnings from compilers.
#ifndef _LIBCPP_CXX03_LANG
  return __builtin_is_constant_evaluated();
#else
  return false;
#endif
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___TYPE_TRAITS_IS_CONSTANT_EVALUATED_H
