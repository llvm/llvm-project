// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_OPTIONAL_HASH_H
#define _LIBCPP_OPTIONAL_HASH_H

#include <__config>
#include <__cstddef/size_t.h>
#include <__functional/hash.h>
#include <__type_traits/remove_const.h>

#include <__fwd/optional.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Tp>
struct hash< __enable_hash_helper<optional<_Tp>, remove_const_t<_Tp>> > {
#  if _LIBCPP_STD_VER <= 17 || defined(_LIBCPP_ENABLE_CXX20_REMOVED_BINDER_TYPEDEFS)
  _LIBCPP_DEPRECATED_IN_CXX17 typedef optional<_Tp> argument_type;
  _LIBCPP_DEPRECATED_IN_CXX17 typedef size_t result_type;
#  endif

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI size_t operator()(const optional<_Tp>& __opt) const {
    return static_cast<bool>(__opt) ? hash<remove_const_t<_Tp>>()(*__opt) : 0;
  }
};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 17

_LIBCPP_POP_MACROS
#endif // _LIBCPP_OPTIONAL_HASH_H
