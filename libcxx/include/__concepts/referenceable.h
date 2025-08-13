//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CONCEPTS_REFERENCEABLE_H
#define _LIBCPP___CONCEPTS_REFERENCEABLE_H

#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER >= 20

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Tp>
using __referenceable_impl _LIBCPP_NODEBUG = _Tp&;

template <class _Tp>
concept __referenceable = requires { typename __referenceable_impl<_Tp>; };

_LIBCPP_END_NAMESPACE_STD

#endif

#endif // _LIBCPP___CONCEPTS_REFERENCEABLE_H
