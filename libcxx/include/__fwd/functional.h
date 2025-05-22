//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#ifndef _LIBCPP___FWD_FUNCTIONAL_H
#define _LIBCPP___FWD_FUNCTIONAL_H

#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

// Work around the visibility on a namespace bleed into user specializations.
// TODO: Remove this workaround once all supported compilers are fixed
namespace std {
inline namespace _LIBCPP_ABI_NAMESPACE {
template <class>
struct hash;
} // namespace _LIBCPP_ABI_NAMESPACE
} // namespace std

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 14
template <class _Tp = void>
#else
template <class _Tp>
#endif
struct less;

template <class>
class reference_wrapper;

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___FWD_FUNCTIONAL_H
