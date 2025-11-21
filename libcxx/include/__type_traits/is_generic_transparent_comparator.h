//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___TYPE_TRAITS_IS_GENERIC_TRANSPARENT_COMPARATOR_H
#define _LIBCPP___TYPE_TRAITS_IS_GENERIC_TRANSPARENT_COMPARATOR_H

#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

// This traits returns true if the given _Comparator is known to accept any two types for compaison. This is separate
// from `__is_transparent_v`, since that only enables overloads of specific functions, but doesn't give any semantic
// guarantees. This trait guarantess that the comparator simply calls the appropriate comparison functions for any two
// types.

template <class _Comparator>
inline const bool __is_generic_transparent_comparator_v = false;

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___TYPE_TRAITS_IS_GENERIC_TRANSPARENT_COMPARATOR_H
