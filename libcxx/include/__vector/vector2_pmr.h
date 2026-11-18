//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___VECTOR_VECTOR2_PMR_H
#define _LIBCPP___VECTOR_VECTOR2_PMR_H

#include <__config>
#include <__cpp_release_namespaces.h>
#include <__fwd/vector2.h>
#include <__memory_resource/polymorphic_allocator.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER >= 17

#if _LIBCPP_STD_VER >= 29

_LIBCPP_BEGIN_NAMESPACE_STD

namespace pmr {
template <class _ValueT>
using vector2 _LIBCPP_AVAILABILITY_PMR = std::vector2<_ValueT, polymorphic_allocator<_ValueT>>;
} // namespace pmr

// PROTOTYPE: P3866R0 standard-release namespace.
namespace cpp29 {
namespace pmr {
template <class _ValueT>
using vector _LIBCPP_AVAILABILITY_PMR = std::pmr::vector2<_ValueT>;
} // namespace pmr
} // namespace cpp29

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 29

#endif

#endif // _LIBCPP___VECTOR_VECTOR2_PMR_H
