// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___RANGES_ELEMENTS_OF_H
#define _LIBCPP___RANGES_ELEMENTS_OF_H

#include <__config>
#include <__cstddef/byte.h>
#include <__memory/allocator.h>
#include <__ranges/concepts.h>
#include <__utility/forward.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 23

namespace ranges {

template <range _Range, class _Allocator = allocator<byte>>
struct elements_of {
  _LIBCPP_NO_UNIQUE_ADDRESS _Range range;
  _LIBCPP_NO_UNIQUE_ADDRESS _Allocator allocator = _Allocator();
};

template <class _Range, class _Allocator = allocator<byte>>
elements_of(_Range&&, _Allocator = _Allocator()) -> elements_of<_Range&&, _Allocator>;

} // namespace ranges

#endif // _LIBCPP_STD_VER >= 23

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___RANGES_ELEMENTS_OF_H
