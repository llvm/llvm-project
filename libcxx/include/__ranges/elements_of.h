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
#include <__memory/allocator.h>
#include <__ranges/concepts.h>
#include <__utility/move.h>
#include <cstddef>

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

#  if defined(_LIBCPP_COMPILER_CLANG_BASED) && _LIBCPP_CLANG_VER < 1600
  // This explicit constructor is required because AppleClang 15 hasn't implement P0960R3
  _LIBCPP_HIDE_FROM_ABI explicit constexpr elements_of(_Range __range, _Allocator __alloc = _Allocator())
      : range(std::move(__range)), allocator(std::move(__alloc)) {}
#  endif
};

template <class _Range, class _Allocator = allocator<byte>>
#  if defined(_LIBCPP_COMPILER_CLANG_BASED) && _LIBCPP_CLANG_VER < 1600
// This explicit constraint is required because AppleClang 15 might not deduce the correct type for `_Range` without it
  requires range<_Range&&>
#  endif
elements_of(_Range&&, _Allocator = _Allocator()) -> elements_of<_Range&&, _Allocator>;

} // namespace ranges

#endif // _LIBCPP_STD_VER >= 23

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___RANGES_ELEMENTS_OF_H
