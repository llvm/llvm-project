// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___NEW_HARDWARE_INTERFERENCE_SIZE_H
#define _LIBCPP___NEW_HARDWARE_INTERFERENCE_SIZE_H

#include <__config>
#include <cstddef>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 17

#  if defined(__GCC_DESTRUCTIVE_SIZE) && defined(__GCC_CONSTRUCTIVE_SIZE)

inline constexpr size_t hardware_destructive_interference_size  = __GCC_DESTRUCTIVE_SIZE;
inline constexpr size_t hardware_constructive_interference_size = __GCC_CONSTRUCTIVE_SIZE;

#  elif defined(__APPLE__) && defined(__arm64__)

inline constexpr size_t hardware_destructive_interference_size  = 128;
inline constexpr size_t hardware_constructive_interference_size = 128;

#  else

// These values are correct for most platforms
inline constexpr size_t hardware_destructive_interference_size  = 64; // TODO: Clang should provide better values
inline constexpr size_t hardware_constructive_interference_size = 64;

#  endif

#endif // _LIBCPP_STD_VER >= 17

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___NEW_HARDWARE_INTERFERENCE_SIZE_H
