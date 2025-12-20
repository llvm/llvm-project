//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___FUNCTIONAL_MOVE_ONLY_FUNCTION_H
#define _LIBCPP___FUNCTIONAL_MOVE_ONLY_FUNCTION_H

#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER >= 23 && !defined(_LIBCPP_COMPILER_GCC) && _LIBCPP_HAS_EXPERIMENTAL_MOVE_ONLY_FUNCTION

// move_only_function design:
//
// move_only_function has a small buffer with a size of `3 * sizeof(void*)` bytes. This buffer can only be used when the
// object to be stored is "trivially relocatable" (currently only when it is trivially move constructible and trivially
// destructible). The vtable entry for the destructor is a null pointer when the stored object is trivially
// destructible.
//
// trivially relocatable: It would also be possible to store nothrow_move_constructible types, but that would mean
// that move_only_function itself would not be trivially relocatable anymore. The decision to keep move_only_function
// trivially relocatable was made because we expect move_only_function to be stored persistently most of the time, since
// std::function_ref can be used for cases where a function object doesn't need to be stored.
//
// buffer size: We did a survey of six implementations from various vendors. Three of them had a buffer size of 24 bytes
// on 64 bit systems. This will also allow storing a function object containing a std::string or std::vector inside the
// small buffer once there is a language definition of "trivially relocatable".
//
// interaction with copyable_function: When converting a copyable_function into a move_only_function we want to avoid
// wrapping the copyable_function inside the move_only_function to avoid a double indirection. Instead, we copy the
// small buffer and use copyable_function's vtable.

// NOLINTBEGIN(readability-duplicate-include)
#  define _LIBCPP_IN_MOVE_ONLY_FUNCTION_H

#  include <__functional/move_only_function_impl.h>

#  define _LIBCPP_MOVE_ONLY_FUNCTION_REF &
#  include <__functional/move_only_function_impl.h>

#  define _LIBCPP_MOVE_ONLY_FUNCTION_REF &&
#  include <__functional/move_only_function_impl.h>

#  define _LIBCPP_MOVE_ONLY_FUNCTION_CV const
#  include <__functional/move_only_function_impl.h>

#  define _LIBCPP_MOVE_ONLY_FUNCTION_CV const
#  define _LIBCPP_MOVE_ONLY_FUNCTION_REF &
#  include <__functional/move_only_function_impl.h>

#  define _LIBCPP_MOVE_ONLY_FUNCTION_CV const
#  define _LIBCPP_MOVE_ONLY_FUNCTION_REF &&
#  include <__functional/move_only_function_impl.h>

#  undef _LIBCPP_IN_MOVE_ONLY_FUNCTION_H
// NOLINTEND(readability-duplicate-include)

#endif // _LIBCPP_STD_VER >= 23 && !defined(_LIBCPP_COMPILER_GCC) && _LIBCPP_HAS_EXPERIMENTAL_MOVE_ONLY_FUNCTION

#endif // _LIBCPP___FUNCTIONAL_MOVE_ONLY_FUNCTION_H
