//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___NEW_NEW_AT_LEAST_H
#define _LIBCPP___NEW_NEW_AT_LEAST_H

#include <__config>
#include <__cstddef/size_t.h>
#include <__new/align_val_t.h>
#include <__new/allocation_result.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_AVAILABILITY_HAS_NEW_AT_LEAST

_LIBCPP_BEGIN_NAMESPACE_STD
_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS

// `__new_at_least` acts like an overload of `operator new` which takes a size (and possibly alignment) and returns
// a pointer as well as the actually allocated amount of memory. If the user replaces the relevant `operator new`
// overload this will fall back to calling that. Otherwise it tries to allocate in a way to get the actually allocated
// size, depending on what the platform provides.

_LIBCPP_MALLOC_SPAN _LIBCPP_EXPORTED_FROM_ABI __allocation_result<void*> __new_at_least(size_t);
#  if _LIBCPP_HAS_LIBRARY_ALIGNED_ALLOCATION
_LIBCPP_MALLOC_SPAN _LIBCPP_EXPORTED_FROM_ABI __allocation_result<void*> __new_at_least(size_t, align_val_t);
#  endif

_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_AVAILABILITY_HAS_NEW_AT_LEAST

#endif // _LIBCPP___NEW_NEW_AT_LEAST_H
