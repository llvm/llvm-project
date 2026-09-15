//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "__cxxabi_config.h"
#include "abort_message.h"
#include <new>

// Perform a few sanity checks on libc++ and libc++abi macros to ensure that
// the code below can be an exact copy of the code in libcxx/src/new.cpp.
#if !defined(_THROW_BAD_ALLOC)
#  error The _THROW_BAD_ALLOC macro should be already defined by libc++
#endif

#if defined(_LIBCXXABI_NO_EXCEPTIONS) != !_LIBCPP_HAS_EXCEPTIONS
#  error libc++ and libc++abi seem to disagree on whether exceptions are enabled
#endif

inline void __throw_bad_alloc_shim() {
#if _LIBCPP_HAS_EXCEPTIONS
  throw std::bad_alloc();
#else
  __abort_message("bad_alloc was thrown in -fno-exceptions mode");
#endif
}

#define _LIBCPP_ASSERT_SHIM(expr, str)                                                                                 \
  do {                                                                                                                 \
    if (!expr)                                                                                                         \
      __abort_message(str);                                                                                            \
  } while (false)

#include "support/new.ipp"
