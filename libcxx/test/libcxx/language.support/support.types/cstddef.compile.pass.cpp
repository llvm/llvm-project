//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Make sure libc++'s <cstddef> defines types like ::nullptr_t in the global namespace.
// This is a conforming extension to be consistent with other implementations, which all
// appear to provide that behavior too.

#include <cstddef>
#include "test_macros.h"

using PtrdiffT = ::ptrdiff_t;
using SizeT = ::size_t;
#if TEST_STD_VER >= 11
using MaxAlignT = ::max_align_t;
#endif

// Supported in C++03 mode too for backwards compatibility with previous versions of libc++
using NullptrT = ::nullptr_t;

// Also ensure that we provide std::nullptr_t in C++03 mode, which is an extension too.
using StdNullptrT = std::nullptr_t;
