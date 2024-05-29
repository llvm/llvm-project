//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03, c++11

// Until we drop support for the synchronization library in C++11/14/17
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

// <barrier>

// explicit barrier(ptrdiff_t __count, _CompletionF __completion = _CompletionF());

// Make sure that the ctor of barrier is explicit.

#include <barrier>

#include "test_convertible.h"

static_assert(!test_convertible<std::barrier<>, std::ptrdiff_t>(), "This constructor must be explicit");
