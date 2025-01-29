//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// XFAIL: FROZEN-CXX03-HEADERS-FIXME

// This test verifies that <stdatomic.h> redirects to <atomic>.

// Before C++23, <stdatomic.h> can be included after <atomic>, but including it
// first doesn't work because its macros break <atomic>. Fixing that is the point
// of the C++23 change that added <stdatomic.h> to C++. Thus, this test verifies
// that <stdatomic.h> can be included first.
#include <stdatomic.h>
#include <atomic>

#include <type_traits>

static_assert(std::is_same<atomic_int, std::atomic<int> >::value, "");
