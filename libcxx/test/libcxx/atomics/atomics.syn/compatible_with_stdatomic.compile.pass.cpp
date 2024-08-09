//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-threads

// This test verifies that <stdatomic.h> redirects to <atomic>. As an extension,
// libc++ enables this redirection even before C++23.

// Ordinarily, <stdatomic.h> can be included after <atomic>, but including it
// first doesn't work because its macros break <atomic>. Verify that
// <stdatomic.h> can be included first.
#include <stdatomic.h>
#include <atomic>

#include <type_traits>

static_assert(std::is_same<atomic_int, std::atomic<int> >::value, "");
