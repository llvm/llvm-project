//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-threads

// This test ensures that we don't hijack the <stdatomic.h> header (e.g. by providing
// an empty header) even when compiling before C++23, since some users were using the
// Clang or platform provided header before libc++ added its own.

// On GCC, the compiler-provided <stdatomic.h> is not C++ friendly, so including <stdatomic.h>
// doesn't work at all if we don't use the <stdatomic.h> provided by libc++ in C++23 and above.
// XFAIL: (c++11 || c++14 || c++17 || c++20) && gcc

#include <stdatomic.h>

void f() {
  atomic_int i; // just make sure the header isn't empty
  (void)i;
}
