//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-threads

// This test verifies that <stdatomic.h> DOES NOT redirect to <atomic> before C++23,
// since doing so is a breaking change. Several things can break when that happens,
// because the type of _Atomic(T) changes from _Atomic(T) to std::atomic<T>.
//
// For example, redeclarations can become invalid depending on whether they
// have been declared with <stdatomic.h> in scope or not.

// REQUIRES: c++03 || c++11 || c++14 || c++17 || c++20

// On GCC, the compiler-provided <stdatomic.h> is not C++ friendly, so including <stdatomic.h>
// doesn't work at all if we don't use the <stdatomic.h> provided by libc++ in C++23 and above.
// XFAIL: (c++11 || c++14 || c++17 || c++20) && gcc

#include <atomic>
#include <stdatomic.h>
#include <type_traits>

static_assert(!std::is_same<_Atomic(int), std::atomic<int> >::value, "");
