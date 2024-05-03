//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03
// UNSUPPORTED: no-threads

// <mutex>

// struct defer_lock_t { explicit defer_lock_t() = default; };
// struct try_to_lock_t { explicit try_to_lock_t() = default; };
// struct adopt_lock_t { explicit adopt_lock_t() = default; };

// This test checks for https://wg21.link/LWG2510.

#include <mutex>

std::defer_lock_t f1() { return {}; } // expected-error 1 {{chosen constructor is explicit in copy-initialization}}
std::try_to_lock_t f2() { return {}; } // expected-error 1 {{chosen constructor is explicit in copy-initialization}}
std::adopt_lock_t f3() { return {}; } // expected-error 1 {{chosen constructor is explicit in copy-initialization}}
