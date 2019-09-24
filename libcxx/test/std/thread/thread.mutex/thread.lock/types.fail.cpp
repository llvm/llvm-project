//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03
// UNSUPPORTED: libcpp-has-no-threads

// Before GCC 6, aggregate initialization kicks in.
// See https://stackoverflow.com/q/41799015/627587.
// UNSUPPORTED: gcc-5

// <mutex>

// struct defer_lock_t { explicit defer_lock_t() = default; };
// struct try_to_lock_t { explicit try_to_lock_t() = default; };
// struct adopt_lock_t { explicit adopt_lock_t() = default; };

// This test checks for LWG 2510.

#include <mutex>


std::defer_lock_t f1() { return {}; } // expected-error 1 {{chosen constructor is explicit in copy-initialization}}
std::try_to_lock_t f2() { return {}; } // expected-error 1 {{chosen constructor is explicit in copy-initialization}}
std::adopt_lock_t f3() { return {}; } // expected-error 1 {{chosen constructor is explicit in copy-initialization}}

int main(int, char**) {
    return 0;
}
