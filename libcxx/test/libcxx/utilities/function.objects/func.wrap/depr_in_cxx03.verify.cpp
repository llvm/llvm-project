//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <functional>

// Check that libc++'s emulation of std::function is deprecated in C++03

// REQUIRES: c++98 || c++03
// REQUIRES: verify-support

#include <functional>
#include "test_macros.h"

int main() {
    // Note:
    // We use sizeof() to require it to be a complete type. We don't create a
    // variable because otherwise we get two warnings for each variable (the
    // second warning is when the destructor is implicitly called).
    (void)sizeof(std::function<void ()>); // expected-warning {{'function<void ()>' is deprecated}}
    (void)sizeof(std::function<void (int)>); // expected-warning {{'function<void (int)>' is deprecated}}
    (void)sizeof(std::function<void (int, int)>); // expected-warning {{'function<void (int, int)>' is deprecated}}
    (void)sizeof(std::function<void (int, int, int)>); // expected-warning {{'function<void (int, int, int)>' is deprecated}}
}
