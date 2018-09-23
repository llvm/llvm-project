//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <functional>

// not1
//  deprecated in C++17

// UNSUPPORTED: clang-4.0
// REQUIRES: verify-support

// MODULES_DEFINES: _LIBCPP_ENABLE_DEPRECATION_WARNINGS
#define _LIBCPP_ENABLE_DEPRECATION_WARNINGS

#include <functional>

#include "test_macros.h"

struct Predicate {
    typedef int argument_type;
    bool operator()(argument_type) const { return true; }
};

int main() {
#if TEST_STD_VER < 17
    // expected-no-diagnostics
#else
    // expected-error@* 1 {{'not1<Predicate>' is deprecated}}
#endif
    std::not1(Predicate());
}
