//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <functional>

// unary_negate
//  deprecated in C++17

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
    // expected-error@* 1 {{'unary_negate<Predicate>' is deprecated}}
#endif
    std::unary_negate<Predicate> f((Predicate()));
    (void)f;
}
