//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <functional>

// binary_negate
//  deprecated in C++17

// UNSUPPORTED: clang-4.0
// REQUIRES: verify-support

// MODULES_DEFINES: _LIBCPP_ENABLE_DEPRECATION_WARNINGS
#define _LIBCPP_ENABLE_DEPRECATION_WARNINGS

#include <functional>

#include "test_macros.h"

struct Predicate {
    typedef int first_argument_type;
    typedef int second_argument_type;
    bool operator()(first_argument_type, second_argument_type) const { return true; }
};

int main() {
#if TEST_STD_VER < 17
    // expected-no-diagnostics
#else
    // expected-error@* 1 {{'binary_negate<Predicate>' is deprecated}}
#endif
    std::binary_negate<Predicate> f((Predicate()));
    (void)f;
}
