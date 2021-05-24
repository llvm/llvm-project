//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>

// not2
//  deprecated in C++17

// UNSUPPORTED: clang-4.0
// UNSUPPORTED: c++03, c++11, c++14
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_CXX20_REMOVED_NEGATORS

#include <functional>

#include "test_macros.h"

struct Predicate {
    typedef int first_argument_type;
    typedef int second_argument_type;
    bool operator()(first_argument_type, second_argument_type) const { return true; }
};

int main(int, char**) {
    std::not2(Predicate()); // expected-warning {{'not2<Predicate>' is deprecated}}

    return 0;
}
