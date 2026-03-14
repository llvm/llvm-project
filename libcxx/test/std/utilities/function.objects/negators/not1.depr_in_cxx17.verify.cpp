//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>

// not1
//  deprecated in C++17

// UNSUPPORTED: c++03, c++11, c++14
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_CXX20_REMOVED_NEGATORS

#include <functional>

struct Predicate {
    typedef int argument_type;
    bool operator()(argument_type) const { return true; }
};

void f() {
    std::not1(Predicate()); // expected-warning {{'not1<Predicate>' is deprecated}}
}
