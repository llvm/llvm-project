//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <list>

// template <class... Args> void emplace(const_iterator p, Args&&... args);

// UNSUPPORTED: c++03
// UNSUPPORTED: libcxx-no-debug-mode

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DEBUG=1

#include <list>

#include "test_macros.h"
#include "debug_macros.h"

struct A {
  explicit A(int i, double d) {
    (void)i;
    (void)d;
  }
};

int main(int, char**)
{
    std::list<A> c1;
    std::list<A> c2;
    TEST_LIBCPP_ASSERT_FAILURE(c1.emplace(c2.cbegin(), 2, 3.5),
                               "list::emplace(iterator, args...) called with an iterator not referring to this list");

    return 0;
}
