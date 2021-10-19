//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <list>

// Call erase(const_iterator position) with iterator from another container

// UNSUPPORTED: libcxx-no-debug-mode

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DEBUG=1

#include <list>

#include "test_macros.h"
#include "debug_macros.h"

int main(int, char**)
{
    int a1[] = {1, 2, 3};
    std::list<int> l1(a1, a1+3);
    std::list<int> l2(a1, a1+3);
    std::list<int>::const_iterator i = l2.begin();
    TEST_LIBCPP_ASSERT_FAILURE(l1.erase(i), "list::erase(iterator) called with an iterator not referring to this list");

    return 0;
}
