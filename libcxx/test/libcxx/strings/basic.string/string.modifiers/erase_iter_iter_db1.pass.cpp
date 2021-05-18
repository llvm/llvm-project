//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// Call erase(const_iterator first, const_iterator last); with first iterator from another container

// UNSUPPORTED: libcxx-no-debug-mode

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DEBUG=1

#include <string>

#include "test_macros.h"
#include "debug_macros.h"

int main(int, char**)
{
    std::string l1("123");
    std::string l2("123");
    TEST_LIBCPP_ASSERT_FAILURE(
        l1.erase(l2.cbegin(), l1.cbegin() + 1),
        "string::erase(iterator,  iterator) called with an iterator not referring to this string");

    return 0;
}
