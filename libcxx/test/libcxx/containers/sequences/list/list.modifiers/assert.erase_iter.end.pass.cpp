//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <list>

// Call erase(const_iterator position) with end()

// UNSUPPORTED: c++03, windows
// UNSUPPORTED: use_system_cxx_lib && target={{.+}}-apple-macosx{{10.9|10.10|10.11|10.12|10.13|10.14|10.15|11|12}}
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_ASSERTIONS=1

#include <list>

#include "check_assertion.h"

int main(int, char**) {
    int a1[] = {1, 2, 3};
    std::list<int> l1(a1, a1+3);
    std::list<int>::const_iterator i = l1.end();
    TEST_LIBCPP_ASSERT_FAILURE(l1.erase(i), "list::erase(iterator) called with a non-dereferenceable iterator");

    return 0;
}
