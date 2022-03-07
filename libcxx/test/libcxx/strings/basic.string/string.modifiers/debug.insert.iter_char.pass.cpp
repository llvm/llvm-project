//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// iterator insert(const_iterator p, charT c);

// UNSUPPORTED: libcxx-no-debug-mode, c++03, windows
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DEBUG=1

// TODO: Since string::insert(iter, char) is intantiated in the dylib, this test doesn't
//       actually work if the dylib hasn't been built with debug assertions enabled.
//       Until we overhaul the debug mode, mark this test as unsupported to avoid
//       spurious CI failures.
// REQUIRES: never-run

#include <string>

#include "check_assertion.h"

int main(int, char**) {
    typedef std::string S;
    S s;
    S s2;
    TEST_LIBCPP_ASSERT_FAILURE(s.insert(s2.begin(), '1'),
        "string::insert(iterator, character) called with an iterator not referring to this string");

    return 0;
}
