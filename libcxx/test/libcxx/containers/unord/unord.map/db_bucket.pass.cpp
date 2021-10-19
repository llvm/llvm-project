//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_map>

// size_type bucket(const key_type& __k) const;

// UNSUPPORTED: libcxx-no-debug-mode

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DEBUG=1

#include <unordered_map>
#include <string>

#include "test_macros.h"
#include "debug_macros.h"

int main(int, char**) {
    typedef std::unordered_map<int, std::string> C;
    C c;
    TEST_LIBCPP_ASSERT_FAILURE(c.bucket(3), "unordered container::bucket(key) called when bucket_count() == 0");

    return 0;
}
