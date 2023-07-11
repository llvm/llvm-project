//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_map>

// Increment local_iterator past end.

// REQUIRES: has-unix-headers
// UNSUPPORTED: !libcpp-has-legacy-debug-mode, c++03

#include <unordered_map>
#include <string>
#include <cassert>

#include "check_assertion.h"
#include "min_allocator.h"

int main(int, char**) {
    {
        typedef std::unordered_map<int, std::string> C;
        C c;
        c.insert(std::make_pair(42, std::string()));
        C::size_type b = c.bucket(42);
        C::local_iterator i = c.begin(b);
        assert(i != c.end(b));
        ++i;
        assert(i == c.end(b));
        TEST_LIBCPP_ASSERT_FAILURE(++i, "Attempted to increment a non-incrementable unordered container local_iterator");
    }

    {
        typedef std::unordered_map<int, std::string, std::hash<int>, std::equal_to<int>,
                                   min_allocator<std::pair<const int, std::string>>> C;
        C c({{42, std::string()}});
        C::size_type b = c.bucket(42);
        C::local_iterator i = c.begin(b);
        assert(i != c.end(b));
        ++i;
        assert(i == c.end(b));
        TEST_LIBCPP_ASSERT_FAILURE(++i, "Attempted to increment a non-incrementable unordered container local_iterator");
    }

    return 0;
}
