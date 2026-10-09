//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// template<class Operation>
// constexpr void resize_and_overwrite(size_type n, Operation op); // since C++23

// The result r of the operation must satisfy 0 <= r <= n.

// REQUIRES: std-at-least-c++23
// REQUIRES: can-test-hardening-assertions-fast
// Execution would continue into an out-of-bounds write.
// UNSUPPORTED: libcpp-assertion-semantic={{ignore|observe}}

#include <cstddef>
#include <string>

#include "check_assertion.h"
#include "min_allocator.h"
#include "test_macros.h"

template <class S>
void test() {
  {
    S s;
    TEST_LIBCPP_ASSERT_FAILURE(s.resize_and_overwrite(5, [](auto*, std::size_t n) { return n + 1; }),
                               "string::resize_and_overwrite: the operation returned a size larger than n");
  }
  {
    S s(40, 'x');
    TEST_LIBCPP_ASSERT_FAILURE(s.resize_and_overwrite(40, [](auto*, std::size_t n) { return n + 4096; }),
                               "string::resize_and_overwrite: the operation returned a size larger than n");
  }
  {
    S s(40, 'x');
    TEST_LIBCPP_ASSERT_FAILURE(s.resize_and_overwrite(40, [](auto*, std::size_t) { return -1; }),
                               "string::resize_and_overwrite: the operation returned a negative size");
  }
  {
    // A narrow signed result must not be compared as its unsigned counterpart, where -1 would be 255.
    S s;
    TEST_LIBCPP_ASSERT_FAILURE(
        s.resize_and_overwrite(300, [](auto*, std::size_t) { return static_cast<signed char>(-1); }),
        "string::resize_and_overwrite: the operation returned a negative size");
  }
#ifndef TEST_HAS_NO_INT128
  {
    // A wide result must not be truncated to size_type before the comparison, where 2^64 + 1 would be 1.
    S s;
    TEST_LIBCPP_ASSERT_FAILURE(
        s.resize_and_overwrite(5, [](auto*, std::size_t) { return (static_cast<__int128_t>(1) << 64) + 1; }),
        "string::resize_and_overwrite: the operation returned a size larger than n");
  }
#endif
}

int main(int, char**) {
  test<std::string>();
  test<std::basic_string<char, std::char_traits<char>, min_allocator<char> > >();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<std::wstring>();
#endif

  return 0;
}
