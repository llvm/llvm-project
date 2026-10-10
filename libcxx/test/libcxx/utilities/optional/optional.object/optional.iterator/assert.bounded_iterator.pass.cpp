//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <optional>

// REQUIRES: std-at-least-c++26, libcpp-has-abi-bounded-iterators-in-optional
// REQUIRES: can-test-hardening-assertions-fast

// Test that an assertion fires for invalid uses of the following operators on a bounded iterator:

// operator++()
// operator++(int),
// operator--(),
// operator--(int),
// operator*
// operator[]
// operator->
// operator+=
// operator-=

#include <optional>

#include "check_assertion.h"

template <typename T>
void test_bounded_iter() {
  { // operator++
    std::optional<T> o{1};
    auto i = o.end();

    TEST_LIBCPP_ASSERT_FAILURE(++i, "__bounded_iter::operator++: Attempt to advance an iterator past the end");
    TEST_LIBCPP_ASSERT_FAILURE(i++, "__bounded_iter::operator++: Attempt to advance an iterator past the end");
  }

  { // operator--
    std::optional<T> o{1};
    auto i = o.begin();

    TEST_LIBCPP_ASSERT_FAILURE(--i, "__bounded_iter::operator--: Attempt to rewind an iterator past the start");
    TEST_LIBCPP_ASSERT_FAILURE(i--, "__bounded_iter::operator--: Attempt to rewind an iterator past the start");
  }

  { // operator*
    std::optional<T> o;
    auto i = o.begin();

    TEST_LIBCPP_ASSERT_FAILURE(*i, "__bounded_iter::operator*: Attempt to dereference an iterator at the end");
  }

  { // operator[]
    std::optional<T> o{1};
    auto i = o.begin();

    TEST_LIBCPP_ASSERT_FAILURE(i[1], "__bounded_iter::operator[]: Attempt to index an iterator at or past the end");
    TEST_LIBCPP_ASSERT_FAILURE(i[-1], "__bounded_iter::operator[]: Attempt to index an iterator past the start");
  }

  { // operator->
    std::optional<T> o{1};
    auto i = o.end();

    TEST_LIBCPP_ASSERT_FAILURE(
        i.operator->(), "__bounded_iter::operator->: Attempt to dereference an iterator at the end");
  }

  { // operator+=
    std::optional<T> o{1};
    auto i = o.begin();

    TEST_LIBCPP_ASSERT_FAILURE(i += 2, "__bounded_iter::operator+=: Attempt to advance an iterator past the end");
    TEST_LIBCPP_ASSERT_FAILURE(i += -1, "__bounded_iter::operator+=: Attempt to rewind an iterator past the start");
  }

  { // operator-=
    std::optional<T> o{1};
    auto i = o.begin();

    TEST_LIBCPP_ASSERT_FAILURE(i -= 1, "__bounded_iter::operator-=: Attempt to rewind an iterator past the start");
    TEST_LIBCPP_ASSERT_FAILURE(i -= -2, "__bounded_iter::operator-=: Attempt to advance an iterator past the end");
  }
}

int main(int, char**) {
  test_bounded_iter<char>();
  test_bounded_iter<int>();

  return 0;
}
