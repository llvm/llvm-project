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

// if T has the alignment required to fit a bounds check, optional<T>
// will always have static_packed_bounded_iter regardless of if the bounded iterator
// option is enabled for it.
// see __static_packed_bounded_iterator.h for more information on how this is determined.

void test_packed_iter() {
  // int has enough room, 3 bits free -> 0-2 range
  { // operator++
    std::optional<int> o{1};
    auto i = o.end();

    TEST_LIBCPP_ASSERT_FAILURE(
        ++i, "__static_packed_bounded_iterator::operator++: Attempt to advance an iterator past the end");
    TEST_LIBCPP_ASSERT_FAILURE(
        i++, "__static_packed_bounded_iterator::operator++: Attempt to advance an iterator past the end");
  }

  { // operator--
    std::optional<int> o{1};
    auto i = o.begin();

    TEST_LIBCPP_ASSERT_FAILURE(
        --i, "__static_packed_bounded_iterator::operator--: Attempt to rewind an iterator past the start");
    TEST_LIBCPP_ASSERT_FAILURE(
        i--, "__static_packed_bounded_iterator::operator--: Attempt to rewind an iterator past the start");
  }

  { // operator*
    std::optional<int> o;
    auto i = o.begin();

    TEST_LIBCPP_ASSERT_FAILURE(
        *i, "__static_packed_bounded_iterator::operator*: Attempt to dereference an iterator at the end");
  }

  { // operator[]
    std::optional<int> o{1};
    auto i = o.begin();

    TEST_LIBCPP_ASSERT_FAILURE(
        i[1], "__static_packed_bounded_iterator::operator[]: Attempt to index an iterator at or past the end");
    TEST_LIBCPP_ASSERT_FAILURE(
        i[-1], "__static_packed_bounded_iterator::operator[]: Attempt to index an iterator past the start");
  }

  { // operator->
    std::optional<int> o{1};
    auto i = o.end();

    TEST_LIBCPP_ASSERT_FAILURE(
        i.operator->(), "__static_packed_bounded_iterator::operator->: Attempt to dereference an iterator at the end");
  }

  { // operator+=
    std::optional<int> o{1};
    auto i = o.begin();

    TEST_LIBCPP_ASSERT_FAILURE(
        i += 2, "__static_packed_bounded_iterator::operator+=: Attempt to advance an iterator past the end");
    TEST_LIBCPP_ASSERT_FAILURE(
        i += -1, "__static_packed_bounded_iterator::operator+=: Attempt to rewind an iterator past the start");
  }

  { // operator-=
    std::optional<int> o{1};
    auto i = o.begin();

    TEST_LIBCPP_ASSERT_FAILURE(
        i -= 1, "__static_packed_bounded_iterator::operator-=: Attempt to rewind an iterator past the start");
    TEST_LIBCPP_ASSERT_FAILURE(
        i -= -2, "__static_packed_bounded_iterator::operator-=: Attempt to advance an iterator past the end");
  }
}

// short doesn't have the alignment required to use static_packed_bounded_iter
// so optional<short>'s iterator is bounded_iter with the option enabled.
void test_bounded_iter() {
  { // operator++
    std::optional<short> o{1};
    auto i = o.end();

    TEST_LIBCPP_ASSERT_FAILURE(++i, "__bounded_iter::operator++: Attempt to advance an iterator past the end");
    TEST_LIBCPP_ASSERT_FAILURE(i++, "__bounded_iter::operator++: Attempt to advance an iterator past the end");
  }

  { // operator--
    std::optional<short> o{1};
    auto i = o.begin();

    TEST_LIBCPP_ASSERT_FAILURE(--i, "__bounded_iter::operator--: Attempt to rewind an iterator past the start");
    TEST_LIBCPP_ASSERT_FAILURE(i--, "__bounded_iter::operator--: Attempt to rewind an iterator past the start");
  }

  { // operator*
    std::optional<short> o;
    auto i = o.begin();

    TEST_LIBCPP_ASSERT_FAILURE(*i, "__bounded_iter::operator*: Attempt to dereference an iterator at the end");
  }

  { // operator[]
    std::optional<short> o{1};
    auto i = o.begin();

    TEST_LIBCPP_ASSERT_FAILURE(i[1], "__bounded_iter::operator[]: Attempt to index an iterator at or past the end");
    TEST_LIBCPP_ASSERT_FAILURE(i[-1], "__bounded_iter::operator[]: Attempt to index an iterator past the start");
  }

  { // operator->
    std::optional<short> o{1};
    auto i = o.end();

    TEST_LIBCPP_ASSERT_FAILURE(
        i.operator->(), "__bounded_iter::operator->: Attempt to dereference an iterator at the end");
  }

  { // operator+=
    std::optional<short> o{1};
    auto i = o.begin();

    TEST_LIBCPP_ASSERT_FAILURE(i += 2, "__bounded_iter::operator+=: Attempt to advance an iterator past the end");
    TEST_LIBCPP_ASSERT_FAILURE(i += -1, "__bounded_iter::operator+=: Attempt to rewind an iterator past the start");
  }

  { // operator-=
    std::optional<short> o{1};
    auto i = o.begin();

    TEST_LIBCPP_ASSERT_FAILURE(i -= 1, "__bounded_iter::operator-=: Attempt to rewind an iterator past the start");
    TEST_LIBCPP_ASSERT_FAILURE(i -= -2, "__bounded_iter::operator-=: Attempt to advance an iterator past the end");
  }
}

int main(int, char**) {
  test_bounded_iter();
  test_packed_iter();

  return 0;
}
