//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <optional>

// Add to iterator out of bounds.

// REQUIRES: std-at-least-c++26
// REQUIRES: can-test-hardening-assertions-fast
// UNSUPPORTED: libcpp-has-abi-bounded-iterators-in-optional

#include <optional>

#include "check_assertion.h"

// If T fits the requirements for __static_packed_bounded_iterator, optional<T>'s iterator will always be that,
// regardless of if the ABI switch is on or off. This is because it has less overhead than bounded_iter while providing the same guarantees for the on case,
// while for the off case, it has more guarantees than __capacity_aware_iterator while occupying the same amount of space.
// Otherwise, optional<T>'s iterator is __capacity_aware_iterator.
void test_packed() {
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

void test_nonpacked() {
  {
    std::optional<short> opt(1);
    auto i = opt.begin();

    TEST_LIBCPP_ASSERT_FAILURE(
        i += 2,
        "__capacity_aware_iterator::operator+=: Attempting to move iterator past its container's possible range");

    TEST_LIBCPP_ASSERT_FAILURE(
        i += -2,
        "__capacity_aware_iterator::operator+=: Attempting to move iterator past its container's possible range");

    TEST_LIBCPP_ASSERT_FAILURE(
        i -= 2,
        "__capacity_aware_iterator::operator-=: Attempting to move iterator past its container's possible range");

    TEST_LIBCPP_ASSERT_FAILURE(
        i -= -2,
        "__capacity_aware_iterator::operator-=: Attempting to move iterator past its container's possible range");

    TEST_LIBCPP_ASSERT_FAILURE(
        i[2],
        "__capacity_aware_iterator::operator[]: Attempting to index iterator past its container's possible range");

    TEST_LIBCPP_ASSERT_FAILURE(
        i[-2],
        "__capacity_aware_iterator::operator[]: Attempting to index iterator past its container's possible range");
  }
}

int main(int, char**) {
  test_packed();
  test_nonpacked();
  return 0;
}
