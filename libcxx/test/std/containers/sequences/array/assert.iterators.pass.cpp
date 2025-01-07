//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-unix-headers, libcpp-has-abi-bounded-iterators-in-std-array
// UNSUPPORTED: c++03
// UNSUPPORTED: libcpp-hardening-mode=none
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

// <array>

// Make sure that std::array's iterators check for OOB accesses when the right hardening settings
// are enabled.

#include <array>
#include <cstddef>
#include <iterator>

#include "check_assertion.h"

template <typename Iter>
void test_iterator(Iter begin, Iter end) {
  std::ptrdiff_t distance = std::distance(begin, end);

  // Dereferencing an iterator at the end.
  {
    TEST_LIBCPP_ASSERT_FAILURE(*end, "__static_bounded_iter::operator*: Attempt to dereference an iterator at the end");
    TEST_LIBCPP_ASSERT_FAILURE(
        end.operator->(), "__static_bounded_iter::operator->: Attempt to dereference an iterator at the end");
  }

  // Incrementing an iterator past the end.
  {
    auto it = end;
    TEST_LIBCPP_ASSERT_FAILURE(it++, "__static_bounded_iter::operator++: Attempt to advance an iterator past the end");
    it = end;
    TEST_LIBCPP_ASSERT_FAILURE(++it, "__static_bounded_iter::operator++: Attempt to advance an iterator past the end");
  }

  // Decrementing an iterator past the start.
  {
    auto it = begin;
    TEST_LIBCPP_ASSERT_FAILURE(it--, "__static_bounded_iter::operator--: Attempt to rewind an iterator past the start");
    it = begin;
    TEST_LIBCPP_ASSERT_FAILURE(--it, "__static_bounded_iter::operator--: Attempt to rewind an iterator past the start");
  }

  // Advancing past the end with operator+= and operator+.
  {
    [[maybe_unused]] const char* msg = "__static_bounded_iter::operator+=: Attempt to advance an iterator past the end";
    auto it                          = end;
    TEST_LIBCPP_ASSERT_FAILURE(it += 1, msg);
    TEST_LIBCPP_ASSERT_FAILURE(end + 1, msg);
    it = begin;
    TEST_LIBCPP_ASSERT_FAILURE(it += (distance + 1), msg);
    TEST_LIBCPP_ASSERT_FAILURE(begin + (distance + 1), msg);
  }

  // Advancing past the end with operator-= and operator-.
  {
    [[maybe_unused]] const char* msg = "__static_bounded_iter::operator-=: Attempt to advance an iterator past the end";
    auto it                          = end;
    TEST_LIBCPP_ASSERT_FAILURE(it -= (-1), msg);
    TEST_LIBCPP_ASSERT_FAILURE(end - (-1), msg);
    it = begin;
    TEST_LIBCPP_ASSERT_FAILURE(it -= (-distance - 1), msg);
    TEST_LIBCPP_ASSERT_FAILURE(begin - (-distance - 1), msg);
  }

  // Rewinding past the start with operator+= and operator+.
  {
    [[maybe_unused]] const char* msg =
        "__static_bounded_iter::operator+=: Attempt to rewind an iterator past the start";
    auto it = begin;
    TEST_LIBCPP_ASSERT_FAILURE(it += (-1), msg);
    TEST_LIBCPP_ASSERT_FAILURE(begin + (-1), msg);
    it = end;
    TEST_LIBCPP_ASSERT_FAILURE(it += (-distance - 1), msg);
    TEST_LIBCPP_ASSERT_FAILURE(end + (-distance - 1), msg);
  }

  // Rewinding past the start with operator-= and operator-.
  {
    [[maybe_unused]] const char* msg =
        "__static_bounded_iter::operator-=: Attempt to rewind an iterator past the start";
    auto it = begin;
    TEST_LIBCPP_ASSERT_FAILURE(it -= 1, msg);
    TEST_LIBCPP_ASSERT_FAILURE(begin - 1, msg);
    it = end;
    TEST_LIBCPP_ASSERT_FAILURE(it -= (distance + 1), msg);
    TEST_LIBCPP_ASSERT_FAILURE(end - (distance + 1), msg);
  }

  // Out-of-bounds operator[].
  {
    [[maybe_unused]] const char* end_msg =
        "__static_bounded_iter::operator[]: Attempt to index an iterator at or past the end";
    [[maybe_unused]] const char* past_end_msg =
        "__static_bounded_iter::operator[]: Attempt to index an iterator at or past the end";
    [[maybe_unused]] const char* past_start_msg =
        "__static_bounded_iter::operator[]: Attempt to index an iterator past the start";
    TEST_LIBCPP_ASSERT_FAILURE(begin[distance], end_msg);
    TEST_LIBCPP_ASSERT_FAILURE(begin[distance + 1], past_end_msg);
    TEST_LIBCPP_ASSERT_FAILURE(begin[-1], past_start_msg);
    TEST_LIBCPP_ASSERT_FAILURE(begin[-99], past_start_msg);

    if (distance > 0) {
      auto it = begin + 1;
      TEST_LIBCPP_ASSERT_FAILURE(it[distance - 1], end_msg);
      TEST_LIBCPP_ASSERT_FAILURE(it[distance], past_end_msg);
      TEST_LIBCPP_ASSERT_FAILURE(it[-2], past_start_msg);
      TEST_LIBCPP_ASSERT_FAILURE(it[-99], past_start_msg);
    }
  }
}

int main(int, char**) {
  // Empty array
  {
    std::array<int, 0> array = {};

    // array::iterator
    test_iterator(array.begin(), array.end());

    // array::const_iterator
    test_iterator(array.cbegin(), array.cend());
  }

  // Non-empty array
  {
    std::array<int, 10> array = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // array::iterator
    test_iterator(array.begin(), array.end());

    // array::const_iterator
    test_iterator(array.cbegin(), array.cend());
  }

  return 0;
}
