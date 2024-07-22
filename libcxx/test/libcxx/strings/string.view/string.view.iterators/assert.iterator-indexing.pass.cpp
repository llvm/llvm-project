//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Make sure that std::string_view's iterators check for OOB accesses when the debug mode is enabled.

// REQUIRES: has-unix-headers, libcpp-has-abi-bounded-iterators
// UNSUPPORTED: libcpp-hardening-mode=none

#include <iterator>
#include <string_view>

#include "check_assertion.h"

template <typename Iter>
void test_iterator(Iter begin, Iter end, bool reverse) {
  ptrdiff_t distance = std::distance(begin, end);

  // Dereferencing an iterator at the end.
  {
    TEST_LIBCPP_ASSERT_FAILURE(
        *end,
        reverse ? "__bounded_iter::operator--: Attempt to rewind an iterator past the start"
                : "__bounded_iter::operator*: Attempt to dereference an iterator at the end");
#if _LIBCPP_STD_VER >= 20
    // In C++20 mode, std::reverse_iterator implements operator->, but not operator*, with
    // std::prev instead of operator--. std::prev ultimately calls operator+
    TEST_LIBCPP_ASSERT_FAILURE(
        end.operator->(),
        reverse ? "__bounded_iter::operator+=: Attempt to rewind an iterator past the start"
                : "__bounded_iter::operator->: Attempt to dereference an iterator at the end");
#else
    TEST_LIBCPP_ASSERT_FAILURE(
        end.operator->(),
        reverse ? "__bounded_iter::operator--: Attempt to rewind an iterator past the start"
                : "__bounded_iter::operator->: Attempt to dereference an iterator at the end");
#endif
  }

  // Incrementing an iterator past the end.
  {
    [[maybe_unused]] const char* msg =
        reverse ? "__bounded_iter::operator--: Attempt to rewind an iterator past the start"
                : "__bounded_iter::operator++: Attempt to advance an iterator past the end";
    auto it = end;
    TEST_LIBCPP_ASSERT_FAILURE(it++, msg);
    it = end;
    TEST_LIBCPP_ASSERT_FAILURE(++it, msg);
  }

  // Decrementing an iterator past the start.
  {
    [[maybe_unused]] const char* msg =
        reverse ? "__bounded_iter::operator++: Attempt to advance an iterator past the end"
                : "__bounded_iter::operator--: Attempt to rewind an iterator past the start";
    auto it = begin;
    TEST_LIBCPP_ASSERT_FAILURE(it--, msg);
    it = begin;
    TEST_LIBCPP_ASSERT_FAILURE(--it, msg);
  }

  // Advancing past the end with operator+= and operator+.
  {
    [[maybe_unused]] const char* msg =
        reverse ? "__bounded_iter::operator-=: Attempt to rewind an iterator past the start"
                : "__bounded_iter::operator+=: Attempt to advance an iterator past the end";
    auto it = end;
    TEST_LIBCPP_ASSERT_FAILURE(it += 1, msg);
    TEST_LIBCPP_ASSERT_FAILURE(end + 1, msg);
    it = begin;
    TEST_LIBCPP_ASSERT_FAILURE(it += (distance + 1), msg);
    TEST_LIBCPP_ASSERT_FAILURE(begin + (distance + 1), msg);
  }

  // Advancing past the end with operator-= and operator-.
  {
    [[maybe_unused]] const char* msg =
        reverse ? "__bounded_iter::operator+=: Attempt to rewind an iterator past the start"
                : "__bounded_iter::operator-=: Attempt to advance an iterator past the end";
    auto it = end;
    TEST_LIBCPP_ASSERT_FAILURE(it -= (-1), msg);
    TEST_LIBCPP_ASSERT_FAILURE(end - (-1), msg);
    it = begin;
    TEST_LIBCPP_ASSERT_FAILURE(it -= (-distance - 1), msg);
    TEST_LIBCPP_ASSERT_FAILURE(begin - (-distance - 1), msg);
  }

  // Rewinding past the start with operator+= and operator+.
  {
    [[maybe_unused]] const char* msg =
        reverse ? "__bounded_iter::operator-=: Attempt to advance an iterator past the end"
                : "__bounded_iter::operator+=: Attempt to rewind an iterator past the start";
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
        reverse ? "__bounded_iter::operator+=: Attempt to advance an iterator past the end"
                : "__bounded_iter::operator-=: Attempt to rewind an iterator past the start";
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
        reverse ? "__bounded_iter::operator--: Attempt to rewind an iterator past the start"
                : "__bounded_iter::operator[]: Attempt to index an iterator at or past the end";
    [[maybe_unused]] const char* past_end_msg =
        reverse ? "__bounded_iter::operator-=: Attempt to rewind an iterator past the start"
                : "__bounded_iter::operator[]: Attempt to index an iterator at or past the end";
    [[maybe_unused]] const char* past_start_msg =
        reverse ? "__bounded_iter::operator-=: Attempt to advance an iterator past the end"
                : "__bounded_iter::operator[]: Attempt to index an iterator past the start";
    TEST_LIBCPP_ASSERT_FAILURE(begin[distance], end_msg);
    TEST_LIBCPP_ASSERT_FAILURE(begin[distance + 1], past_end_msg);
    TEST_LIBCPP_ASSERT_FAILURE(begin[-1], past_start_msg);
    TEST_LIBCPP_ASSERT_FAILURE(begin[-99], past_start_msg);

    auto it = begin + 1;
    TEST_LIBCPP_ASSERT_FAILURE(it[distance - 1], end_msg);
    TEST_LIBCPP_ASSERT_FAILURE(it[distance], past_end_msg);
    TEST_LIBCPP_ASSERT_FAILURE(it[-2], past_start_msg);
    TEST_LIBCPP_ASSERT_FAILURE(it[-99], past_start_msg);
  }
}

int main(int, char**) {
  std::string_view const str("hello world");

  // string_view::iterator
  test_iterator(str.begin(), str.end(), /*reverse=*/false);

  // string_view::const_iterator
  test_iterator(str.cbegin(), str.cend(), /*reverse=*/false);

  // string_view::reverse_iterator
  test_iterator(str.rbegin(), str.rend(), /*reverse=*/true);

  // string_view::const_reverse_iterator
  test_iterator(str.crbegin(), str.crend(), /*reverse=*/true);

  return 0;
}
