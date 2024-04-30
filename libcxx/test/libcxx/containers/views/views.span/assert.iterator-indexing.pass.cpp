//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// Make sure that std::span's iterators check for OOB accesses when the debug mode is enabled.

// REQUIRES: has-unix-headers, libcpp-has-abi-bounded-iterators
// UNSUPPORTED: libcpp-hardening-mode=none

#include <span>

#include "check_assertion.h"

struct Foo {
  int x;
};

template <typename Iter>
void test_iterator(Iter begin, Iter end, bool reverse) {
  std::ptrdiff_t distance = std::distance(begin, end);

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
        end->x,
        reverse ? "__bounded_iter::operator+=: Attempt to rewind an iterator past the start"
                : "__bounded_iter::operator->: Attempt to dereference an iterator at the end");
#else
    TEST_LIBCPP_ASSERT_FAILURE(
        end->x,
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
    TEST_LIBCPP_ASSERT_FAILURE(++it, msg);
  }

  // Decrementing an iterator past the start.
  {
    [[maybe_unused]] const char* msg =
        reverse ? "__bounded_iter::operator++: Attempt to advance an iterator past the end"
                : "__bounded_iter::operator--: Attempt to rewind an iterator past the start";
    auto it = begin;
    TEST_LIBCPP_ASSERT_FAILURE(it--, msg);
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
  // span<T>::iterator
  {
    Foo array[] = {{0}, {1}, {2}};
    std::span<Foo> const span(array, 3);
    test_iterator(span.begin(), span.end(), /*reverse=*/false);
  }

  // span<T, N>::iterator
  {
    Foo array[] = {{0}, {1}, {2}};
    std::span<Foo, 3> const span(array, 3);
    test_iterator(span.begin(), span.end(), /*reverse=*/false);
  }

  // span<T>::reverse_iterator
  {
    Foo array[] = {{0}, {1}, {2}};
    std::span<Foo> const span(array, 3);
    test_iterator(span.rbegin(), span.rend(), /*reverse=*/true);
  }

  // span<T, N>::reverse_iterator
  {
    Foo array[] = {{0}, {1}, {2}};
    std::span<Foo, 3> const span(array, 3);
    test_iterator(span.rbegin(), span.rend(), /*reverse=*/true);
  }

  return 0;
}
