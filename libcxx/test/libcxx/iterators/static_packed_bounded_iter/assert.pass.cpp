//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// UNSUPPORTED: libcpp-hardening-mode=none

// template <class _Ptr, class _Tag, class _RangeCapacity>
// class __static_packed_bounded_iterator;

// Check assert failure if advancing, rewinding past start or past end, and dereferencing if at end

#include <__iterator/static_packed_bounded_iter.h>
#include <iterator>

#include "check_assertion.h"
#include "test_macros.h"

struct TEST_ALIGNAS(4) Foo {
  char x;
  Foo(int y) : x(y) {}
};

template <typename Ptr, typename Ty = Foo>
void test() {
  Ty arr[] = {1, 2};

  constexpr long sz = std::size(arr);

  using BoundedIter = std::__static_packed_bounded_iterator<Ptr, decltype(arr), sz>;

  BoundedIter it = std::__make_static_packed_bounded_iter<Ptr, decltype(arr), sz>(Ptr(arr));

  TEST_LIBCPP_ASSERT_FAILURE(
      it--, "__static_packed_bounded_iterator::operator--: Attempt to rewind an iterator past the start");
  TEST_LIBCPP_ASSERT_FAILURE(
      --it, "__static_packed_bounded_iterator::operator--: Attempt to rewind an iterator past the start");

  TEST_LIBCPP_ASSERT_FAILURE(
      it += -1, "__static_packed_bounded_iterator::operator+=: Attempt to rewind an iterator past the start");
  TEST_LIBCPP_ASSERT_FAILURE(
      it += (sz + 1), "__static_packed_bounded_iterator::operator+=: Attempt to advance an iterator past the end");

  TEST_LIBCPP_ASSERT_FAILURE(
      it -= 1, "__static_packed_bounded_iterator::operator-=: Attempt to rewind an iterator past the start");
  TEST_LIBCPP_ASSERT_FAILURE(
      it -= -(sz + 1), "__static_packed_bounded_iterator::operator-=: Attempt to advance an iterator past the end");

  TEST_LIBCPP_ASSERT_FAILURE(
      it[sz], "__static_packed_bounded_iterator::operator[]: Attempt to index an iterator at or past the end");
  TEST_LIBCPP_ASSERT_FAILURE(
      it[-1], "__static_packed_bounded_iterator::operator[]: Attempt to index an iterator past the start");

  it += sz;

  TEST_LIBCPP_ASSERT_FAILURE(
      it++, "__static_packed_bounded_iterator::operator++: Attempt to advance an iterator past the end");
  TEST_LIBCPP_ASSERT_FAILURE(
      ++it, "__static_packed_bounded_iterator::operator++: Attempt to advance an iterator past the end");

  TEST_LIBCPP_ASSERT_FAILURE(
      *it, "__static_packed_bounded_iterator::operator*: Attempt to dereference an iterator at the end");
  TEST_LIBCPP_ASSERT_FAILURE(
      it.operator->(), "__static_packed_bounded_iterator::operator->: Attempt to dereference an iterator at the end");
}

int main(int, char**) {
  test<Foo*>();

  return 0;
}
