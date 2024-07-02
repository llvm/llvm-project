//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// template <class _Iterator>
// struct __bounded_iter;
//
// Dereference and indexing operators

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03
// UNSUPPORTED: libcpp-hardening-mode=none
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

#include <iterator>

#include "check_assertion.h"
#include "test_iterators.h"
#include "test_macros.h"

struct Foo {
  int x;
  TEST_CONSTEXPR bool operator==(Foo const& other) const { return x == other.x; }
};

template <class Iter>
TEST_CONSTEXPR_CXX14 bool tests() {
  Foo array[]                           = {Foo{40}, Foo{41}, Foo{42}, Foo{43}, Foo{44}};
  Foo* b                                = array + 0;
  Foo* e                                = array + 5;
  std::__bounded_iter<Iter> const iter1 = std::__make_bounded_iter(Iter(b), Iter(b), Iter(e));
  std::__bounded_iter<Iter> const iter2 = std::__make_bounded_iter(Iter(e), Iter(b), Iter(e));

  // operator*
  assert(*iter1 == Foo{40});
  // operator->
  assert(iter1->x == 40);
  // operator[]
  assert(iter1[0] == Foo{40});
  assert(iter1[1] == Foo{41});
  assert(iter1[2] == Foo{42});
  assert(iter2[-1] == Foo{44});
  assert(iter2[-2] == Foo{43});

  return true;
}

template <class Iter>
void test_death() {
  Foo array[]                          = {Foo{0}, Foo{1}, Foo{2}, Foo{3}, Foo{4}};
  Foo* b                               = array + 0;
  Foo* e                               = array + 5;
  std::__bounded_iter<Iter> const iter = std::__make_bounded_iter(Iter(b), Iter(b), Iter(e));
  std::__bounded_iter<Iter> const oob  = std::__make_bounded_iter(Iter(e), Iter(b), Iter(e));

  // operator*
  TEST_LIBCPP_ASSERT_FAILURE(*oob, "__bounded_iter::operator*: Attempt to dereference an iterator at the end");
  // operator->
  TEST_LIBCPP_ASSERT_FAILURE(oob->x, "__bounded_iter::operator->: Attempt to dereference an iterator at the end");
  // operator[]
  TEST_LIBCPP_ASSERT_FAILURE(iter[-1], "__bounded_iter::operator[]: Attempt to index an iterator past the start");
  TEST_LIBCPP_ASSERT_FAILURE(iter[5], "__bounded_iter::operator[]: Attempt to index an iterator at or past the end");
  TEST_LIBCPP_ASSERT_FAILURE(oob[0], "__bounded_iter::operator[]: Attempt to index an iterator at or past the end");
  TEST_LIBCPP_ASSERT_FAILURE(oob[1], "__bounded_iter::operator[]: Attempt to index an iterator at or past the end");
  TEST_LIBCPP_ASSERT_FAILURE(oob[-6], "__bounded_iter::operator[]: Attempt to index an iterator past the start");
}

int main(int, char**) {
  tests<Foo*>();
  test_death<Foo*>();
#if TEST_STD_VER > 11
  static_assert(tests<Foo*>(), "");
#endif

#if TEST_STD_VER > 17
  tests<contiguous_iterator<Foo*> >();
  test_death<contiguous_iterator<Foo*> >();
  static_assert(tests<contiguous_iterator<Foo*> >(), "");
#endif

  return 0;
}
