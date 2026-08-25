//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <inplace_vector>

// template<class InputIterator>
//   constexpr iterator insert(const_iterator position, InputIterator first, InputIterator last);

#include <cassert>
#include <concepts>
#include <inplace_vector>

#include "../common.h"
#include "test_iterators.h"
#include "test_macros.h"

namespace adl {
struct S {};
void make_move_iterator(S*) {}
} // namespace adl

template <class Iter>
constexpr void test_iterators() {
  int a[] = {3, 4};
  std::inplace_vector<int, 8> c{1, 2, 5};
  std::same_as<std::inplace_vector<int, 8>::iterator> decltype(auto) i = c.insert(c.begin() + 2, Iter(a), Iter(a + 2));
  assert(i == c.begin() + 2);
  assert_inplace_vector_equal(c, {1, 2, 3, 4, 5});

  i = c.insert(c.end(), Iter(a), Iter(a));
  assert(i == c.end());
  assert_inplace_vector_equal(c, {1, 2, 3, 4, 5});
}

constexpr bool test() {
  test_iterators<cpp17_input_iterator<int*> >();
  test_iterators<forward_iterator<int*> >();
  test_iterators<int*>();

  if (!TEST_IS_CONSTANT_EVALUATED || TEST_INPLACE_VECTOR_NONTRIVIAL_CONSTEXPR) {
    struct Wrapper {
      constexpr Wrapper(int n) : n_(n) {}
      void operator=(int) = delete;
      int n_;
    };

    int a[] = {1, 2, 3};
    std::inplace_vector<Wrapper, 4> c;
    c.insert(c.end(), a, a + 3);
    assert(c.size() == 3);
    assert(c[0].n_ == 1);
    assert(c[1].n_ == 2);
    assert(c[2].n_ == 3);
  }
  {
    std::inplace_vector<adl::S, 4> s;
    s.insert(s.end(), cpp17_input_iterator<adl::S*>(nullptr), cpp17_input_iterator<adl::S*>(nullptr));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

#ifndef TEST_HAS_NO_EXCEPTIONS
  int a[] = {3, 4};
  std::inplace_vector<int, 4> c{1, 2, 5};
  assert_throws_bad_alloc([&] { c.insert(c.begin() + 2, a, a + 2); });
  assert_inplace_vector_equal(c, {1, 2, 5});
#endif

  return 0;
}
