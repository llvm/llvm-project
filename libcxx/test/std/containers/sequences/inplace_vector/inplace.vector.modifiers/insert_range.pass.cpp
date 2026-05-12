//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-steps): -fconstexpr-steps=2000000

// <inplace_vector>

// template<container-compatible-range<T> R>
//   constexpr iterator insert_range(const_iterator position, R&& rg);

#include <array>
#include <cassert>
#include <concepts>
#include <inplace_vector>

#include "../../insert_range_sequence_containers.h"
#include "../common.h"
#include "test_macros.h"

template <class Iter, class Sent>
constexpr void test_iterators() {
  test_sequence_insert_range<InplaceVector<int>, Iter, Sent>([]([[maybe_unused]] auto&& c) {});
}

constexpr bool test() {
  static_assert(test_constraints_insert_range<InplaceVector, int, double>());

  test_iterators<cpp20_input_iterator<const int*>, sentinel_wrapper<cpp20_input_iterator<const int*> > >();
  test_iterators<forward_iterator<const int*>, sentinel_wrapper<forward_iterator<const int*> > >();
  test_iterators<const int*, const int*>();

  if (!TEST_IS_CONSTANT_EVALUATED || TEST_INPLACE_VECTOR_NONTRIVIAL_CONSTEXPR) {
    test_sequence_insert_range_move_only<InplaceVector>();
  }

  {
    int in[] = {-1, -2, -3};
    std::inplace_vector<int, 8> c{1, 2, 6};
    std::same_as<std::inplace_vector<int, 8>::iterator> decltype(auto) i = c.insert_range(c.begin() + 2, in);
    assert(i == c.begin() + 2);
    assert_inplace_vector_equal(c, {1, 2, -1, -2, -3, 6});
  }

  // Ensure that insert_range doesn't use unexpected assignment.
  if (!TEST_IS_CONSTANT_EVALUATED || TEST_INPLACE_VECTOR_NONTRIVIAL_CONSTEXPR) {
    struct Wrapper {
      constexpr Wrapper(int n) : n_(n) {}
      void operator=(int) = delete;

      int n_;
    };

    int a[]{1, 2, 3, 4, 5};
    std::inplace_vector<Wrapper, 8> c;
    c.insert_range(c.end(), a);
    assert(c.size() == std::size(a));
    for (std::size_t i = 0; i != std::size(a); ++i)
      assert(c[i].n_ == a[i]);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  test_insert_range_exception_safety_throwing_copy<InplaceVector>();

#ifndef TEST_HAS_NO_EXCEPTIONS
  int in[] = {3, 4};
  std::inplace_vector<int, 3> c{1, 2};
  assert_throws_bad_alloc([&] { c.insert_range(c.begin(), in); });
  assert_inplace_vector_equal(c, {1, 2});
#endif

  return 0;
}
