//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// UNSUPPORTED: no-exceptions

// <inplace_vector>

// constexpr explicit inplace_vector(size_type n);
// constexpr inplace_vector(size_type n, const T& value);
// template<class InputIterator>
//   constexpr inplace_vector(InputIterator first, InputIterator last);
// constexpr inplace_vector(const inplace_vector&);
// constexpr inplace_vector(initializer_list<T> il);

#include <inplace_vector>

#include "../common.h"

void test_over_capacity_exceptions() {
  assert_throws_bad_alloc([] { std::inplace_vector<int, 2> c(3); });
  assert_throws_bad_alloc([] { std::inplace_vector<int, 2> c(3, 1); });
  assert_throws_bad_alloc([] { std::inplace_vector<int, 2> c{1, 2, 3}; });

  int a[] = {1, 2, 3};
  assert_throws_bad_alloc([&] { std::inplace_vector<int, 2> c(a, a + 3); });
}

void test_element_constructor_exceptions() {
  ThrowingValue::reset();
  ThrowingValue::throw_after = 0;
  try {
    std::inplace_vector<ThrowingValue, 4> c(1);
    assert(false);
  } catch (int) {
    assert(ThrowingValue::alive == 0);
  }

  ThrowingValue::reset();
  {
    ThrowingValue value(1);
    ThrowingValue::throw_after = 0;
    try {
      std::inplace_vector<ThrowingValue, 4> c(1, value);
      assert(false);
    } catch (int) {
      assert(ThrowingValue::alive == 1);
    }
  }
  assert(ThrowingValue::alive == 0);
}

void test_iterator_constructor_exceptions() {
  ThrowingValue::reset();
  {
    ThrowingValue values[] = {ThrowingValue(1), ThrowingValue(2)};
    try {
      std::inplace_vector<ThrowingValue, 4> c(
          throwing_iterator<ThrowingValue, std::input_iterator_tag>(values),
          throwing_iterator<ThrowingValue, std::input_iterator_tag>(values + 2, 2));
      assert(false);
    } catch (int) {
      assert(ThrowingValue::alive == 2);
    }
  }
  assert(ThrowingValue::alive == 0);

  ThrowingValue::reset();
  {
    ThrowingValue values[] = {ThrowingValue(1), ThrowingValue(2)};
    try {
      std::inplace_vector<ThrowingValue, 4> c(
          throwing_iterator<ThrowingValue, std::forward_iterator_tag>(values),
          throwing_iterator<ThrowingValue, std::forward_iterator_tag>(values + 2, 2));
      assert(false);
    } catch (int) {
      assert(ThrowingValue::alive == 2);
    }
  }
  assert(ThrowingValue::alive == 0);
}

void test_copy_constructor_exceptions() {
  ThrowingValue::reset();
  {
    std::inplace_vector<ThrowingValue, 4> source;
    source.emplace_back(1);
    source.emplace_back(2);

    ThrowingValue::throw_after = 1;
    try {
      std::inplace_vector<ThrowingValue, 4> copy(source);
      assert(false);
    } catch (int) {
      assert(source.size() == 2);
      assert(source[0].value == 1);
      assert(source[1].value == 2);
      assert(ThrowingValue::alive == 2);
    }
  }
  assert(ThrowingValue::alive == 0);
}

void test_initializer_list_constructor_exceptions() {
  ThrowingValue::reset();
  try {
    ThrowingValue::throw_after = 3;
    std::inplace_vector<ThrowingValue, 4> c{ThrowingValue(1), ThrowingValue(2)};
    assert(false);
  } catch (int) {
    assert(ThrowingValue::alive == 0);
  }
}

int main(int, char**) {
  test_over_capacity_exceptions();
  test_element_constructor_exceptions();
  test_iterator_constructor_exceptions();
  test_copy_constructor_exceptions();
  test_initializer_list_constructor_exceptions();

  return 0;
}
