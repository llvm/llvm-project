//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_TRANSPARENT_ASSOCIATIVE_H
#define TEST_TRANSPARENT_ASSOCIATIVE_H

#include "test_macros.h"

#include <cassert>

#if TEST_STD_VER > 23

template <typename T>
struct StoredType;

template <typename T>
struct SearchedType {
  explicit SearchedType(T value, int* counter) : value_(value), conversions_(counter) {}

  operator StoredType<T>() const {
    ++*conversions_;
    return StoredType<T>{value_};
  }

  T get_value() const { return value_; }

  auto operator<=>(const SearchedType<T>&) const = default;

private:
  T value_;
  int* conversions_;
};

template <typename T>
struct StoredType {
  StoredType() = default;
  StoredType(T value) : value_(value) {}

  T get_value() const { return value_; }

  auto operator<=>(const StoredType<T>&) const = default;

private:
  T value_;
};

struct transparent_comparator_base {
  using is_transparent = void;

  template <typename T>
  bool operator()(const SearchedType<T>& lhs, const StoredType<T>& rhs) const {
    return lhs.get_value() < rhs.get_value();
  }

  template <typename T>
  bool operator()(const StoredType<T>& lhs, const SearchedType<T>& rhs) const {
    return lhs.get_value() < rhs.get_value();
  }

  template <typename T>
  bool operator()(const StoredType<T>& lhs, const StoredType<T>& rhs) const {
    return lhs < rhs;
  }
};

struct transparent_comparator_final final : public transparent_comparator_base {};

template <class Container>
void test_transparent_erase(Container c) {
  int conversions = 0;
  assert(c.erase(SearchedType<int>(1, &conversions)) != 0);
  assert(c.erase(SearchedType<int>(2, &conversions)) != 0);
  assert(c.erase(SearchedType<int>(3, &conversions)) == 0);

  assert(conversions == 0);

  c.erase(c.begin());
  c.erase(c.cbegin());

  assert(c.empty());
}

template <class Container>
void test_non_transparent_erase(Container c) {
  int conversions = 0;
  assert(c.erase(SearchedType<int>(1, &conversions)) != 0);
  assert(conversions == 1);
  assert(c.erase(SearchedType<int>(2, &conversions)) != 0);
  assert(conversions == 2);
  assert(c.erase(SearchedType<int>(3, &conversions)) == 0);
  assert(conversions == 3);
}

template <class Container>
void test_transparent_extract(Container c) {
  int conversions = 0;
  assert(!c.extract(SearchedType<int>(1, &conversions)).empty());
  assert(!c.extract(SearchedType<int>(2, &conversions)).empty());
  assert(c.extract(SearchedType<int>(3, &conversions)).empty());
  assert(conversions == 0);

  assert(!c.extract(c.cbegin()).empty());
  assert(c.empty());
}

template <class Container>
void test_non_transparent_extract(Container c) {
  int conversions = 0;
  assert(!c.extract(SearchedType<int>(1, &conversions)).empty());
  assert(conversions == 1);
  assert(!c.extract(SearchedType<int>(2, &conversions)).empty());
  assert(conversions == 2);
  assert(c.extract(SearchedType<int>(3, &conversions)).empty());
  assert(conversions == 3);
}

#endif // TEST_STD_VER > 23

#endif // TEST_TRANSPARENT_ASSOCIATIVE_H
