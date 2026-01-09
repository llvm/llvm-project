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

#if TEST_STD_VER >= 23

#  include <concepts>
#  include <type_traits>

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

  friend bool operator==(StoredType const& lhs, StoredType const& rhs) { return lhs.value_ == rhs.value_; }

  friend bool operator==(StoredType const& lhs, SearchedType<T> const& rhs) { return lhs.value_ == rhs.get_value(); }

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
  static_assert(
      std::same_as<
          typename Container::size_type,
          std::invoke_result_t<decltype(&Container::template erase<SearchedType<int>>), Container, SearchedType<int>>>);

  int conversions = 0;

  assert(c.erase(SearchedType<int>(1, &conversions)) != 0);
  assert(c.erase(SearchedType<int>(2, &conversions)) != 0);
  assert(c.erase(SearchedType<int>(3, &conversions)) == 0);
  assert(conversions == 0);
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

template <typename NodeHandle>
concept node_handle_has_key = requires(NodeHandle nh) {
  { nh.key() };
};

template <class T, class Container>
void test_single_extract(SearchedType<T> key, Container& c) {
  auto node_handle = c.extract(key);

  assert(!node_handle.empty());

  if constexpr (node_handle_has_key<typename Container::node_type>) {
    assert(node_handle.key() == key);
  } else {
    assert(node_handle.value() == key);
  }
}

template <class Container>
void test_transparent_extract(Container c) {
  static_assert(std::same_as< typename Container::node_type,
                              std::invoke_result_t<decltype(&Container::template extract<SearchedType<int>>),
                                                   Container,
                                                   SearchedType<int>>>);

  int conversions = 0;

  test_single_extract(SearchedType<int>(1, &conversions), c);
  test_single_extract(SearchedType<int>(2, &conversions), c);

  assert(c.extract(SearchedType<int>(3, &conversions)).empty());
  assert(conversions == 0);
}

template <class Container>
void test_non_transparent_extract(Container c) {
  int conversions = 0;

  test_single_extract(SearchedType<int>(1, &conversions), c);
  test_single_extract(SearchedType<int>(2, &conversions), c);

  assert(c.extract(SearchedType<int>(3, &conversions)).empty());
  assert(conversions == 3);
}

#endif // TEST_STD_VER >= 23

#endif // TEST_TRANSPARENT_ASSOCIATIVE_H
