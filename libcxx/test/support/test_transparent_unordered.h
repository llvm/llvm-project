//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_TRANSPARENT_UNORDERED_H
#define TEST_TRANSPARENT_UNORDERED_H

#include "test_macros.h"
#include "is_transparent.h"

#include <cassert>

#if TEST_STD_VER > 17

#  include <concepts>
#  include <type_traits>

template <typename T>
struct StoredType;

template <typename T>
struct SearchedType;

struct hash_impl {
  template <typename T>
  constexpr std::size_t operator()(SearchedType<T> const& t) const {
    return static_cast<std::size_t>(t.get_value());
  }

  template <typename T>
  constexpr std::size_t operator()(StoredType<T> const& t) const {
    return static_cast<std::size_t>(t.get_value());
  }
};

struct non_transparent_hash : hash_impl {};

struct transparent_hash : hash_impl {
  using is_transparent = void;
};

struct transparent_hash_final final : transparent_hash {};

struct transparent_equal_final final : std::equal_to<> {};

template <typename T>
struct SearchedType {
  explicit SearchedType(T value, int *counter) : value_(value), conversions_(counter) { }

  // Whenever a conversion is performed, increment the counter to keep track
  // of conversions.
  operator StoredType<T>() const {
    ++*conversions_;
    return StoredType<T>{value_};
  }

  int get_value() const {
    return value_;
  }

private:
  T value_;
  int *conversions_;
};

template <typename T>
struct StoredType {
  StoredType() = default;
  StoredType(T value) : value_(value) { }

  friend bool operator==(StoredType const& lhs, StoredType const& rhs) {
    return lhs.value_ == rhs.value_;
  }

  // If we're being passed a SearchedType<T> object, avoid the conversion
  // to T. This allows testing that the transparent operations are correctly
  // forwarding the SearchedType all the way to this comparison by checking
  // that we didn't have a conversion when we search for a SearchedType<T>
  // in a container full of StoredType<T>.
  friend bool operator==(StoredType const& lhs, SearchedType<T> const& rhs) {
    return lhs.value_ == rhs.get_value();
  }

  int get_value() const {
    return value_;
  }

private:
  T value_;
};

template<template<class...> class UnorderedSet, class Hash, class Equal>
using unord_set_type = UnorderedSet<StoredType<int>, Hash, Equal>;

template<template<class...> class UnorderedMap, class Hash, class Equal>
using unord_map_type = UnorderedMap<StoredType<int>, int, Hash, Equal>;

template<class Container>
void test_transparent_find(Container c) {
  int conversions = 0;
  assert(c.find(SearchedType<int>(1, &conversions)) != c.end());
  assert(c.find(SearchedType<int>(2, &conversions)) != c.end());
  assert(c.find(SearchedType<int>(3, &conversions)) == c.end());
  assert(conversions == 0);
}

template<class Container>
void test_non_transparent_find(Container c) {
  int conversions = 0;
  assert(c.find(SearchedType<int>(1, &conversions)) != c.end());
  assert(conversions == 1);
  assert(c.find(SearchedType<int>(2, &conversions)) != c.end());
  assert(conversions == 2);
  assert(c.find(SearchedType<int>(3, &conversions)) == c.end());
  assert(conversions == 3);
}

template<class Container>
void test_transparent_count(Container c) {
  int conversions = 0;
  assert(c.count(SearchedType<int>(1, &conversions)) > 0);
  assert(c.count(SearchedType<int>(2, &conversions)) > 0);
  assert(c.count(SearchedType<int>(3, &conversions)) == 0);
  assert(conversions == 0);
}

template<class Container>
void test_non_transparent_count(Container c) {
  int conversions = 0;
  assert(c.count(SearchedType<int>(1, &conversions)) > 0);
  assert(conversions == 1);
  assert(c.count(SearchedType<int>(2, &conversions)) > 0);
  assert(conversions == 2);
  assert(c.count(SearchedType<int>(3, &conversions)) == 0);
  assert(conversions == 3);
}

template<class Container>
void test_transparent_contains(Container c) {
  int conversions = 0;
  assert(c.contains(SearchedType<int>(1, &conversions)));
  assert(c.contains(SearchedType<int>(2, &conversions)));
  assert(!c.contains(SearchedType<int>(3, &conversions)));
  assert(conversions == 0);
}

template<class Container>
void test_non_transparent_contains(Container c) {
  int conversions = 0;
  assert(c.contains(SearchedType<int>(1, &conversions)));
  assert(conversions == 1);
  assert(c.contains(SearchedType<int>(2, &conversions)));
  assert(conversions == 2);
  assert(!c.contains(SearchedType<int>(3, &conversions)));
  assert(conversions == 3);
}

template<class Container>
void test_transparent_equal_range(Container c) {
  int conversions = 0;
  auto iters = c.equal_range(SearchedType<int>(1, &conversions));
  assert(std::distance(iters.first, iters.second) > 0);
  iters = c.equal_range(SearchedType<int>(2, &conversions));
  assert(std::distance(iters.first, iters.second) > 0);
  iters = c.equal_range(SearchedType<int>(3, &conversions));
  assert(std::distance(iters.first, iters.second) == 0);
  assert(conversions == 0);
}

template<class Container>
void test_non_transparent_equal_range(Container c) {
  int conversions = 0;
  auto iters = c.equal_range(SearchedType<int>(1, &conversions));
  assert(std::distance(iters.first, iters.second) > 0);
  assert(conversions == 1);
  iters = c.equal_range(SearchedType<int>(2, &conversions));
  assert(std::distance(iters.first, iters.second) > 0);
  assert(conversions == 2);
  iters = c.equal_range(SearchedType<int>(3, &conversions));
  assert(std::distance(iters.first, iters.second) == 0);
  assert(conversions == 3);
}

#  if TEST_STD_VER >= 23

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

#  endif // TEST_STD_VER >= 23

#endif // TEST_STD_VER > 17

#endif // TEST_TRANSPARENT_UNORDERED_H
