//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// UNSUPPORTED: libcpp-has-no-experimental-hive

// <hive>

// __hive_iterator: member types, size, triviality, the const conversion, the bidirectional iterator concept, and
// construction from a block and an index through std::__make_hive_iterator only.

#include <cassert>
#include <concepts>
#include <cstddef>
#include <hive>
#include <iterator>
#include <memory>
#include <type_traits>

#include "min_allocator.h"
#include "test_macros.h"

template <class T, class VoidPtr>
void check() {
  using It   = std::__hive_iterator<T, VoidPtr, false>;
  using CIt  = std::__hive_iterator<T, VoidPtr, true>;
  using Ptr  = typename std::pointer_traits<VoidPtr>::template rebind<T>;
  using CPtr = typename std::pointer_traits<VoidPtr>::template rebind<const T>;

  static_assert(std::same_as<typename It::iterator_category, std::bidirectional_iterator_tag>);
  static_assert(std::same_as<typename It::value_type, T>);
  static_assert(std::same_as<typename It::difference_type, typename std::pointer_traits<VoidPtr>::difference_type>);
  static_assert(std::same_as<typename It::reference, T&>);
  static_assert(std::same_as<typename It::pointer, Ptr>);
  static_assert(std::same_as<typename CIt::value_type, T>);
  static_assert(std::same_as<typename CIt::reference, const T&>);
  static_assert(std::same_as<typename CIt::pointer, CPtr>);

  static_assert(std::bidirectional_iterator<It>);
  static_assert(std::bidirectional_iterator<CIt>);
  static_assert(std::is_trivially_copyable_v<It>);
  static_assert(std::is_trivially_copyable_v<CIt>);
  static_assert(std::is_nothrow_default_constructible_v<It>);
  static_assert(std::is_convertible_v<It, CIt>);
  static_assert(!std::is_convertible_v<CIt, It>);
  static_assert(sizeof(It) == 2 * sizeof(Ptr));

  // A block and an index make an iterator only through std::__make_hive_iterator.
  using BlockPtr = typename std::pointer_traits<VoidPtr>::template rebind<std::__hive_block<T, VoidPtr>>;
  static_assert(!std::is_constructible_v<It, BlockPtr, std::size_t>);
  static_assert(!std::is_constructible_v<CIt, BlockPtr, std::size_t>);
  static_assert(!std::is_constructible_v<It, std::nullptr_t, std::size_t>);
  static_assert(noexcept(std::__make_hive_iterator<It>(BlockPtr(), 0)));

  std::same_as<It> decltype(auto) made = std::__make_hive_iterator<It>(BlockPtr(), 0);
  assert(made == It());
  std::same_as<CIt> decltype(auto) const_made = std::__make_hive_iterator<CIt>(BlockPtr(), 0);
  assert(const_made == CIt());
}

// Value-initialized iterators are the empty hive's end, and equal, across constness too.
void test_value_initialized() {
  using It  = std::__hive_iterator<int, void*, false>;
  using CIt = std::__hive_iterator<int, void*, true>;
  It a;
  It b;
  CIt c;
  assert(a == b);
  assert(a == c);
  assert(c == a);
}

int main(int, char**) {
  check<int, void*>();
  check<int, min_pointer<void>>();
  test_value_initialized();

  return 0;
}
