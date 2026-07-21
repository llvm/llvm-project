//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr iterator(iterator<!Const> i)
//    requires Const && convertible_to<iterator_t<V>, iterator_t<Base>> &&
//                 convertible_to<sentinel_t<V>, sentinel_t<Base>>

#include <cassert>
#include <concepts>
#include <ranges>
#include <utility>

#include "../types.h"

// A non-simple view over actual data. begin()/end() return int* when non-const
// and const int* when const, so iterator<false> and iterator<true> are distinct
// and the converting constructor is available (int* converts to const int*).
struct NonSimpleDataView : std::ranges::view_base {
  int* data_;
  int size_;

  constexpr NonSimpleDataView(int* d, int s) : data_(d), size_(s) {}
  NonSimpleDataView(NonSimpleDataView&&)            = default;
  NonSimpleDataView& operator=(NonSimpleDataView&&) = default;

  constexpr int* begin() { return data_; }
  constexpr const int* begin() const { return data_; }
  constexpr int* end() { return data_ + size_; }
  constexpr const int* end() const { return data_ + size_; }
};

static_assert(!simple_view<NonSimpleDataView>);

// Two unrelated input iterator types (no conversion between them).
struct IterA : InputIter<IterA> {};
struct IterB : InputIter<IterB> {
  friend constexpr bool operator==(const IterB&, const IterA&) { return true; }
};

// Two unrelated sentinel types (no conversion between them).
struct SentA {
  friend constexpr bool operator==(const SentA&, const IterA&) { return true; }
  friend constexpr bool operator==(const SentA&, const IterB&) { return true; }
};
struct SentB {
  friend constexpr bool operator==(const SentB&, const IterA&) { return true; }
  friend constexpr bool operator==(const SentB&, const IterB&) { return true; }
};

// Non-simple view where iterator conversion fails (IterA does not convert to IterB).
struct IterNonConvertibleView : std::ranges::view_base {
  constexpr IterA begin() { return {}; }
  constexpr IterB begin() const { return {}; }
  constexpr SentA end() const { return {}; }
};

// Non-simple view where sentinel conversion fails (SentA does not convert to SentB).
struct SentNonConvertibleView : std::ranges::view_base {
  constexpr IterA begin() const { return {}; }
  constexpr SentA end() { return {}; }
  constexpr SentB end() const { return {}; }
};

// Conversion succeeds: both iterator and sentinel types are convertible (int* -> const int*).
using ConvertibleSV    = std::ranges::stride_view<NonSimpleDataView>;
using ConvertibleIter  = std::ranges::iterator_t<ConvertibleSV>;
using ConvertibleCIter = std::ranges::iterator_t<const ConvertibleSV>;
static_assert(!std::same_as<ConvertibleIter, ConvertibleCIter>);
static_assert(std::convertible_to<ConvertibleIter, ConvertibleCIter>);
static_assert(std::constructible_from<ConvertibleCIter, ConvertibleIter>);

// Conversion fails: underlying iterator types are not convertible (IterA -> IterB).
using IterNCSV    = std::ranges::stride_view<IterNonConvertibleView>;
using IterNCIter  = std::ranges::iterator_t<IterNCSV>;
using IterNCCIter = std::ranges::iterator_t<const IterNCSV>;
static_assert(!std::same_as<IterNCIter, IterNCCIter>);
static_assert(!std::convertible_to<IterNCIter, IterNCCIter>);
static_assert(!std::constructible_from<IterNCCIter, IterNCIter>);

// Conversion fails: underlying sentinel types are not convertible (SentA -> SentB).
using SentNCSV    = std::ranges::stride_view<SentNonConvertibleView>;
using SentNCIter  = std::ranges::iterator_t<SentNCSV>;
using SentNCCIter = std::ranges::iterator_t<const SentNCSV>;
static_assert(!std::same_as<SentNCIter, SentNCCIter>);
static_assert(!std::convertible_to<SentNCIter, SentNCCIter>);
static_assert(!std::constructible_from<SentNCCIter, SentNCIter>);

constexpr bool test() {
  {
    // Convert non-const iterator to const iterator with stride 1.
    int arr[] = {10, 20, 30, 40, 50};
    auto sv   = ConvertibleSV(NonSimpleDataView(arr, 5), 1);
    auto it   = sv.begin();
    assert(*it == 10);

    ConvertibleCIter cit{it};
    assert(*cit == 10);

    ++cit;
    assert(*cit == 20);
    ++cit;
    assert(*cit == 30);
    ++cit;
    assert(*cit == 40);
    ++cit;
    assert(*cit == 50);
    ++cit;
    assert(cit == std::as_const(sv).end());
  }
  {
    // Convert non-const iterator to const iterator with stride 2.
    int arr[] = {10, 20, 30, 40, 50};
    auto sv   = ConvertibleSV(NonSimpleDataView(arr, 5), 2);
    auto it   = sv.begin();

    ConvertibleCIter cit{it};
    assert(*cit == 10);
    ++cit;
    assert(*cit == 30);
    ++cit;
    assert(*cit == 50);
    ++cit;
    assert(cit == std::as_const(sv).end());
  }
  {
    // Convert after advancing the non-const iterator.
    int arr[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto sv   = ConvertibleSV(NonSimpleDataView(arr, 9), 3);
    auto it   = sv.begin();
    ++it;
    assert(*it == 4);

    ConvertibleCIter cit{it};
    assert(*cit == 4);
    ++cit;
    assert(*cit == 7);
    ++cit;
    assert(cit == std::as_const(sv).end());
  }
  {
    // Convert with stride larger than range size.
    int arr[] = {42, 99};
    auto sv   = ConvertibleSV(NonSimpleDataView(arr, 2), 5);
    auto it   = sv.begin();

    ConvertibleCIter cit{it};
    assert(*cit == 42);
    ++cit;
    assert(cit == std::as_const(sv).end());
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
