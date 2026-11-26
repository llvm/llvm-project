//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// XFAIL: msvc

// <ranges>

// template <input_range _View>
//   requires view<_View>
// class cache_latest_view : public view_interface<cache_latest_view<_View>> {
//   _LIBCPP_NO_UNIQUE_ADDRESS _View __base_;
//   _LIBCPP_NO_UNIQUE_ADDRESS __non_propagating_cache<__cache_t> __cache_;

// Check that members are marked [[no_unique_address]].

#include <ranges>

struct EmptyView : std::ranges::view_base {
  int* begin() const;
  int* end() const;
};

struct NonEmptyView : std::ranges::view_base {
  int* begin() const;
  int* end() const;
  int i;
};

struct EmptyReference {
  friend bool operator==(const EmptyReference&, const EmptyReference&) = default;
};

struct EmptyReferenceIterator {
  using iterator_concept = std::input_iterator_tag;
  using value_type       = EmptyReference;
  using difference_type  = std::ptrdiff_t;

  EmptyReference operator*() const { return {}; }
  EmptyReferenceIterator& operator++() { return *this; }
  void operator++(int) {}
  friend bool operator==(const EmptyReferenceIterator&, const EmptyReferenceIterator&) = default;
};

struct EmptyReferenceView : std::ranges::view_base {
  EmptyReferenceIterator begin() const;
  EmptyReferenceIterator end() const;
};

using EmptyCache = std::ranges::__non_propagating_cache< std::add_pointer_t<std::ranges::range_reference_t<EmptyView>>>;

// Empty _View members can share storage with the cache.
static_assert(sizeof(std::ranges::cache_latest_view<EmptyView>) == sizeof(EmptyCache));

template <class View>
struct Test {
  [[no_unique_address]] View view;
  char c;
};

static_assert(sizeof(Test<std::ranges::cache_latest_view<EmptyView>>) ==
              sizeof(EmptyCache) + alignof(Test<std::ranges::cache_latest_view<EmptyView>>));

using EmptyReferenceCache = std::ranges::__non_propagating_cache<EmptyReference>;

// Empty cache and empty base subobjects can share storage.
static_assert(sizeof(std::ranges::cache_latest_view<EmptyReferenceView>) == sizeof(EmptyReferenceCache));

using NonEmptyCache =
    std::ranges::__non_propagating_cache< std::add_pointer_t<std::ranges::range_reference_t<NonEmptyView>>>;

// With a non-empty base, verify the cache layout against a type with the
// corresponding no-unique-address members.
struct NonEmptyLayout {
  [[no_unique_address]] NonEmptyView view;
  [[no_unique_address]] NonEmptyCache cache;
};

static_assert(sizeof(std::ranges::cache_latest_view<NonEmptyView>) == sizeof(NonEmptyLayout));
