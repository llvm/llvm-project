//===-- include/flang/Common/static-multimap-view.h -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_COMMON_STATIC_MULTIMAP_VIEW_H_
#define FORTRAN_COMMON_STATIC_MULTIMAP_VIEW_H_
#include <algorithm>
#include <utility>

/// StaticMultimapView is a constexpr friendly multimap implementation over
/// sorted constexpr arrays. As the View name suggests, it does not duplicate
/// the sorted array but only brings range and search concepts over it. It
/// mainly erases the array size from the type and ensures the array is sorted
/// at compile time. When C++20 brings std::span and constexpr std::is_sorted,
/// this can most likely be replaced by those.

namespace Fortran::common {

template <typename V> class StaticMultimapView {
public:
  using Key = typename V::Key;
  using const_iterator = const V *;

  constexpr const_iterator begin() const { return begin_; }
  constexpr const_iterator end() const { return end_; }
  // Be sure to static_assert(map.Verify(), "must be sorted"); for
  // every instance constexpr created. Sadly this cannot be done in
  // the ctor since there is no way to know whether the ctor is actually
  // called at compile time or not.
  template <std::size_t N>
  constexpr StaticMultimapView(const V (&array)[N])
      : begin_{&array[0]}, end_{&array[0] + N} {}

  // std::equal_range will be constexpr in C++20 only, so far there is actually
  // no need for equal_range to be constexpr anyway.
  std::pair<const_iterator, const_iterator> equal_range(const Key &key) const {
    return std::equal_range(begin_, end_, key);
  }

  // Check that the array is sorted. This used to assert at compile time that
  // the array is indeed sorted. When C++20 is required for flang,
  // std::is_sorted can be used here since it will be constexpr.
  constexpr bool Verify() const {
    const V *lastSeen{begin_};
    bool isSorted{true};
    for (const auto *x{begin_}; x != end_; ++x) {
      isSorted &= lastSeen->key <= x->key;
      lastSeen = x;
    }
    return isSorted;
  }

private:
  const_iterator begin_{nullptr};
  const_iterator end_{nullptr};
};
} // namespace Fortran::common
#endif // FORTRAN_COMMON_STATIC_MULTIMAP_VIEW_H_
