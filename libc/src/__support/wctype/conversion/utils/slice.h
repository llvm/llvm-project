//===-- Internal utils for wctype conversion code - slice -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Similar to cpp::span with additional functionality

#ifndef LLVM_LIBC_SRC___SUPPORT_WCTYPE_CONVERSION_UTILS_SLICE_H
#define LLVM_LIBC_SRC___SUPPORT_WCTYPE_CONVERSION_UTILS_SLICE_H

#include "hdr/types/size_t.h"
#include "src/__support/CPP/algorithm.h"
#include "src/__support/CPP/expected.h"
#include "src/__support/CPP/span.h"
#include "src/__support/libc_assert.h"

namespace LIBC_NAMESPACE_DECL {

namespace wctype_internal {

namespace conversion_utils {

enum class Ordering {
  /// An ordering where a compared value is less than another.
  Less = -1,
  /// An ordering where a compared value is equal to another.
  Equal = 0,
  /// An ordering where a compared value is greater than another.
  Greater = 1,
};

template <typename T> struct Slice : public cpp::span<T> {
  LIBC_INLINE constexpr Slice() : cpp::span<T>() {}

  LIBC_INLINE constexpr Slice(T *ptr, size_t len) : cpp::span<T>(ptr, len) {}
  LIBC_INLINE constexpr Slice(T const *ptr, size_t len)
      : cpp::span<T>(ptr, len) {}

  template <typename U, size_t N>
  LIBC_INLINE constexpr Slice(cpp::array<U, N> &arr) : cpp::span<T>(arr) {}

  LIBC_INLINE constexpr Slice(const Slice<T> &) = default;
  LIBC_INLINE constexpr Slice(Slice<T> &&) = default;

  LIBC_INLINE constexpr Slice(const cpp::span<T> &s) : cpp::span<T>(s) {}
  LIBC_INLINE constexpr Slice(cpp::span<T> &&s) : cpp::span<T>(s) {}

  LIBC_INLINE constexpr Slice &operator=(const Slice<T> &n) = default;
  LIBC_INLINE constexpr Slice &operator=(Slice<T> &&n) = default;

  LIBC_INLINE constexpr Slice &operator=(const cpp::span<T> &n) {
    cpp::span<T>::operator=(n);
    return *this;
  }
  LIBC_INLINE constexpr Slice &operator=(cpp::span<T> &&n) {
    cpp::span<T>::operator=(n);
    return *this;
  }

  // Binary searches this slice with a comparator function.
  //
  // The comparator function should return an order code that indicates whether
  // its argument is `Less`, `Equal` or `Greater` the desired target.
  // If the slice is not sorted or if the comparator function does not
  // implement an order consistent with the sort order of the underlying
  // slice, the returned result is unspecified and meaningless.
  //
  // If the value is found then `cpp::expected<size_t>` is returned,
  // containing the index of the matching element. If there are multiple
  // matches, then any one of the matches could be returned. The index is chosen
  // deterministically.
  // If the value is not found then `cpp::unexpected<size_t>` is returned,
  // containing the index where a matching element could be inserted while
  // maintaining sorted order.
  template <typename Fn>
  LIBC_INLINE constexpr cpp::expected<size_t, size_t>
  binary_search_by(Fn func) const {
    auto size = this->size();
    if (size == 0) {
      return cpp::unexpected<size_t>(0);
    }

    size_t base = 0;

    while (size > 1) {
      auto half = size / 2;
      auto mid = base + half;
      auto cmp = func(this->operator[](mid));
      base = (cmp == Ordering::Greater) ? base : mid;
      size -= half;
    }

    auto cmp = func(this->operator[](base));
    if (cmp == Ordering::Equal) {
      LIBC_ASSERT(base < this->size());
      return base;
    } else {
      auto result = base + static_cast<size_t>(cmp == Ordering::Less);
      LIBC_ASSERT(result <= this->size());
      return cpp::unexpected(result);
    }
  }

  LIBC_INLINE constexpr Slice<T> slice_form_range(size_t start,
                                                  size_t end) const {
    LIBC_ASSERT(start <= end && end <= this->size());
    return Slice<T>(this->data() + start, end - start);
  }

  LIBC_INLINE constexpr bool contains(T elm) const {
    for (auto it : *this) {
      if (elm == it) {
        return true;
      }
    }
    return false;
  }

  LIBC_INLINE constexpr void copy_from_slice(Slice<T> other) const {
    for (size_t i = 0; i < cpp::min(this->size(), other.size()); i++) {
      this->data()[i] = other.data()[i];
    }
  }
};

} // namespace conversion_utils

} // namespace wctype_internal

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_WCTYPE_CONVERSION_UTILS_SLICE_H
