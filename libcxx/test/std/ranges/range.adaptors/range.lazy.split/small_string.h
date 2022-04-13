//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_LAZY_SPLIT_SMALL_STRING_H
#define TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_LAZY_SPLIT_SMALL_STRING_H

#include <algorithm>
#include <cassert>
#include <ranges>
#include <string_view>

// A constexpr-friendly lightweight string, primarily useful for comparisons.
// Unlike `std::string`, all functions are `constexpr`. Unlike `std::string_view`, it copies the given string into an
// internal buffer and can work with non-contiguous inputs.
//
// TODO(var-const): remove once https://reviews.llvm.org/D110598 lands and `std::string` can be used instead of this
// class.
template <class Char>
class BasicSmallString {
  constexpr static int N = 32;
  Char buffer_[N] = {};
  size_t size_ = 0;

public:
  // Main constructors.

  constexpr BasicSmallString() = default;

  constexpr BasicSmallString(std::basic_string_view<Char> v) : size_(v.size()) {
    assert(size_ < N);
    if (size_ == 0) return;

    std::copy(v.begin(), v.end(), buffer_);
  }

  template <class I, class S>
  constexpr BasicSmallString(I b, const S& e) {
    for (; b != e; ++b) {
      buffer_[size_++] = *b;
      assert(size_ < N);
    }
  }

  // Delegating constructors.

  constexpr BasicSmallString(const Char* ptr, size_t size) : BasicSmallString(std::basic_string_view<Char>(ptr, size)) {
  }

  template <std::ranges::range R>
  constexpr BasicSmallString(R&& from) : BasicSmallString(from.begin(), from.end()) {
  }

  // Iterators.

  constexpr Char* begin() { return buffer_; }
  constexpr Char* end() { return buffer_ + size_; }
  constexpr const Char* begin() const { return buffer_; }
  constexpr const Char* end() const { return buffer_ + size_; }

  friend constexpr bool operator==(const BasicSmallString& lhs, const BasicSmallString& rhs) {
    return lhs.size_ == rhs.size_ && std::equal(lhs.buffer_, lhs.buffer_ + lhs.size_, rhs.buffer_);
  }
  friend constexpr bool operator==(const BasicSmallString& lhs, std::string_view rhs) {
    return lhs == BasicSmallString(rhs);
  }
};

using SmallString = BasicSmallString<char>;

inline constexpr SmallString operator "" _str(const char* ptr, size_t size) {
  return SmallString(ptr, size);
}

#endif // TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_LAZY_SPLIT_SMALL_STRING_H
