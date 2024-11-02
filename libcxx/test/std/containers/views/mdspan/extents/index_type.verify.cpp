//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <mdspan>

// template<class IndexType, size_t... Extents>
//  class extents;
//
// Mandates:
//   - IndexType is a signed or unsigned integer type, and
//   - each element of Extents is either equal to dynamic_extent, or is representable as a value of type IndexType.

#include <cstddef>
#include <climits>
#include <mdspan>
#include <span> // dynamic_extent

void invalid_index_types() {
  // expected-error@*:* {{static assertion failed: extents::index_type must be a signed or unsigned integer type}}
  [[maybe_unused]] std::extents<char, '*'> ec;
#ifndef TEST_HAS_NO_CHAR8_T
  // expected-error@*:* {{static assertion failed: extents::index_type must be a signed or unsigned integer type}}
  [[maybe_unused]] std::extents<char8_t, u8'*'> ec8;
#endif
  // expected-error@*:* {{static assertion failed: extents::index_type must be a signed or unsigned integer type}}
  [[maybe_unused]] std::extents<char16_t, u'*'> ec16;
  // expected-error@*:* {{static assertion failed: extents::index_type must be a signed or unsigned integer type}}
  [[maybe_unused]] std::extents<char32_t, U'*'> ec32;
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  // expected-error@*:* {{static assertion failed: extents::index_type must be a signed or unsigned integer type}}
  [[maybe_unused]] std::extents<wchar_t, L'*'> ewc;
#endif
}

void invalid_extent_values() {
  // expected-error-re@*:* {{static assertion failed {{.*}}extents ctor: arguments must be representable as index_type and nonnegative}}
  [[maybe_unused]] std::extents<signed char, static_cast<std::size_t>(SCHAR_MAX) + 1> esc1;
  // expected-error-re@*:* {{static assertion failed {{.*}}extents ctor: arguments must be representable as index_type and nonnegative}}
  [[maybe_unused]] std::extents<signed char, std::dynamic_extent - 1> esc2;
  // expected-error-re@*:* {{static assertion failed {{.*}}extents ctor: arguments must be representable as index_type and nonnegative}}
  [[maybe_unused]] std::extents<unsigned char, static_cast<std::size_t>(UCHAR_MAX) + 1> euc1;
  // expected-error-re@*:* {{static assertion failed {{.*}}extents ctor: arguments must be representable as index_type and nonnegative}}
  [[maybe_unused]] std::extents<unsigned char, std::dynamic_extent - 1> euc2;
}
