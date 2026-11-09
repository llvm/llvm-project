//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// <string_view>

// template <class It, class End>
//   constexpr basic_string_view(It begin, End end);

// Test the libc++ extension that the iterator/sentinel constructor of std::basic_string_view
// is noexcept when `end - begin` is noexcept, and potentially-throwing otherwise.

#include <cstddef>
#include <string_view>

struct ThrowingSentinel {
  friend bool operator==(const char*, ThrowingSentinel) noexcept { return true; }
  friend std::ptrdiff_t operator-(const char*, ThrowingSentinel) noexcept { return 0; }
  friend std::ptrdiff_t operator-(ThrowingSentinel, const char*) noexcept(false) { return 0; }
};

static_assert(noexcept(std::string_view(static_cast<const char*>(nullptr), static_cast<const char*>(nullptr))));
static_assert(!noexcept(std::string_view(static_cast<const char*>(nullptr), ThrowingSentinel{})));
