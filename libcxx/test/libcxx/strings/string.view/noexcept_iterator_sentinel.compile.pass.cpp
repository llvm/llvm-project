//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <concepts>
#include <cstddef>
#include <ranges>
#include <string_view>

struct ThrowingSentinel {
  friend bool operator==(const char*, ThrowingSentinel) noexcept { return true; }
  friend std::ptrdiff_t operator-(const char*, ThrowingSentinel) noexcept { return 0; }
  friend std::ptrdiff_t operator-(ThrowingSentinel, const char*) noexcept(false) { return 0; }
};

struct ThrowingCount {
  operator std::ptrdiff_t() const noexcept(false) { return 1; }
};

struct ThrowingReferenceCount {
  operator std::ptrdiff_t() const noexcept { return 1; }
  template <class _Tp>
    requires std::same_as<_Tp, const std::ptrdiff_t>
  operator _Tp&() const noexcept(false) {
    return value;
  }

  std::ptrdiff_t value = 1;
};

static_assert(noexcept(std::string_view(static_cast<const char*>(nullptr), static_cast<const char*>(nullptr))));
static_assert(!noexcept(std::string_view(static_cast<const char*>(nullptr), ThrowingSentinel{})));
static_assert(noexcept(std::views::drop(std::string_view{}, 1)));
static_assert(noexcept(std::views::take(std::string_view{}, 1)));
static_assert(!noexcept(std::views::drop(std::string_view{}, ThrowingCount{})));
static_assert(!noexcept(std::views::take(std::string_view{}, ThrowingCount{})));
static_assert(!noexcept(std::views::drop(std::string_view{}, ThrowingReferenceCount{})));
static_assert(!noexcept(std::views::take(std::string_view{}, ThrowingReferenceCount{})));
