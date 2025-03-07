//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// UNSUPPORTED: libcpp-hardening-mode=none, no-exceptions

#include <ranges>
#include <utility>

#include "check_assertion.h"

int globalArray[8] = {0, 1, 2, 3, 4, 5, 6, 7};

struct ThrowOnCopyView : std::ranges::view_base {
  int start_;
  int* ptr_;
  constexpr explicit ThrowOnCopyView(int* ptr = globalArray, int start = 0) : start_(start), ptr_(ptr) {}
  constexpr ThrowOnCopyView(ThrowOnCopyView&&) = default;
  constexpr ThrowOnCopyView(const ThrowOnCopyView&) { throw 42; };
  constexpr ThrowOnCopyView& operator=(ThrowOnCopyView&&) = default;
  constexpr ThrowOnCopyView& operator=(const ThrowOnCopyView&) {
    throw 42;
    return *this;
  };
  constexpr int* begin() const { return ptr_ + start_; }
  constexpr int* end() const { return ptr_ + 8; }
};

int main() {
  {
    //valueless by exception test constructor
    std::ranges::concat_view<ThrowOnCopyView> concatView_2;
    std::ranges::iterator_t<std::ranges::concat_view<ThrowOnCopyView>> it1;
    std::ranges::iterator_t<std::ranges::concat_view<ThrowOnCopyView>> it2;
    try {
      it1 = concatView_2.begin();
    } catch (...) {
      TEST_LIBCPP_ASSERT_FAILURE(
          [&] {
            it2 = it1;
            (void)it2;
          }(),
          "valueless by exception");
    }
  }

  {
    //valueless by exception test operator==
    std::ranges::concat_view<ThrowOnCopyView> concatView_2;
    std::ranges::iterator_t<std::ranges::concat_view<ThrowOnCopyView>> it1;
    std::ranges::iterator_t<std::ranges::concat_view<ThrowOnCopyView>> it2;
    try {
      it1 = concatView_2.begin();
    } catch (...) {
      TEST_LIBCPP_ASSERT_FAILURE([&] { (void)(it1 == it2); }(), "valueless by exception");
    }
  }

  {
    //valueless by exception test operator--
    std::ranges::concat_view<ThrowOnCopyView> concatView_2;
    std::ranges::iterator_t<std::ranges::concat_view<ThrowOnCopyView>> it1;
    try {
      it1 = concatView_2.begin();
    } catch (...) {
      TEST_LIBCPP_ASSERT_FAILURE([&] { (void)--*it1; }(), "valueless by exception");
    }
  }

  {
    //valueless by exception test operator*
    std::ranges::concat_view<ThrowOnCopyView> concatView_2;
    std::ranges::iterator_t<std::ranges::concat_view<ThrowOnCopyView>> it1;
    try {
      it1 = concatView_2.begin();
    } catch (...) {
      TEST_LIBCPP_ASSERT_FAILURE([&] { (void)*it1; }(), "valueless by exception");
    }
  }

  {
    //valueless by exception test operator++
    std::ranges::concat_view<ThrowOnCopyView> concatView_2;
    std::ranges::iterator_t<std::ranges::concat_view<ThrowOnCopyView>> it1;
    try {
      it1 = concatView_2.begin();
    } catch (...) {
      TEST_LIBCPP_ASSERT_FAILURE([&] { (void)++*it1; }(), "valueless by exception");
    }
  }

  {
    //valueless by exception test operator+=
    std::ranges::concat_view<ThrowOnCopyView> concatView_2;
    std::ranges::iterator_t<std::ranges::concat_view<ThrowOnCopyView>> it1;
    try {
      it1 = concatView_2.begin();
    } catch (...) {
      TEST_LIBCPP_ASSERT_FAILURE([&] { (void)(it1 += 1); }(), "valueless by exception");
    }
  }
}