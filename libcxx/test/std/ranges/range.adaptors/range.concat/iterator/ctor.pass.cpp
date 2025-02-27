//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

#include <ranges>

#include "test_macros.h"
#include "test_iterators.h"
#include "check_assertion.h"
#include "../types.h"

int globalBuff[8] = {0, 1, 2, 3, 4, 5, 6, 7};

struct MoveOnlyView : std::ranges::view_base {
  int start_;
  int* ptr_;
  constexpr explicit MoveOnlyView(int* ptr = globalBuff, int start = 0) : start_(start), ptr_(ptr) {}
  constexpr MoveOnlyView(MoveOnlyView&&)            = default;
  constexpr MoveOnlyView& operator=(MoveOnlyView&&) = default;
  constexpr int* begin() const { return ptr_ + start_; }
  constexpr int* end() const { return ptr_ + 8; }
};

struct NoDefaultInit {
  typedef std::random_access_iterator_tag iterator_category;
  typedef int value_type;
  typedef std::ptrdiff_t difference_type;
  typedef int* pointer;
  typedef int& reference;
  typedef NoDefaultInit self;

  NoDefaultInit(int*);

  reference operator*() const;
  pointer operator->() const;
  auto operator<=>(const self&) const = default;
  bool operator==(int*) const;

  self& operator++();
  self operator++(int);

  self& operator--();
  self operator--(int);

  self& operator+=(difference_type n);
  self operator+(difference_type n) const;
  friend self operator+(difference_type n, self x);

  self& operator-=(difference_type n);
  self operator-(difference_type n) const;
  difference_type operator-(const self&) const;

  reference operator[](difference_type n) const;
};

struct IterNoDefaultInitView : std::ranges::view_base {
  NoDefaultInit begin() const;
  int* end() const;
  NoDefaultInit begin();
  int* end();
};

struct ThrowOnCopyView : std::ranges::view_base {
  int start_;
  int* ptr_;
  constexpr explicit ThrowOnCopyView(int* ptr = globalBuff, int start = 0) : start_(start), ptr_(ptr) {}
  constexpr ThrowOnCopyView(ThrowOnCopyView&&) = default;
  constexpr ThrowOnCopyView(const ThrowOnCopyView&) { throw 42; };
  constexpr ThrowOnCopyView& operator=(ThrowOnCopyView&&) = default;
  constexpr ThrowOnCopyView& operator=(const ThrowOnCopyView&) { throw 42; };
  constexpr int* begin() const { return ptr_ + start_; }
  constexpr int* end() const { return ptr_ + 8; }
};

constexpr bool test() {
  std::ranges::concat_view<MoveOnlyView> concatView;
  auto iter = std::move(concatView).begin();
  std::ranges::iterator_t<std::ranges::concat_view<MoveOnlyView>> i2(iter);
  (void)i2;
  std::ranges::iterator_t<const std::ranges::concat_view<MoveOnlyView>> constIter(iter);
  (void)constIter;

  static_assert(std::default_initializable<std::ranges::iterator_t<std::ranges::concat_view<MoveOnlyView>>>);
  static_assert(!std::default_initializable<std::ranges::iterator_t<std::ranges::concat_view<IterNoDefaultInitView>>>);

  {
    //valueless by exception test
    std::ranges::concat_view<ThrowOnCopyView> concatView;
    std::ranges::iterator_t<std::ranges::concat_view<ThrowOnCopyView>> it1;
    std::ranges::iterator_t<std::ranges::concat_view<ThrowOnCopyView>> it2;
    try {
      it1 = concatView.begin();
    } catch (...) {
      TEST_LIBCPP_ASSERT_FAILURE(
          [&] {
            it2 = it1;
            (void)it2;
          }(),
          "valueless by exception");
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
