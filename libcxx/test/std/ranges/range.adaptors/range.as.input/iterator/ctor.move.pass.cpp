//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <ranges>

//   template<input_range V>
//     requires view<V>
//   template<bool Const>
//   class as_input_view<V>::iterator

//    iterator(iterator&&) = default;
//    iterator& operator=(iterator&&) = default;

#include <cassert>
#include <concepts>
#include <cstdint>
#include <ranges>
#include <type_traits>
#include <utility>

#include "test_iterators.h"

class MoveOnlyInputIterator;

class MoveOnlyInputSentinel {
  int* ptr_ = nullptr;

public:
  constexpr MoveOnlyInputSentinel() = default;
  constexpr explicit MoveOnlyInputSentinel(int* ptr) : ptr_(ptr) {}

  constexpr int* base() const { return ptr_; }
};

class MoveOnlyInputIterator {
  int* ptr_ = nullptr;

public:
  constexpr MoveOnlyInputIterator() = default;
  constexpr explicit MoveOnlyInputIterator(int* ptr) : ptr_(ptr) {}

  MoveOnlyInputIterator(const MoveOnlyInputIterator&)            = delete;
  MoveOnlyInputIterator& operator=(const MoveOnlyInputIterator&) = delete;

  MoveOnlyInputIterator(MoveOnlyInputIterator&&)            = default;
  MoveOnlyInputIterator& operator=(MoveOnlyInputIterator&&) = default;

  using difference_type   = std::intptr_t;
  using value_type        = int;
  using iterator_category = std::input_iterator_tag;

  constexpr int operator*() const { return *ptr_; }

  constexpr MoveOnlyInputIterator& operator++() {
    ++ptr_;
    return *this;
  }
  constexpr MoveOnlyInputIterator operator++(int) {
    MoveOnlyInputIterator __tmp = std::move(*this);
    ++*this;
    return __tmp;
  }

  friend constexpr bool operator==(const MoveOnlyInputIterator& x, const MoveOnlyInputSentinel& y) {
    return x.ptr_ == y.base();
  }
  friend constexpr bool operator==(const MoveOnlyInputSentinel& y, const MoveOnlyInputIterator& x) { return x == y; }
};

static_assert(std::input_iterator<MoveOnlyInputIterator>);

class MoveOnlyInputView : public std::ranges::view_interface<MoveOnlyInputView> {
  int* data_;
  std::size_t size_;

public:
  constexpr MoveOnlyInputView(int* data, std::size_t size) : data_(data), size_(size) {}

  constexpr MoveOnlyInputIterator begin() const { return MoveOnlyInputIterator(data_); }
  constexpr MoveOnlyInputSentinel end() const { return MoveOnlyInputSentinel(data_ + size_); }
};

static_assert(std::ranges::input_range<MoveOnlyInputView>);

constexpr bool test() {
  int arr[] = {1, 2, 3};
  std::ranges::as_input_view<MoveOnlyInputView> view{MoveOnlyInputView{arr, 3}};

  using IteratorT = std::ranges::iterator_t<decltype(view)>;

  static_assert(!std::copy_constructible<IteratorT>);
  static_assert(!std::is_copy_assignable_v<IteratorT>);
  static_assert(std::move_constructible<IteratorT>);
  static_assert(std::is_move_assignable_v<IteratorT>);

  IteratorT it1 = view.begin();

  // Move constructor
  IteratorT it2{std::move(it1)};
  assert(*it2 == 1);

  // Move assignment
  IteratorT it3 = std::move(it2);
  assert(*it3 == 1);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
