//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// UNSUPPORTED: no-exceptions

#include <iostream>
#include <ranges>
#include <utility>
#include <vector>

#include "check_assertion.h"
#include "double_move_tracker.h"
#include "test_iterators.h"

int globalArray[8] = {0, 1, 2, 3, 4, 5, 6, 7};
bool already_moved = false;

template <class It, class ItTraits = It >
class ThrowOnMoveIterator {
  using Traits = std::iterator_traits<ItTraits>;
  It it_;
  support::double_move_tracker tracker_;

  template <class U, class T>
  friend class ThrowOnMoveIterator;

public:
  using iterator_category = std::input_iterator_tag;
  using value_type        = typename Traits::value_type;
  using difference_type   = typename Traits::difference_type;
  using pointer           = It;
  using eference          = typename Traits::reference;

  constexpr ThrowOnMoveIterator() {}

  constexpr ThrowOnMoveIterator(It it) : it_(it) {}

  template <class U, class T>
  constexpr ThrowOnMoveIterator(const ThrowOnMoveIterator<U, T>& u) : it_(u.it_), tracker_(u.tracker_) {}

  constexpr ThrowOnMoveIterator(const ThrowOnMoveIterator&) {}

  constexpr ThrowOnMoveIterator(ThrowOnMoveIterator&&) {
    std::cout << "moved ctor called with: " << already_moved << std::endl;
    if (!already_moved) {
      already_moved = true;
      throw std::runtime_error("Move failed in iter");
    }
  }

  ThrowOnMoveIterator& operator=(ThrowOnMoveIterator&&) {
    if (!already_moved) {
      already_moved = true;
      throw std::runtime_error("Move assignment failed in iter");
    }
    return *this;
  }

  constexpr reference operator*() const { return *it_; }

  constexpr ThrowOnMoveIterator& operator++() {
    ++it_;
    return *this;
  }
  constexpr ThrowOnMoveIterator operator++(int) { return ThrowOnMoveIterator(it_++); }

  friend constexpr bool operator==(const ThrowOnMoveIterator& x, const ThrowOnMoveIterator& y) {
    return x.it_ == y.it_;
  }
  friend constexpr bool operator!=(const ThrowOnMoveIterator& x, const ThrowOnMoveIterator& y) {
    return x.it_ != y.it_;
  }

  friend constexpr It base(const ThrowOnMoveIterator& i) { return i.it_; }

  template <class T>
  void operator,(T const&) = delete;
};

template <class T>
struct BufferView : std::ranges::view_base {
  T* buffer_;
  std::size_t size_;

  template <std::size_t N>
  constexpr BufferView(T (&b)[N]) : buffer_(b), size_(N) {}
};

using IntBufferView = BufferView<int>;

template <bool Simple>
struct Common : IntBufferView {
  using IntBufferView::IntBufferView;
  using Iter      = ThrowOnMoveIterator<int*>;
  using ConstIter = ThrowOnMoveIterator<const int*>;
  using Sent      = sentinel_wrapper<Iter>;
  using ConstSent = sentinel_wrapper<ConstIter>;

  constexpr Iter begin()
    requires(!Simple)
  {
    return Iter{buffer_};
  }
  constexpr const ConstIter begin() const { return ConstIter{buffer_}; }
  constexpr Sent end()
    requires(!Simple)
  {
    return Sent(buffer_ + size_);
  }
  constexpr ConstSent end() const { return ConstSent(buffer_ + size_); }
};

using SimpleCommon    = Common<true>;
using NonSimpleCommon = Common<false>;

int main() {
  {
    int buffer[3]  = {1, 2, 3};
    int buffer2[3] = {1, 2, 3};
    NonSimpleCommon view(buffer);
    NonSimpleCommon view2(buffer2);
    std::ranges::concat_view v(view, view2);
    std::ranges::iterator_t<std::ranges::concat_view<decltype(view), decltype(view2)>> it1;
    try {
      it1 = v.begin();
    } catch (...) {
      std::cout << "hit catch" << std::endl;
      //ASSERT_SAME_TYPE(std::ranges::iterator_t<const decltype(v)>, int);
      std::ranges::iterator_t<const decltype(v)> it2(it1);
      TEST_LIBCPP_ASSERT_FAILURE(
          [=] {
            std::ranges::iterator_t<const decltype(v)> it2(it1);
            (void)it2;
          }(),
          "valueless by exception");
    }
  }
  /*
  {
    //valueless by exception test operator==
    ThrowingRange<int*> throwing{3, 5};
    auto concatView_2 = std::views::concat(throwing);
    decltype(concatView_2.begin()) it1{};
    decltype(concatView_2.begin()) it2{};
    try {
      it1 = concatView_2.begin();
    } catch (...) {
      std::cout << "Catch is hit" << std::endl;
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
      std::cout << "Catch is hit" << std::endl;
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
      std::cout << "Catch is hit" << std::endl;
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
      std::cout << "Catch is hit" << std::endl;
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
      std::cout << "Catch is hit" << std::endl;
      TEST_LIBCPP_ASSERT_FAILURE([&] { (void)(it1 += 1); }(), "valueless by exception");
    }
  }
  */
}
