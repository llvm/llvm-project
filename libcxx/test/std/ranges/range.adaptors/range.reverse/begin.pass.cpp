//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr reverse_iterator<iterator_t<V>> begin();
// constexpr reverse_iterator<iterator_t<V>> begin() requires common_range<V>;
// constexpr auto begin() const requires common_range<const V>;

#include <ranges>
#include <cassert>

#include "test_macros.h"
#include "types.h"

static int globalCount = 0;

struct CountedIter {
    typedef std::bidirectional_iterator_tag iterator_category;
    typedef int                             value_type;
    typedef std::ptrdiff_t                  difference_type;
    typedef int*                            pointer;
    typedef int&                            reference;
    typedef CountedIter                     self;

    pointer ptr_;
    CountedIter(pointer ptr) : ptr_(ptr) {}
    CountedIter() = default;

    reference operator*() const;
    pointer operator->() const;
    auto operator<=>(const self&) const = default;

    self& operator++() { globalCount++; ++ptr_; return *this; }
    self operator++(int) {
      auto tmp = *this;
      ++*this;
      return tmp;
    }

    self& operator--();
    self operator--(int);
};

struct CountedView : std::ranges::view_base {
  int* begin_;
  int* end_;

  CountedView(int* b, int* e) : begin_(b), end_(e) { }

  auto begin() { return CountedIter(begin_); }
  auto begin() const { return CountedIter(begin_); }
  auto end() { return sentinel_wrapper<CountedIter>(CountedIter(end_)); }
  auto end() const { return sentinel_wrapper<CountedIter>(CountedIter(end_)); }
};

struct RASentRange : std::ranges::view_base {
  using sent_t = sentinel_wrapper<random_access_iterator<int*>>;
  using sent_const_t = sentinel_wrapper<random_access_iterator<const int*>>;

  int* begin_;
  int* end_;

  constexpr RASentRange(int* b, int* e) : begin_(b), end_(e) { }

  constexpr random_access_iterator<int*> begin() { return random_access_iterator<int*>{begin_}; }
  constexpr random_access_iterator<const int*> begin() const { return random_access_iterator<const int*>{begin_}; }
  constexpr sent_t end() { return sent_t{random_access_iterator<int*>{end_}}; }
  constexpr sent_const_t end() const { return sent_const_t{random_access_iterator<const int*>{end_}}; }
};

template<class T>
concept BeginInvocable = requires(T t) { t.begin(); };

constexpr bool test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  // Common bidirectional range.
  {
    auto rev = std::ranges::reverse_view(BidirRange{buffer, buffer + 8});
    assert(base(rev.begin().base()) == buffer + 8);
    assert(base(std::move(rev).begin().base()) == buffer + 8);

    ASSERT_SAME_TYPE(decltype(rev.begin()), std::reverse_iterator<bidirectional_iterator<int*>>);
    ASSERT_SAME_TYPE(decltype(std::move(rev).begin()), std::reverse_iterator<bidirectional_iterator<int*>>);
  }
  // Const common bidirectional range.
  {
    const auto rev = std::ranges::reverse_view(BidirRange{buffer, buffer + 8});
    assert(base(rev.begin().base()) == buffer + 8);
    assert(base(std::move(rev).begin().base()) == buffer + 8);

    ASSERT_SAME_TYPE(decltype(rev.begin()), std::reverse_iterator<bidirectional_iterator<const int*>>);
    ASSERT_SAME_TYPE(decltype(std::move(rev).begin()), std::reverse_iterator<bidirectional_iterator<const int*>>);
  }
  // Non-common, non-const (move only) bidirectional range.
  {
    auto rev = std::ranges::reverse_view(BidirSentRange<MoveOnly>{buffer, buffer + 8});
    assert(base(std::move(rev).begin().base()) == buffer + 8);

    ASSERT_SAME_TYPE(decltype(std::move(rev).begin()), std::reverse_iterator<bidirectional_iterator<int*>>);
  }
  // Non-common, non-const bidirectional range.
  {
    auto rev = std::ranges::reverse_view(BidirSentRange<Copyable>{buffer, buffer + 8});
    assert(base(rev.begin().base()) == buffer + 8);
    assert(base(std::move(rev).begin().base()) == buffer + 8);

    ASSERT_SAME_TYPE(decltype(rev.begin()), std::reverse_iterator<bidirectional_iterator<int*>>);
    ASSERT_SAME_TYPE(decltype(std::move(rev).begin()), std::reverse_iterator<bidirectional_iterator<int*>>);
  }
  // Non-common random access range.
  // Note: const overload invalid for non-common ranges, though it would not be imposible
  // to implement for random access ranges.
  {
    auto rev = std::ranges::reverse_view(RASentRange{buffer, buffer + 8});
    assert(base(rev.begin().base()) == buffer + 8);
    assert(base(std::move(rev).begin().base()) == buffer + 8);

    ASSERT_SAME_TYPE(decltype(rev.begin()), std::reverse_iterator<random_access_iterator<int*>>);
    ASSERT_SAME_TYPE(decltype(std::move(rev).begin()), std::reverse_iterator<random_access_iterator<int*>>);
  }
  {
    static_assert( BeginInvocable<      std::ranges::reverse_view<BidirSentRange<Copyable>>>);
    static_assert(!BeginInvocable<const std::ranges::reverse_view<BidirSentRange<Copyable>>>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  {
    // Make sure we cache begin.
    int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    CountedView view{buffer, buffer + 8};
    std::ranges::reverse_view rev(view);
    assert(rev.begin().base().ptr_ == buffer + 8);
    assert(globalCount == 8);
    assert(rev.begin().base().ptr_ == buffer + 8);
    assert(globalCount == 8);
  }

  return 0;
}
