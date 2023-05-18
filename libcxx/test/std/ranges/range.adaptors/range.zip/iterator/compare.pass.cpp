//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// friend constexpr bool operator==(const iterator& x, const iterator& y)
//   requires (equality_comparable<iterator_t<maybe-const<Const, Views>>> && ...);
// friend constexpr bool operator<(const iterator& x, const iterator& y)
//   requires all-random-access<Const, Views...>;
// friend constexpr bool operator>(const iterator& x, const iterator& y)
//   requires all-random-access<Const, Views...>;
// friend constexpr bool operator<=(const iterator& x, const iterator& y)
//   requires all-random-access<Const, Views...>;
// friend constexpr bool operator>=(const iterator& x, const iterator& y)
//   requires all-random-access<Const, Views...>;
// friend constexpr auto operator<=>(const iterator& x, const iterator& y)
//   requires all-random-access<Const, Views...> &&
//            (three_way_comparable<iterator_t<maybe-const<Const, Views>>> && ...);

#include <ranges>
#include <compare>

#include "test_iterators.h"
#include "../types.h"

// This is for testing that zip iterator never calls underlying iterator's >, >=, <=, !=.
// The spec indicates that zip iterator's >= is negating zip iterator's < instead of calling underlying iterator's >=.
// Declare all the operations >, >=, <= etc to make it satisfy random_access_iterator concept,
// but not define them. If the zip iterator's >,>=, <=, etc isn't implemented in the way defined by the standard
// but instead calling underlying iterator's >,>=,<=, we will get a linker error for the runtime tests and
// non-constant expression for the compile time tests.
struct LessThanIterator {
  int* it_ = nullptr;
  LessThanIterator() = default;
  constexpr LessThanIterator(int* it) : it_(it) {}

  using iterator_category = std::random_access_iterator_tag;
  using value_type = int;
  using difference_type = std::intptr_t;

  constexpr int& operator*() const { return *it_; }
  constexpr int& operator[](difference_type n) const { return it_[n]; }
  constexpr LessThanIterator& operator++() {
    ++it_;
    return *this;
  }
  constexpr LessThanIterator& operator--() {
    --it_;
    return *this;
  }
  constexpr LessThanIterator operator++(int) { return LessThanIterator(it_++); }
  constexpr LessThanIterator operator--(int) { return LessThanIterator(it_--); }

  constexpr LessThanIterator& operator+=(difference_type n) {
    it_ += n;
    return *this;
  }
  constexpr LessThanIterator& operator-=(difference_type n) {
    it_ -= n;
    return *this;
  }

  constexpr friend LessThanIterator operator+(LessThanIterator x, difference_type n) {
    x += n;
    return x;
  }
  constexpr friend LessThanIterator operator+(difference_type n, LessThanIterator x) {
    x += n;
    return x;
  }
  constexpr friend LessThanIterator operator-(LessThanIterator x, difference_type n) {
    x -= n;
    return x;
  }
  constexpr friend difference_type operator-(LessThanIterator x, LessThanIterator y) { return x.it_ - y.it_; }

  constexpr friend bool operator==(LessThanIterator const&, LessThanIterator const&) = default;
  friend bool operator!=(LessThanIterator const&, LessThanIterator const&);

  constexpr friend bool operator<(LessThanIterator const& x, LessThanIterator const& y) { return x.it_ < y.it_; }
  friend bool operator<=(LessThanIterator const&, LessThanIterator const&);
  friend bool operator>(LessThanIterator const&, LessThanIterator const&);
  friend bool operator>=(LessThanIterator const&, LessThanIterator const&);
};
static_assert(std::random_access_iterator<LessThanIterator>);

struct SmallerThanRange : IntBufferView {
  using IntBufferView::IntBufferView;
  constexpr LessThanIterator begin() const { return {buffer_}; }
  constexpr LessThanIterator end() const { return {buffer_ + size_}; }
};
static_assert(std::ranges::random_access_range<SmallerThanRange>);

struct ForwardCommonView : IntBufferView {
  using IntBufferView::IntBufferView;
  using iterator = forward_iterator<int*>;

  constexpr iterator begin() const { return iterator(buffer_); }
  constexpr iterator end() const { return iterator(buffer_ + size_); }
};

constexpr void compareOperatorTest(auto&& iter1, auto&& iter2) {
  assert(!(iter1 < iter1));
  assert(iter1 < iter2);
  assert(!(iter2 < iter1));
  assert(iter1 <= iter1);
  assert(iter1 <= iter2);
  assert(!(iter2 <= iter1));
  assert(!(iter1 > iter1));
  assert(!(iter1 > iter2));
  assert(iter2 > iter1);
  assert(iter1 >= iter1);
  assert(!(iter1 >= iter2));
  assert(iter2 >= iter1);
  assert(iter1 == iter1);
  assert(!(iter1 == iter2));
  assert(iter2 == iter2);
  assert(!(iter1 != iter1));
  assert(iter1 != iter2);
  assert(!(iter2 != iter2));
}

constexpr void inequalityOperatorsDoNotExistTest(auto&& iter1, auto&& iter2) {
  using Iter1 = decltype(iter1);
  using Iter2 = decltype(iter2);
  static_assert(!std::is_invocable_v<std::less<>, Iter1, Iter2>);
  static_assert(!std::is_invocable_v<std::less_equal<>, Iter1, Iter2>);
  static_assert(!std::is_invocable_v<std::greater<>, Iter1, Iter2>);
  static_assert(!std::is_invocable_v<std::greater_equal<>, Iter1, Iter2>);
}

constexpr bool test() {
  {
    // Test a new-school iterator with operator<=>; the iterator should also have operator<=>.
    using It = three_way_contiguous_iterator<int*>;
    using SubRange = std::ranges::subrange<It>;
    static_assert(std::three_way_comparable<It>);
    using R = std::ranges::zip_view<SubRange, SubRange>;
    static_assert(std::three_way_comparable<std::ranges::iterator_t<R>>);

    int a[] = {1, 2, 3, 4};
    int b[] = {5, 6, 7, 8, 9};
    auto r = std::views::zip(SubRange(It(a), It(a + 4)), SubRange(It(b), It(b + 5)));
    auto iter1 = r.begin();
    auto iter2 = iter1 + 1;

    compareOperatorTest(iter1, iter2);

    assert((iter1 <=> iter2) == std::strong_ordering::less);
    assert((iter1 <=> iter1) == std::strong_ordering::equal);
    assert((iter2 <=> iter1) == std::strong_ordering::greater);
  }

  {
    // Test an old-school iterator with no operator<=>; the transform iterator shouldn't have
    // operator<=> either.
    using It = random_access_iterator<int*>;
    using Subrange = std::ranges::subrange<It>;
    static_assert(!std::three_way_comparable<It>);
    using R = std::ranges::zip_view<Subrange, Subrange>;
#ifdef _LIBCPP_VERSION
    // libc++ hasn't implemented LWG-3692 "zip_view::iterator's operator<=> is overconstrained"
    static_assert(!std::three_way_comparable<std::ranges::iterator_t<R>>);
#else
    static_assert(std::three_way_comparable<std::ranges::iterator_t<R>>);
#endif

    int a[] = {1, 2, 3, 4};
    int b[] = {5, 6, 7, 8, 9};
    auto r = std::views::zip(Subrange(It(a), It(a + 4)), Subrange(It(b), It(b + 5)));
    auto iter1 = r.begin();
    auto iter2 = iter1 + 1;

    compareOperatorTest(iter1, iter2);
  }

  {
    // non random_access_range
    int buffer1[1] = {1};
    int buffer2[2] = {1, 2};

    std::ranges::zip_view v{InputCommonView(buffer1), InputCommonView(buffer2)};
    using View = decltype(v);
    static_assert(!std::ranges::forward_range<View>);
    static_assert(std::ranges::input_range<View>);
    static_assert(std::ranges::common_range<View>);

    auto it1 = v.begin();
    auto it2 = v.end();
    assert(it1 != it2);

    ++it1;
    assert(it1 == it2);

    inequalityOperatorsDoNotExistTest(it1, it2);
  }

  {
    // in this case sentinel is computed by getting each of the underlying sentinel, so only one
    // underlying iterator is comparing equal
    int buffer1[1] = {1};
    int buffer2[2] = {1, 2};
    std::ranges::zip_view v{ForwardCommonView(buffer1), ForwardCommonView(buffer2)};
    using View = decltype(v);
    static_assert(std::ranges::common_range<View>);
    static_assert(!std::ranges::bidirectional_range<View>);

    auto it1 = v.begin();
    auto it2 = v.end();
    assert(it1 != it2);

    ++it1;
    // it1:  <buffer1 + 1, buffer2 + 1>
    // it2:  <buffer1 + 1, buffer2 + 2>
    assert(it1 == it2);

    inequalityOperatorsDoNotExistTest(it1, it2);
  }

  {
    // only < and == are needed
    int a[] = {1, 2, 3, 4};
    int b[] = {5, 6, 7, 8, 9};
    auto r = std::views::zip(SmallerThanRange(a), SmallerThanRange(b));
    auto iter1 = r.begin();
    auto iter2 = iter1 + 1;

    compareOperatorTest(iter1, iter2);
  }

  {
    // underlying iterator does not support ==
    using IterNoEqualView = BasicView<cpp20_input_iterator<int*>, sentinel_wrapper<cpp20_input_iterator<int*>>>;
    int buffer[] = {1};
    std::ranges::zip_view r(IterNoEqualView{buffer});
    auto it = r.begin();
    using Iter = decltype(it);
    static_assert(!std::invocable<std::equal_to<>, Iter, Iter>);
    inequalityOperatorsDoNotExistTest(it, it);
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
