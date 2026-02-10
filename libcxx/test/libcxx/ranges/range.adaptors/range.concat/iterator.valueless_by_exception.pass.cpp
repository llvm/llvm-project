//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-unix-headers, libcpp-hardening-mode={{extensive|debug}}
// REQUIRES: std-at-least-c++26
// UNSUPPORTED: libcpp-hardening-mode=none

#include <iostream>
#include <ranges>
#include <utility>
#include <vector>

#include "check_assertion.h"
#include "double_move_tracker.h"
#include "test_iterators.h"

int val[] = {1, 2, 3};

bool flag = false;

template <std::size_t N>
struct Iter;

template <std::size_t N>
struct Iter {
  using value_type        = int;
  using difference_type   = std::ptrdiff_t;
  using reference         = int&;
  using pointer           = int*;
  using iterator_category = std::random_access_iterator_tag;
  using iterator_concept  = std::random_access_iterator_tag;

private:
  int* ptr_ = nullptr;

  template <std::size_t M>
  friend struct Iter;

public:
  Iter() = default;
  Iter(int* ptr) : ptr_(ptr) {}
  Iter(const Iter&) = default;
  Iter(Iter&& other) : ptr_(other.ptr_) {
    if (flag)
      throw 5;
  }

  Iter& operator=(const Iter&) = default;
  Iter& operator=(Iter&& o) {
    ptr_ = o.ptr_;
    if (flag)
      throw 5;
    return *this;
  }

  reference operator*() const { return *ptr_; }
  pointer operator->() const { return ptr_; }
  reference operator[](difference_type n) const { return ptr_[n]; }

  Iter& operator++() {
    ++ptr_;
    return *this;
  }
  Iter operator++(int) {
    auto tmp = *this;
    ++*this;
    return tmp;
  }
  Iter& operator--() {
    --ptr_;
    return *this;
  }
  Iter operator--(int) {
    auto tmp = *this;
    --*this;
    return tmp;
  }

  Iter& operator+=(difference_type n) {
    ptr_ += n;
    return *this;
  }
  Iter& operator-=(difference_type n) {
    ptr_ -= n;
    return *this;
  }

  template <std::size_t X>
  friend Iter<X> operator+(Iter<X> it, difference_type n);

  template <std::size_t X>
  friend Iter<X> operator+(difference_type n, Iter<X> it);

  template <std::size_t X>
  friend Iter<X> operator-(Iter<X> it, difference_type n);

  template <std::size_t X, std::size_t Y>
  friend difference_type operator-(Iter<X> a, Iter<Y> b);

  friend bool operator==(Iter a, Iter b) { return a.ptr_ == b.ptr_; };
  friend bool operator<(Iter a, Iter b) { return a.ptr_ < b.ptr_; }
  friend bool operator>(Iter a, Iter b) { return a.ptr_ > b.ptr_; }
  friend bool operator<=(Iter a, Iter b) { return a.ptr_ <= b.ptr_; }
  friend bool operator>=(Iter a, Iter b) { return a.ptr_ >= b.ptr_; }
  friend auto operator<=>(Iter a, Iter b) { return a.ptr_ <=> b.ptr_; }
};

template <std::size_t X>
inline Iter<X> operator+(Iter<X> it, std::ptrdiff_t n) {
  return Iter<X>(it.ptr_ + n);
}

template <std::size_t X>
inline Iter<X> operator+(std::ptrdiff_t n, Iter<X> it) {
  return Iter<X>(it.ptr_ + n);
}

template <std::size_t X>
inline Iter<X> operator-(Iter<X> it, std::ptrdiff_t n) {
  return Iter<X>(it.ptr_ - n);
}

template <std::size_t X, std::size_t Y>
inline std::ptrdiff_t operator-(Iter<X> a, Iter<Y> b) {
  return a.ptr_ - b.ptr_;
}

template <std::size_t N>
struct Range : std::ranges::view_base {
  using iterator       = Iter<N>;
  using const_iterator = Iter<N>;
  using sentinel       = sentinel_wrapper<iterator>;

  int* data_;
  std::size_t size_;

  Range() : data_(val), size_(4) {}

  Range(int* data, std::size_t size) : data_(data), size_(size) {}

  iterator begin() { return iterator(data_); }
  iterator end() { return iterator(data_ + size_); }

  const_iterator begin() const { return const_iterator(data_); }
  const_iterator end() const { return const_iterator(data_ + size_); }
};

template <std::size_t N, typename T>
struct NonSimpleIter {
  using difference_type = ptrdiff_t;
  using value_type      = T;

  T* ptr_ = nullptr;

  NonSimpleIter() = default;
  NonSimpleIter(T* ptr) : ptr_(ptr) {}

  template <class U>
    requires(std::is_const_v<T> && !std::is_const_v<U> &&
             std::is_same_v<std::remove_const_t<U>, std::remove_const_t<T>>)
  NonSimpleIter(const NonSimpleIter<N, U>& other) : ptr_(other.ptr_) {}

  NonSimpleIter(const NonSimpleIter&) = default;
  NonSimpleIter(NonSimpleIter&& other) : ptr_(other.ptr_) {
    if (flag)
      throw 5;
  }

  NonSimpleIter& operator=(const NonSimpleIter&) = default;
  NonSimpleIter& operator=(NonSimpleIter&& o) {
    ptr_ = o.ptr_;
    if (flag)
      throw 5;
    return *this;
  }

  T& operator*() const { return *ptr_; }
  NonSimpleIter& operator++() {
    ++ptr_;
    return *this;
  }
  NonSimpleIter operator++(int) {
    auto tmp = *this;
    ++*this;
    return tmp;
  }

  friend bool operator==(NonSimpleIter, NonSimpleIter) = default;
};

template <std::size_t N, typename T>
struct NonSimpleRange : std::ranges::view_base {
  NonSimpleIter<N, T> begin() { return &val[0]; }
  NonSimpleIter<N, T> end() { return &val[3]; }
  NonSimpleIter<N, const T> begin() const { return &val[0]; }
  NonSimpleIter<N, const T> end() const { return &val[3]; }
};

static_assert(std::ranges::range<Range<0>>);
static_assert(std::ranges::sized_range<Range<0>>);

int main() {
  {
    // valueless by exception test operator*
    Range<0> r1;
    Range<1> r2;

    auto cv    = std::views::concat(r1, r2);
    auto iter1 = cv.begin();
    auto iter2 = std::ranges::next(cv.begin(), 4);
    flag       = true;
    try {
      iter1 = std::move(iter2);
      assert(false);
    } catch (...) {
      TEST_LIBCPP_ASSERT_FAILURE([=] { *iter1; }(), "Trying to dereference a valueless iterator of concat_view.");
    }
  }

  {
    // valueless by exception test operator==
    flag = false;
    Range<0> r1;
    Range<1> r2;

    auto cv    = std::views::concat(r1, r2);
    auto iter1 = cv.begin();
    auto iter2 = std::ranges::next(cv.begin(), 4);
    auto iter3 = cv.begin();
    flag       = true;
    try {
      iter1 = std::move(iter2);
      assert(false);
    } catch (...) {
      TEST_LIBCPP_ASSERT_FAILURE(
          [=] { (void)(iter1 == iter3); }(), "Trying to compare a valueless iterator of concat_view.");
    }
  }

  {
    // valueless by exception test operator== with a sentinel
    flag = false;
    Range<0> r1;
    Range<1> r2;
    auto cv    = std::views::concat(r1, r2);
    auto iter1 = cv.begin();
    auto iter2 = std::ranges::next(cv.begin(), 4);
    flag       = true;
    try {
      iter1 = std::move(iter2);
      assert(false);
    } catch (...) {
      TEST_LIBCPP_ASSERT_FAILURE([=] { (void)(iter1 == std::default_sentinel); }(),
                                 "Trying to compare a valueless iterator of concat_view with the default sentinel.");
    }
  }

  {
    // valueless by exception test operator--
    flag = false;
    Range<0> r1;
    Range<1> r2;

    auto cv    = std::views::concat(r1, r2);
    auto iter1 = cv.begin();
    auto iter2 = std::ranges::next(cv.begin(), 4);
    flag       = true;
    try {
      iter1 = std::move(iter2);
      assert(false);
    } catch (...) {
      TEST_LIBCPP_ASSERT_FAILURE([&] { --iter1; }(), "Trying to decrement a valueless iterator of concat_view.");
    }
  }

  {
    // valueless by exception test operator--(int)
    flag = false;
    Range<0> r1;
    Range<1> r2;

    auto cv    = std::views::concat(r1, r2);
    auto iter1 = cv.begin();
    auto iter2 = std::ranges::next(cv.begin(), 4);
    flag       = true;
    try {
      iter1 = std::move(iter2);
      assert(false);
    } catch (...) {
      TEST_LIBCPP_ASSERT_FAILURE([&] { iter1--; }(), "Trying to decrement a valueless iterator of concat_view.");
    }
  }

  {
    // valueless by exception test operator++(int)
    flag = false;
    Range<0> r1;
    Range<1> r2;

    auto cv    = std::views::concat(r1, r2);
    auto iter1 = cv.begin();
    auto iter2 = std::ranges::next(cv.begin(), 4);
    flag       = true;
    try {
      iter1 = std::move(iter2);
      assert(false);
    } catch (...) {
      TEST_LIBCPP_ASSERT_FAILURE([&] { iter1++; }(), "Trying to increment a valueless iterator of concat_view.");
    }
  }

  {
    // valueless by exception test operator++
    flag = false;
    Range<0> r1;
    Range<1> r2;

    auto cv    = std::views::concat(r1, r2);
    auto iter1 = cv.begin();
    auto iter2 = std::ranges::next(cv.begin(), 4);
    flag       = true;
    try {
      iter1 = std::move(iter2);
      assert(false);
    } catch (...) {
      TEST_LIBCPP_ASSERT_FAILURE([&] { ++iter1; }(), "Trying to increment a valueless iterator of concat_view.");
    }
  }

  {
    // valueless by exception test operator+=
    flag = false;
    Range<0> r1;
    Range<1> r2;

    auto cv    = std::views::concat(r1, r2);
    auto iter1 = cv.begin();
    auto iter2 = std::ranges::next(cv.begin(), 4);
    flag       = true;
    try {
      iter1 = std::move(iter2);
      assert(false);
    } catch (...) {
      TEST_LIBCPP_ASSERT_FAILURE([&] { iter1 += 1; }(), "Trying to increment a valueless iterator of concat_view.");
    }
  }

  {
    // valueless by exception test operator-=
    // this one eventually calls operator+= inside the function
    flag = false;
    Range<0> r1;
    Range<1> r2;

    auto cv    = std::views::concat(r1, r2);
    auto iter1 = cv.begin();
    auto iter2 = std::ranges::next(cv.begin(), 4);
    flag       = true;
    try {
      iter1 = std::move(iter2);
      assert(false);
    } catch (...) {
      TEST_LIBCPP_ASSERT_FAILURE([&] { iter1 -= 1; }(), "Trying to increment a valueless iterator of concat_view.");
    }
  }

  {
    // valueless by exception test operator-=
    // this one eventually calls operator+= inside the function
    flag = false;
    Range<0> r1;
    Range<1> r2;

    auto cv    = std::views::concat(r1, r2);
    auto iter1 = cv.begin();
    auto iter2 = std::ranges::next(cv.begin(), 4);
    flag       = true;
    try {
      iter1 = std::move(iter2);
      assert(false);
    } catch (...) {
      TEST_LIBCPP_ASSERT_FAILURE([&] { iter1[1]; }(), "Trying to increment a valueless iterator of concat_view.");
    }
  }

  {
    // valueless by exception test operator+(it, n)
    flag = false;
    Range<0> r1;
    Range<1> r2;

    auto cv    = std::views::concat(r1, r2);
    auto iter1 = cv.begin();
    auto iter2 = std::ranges::next(cv.begin(), 4);
    flag       = true;
    try {
      iter1 = std::move(iter2);
      assert(false);
    } catch (...) {
      TEST_LIBCPP_ASSERT_FAILURE([&] { [[maybe_unused]] auto iter3 = iter1 + 1; }(),
                                 "Trying to increment a valueless iterator of concat_view.");
    }
  }

  {
    // valueless by exception test operator+(n, it)
    flag = false;
    Range<0> r1;
    Range<1> r2;

    auto cv    = std::views::concat(r1, r2);
    auto iter1 = cv.begin();
    auto iter2 = std::ranges::next(cv.begin(), 4);
    flag       = true;
    try {
      iter1 = std::move(iter2);
      assert(false);
    } catch (...) {
      TEST_LIBCPP_ASSERT_FAILURE([&] { [[maybe_unused]] auto iter3 = iter1 + 1; }(),
                                 "Trying to increment a valueless iterator of concat_view.");
    }
  }

  {
    // valueless by exception test operator-(it, default_sentinel)
    flag = false;
    Range<0> r1;
    Range<1> r2;

    auto cv    = std::views::concat(r1, r2);
    auto iter1 = cv.begin();
    auto iter2 = std::ranges::next(cv.begin(), 4);
    flag       = true;
    try {
      iter1 = std::move(iter2);
      assert(false);
    } catch (...) {
      TEST_LIBCPP_ASSERT_FAILURE([&] { [[maybe_unused]] auto iter3 = iter1 - std::default_sentinel_t{}; }(),
                                 "Trying to subtract a valuess iterators of concat_view from the default sentinel.");
    }
  }

  {
    // valueless by exception test operator-(default_sentinel, it)
    flag = false;
    Range<0> r1;
    Range<1> r2;

    auto cv    = std::views::concat(r1, r2);
    auto iter1 = cv.begin();
    auto iter2 = std::ranges::next(cv.begin(), 4);
    flag       = true;
    try {
      iter1 = std::move(iter2);
      assert(false);
    } catch (...) {
      TEST_LIBCPP_ASSERT_FAILURE([&] { [[maybe_unused]] auto iter3 = iter1 - std::default_sentinel_t{}; }(),
                                 "Trying to subtract a valuess iterators of concat_view from the default sentinel.");
    }
  }

  {
    // valueless by exception test operator>
    flag = false;
    Range<0> r1;
    Range<1> r2;

    auto cv    = std::views::concat(r1, r2);
    auto iter1 = cv.begin();
    auto iter2 = std::ranges::next(cv.begin(), 4);
    flag       = true;
    try {
      iter1 = std::move(iter2);
      assert(false);
    } catch (...) {
      TEST_LIBCPP_ASSERT_FAILURE(
          [&] { (void)(iter1 > iter2); }(), "Trying to compare a valueless iterator of concat_view.");
    }
  }

  {
    // valueless by exception test operator>=
    flag = false;
    Range<0> r1;
    Range<1> r2;

    auto cv    = std::views::concat(r1, r2);
    auto iter1 = cv.begin();
    auto iter2 = std::ranges::next(cv.begin(), 4);
    flag       = true;
    try {
      iter1 = std::move(iter2);
      assert(false);
    } catch (...) {
      TEST_LIBCPP_ASSERT_FAILURE(
          [&] { (void)(iter1 >= iter2); }(), "Trying to compare a valueless iterator of concat_view.");
    }
  }

  {
    // valueless by exception test operator<
    flag = false;
    Range<0> r1;
    Range<1> r2;

    auto cv    = std::views::concat(r1, r2);
    auto iter1 = cv.begin();
    auto iter2 = std::ranges::next(cv.begin(), 4);
    flag       = true;
    try {
      iter1 = std::move(iter2);
      assert(false);
    } catch (...) {
      TEST_LIBCPP_ASSERT_FAILURE(
          [&] { (void)(iter1 < iter2); }(), "Trying to compare a valueless iterator of concat_view.");
    }
  }

  {
    // valueless by exception test operator<=
    flag = false;
    Range<0> r1;
    Range<1> r2;

    auto cv    = std::views::concat(r1, r2);
    auto iter1 = cv.begin();
    auto iter2 = std::ranges::next(cv.begin(), 4);
    flag       = true;
    try {
      iter1 = std::move(iter2);
      assert(false);
    } catch (...) {
      TEST_LIBCPP_ASSERT_FAILURE(
          [&] { (void)(iter1 <= iter2); }(), "Trying to compare a valueless iterator of concat_view.");
    }
  }

  {
    // valueless by exception test operator<=>
    flag = false;
    Range<0> r1;
    Range<1> r2;

    auto cv    = std::views::concat(r1, r2);
    auto iter1 = cv.begin();
    auto iter2 = std::ranges::next(cv.begin(), 4);
    flag       = true;
    try {
      iter1 = std::move(iter2);
      assert(false);
    } catch (...) {
      TEST_LIBCPP_ASSERT_FAILURE(
          [&] { (void)(iter1 <=> iter2); }(), "Trying to compare a valueless iterator of concat_view.");
    }
  }

  {
    // valueless by exception test operator- between two iterators
    flag = false;
    Range<0> r1;
    Range<1> r2;

    auto cv    = std::views::concat(r1, r2);
    auto iter1 = cv.begin();
    auto iter2 = std::ranges::next(cv.begin(), 4);
    flag       = true;
    try {
      iter1 = std::move(iter2);
      assert(false);
    } catch (...) {
      TEST_LIBCPP_ASSERT_FAILURE(
          [&] { (void)(iter1 - iter2); }(),
          "Trying to subtract two iterators of concat_view where at least one iterator is valueless.");
    }
  }

  {
    // valueless by exception test operator- with a constant
    flag = false;
    Range<0> r1;
    Range<1> r2;

    auto cv    = std::views::concat(r1, r2);
    auto iter1 = cv.begin();
    auto iter2 = std::ranges::next(cv.begin(), 4);
    flag       = true;
    try {
      iter1 = std::move(iter2);
    } catch (...) {
      TEST_LIBCPP_ASSERT_FAILURE(
          [&] { (void)(iter1 - 1); }(), "Trying to subtract a valuess iterators of concat_view.");
    }
  }

  {
    // valueless by exception test iter_move(it)
    flag = false;
    Range<0> r1;
    Range<1> r2;

    auto cv    = std::views::concat(r1, r2);
    auto iter1 = cv.begin();
    auto iter2 = std::ranges::next(cv.begin(), 4);
    flag       = true;
    try {
      iter1 = std::move(iter2);
      assert(false);
    } catch (...) {
      TEST_LIBCPP_ASSERT_FAILURE([&] { [[maybe_unused]] auto iter3 = std::ranges::iter_move(iter1); }(),
                                 "Trying to apply iter_move to a valueless iterator of concat_view.");
    }
  }

  {
    // valueless by exception test iter_swap(iter1, iter2)
    flag = false;
    Range<0> r1;
    Range<1> r2;

    auto cv    = std::views::concat(r1, r2);
    auto iter1 = cv.begin();
    auto iter2 = std::ranges::next(cv.begin(), 4);
    flag       = true;
    try {
      iter1 = std::move(iter2);
      assert(false);
    } catch (...) {
      TEST_LIBCPP_ASSERT_FAILURE([&] { std::ranges::iter_swap(iter1, iter2); }(),
                                 "Trying to swap iterators of concat_view where at least one iterator is valueless.");
    }
  }

  {
    // valueless by exception test constructor
    flag = false;
    NonSimpleRange<0, int> r1;
    NonSimpleRange<1, int> r2;

    auto cv     = std::views::concat(r1, r2);
    auto iter1  = cv.begin();
    auto iter2  = std::ranges::next(cv.begin(), 4);
    flag        = true;
    using Iter  = std::ranges::iterator_t<decltype(cv)>;       // iterator<false>
    using CIter = std::ranges::iterator_t<const decltype(cv)>; // iterator<true>

    try {
      iter1 = std::move(iter2);
      assert(false);
    } catch (...) {
      TEST_LIBCPP_ASSERT_FAILURE(
          [&] { [[maybe_unused]] CIter it3(iter1); }(), "Trying to convert from a valueless iterator of concat_view.");
    }
  }
}
