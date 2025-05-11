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

int val[] = {1, 2, 3};

bool flag = false;

template<std::size_t N> struct Iter;

template<std::size_t N>
struct Iter {
    using value_type = int;
    using difference_type = std::ptrdiff_t;
    using reference = int&;
    using pointer = int*;
    using iterator_category = std::random_access_iterator_tag;
    using iterator_concept = std::random_access_iterator_tag;

private:
    int* ptr_ = nullptr;

    template<std::size_t M> friend struct Iter;

public:
    Iter() = default;
    Iter(int* ptr) : ptr_(ptr) {}
    Iter(const Iter&) = default;
    Iter(Iter&& other) noexcept : ptr_(other.ptr_) {}

    template<std::size_t M>
    Iter(const Iter<M>& other) : ptr_(other.ptr_) {}

    Iter& operator=(const Iter&) = default;
    Iter& operator=(Iter&& other) noexcept {
        ptr_ = other.ptr_;
        return *this;
    }

    reference operator*() const { return *ptr_; }
    pointer operator->() const { return ptr_; }
    reference operator[](difference_type n) const { return ptr_[n]; }

    Iter& operator++() { ++ptr_; return *this; }
    Iter operator++(int) { auto tmp = *this; ++*this; return tmp; }
    Iter& operator--() { --ptr_; return *this; }
    Iter operator--(int) { auto tmp = *this; --*this; return tmp; }

    Iter& operator+=(difference_type n) { ptr_ += n; return *this; }
    Iter& operator-=(difference_type n) { ptr_ -= n; return *this; }

    template<std::size_t X>
    friend Iter<X> operator+(Iter<X> it, difference_type n);

    template<std::size_t X>
    friend Iter<X> operator+(difference_type n, Iter<X> it);

    template<std::size_t X>
    friend Iter<X> operator-(Iter<X> it, difference_type n);

    template<std::size_t X, std::size_t Y>
    friend difference_type operator-(Iter<X> a, Iter<Y> b);

    friend bool operator==(Iter a, Iter b) = default;
    friend bool operator<(Iter a, Iter b) { return a.ptr_ < b.ptr_; }
    friend bool operator>(Iter a, Iter b) { return a.ptr_ > b.ptr_; }
    friend bool operator<=(Iter a, Iter b) { return a.ptr_ <= b.ptr_; }
    friend bool operator>=(Iter a, Iter b) { return a.ptr_ >= b.ptr_; }
};

template<std::size_t X>
inline Iter<X> operator+(Iter<X> it, std::ptrdiff_t n) {
    return Iter<X>(it.ptr_ + n);
}

template<std::size_t X>
inline Iter<X> operator+(std::ptrdiff_t n, Iter<X> it) {
    return Iter<X>(it.ptr_ + n);
}

template<std::size_t X>
inline Iter<X> operator-(Iter<X> it, std::ptrdiff_t n) {
    return Iter<X>(it.ptr_ - n);
}

template<std::size_t X, std::size_t Y>
inline std::ptrdiff_t operator-(Iter<X> a, Iter<Y> b) {
    return a.ptr_ - b.ptr_;
}

template<std::size_t N>
struct Range : std::ranges::view_base {
    using iterator = Iter<N>;
    using const_iterator = Iter<N>;

    int* data_;
    std::size_t size_;

    Range() : data_(val), size_(4) {}

    Range(int* data, std::size_t size) : data_(data), size_(size) {}

    iterator begin() { return iterator(data_); }
    iterator end() { return iterator(data_ + size_); }

    const_iterator begin() const { return const_iterator(data_); }
    const_iterator end() const { return const_iterator(data_ + size_); }

    std::size_t size() const { return size_; }
};

static_assert(std::ranges::range<Range<0>>);
static_assert(std::ranges::sized_range<Range<0>>);

int main() {

  {
    //valueless by exception test operator*
    Range<0> r1;
    Range<1> r2;

    auto cv = std::views::concat(r1, r2);
    auto iter1 = cv.begin();
    auto iter2 = std::ranges::next(cv.begin(), 4);
    flag = true;
    try {
        iter1 = std::move(iter2);
    } catch (...) {
      auto f = std::ranges::distance(r1);
      (void)f;
      TEST_LIBCPP_ASSERT_FAILURE(
          [=] {
            *iter1;
          }(),
          "valueless by exception");
    }
  }

  {
    //valueless by exception test operator==
    flag = false;
    Range<0> r1;
    Range<1> r2;

    auto cv = std::views::concat(r1, r2);
    auto iter1 = cv.begin();
    auto iter2 = std::ranges::next(cv.begin(), 4);
    auto iter3 = cv.begin();
    flag = true;
    try {
        iter1 = std::move(iter2);
    } catch (...) {
      TEST_LIBCPP_ASSERT_FAILURE(
          [=] {
            (void)(iter1 == iter3);
          }(),
          "valueless by exception");
    }
  }

  {
    //valueless by exception test operator--
    flag = false;
    Range<0> r1;
    Range<1> r2;

    auto cv = std::views::concat(r1, r2);
    auto iter1 = cv.begin();
    auto iter2 = std::ranges::next(cv.begin(), 4);
    flag = true;
    try {
        iter1 = std::move(iter2);
    } catch (...) {
      //ASSERT_SAME_TYPE(decltype(iter1), int);
      TEST_LIBCPP_ASSERT_FAILURE(
          [&] {
            iter1--;
          }(),
          "valueless by exception");
    }
  }

  {
    //valueless by exception test operator++
    flag = false;
    Range<0> r1;
    Range<1> r2;

    auto cv = std::views::concat(r1, r2);
    auto iter1 = cv.begin();
    auto iter2 = std::ranges::next(cv.begin(), 4);
    flag = true;
    try {
        iter1 = std::move(iter2);
    } catch (...) {
      TEST_LIBCPP_ASSERT_FAILURE(
          [&] {
            ++iter1;
          }(),
          "valueless by exception");
    }
  }

  {
    //valueless by exception test operator+=
    flag = false;
    Range<0> r1;
    Range<1> r2;

    auto cv = std::views::concat(r1, r2);
    auto iter1 = cv.begin();
    auto iter2 = std::ranges::next(cv.begin(), 4);
    flag = true;
    try {
        iter1 = std::move(iter2);
    } catch (...) {
      TEST_LIBCPP_ASSERT_FAILURE(
          [&] {
            iter1 += 1;
          }(),
          "valueless by exception");
    }
  }
}
