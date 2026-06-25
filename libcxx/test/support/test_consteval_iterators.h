//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_TEST_CONSTEVAL_ITERATORS_H
#define SUPPORT_TEST_CONSTEVAL_ITERATORS_H

#include <iterator>
#include <type_traits>
#include <utility>

#include "double_move_tracker.h"
#include "test_macros.h"

#if TEST_STD_VER >= 20

template <std::random_access_iterator It>
class consteval_random_access_iterator {
  It it_;
  support::double_move_tracker tracker_;

  template <std::random_access_iterator>
  friend class consteval_random_access_iterator;

public:
  using iterator_category = std::input_iterator_tag;
  using iterator_concept  = std::random_access_iterator_tag;
  using value_type        = std::iterator_traits<It>::value_type;
  using difference_type   = std::iterator_traits<It>::difference_type;

  consteval consteval_random_access_iterator() : it_() {}
  consteval explicit consteval_random_access_iterator(It it) : it_(it) {}
  consteval consteval_random_access_iterator(const consteval_random_access_iterator&) = default;
  consteval consteval_random_access_iterator(consteval_random_access_iterator&&)      = default;

  consteval consteval_random_access_iterator& operator=(const consteval_random_access_iterator&) = default;
  consteval consteval_random_access_iterator& operator=(consteval_random_access_iterator&&)      = default;

  template <class U>
  consteval consteval_random_access_iterator(const consteval_random_access_iterator<U>& u)
      : it_(u.it_), tracker_(u.tracker_) {}

  template <class U>
  consteval consteval_random_access_iterator(consteval_random_access_iterator<U>&& u)
      : it_(std::move(u.it_)), tracker_(std::move(u.tracker_)) {
    u.it_ = U();
  }

  consteval decltype(auto) operator*() const { return *it_; }
  consteval decltype(auto) operator[](difference_type n) const { return it_[n]; }

  consteval consteval_random_access_iterator& operator++() {
    ++it_;
    return *this;
  }
  consteval consteval_random_access_iterator& operator--() {
    --it_;
    return *this;
  }
  consteval consteval_random_access_iterator operator++(int) { return consteval_random_access_iterator(it_++); }
  consteval consteval_random_access_iterator operator--(int) { return consteval_random_access_iterator(it_--); }

  consteval consteval_random_access_iterator& operator+=(difference_type n) {
    it_ += n;
    return *this;
  }
  consteval consteval_random_access_iterator& operator-=(difference_type n) {
    it_ -= n;
    return *this;
  }
  friend consteval consteval_random_access_iterator operator+(consteval_random_access_iterator x, difference_type n) {
    x += n;
    return x;
  }
  friend consteval consteval_random_access_iterator operator+(difference_type n, consteval_random_access_iterator x) {
    x += n;
    return x;
  }
  friend consteval consteval_random_access_iterator operator-(consteval_random_access_iterator x, difference_type n) {
    x -= n;
    return x;
  }
  friend consteval difference_type operator-(consteval_random_access_iterator x, consteval_random_access_iterator y) {
    return x.it_ - y.it_;
  }

  friend consteval bool
  operator==(const consteval_random_access_iterator& x, const consteval_random_access_iterator& y) {
    return x.it_ == y.it_;
  }
  friend consteval bool
  operator!=(const consteval_random_access_iterator& x, const consteval_random_access_iterator& y) {
    return x.it_ != y.it_;
  }
  friend consteval bool
  operator<(const consteval_random_access_iterator& x, const consteval_random_access_iterator& y) {
    return x.it_ < y.it_;
  }
  friend consteval bool
  operator<=(const consteval_random_access_iterator& x, const consteval_random_access_iterator& y) {
    return x.it_ <= y.it_;
  }
  friend consteval bool
  operator>(const consteval_random_access_iterator& x, const consteval_random_access_iterator& y) {
    return x.it_ > y.it_;
  }
  friend consteval bool
  operator>=(const consteval_random_access_iterator& x, const consteval_random_access_iterator& y) {
    return x.it_ >= y.it_;
  }

  friend consteval It base(const consteval_random_access_iterator& i) { return i.it_; }

  template <class T>
  void operator,(T const&) = delete;
};
template <class It>
consteval_random_access_iterator(It) -> consteval_random_access_iterator<It>;

static_assert(std::random_access_iterator<consteval_random_access_iterator<int*>>);

template <std::contiguous_iterator It>
class consteval_contiguous_iterator {
  It it_;
  support::double_move_tracker tracker_;

  template <std::contiguous_iterator U>
  friend class consteval_contiguous_iterator;

public:
  using iterator_category = std::contiguous_iterator_tag;
  using value_type        = std::iterator_traits<It>::value_type;
  using difference_type   = std::iterator_traits<It>::difference_type;
  using pointer           = std::iterator_traits<It>::pointer;
  using reference         = std::iterator_traits<It>::reference;
  using element_type      = value_type;

  consteval It base() const { return it_; }

  consteval consteval_contiguous_iterator() : it_() {}
  consteval explicit consteval_contiguous_iterator(It it) : it_(it) {}
  consteval consteval_contiguous_iterator(const consteval_contiguous_iterator&) = default;
  consteval consteval_contiguous_iterator(consteval_contiguous_iterator&&)      = default;

  consteval consteval_contiguous_iterator& operator=(const consteval_contiguous_iterator&) = default;
  consteval consteval_contiguous_iterator& operator=(consteval_contiguous_iterator&&)      = default;

  template <class U>
  consteval consteval_contiguous_iterator(const consteval_contiguous_iterator<U>& u)
      : it_(u.it_), tracker_(u.tracker_) {}

  template <class U, std::enable_if_t<std::is_default_constructible_v<U>, int> = 0>
  consteval consteval_contiguous_iterator(consteval_contiguous_iterator<U>&& u)
      : it_(std::move(u.it_)), tracker_(std::move(u.tracker_)) {
    u.it_ = U();
  }

  consteval reference operator*() const { return *it_; }
  consteval pointer operator->() const { return it_; }
  consteval reference operator[](difference_type n) const { return it_[n]; }

  consteval consteval_contiguous_iterator& operator++() {
    ++it_;
    return *this;
  }
  consteval consteval_contiguous_iterator& operator--() {
    --it_;
    return *this;
  }
  consteval consteval_contiguous_iterator operator++(int) { return consteval_contiguous_iterator(it_++); }
  consteval consteval_contiguous_iterator operator--(int) { return consteval_contiguous_iterator(it_--); }

  consteval consteval_contiguous_iterator& operator+=(difference_type n) {
    it_ += n;
    return *this;
  }
  consteval consteval_contiguous_iterator& operator-=(difference_type n) {
    it_ -= n;
    return *this;
  }
  friend consteval consteval_contiguous_iterator operator+(consteval_contiguous_iterator x, difference_type n) {
    x += n;
    return x;
  }
  friend consteval consteval_contiguous_iterator operator+(difference_type n, consteval_contiguous_iterator x) {
    x += n;
    return x;
  }
  friend consteval consteval_contiguous_iterator operator-(consteval_contiguous_iterator x, difference_type n) {
    x -= n;
    return x;
  }
  friend consteval difference_type operator-(consteval_contiguous_iterator x, consteval_contiguous_iterator y) {
    return x.it_ - y.it_;
  }

  friend consteval bool operator==(const consteval_contiguous_iterator& x, const consteval_contiguous_iterator& y) {
    return x.it_ == y.it_;
  }
  friend consteval bool operator!=(const consteval_contiguous_iterator& x, const consteval_contiguous_iterator& y) {
    return x.it_ != y.it_;
  }
  friend consteval bool operator<(const consteval_contiguous_iterator& x, const consteval_contiguous_iterator& y) {
    return x.it_ < y.it_;
  }
  friend consteval bool operator<=(const consteval_contiguous_iterator& x, const consteval_contiguous_iterator& y) {
    return x.it_ <= y.it_;
  }
  friend consteval bool operator>(const consteval_contiguous_iterator& x, const consteval_contiguous_iterator& y) {
    return x.it_ > y.it_;
  }
  friend consteval bool operator>=(const consteval_contiguous_iterator& x, const consteval_contiguous_iterator& y) {
    return x.it_ >= y.it_;
  }

  // Note no operator<=>, use three_way_contiguous_iterator for testing operator<=>

  friend consteval It base(const consteval_contiguous_iterator& i) { return i.it_; }

  template <class T>
  void operator,(T const&) = delete;
};
template <class It>
consteval_contiguous_iterator(It) -> consteval_contiguous_iterator<It>;

static_assert(std::contiguous_iterator<consteval_contiguous_iterator<int*>>);

#endif // TEST_STD_VER >= 20

#endif // SUPPORT_TEST_CONSTEVAL_ITERATORS_H
