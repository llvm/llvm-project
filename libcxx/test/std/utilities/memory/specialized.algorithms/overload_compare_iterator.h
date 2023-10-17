// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBCPP_TEST_STD_UTILITIES_MEMORY_SPECIALIZED_ALGORITHMS_OVERLOAD_COMPARE_ITERATOR_H
#define LIBCPP_TEST_STD_UTILITIES_MEMORY_SPECIALIZED_ALGORITHMS_OVERLOAD_COMPARE_ITERATOR_H

#include <iterator>
#include <memory>

// An iterator type that overloads operator== and operator!= without any constraints, which
// can trip up some algorithms if we compare iterators against types that we're not allowed to.
//
// See https://github.com/llvm/llvm-project/issues/69334 for details.
template <class Iterator>
struct overload_compare_iterator {
  using value_type        = typename std::iterator_traits<Iterator>::value_type;
  using difference_type   = typename std::iterator_traits<Iterator>::difference_type;
  using reference         = typename std::iterator_traits<Iterator>::reference;
  using pointer           = typename std::iterator_traits<Iterator>::pointer;
  using iterator_category = typename std::iterator_traits<Iterator>::iterator_category;

  overload_compare_iterator() = default;

  explicit overload_compare_iterator(Iterator it) : it_(it) {}

  overload_compare_iterator(overload_compare_iterator const&)            = default;
  overload_compare_iterator(overload_compare_iterator&&)                 = default;
  overload_compare_iterator& operator=(overload_compare_iterator const&) = default;
  overload_compare_iterator& operator=(overload_compare_iterator&&)      = default;

  constexpr reference operator*() const noexcept { return *it_; }

  constexpr pointer operator->() const noexcept { return std::addressof(*it_); }

  constexpr overload_compare_iterator& operator++() noexcept {
    ++it_;
    return *this;
  }

  constexpr overload_compare_iterator operator++(int) const noexcept {
    overload_compare_iterator old{*this};
    ++(*this);
    return old;
  }

  constexpr bool operator==(overload_compare_iterator const& other) const noexcept { return this->it_ == other.it_; }

  constexpr bool operator!=(overload_compare_iterator const& other) const noexcept { return !this->operator==(other); }

  // Hostile overloads
  template <class Sentinel>
  friend bool operator==(overload_compare_iterator const& lhs, Sentinel const& rhs) noexcept {
    return static_cast<Iterator const&>(lhs) == rhs;
  }

  template <class Sentinel>
  friend bool operator!=(overload_compare_iterator const& lhs, Sentinel const& rhs) noexcept {
    return static_cast<Iterator const&>(lhs) != rhs;
  }

private:
  Iterator it_;
};

#endif // LIBCPP_TEST_STD_UTILITIES_MEMORY_SPECIALIZED_ALGORITHMS_OVERLOAD_COMPARE_ITERATOR_H
