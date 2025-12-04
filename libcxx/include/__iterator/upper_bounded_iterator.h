// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/*
 * __upper_bounded_iterator is an iterator that wraps an underlying iterator.
 * It stores the underlying container type to prevent mixing iterators, and allow algorithms
 * to optimize based on the underlying container type.
 * It also encodes the container's (known at compile-time) maximum amount of elements as part of the type.
 * As of writing, the only standard library containers which have this property are inplace_vector and optional.
 */

#ifndef _LIBCPP___ITERATOR_UPPER_BOUNDED_ITERATOR_H
#define _LIBCPP___ITERATOR_UPPER_BOUNDED_ITERATOR_H

#include <__compare/ordering.h>
#include <__compare/three_way_comparable.h>
#include <__config>
#include <__cstddef/size_t.h>
#include <__iterator/incrementable_traits.h>
#include <__iterator/iterator_traits.h>
#include <__memory/pointer_traits.h>
#include <__type_traits/is_constructible.h>
#include <__type_traits/is_convertible.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 26

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Iter, class _Container, std::size_t _Max_Elements>
class __upper_bounded_iterator {
private:
  _Iter __iter_;

  friend _Container;

public:
  using iterator_category = iterator_traits<_Iter>::iterator_category;
  using iterator_concept  = _Iter::iterator_concept;
  using value_type        = iter_value_t<_Iter>;
  using difference_type   = iter_difference_t<_Iter>;
  using reference         = iter_reference_t<_Iter>;

  _LIBCPP_HIDE_FROM_ABI constexpr __upper_bounded_iterator()
    requires is_default_constructible_v<_Iter>
  = default;

  _LIBCPP_HIDE_FROM_ABI constexpr explicit __upper_bounded_iterator(_Iter __iter) : __iter_(std::move(__iter)) {}

  template <typename _Iter2>
    requires is_convertible_v<_Iter2, _Iter>
  _LIBCPP_HIDE_FROM_ABI constexpr __upper_bounded_iterator(
      const __upper_bounded_iterator<_Iter2, _Container, _Max_Elements>& __y)
      : __iter_(__y.__iter_) {}

  _LIBCPP_HIDE_FROM_ABI constexpr _Iter __base() const noexcept(noexcept(_Iter(__iter_))) { return __iter_; }
  _LIBCPP_HIDE_FROM_ABI constexpr auto __max_elements() const noexcept { return _Max_Elements; }

  _LIBCPP_HIDE_FROM_ABI constexpr decltype(auto) operator*() const { return *__iter_; }
  _LIBCPP_HIDE_FROM_ABI constexpr decltype(auto) operator->() const
    requires requires { __iter_.operator->(); }
  {
    return __iter_.operator->();
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __upper_bounded_iterator& operator++() {
    ++__iter_;
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __upper_bounded_iterator operator++(int) {
    __upper_bounded_iterator __tmp(*this);
    ++*this;
    return __tmp;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __upper_bounded_iterator& operator--() {
    --__iter_;
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __upper_bounded_iterator operator--(int) {
    __upper_bounded_iterator __tmp(*this);
    --*this;
    return __tmp;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __upper_bounded_iterator& operator+=(difference_type __x) {
    __iter_ += __x;
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __upper_bounded_iterator& operator-=(difference_type __x) {
    __iter_ -= __x;
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr decltype(auto) operator[](difference_type __n) const { return *(*this + __n); }

  _LIBCPP_HIDE_FROM_ABI friend constexpr bool
  operator==(const __upper_bounded_iterator& __x, const __upper_bounded_iterator& __y) {
    return __x.__iter_ == __y.__iter_;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr auto
  operator<=>(const __upper_bounded_iterator& __x, const __upper_bounded_iterator& __y) {
    if constexpr (three_way_comparable_with<_Iter, _Iter, strong_ordering>) {
      return __x.__iter_ <=> __y.__iter_;
    } else {
      if (__x.__iter_ < __x.__iter_) {
        return strong_ordering::less;
      } else if (__x.__iter_ == __y.__iter_) {
        return strong_ordering::equal;
      }
      return strong_ordering::greater;
    }
  }

  template <class _Iter2>
  _LIBCPP_HIDE_FROM_ABI friend constexpr auto
  operator<=>(const __upper_bounded_iterator& __x,
              const __upper_bounded_iterator<_Iter2, _Container, _Max_Elements> __y) {
    if constexpr (three_way_comparable_with<_Iter, _Iter2, strong_ordering>) {
      return __x.__iter_ <=> __y.__iter_;
    } else {
      if (__x.__iter_ < __x.__iter_) {
        return strong_ordering::less;
      } else if (__x.__iter_ == __y.__iter_) {
        return strong_ordering::equal;
      }
      return strong_ordering::greater;
    }
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr __upper_bounded_iterator
  operator+(const __upper_bounded_iterator& __i, difference_type __n) {
    auto __tmp = __i;
    __tmp += __n;
    return __tmp;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr __upper_bounded_iterator
  operator+(difference_type __n, const __upper_bounded_iterator& __i) {
    auto __tmp = __i;
    __tmp += __n;
    return __tmp;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr __upper_bounded_iterator
  operator-(const __upper_bounded_iterator& __i, difference_type __n) {
    return __i.__iter_ + __n;
  }

  template <class _Iter2>
  _LIBCPP_HIDE_FROM_ABI friend constexpr difference_type
  operator-(const __upper_bounded_iterator& __x,
            const __upper_bounded_iterator<_Iter2, _Container, _Max_Elements>& __y) {
    return difference_type(__x.__base() - __y.__base());
  }
};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 26

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ITERATOR_UPPER_BOUNDED_ITERATOR_H
