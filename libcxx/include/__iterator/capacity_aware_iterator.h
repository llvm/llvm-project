// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ITERATOR_UPPER_BOUNDED_ITERATOR_H
#define _LIBCPP___ITERATOR_UPPER_BOUNDED_ITERATOR_H

#include <__assert>
#include <__compare/ordering.h>
#include <__compare/three_way_comparable.h>
#include <__config>
#include <__cstddef/size_t.h>
#include <__iterator/concepts.h>
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

// __capacity_aware_iterator is an iterator that wraps an underlying iterator.
// It stores the underlying container type to prevent mixing iterators, and allow algorithms
// to optimize based on the underlying container type.
// It also encodes the container's (known at compile-time) maximum amount of elements as part of the type.
// As of writing, the only standard library containers which have this property are inplace_vector and optional.

template <class _Iter, class _Container, std::size_t _ContainerMaxElements>
class __capacity_aware_iterator {
private:
  _Iter __iter_;

  friend _Container;

  _LIBCPP_HIDE_FROM_ABI static constexpr auto __get_iter_concept() {
    if constexpr (contiguous_iterator<_Iter>) {
      return contiguous_iterator_tag{};
    } else if constexpr (random_access_iterator<_Iter>) {
      return random_access_iterator_tag{};
    } else if constexpr (bidirectional_iterator<_Iter>) {
      return bidirectional_iterator_tag{};
    } else if constexpr (forward_iterator<_Iter>) {
      return forward_iterator_tag{};
    } else {
      return input_iterator_tag{};
    }
  }

public:
  using iterator_category = iterator_traits<_Iter>::iterator_category;
  using iterator_concept  = decltype(__get_iter_concept());
  using difference_type   = iter_difference_t<_Iter>;
  using pointer           = iterator_traits<_Iter>::pointer;
  using reference         = iter_reference_t<_Iter>;
  using value_type        = iter_value_t<_Iter>;

  _LIBCPP_HIDE_FROM_ABI constexpr __capacity_aware_iterator()
    requires is_default_constructible_v<_Iter>
  = default;

  template <typename _Iter2>
    requires is_convertible_v<_Iter2, _Iter>
  _LIBCPP_HIDE_FROM_ABI constexpr __capacity_aware_iterator(
      const __capacity_aware_iterator<_Iter2, _Container, _ContainerMaxElements>& __y)
      : __iter_(__y.base()) {}

private:
  _LIBCPP_HIDE_FROM_ABI constexpr explicit __capacity_aware_iterator(_Iter __iter) : __iter_(std::move(__iter)) {}

  template <typename _Tp, class>
  friend struct __optional_iterator;

  template <class _It, class _Container2, size_t _ContainerMaxElems2>
  _LIBCPP_HIDE_FROM_ABI friend constexpr auto __make_capacity_aware_iterator(_It __iter);

public:
  _LIBCPP_HIDE_FROM_ABI constexpr _Iter base() const noexcept { return __iter_; }
  _LIBCPP_HIDE_FROM_ABI constexpr decltype(auto) operator*() const { return *__iter_; }
  _LIBCPP_HIDE_FROM_ABI constexpr decltype(auto) operator->() const
    requires requires { __iter_.operator->(); }
  {
    return __iter_.operator->();
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __capacity_aware_iterator& operator++() {
    ++__iter_;
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __capacity_aware_iterator operator++(int) {
    __capacity_aware_iterator __tmp(*this);
    ++*this;
    return __tmp;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __capacity_aware_iterator& operator--() {
    --__iter_;
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __capacity_aware_iterator operator--(int) {
    __capacity_aware_iterator __tmp(*this);
    --*this;
    return __tmp;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __capacity_aware_iterator& operator+=(difference_type __n) {
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
        static_cast<size_t>((__n >= 0 ? __n : -__n)) <= _ContainerMaxElements,
        "__capacity_aware_iterator::operator+=: Attempting to move iterator past its container's possible range");

    __iter_ += __n;
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __capacity_aware_iterator& operator-=(difference_type __n) {
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
        static_cast<size_t>((__n >= 0 ? __n : -__n)) <= _ContainerMaxElements,
        "__capacity_aware_iterator::operator-=: Attempting to move iterator past its container's possible range");

    __iter_ -= __n;
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr decltype(auto) operator[](difference_type __n) const {
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
        static_cast<size_t>(__n >= 0 ? __n : -__n) < _ContainerMaxElements,
        "__capacity_aware_iterator::operator[]: Attempting to index iterator past its container's possible range");
    return *(*this + __n);
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr bool
  operator==(const __capacity_aware_iterator& __x, const __capacity_aware_iterator& __y) {
    return __x.__iter_ == __y.__iter_;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr auto
  operator<=>(const __capacity_aware_iterator& __x, const __capacity_aware_iterator& __y) {
    if constexpr (three_way_comparable_with<_Iter, _Iter, strong_ordering>) {
      return __x.__iter_ <=> __y.__iter_;
    } else {
      if (__x.__iter_ < __y.__iter_) {
        return strong_ordering::less;
      } else if (__x.__iter_ == __y.__iter_) {
        return strong_ordering::equal;
      }
      return strong_ordering::greater;
    }
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr __capacity_aware_iterator
  operator+(const __capacity_aware_iterator& __i, difference_type __n) {
    auto __tmp = __i;
    __tmp += __n;
    return __tmp;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr __capacity_aware_iterator
  operator+(difference_type __n, const __capacity_aware_iterator& __i) {
    auto __tmp = __i;
    __tmp += __n;
    return __tmp;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr __capacity_aware_iterator
  operator-(const __capacity_aware_iterator& __i, difference_type __n) {
    auto __tmp = __i;
    __tmp -= __n;
    return __tmp;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr difference_type
  operator-(const __capacity_aware_iterator& __x, const __capacity_aware_iterator& __y) {
    return difference_type(__x.base() - __y.base());
  }
};

template <class _It, class _Container2, size_t _ContainerMaxElems2>
_LIBCPP_HIDE_FROM_ABI constexpr auto __make_capacity_aware_iterator(_It __iter) {
  return __capacity_aware_iterator<_It, _Container2, _ContainerMaxElems2>(__iter);
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 26

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ITERATOR_UPPER_BOUNDED_ITERATOR_H
