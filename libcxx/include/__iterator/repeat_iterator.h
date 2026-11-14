//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ITERATOR_REPEAT_ITERATOR_H
#define _LIBCPP___ITERATOR_REPEAT_ITERATOR_H

#include <__assert>
#include <__config>
#include <__cstddef/ptrdiff_t.h>
#include <__cstddef/size_t.h>
#include <__iterator/iterator_traits.h>
#include <__memory/addressof.h>
#include <__type_traits/remove_const.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

// This iterator repeats a given value N times. This is essentially repeat_view::iterator with a bound of type size_t,
// but available before C++20.

template <class _Tp>
class __repeat_iterator {
  _Tp* __value_;
  size_t __count_;

public:
  using value_type        = __remove_const_t<_Tp>;
  using difference_type   = ptrdiff_t;
  using pointer           = _Tp*;
  using reference         = _Tp&;
  using iterator_category = random_access_iterator_tag;

  _LIBCPP_CONSTEXPR_SINCE_CXX14 __repeat_iterator(_Tp& __val, size_t __count)
      : __value_(std::addressof(__val)), __count_(__count) {}

  [[__nodiscard__]] _LIBCPP_CONSTEXPR_SINCE_CXX14 _Tp& operator*() const _NOEXCEPT { return *__value_; }

  _LIBCPP_CONSTEXPR_SINCE_CXX14 _Tp* operator->() const _NOEXCEPT { return __value_; }

  _LIBCPP_CONSTEXPR_SINCE_CXX14 __repeat_iterator& operator++() {
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(__count_ > 0, "Iterator already at or past end.");
    --__count_;
    return *this;
  }

  _LIBCPP_CONSTEXPR_SINCE_CXX14 __repeat_iterator operator++(int) {
    auto __tmp = *this;
    ++*this;
    return __tmp;
  }

  _LIBCPP_CONSTEXPR_SINCE_CXX14 __repeat_iterator& operator--() {
    ++__count_;
    return *this;
  }

  _LIBCPP_CONSTEXPR_SINCE_CXX14 __repeat_iterator operator--(int) {
    auto __tmp = *this;
    --*this;
    return __tmp;
  }

  [[__nodiscard__]] _LIBCPP_CONSTEXPR_SINCE_CXX14 __repeat_iterator operator+(ptrdiff_t __n) const {
    return __repeat_iterator(*__value_, __count_ - __n);
  }

  [[__nodiscard__]] friend _LIBCPP_CONSTEXPR_SINCE_CXX14 __repeat_iterator
  operator+(ptrdiff_t __n, const __repeat_iterator& __x) {
    return __x + __n;
  }

  _LIBCPP_CONSTEXPR_SINCE_CXX14 __repeat_iterator& operator+=(ptrdiff_t __n) {
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(__n <= __count_, "Cannot advance iterator past end.");
    __count_ -= __n;
    return *this;
  }

  [[__nodiscard__]] _LIBCPP_CONSTEXPR_SINCE_CXX14 __repeat_iterator operator-(ptrdiff_t __n) const {
    return __repeat_iterator(*__value_, __count_ + __n);
  }

  [[__nodiscard__]] friend _LIBCPP_CONSTEXPR_SINCE_CXX14 ptrdiff_t
  operator-(const __repeat_iterator& __lhs, const __repeat_iterator& __rhs) _NOEXCEPT {
    return __rhs.__count_ - __lhs.__count_;
  }

  _LIBCPP_CONSTEXPR_SINCE_CXX14 __repeat_iterator& operator-=(ptrdiff_t __n) {
    __count_ += __n;
    return *this;
  }

  [[__nodiscard__]] _LIBCPP_CONSTEXPR_SINCE_CXX14 _Tp& operator[]([[__maybe_unused__]] ptrdiff_t __n) const {
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(__n < __count_, "Subscript argument must be less than size.");
    return *__value_;
  }

  friend _LIBCPP_CONSTEXPR_SINCE_CXX14 bool
  operator==(const __repeat_iterator& __lhs, const __repeat_iterator& __rhs) _NOEXCEPT {
    return __lhs.__count_ == __rhs.__count_;
  }

  friend _LIBCPP_CONSTEXPR_SINCE_CXX14 bool
  operator!=(const __repeat_iterator& __lhs, const __repeat_iterator& __rhs) _NOEXCEPT {
    return !(__lhs == __rhs);
  }
};

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ITERATOR_REPEAT_ITERATOR_H
