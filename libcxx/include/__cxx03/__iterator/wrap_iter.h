// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CXX03___ITERATOR_WRAP_ITER_H
#define _LIBCPP___CXX03___ITERATOR_WRAP_ITER_H

#include <__cxx03/__config>
#include <__cxx03/__iterator/iterator_traits.h>
#include <__cxx03/__memory/addressof.h>
#include <__cxx03/__memory/pointer_traits.h>
#include <__cxx03/__type_traits/enable_if.h>
#include <__cxx03/__type_traits/is_convertible.h>
#include <__cxx03/cstddef>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Iter>
class __wrap_iter {
public:
  typedef _Iter iterator_type;
  typedef typename iterator_traits<iterator_type>::value_type value_type;
  typedef typename iterator_traits<iterator_type>::difference_type difference_type;
  typedef typename iterator_traits<iterator_type>::pointer pointer;
  typedef typename iterator_traits<iterator_type>::reference reference;
  typedef typename iterator_traits<iterator_type>::iterator_category iterator_category;

private:
  iterator_type __i_;

public:
  _LIBCPP_HIDE_FROM_ABI __wrap_iter() _NOEXCEPT : __i_() {}
  template <class _Up, __enable_if_t<is_convertible<_Up, iterator_type>::value, int> = 0>
  _LIBCPP_HIDE_FROM_ABI __wrap_iter(const __wrap_iter<_Up>& __u) _NOEXCEPT : __i_(__u.base()) {}
  _LIBCPP_HIDE_FROM_ABI reference operator*() const _NOEXCEPT { return *__i_; }
  _LIBCPP_HIDE_FROM_ABI pointer operator->() const _NOEXCEPT { return std::__to_address(__i_); }
  _LIBCPP_HIDE_FROM_ABI __wrap_iter& operator++() _NOEXCEPT {
    ++__i_;
    return *this;
  }
  _LIBCPP_HIDE_FROM_ABI __wrap_iter operator++(int) _NOEXCEPT {
    __wrap_iter __tmp(*this);
    ++(*this);
    return __tmp;
  }

  _LIBCPP_HIDE_FROM_ABI __wrap_iter& operator--() _NOEXCEPT {
    --__i_;
    return *this;
  }
  _LIBCPP_HIDE_FROM_ABI __wrap_iter operator--(int) _NOEXCEPT {
    __wrap_iter __tmp(*this);
    --(*this);
    return __tmp;
  }
  _LIBCPP_HIDE_FROM_ABI __wrap_iter operator+(difference_type __n) const _NOEXCEPT {
    __wrap_iter __w(*this);
    __w += __n;
    return __w;
  }
  _LIBCPP_HIDE_FROM_ABI __wrap_iter& operator+=(difference_type __n) _NOEXCEPT {
    __i_ += __n;
    return *this;
  }
  _LIBCPP_HIDE_FROM_ABI __wrap_iter operator-(difference_type __n) const _NOEXCEPT { return *this + (-__n); }
  _LIBCPP_HIDE_FROM_ABI __wrap_iter& operator-=(difference_type __n) _NOEXCEPT {
    *this += -__n;
    return *this;
  }
  _LIBCPP_HIDE_FROM_ABI reference operator[](difference_type __n) const _NOEXCEPT { return __i_[__n]; }

  _LIBCPP_HIDE_FROM_ABI iterator_type base() const _NOEXCEPT { return __i_; }

private:
  _LIBCPP_HIDE_FROM_ABI explicit __wrap_iter(iterator_type __x) _NOEXCEPT : __i_(__x) {}

  template <class _Up>
  friend class __wrap_iter;
  template <class _CharT, class _Traits, class _Alloc>
  friend class basic_string;
  template <class _CharT, class _Traits>
  friend class basic_string_view;
  template <class _Tp, class _Alloc>
  friend class _LIBCPP_TEMPLATE_VIS vector;
  template <class _Tp, size_t>
  friend class _LIBCPP_TEMPLATE_VIS span;
  template <class _Tp, size_t _Size>
  friend struct array;
};

template <class _Iter1>
_LIBCPP_HIDE_FROM_ABI bool operator==(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter1>& __y) _NOEXCEPT {
  return __x.base() == __y.base();
}

template <class _Iter1, class _Iter2>
_LIBCPP_HIDE_FROM_ABI bool operator==(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter2>& __y) _NOEXCEPT {
  return __x.base() == __y.base();
}

template <class _Iter1>
_LIBCPP_HIDE_FROM_ABI bool operator<(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter1>& __y) _NOEXCEPT {
  return __x.base() < __y.base();
}

template <class _Iter1, class _Iter2>
_LIBCPP_HIDE_FROM_ABI bool operator<(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter2>& __y) _NOEXCEPT {
  return __x.base() < __y.base();
}

template <class _Iter1>
_LIBCPP_HIDE_FROM_ABI bool operator!=(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter1>& __y) _NOEXCEPT {
  return !(__x == __y);
}

template <class _Iter1, class _Iter2>
_LIBCPP_HIDE_FROM_ABI bool operator!=(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter2>& __y) _NOEXCEPT {
  return !(__x == __y);
}

// TODO(mordante) disable these overloads in the LLVM 20 release.
template <class _Iter1>
_LIBCPP_HIDE_FROM_ABI bool operator>(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter1>& __y) _NOEXCEPT {
  return __y < __x;
}

template <class _Iter1, class _Iter2>
_LIBCPP_HIDE_FROM_ABI bool operator>(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter2>& __y) _NOEXCEPT {
  return __y < __x;
}

template <class _Iter1>
_LIBCPP_HIDE_FROM_ABI bool operator>=(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter1>& __y) _NOEXCEPT {
  return !(__x < __y);
}

template <class _Iter1, class _Iter2>
_LIBCPP_HIDE_FROM_ABI bool operator>=(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter2>& __y) _NOEXCEPT {
  return !(__x < __y);
}

template <class _Iter1>
_LIBCPP_HIDE_FROM_ABI bool operator<=(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter1>& __y) _NOEXCEPT {
  return !(__y < __x);
}

template <class _Iter1, class _Iter2>
_LIBCPP_HIDE_FROM_ABI bool operator<=(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter2>& __y) _NOEXCEPT {
  return !(__y < __x);
}

template <class _Iter1, class _Iter2>
_LIBCPP_HIDE_FROM_ABI typename __wrap_iter<_Iter1>::difference_type
operator-(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter2>& __y) _NOEXCEPT {
  return __x.base() - __y.base();
}

template <class _Iter1>
_LIBCPP_HIDE_FROM_ABI __wrap_iter<_Iter1>
operator+(typename __wrap_iter<_Iter1>::difference_type __n, __wrap_iter<_Iter1> __x) _NOEXCEPT {
  __x += __n;
  return __x;
}

template <class _It>
struct __libcpp_is_contiguous_iterator<__wrap_iter<_It> > : true_type {};

template <class _It>
struct _LIBCPP_TEMPLATE_VIS pointer_traits<__wrap_iter<_It> > {
  typedef __wrap_iter<_It> pointer;
  typedef typename pointer_traits<_It>::element_type element_type;
  typedef typename pointer_traits<_It>::difference_type difference_type;

  _LIBCPP_HIDE_FROM_ABI static element_type* to_address(pointer __w) _NOEXCEPT { return std::__to_address(__w.base()); }
};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___CXX03___ITERATOR_WRAP_ITER_H
