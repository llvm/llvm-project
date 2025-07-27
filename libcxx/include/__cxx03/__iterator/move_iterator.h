// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CXX03___ITERATOR_MOVE_ITERATOR_H
#define _LIBCPP___CXX03___ITERATOR_MOVE_ITERATOR_H

#include <__cxx03/__config>
#include <__cxx03/__iterator/iterator_traits.h>
#include <__cxx03/__type_traits/conditional.h>
#include <__cxx03/__type_traits/enable_if.h>
#include <__cxx03/__type_traits/is_assignable.h>
#include <__cxx03/__type_traits/is_constructible.h>
#include <__cxx03/__type_traits/is_convertible.h>
#include <__cxx03/__type_traits/is_reference.h>
#include <__cxx03/__type_traits/is_same.h>
#include <__cxx03/__type_traits/remove_reference.h>
#include <__cxx03/__utility/declval.h>
#include <__cxx03/__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__cxx03/__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Iter>
class _LIBCPP_TEMPLATE_VIS move_iterator {
public:
  typedef _Iter iterator_type;
  typedef _If< __has_random_access_iterator_category<_Iter>::value,
               random_access_iterator_tag,
               typename iterator_traits<_Iter>::iterator_category >
      iterator_category;
  typedef typename iterator_traits<iterator_type>::value_type value_type;
  typedef typename iterator_traits<iterator_type>::difference_type difference_type;
  typedef iterator_type pointer;

  typedef typename iterator_traits<iterator_type>::reference __reference;
  typedef __conditional_t<is_reference<__reference>::value, __libcpp_remove_reference_t<__reference>&&, __reference>
      reference;

  _LIBCPP_HIDE_FROM_ABI explicit move_iterator(_Iter __i) : __current_(std::move(__i)) {}

  _LIBCPP_HIDE_FROM_ABI move_iterator& operator++() {
    ++__current_;
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI pointer operator->() const { return __current_; }

  _LIBCPP_HIDE_FROM_ABI move_iterator() : __current_() {}

  template <class _Up, __enable_if_t< !is_same<_Up, _Iter>::value && is_convertible<const _Up&, _Iter>::value, int> = 0>
  _LIBCPP_HIDE_FROM_ABI move_iterator(const move_iterator<_Up>& __u) : __current_(__u.base()) {}

  template <class _Up,
            __enable_if_t< !is_same<_Up, _Iter>::value && is_convertible<const _Up&, _Iter>::value &&
                               is_assignable<_Iter&, const _Up&>::value,
                           int> = 0>
  _LIBCPP_HIDE_FROM_ABI move_iterator& operator=(const move_iterator<_Up>& __u) {
    __current_ = __u.base();
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI _Iter base() const { return __current_; }

  _LIBCPP_HIDE_FROM_ABI reference operator*() const { return static_cast<reference>(*__current_); }
  _LIBCPP_HIDE_FROM_ABI reference operator[](difference_type __n) const {
    return static_cast<reference>(__current_[__n]);
  }

  _LIBCPP_HIDE_FROM_ABI move_iterator operator++(int) {
    move_iterator __tmp(*this);
    ++__current_;
    return __tmp;
  }

  _LIBCPP_HIDE_FROM_ABI move_iterator& operator--() {
    --__current_;
    return *this;
  }
  _LIBCPP_HIDE_FROM_ABI move_iterator operator--(int) {
    move_iterator __tmp(*this);
    --__current_;
    return __tmp;
  }
  _LIBCPP_HIDE_FROM_ABI move_iterator operator+(difference_type __n) const { return move_iterator(__current_ + __n); }
  _LIBCPP_HIDE_FROM_ABI move_iterator& operator+=(difference_type __n) {
    __current_ += __n;
    return *this;
  }
  _LIBCPP_HIDE_FROM_ABI move_iterator operator-(difference_type __n) const { return move_iterator(__current_ - __n); }
  _LIBCPP_HIDE_FROM_ABI move_iterator& operator-=(difference_type __n) {
    __current_ -= __n;
    return *this;
  }

private:
  template <class _It2>
  friend class move_iterator;

  _Iter __current_;
};
_LIBCPP_CTAD_SUPPORTED_FOR_TYPE(move_iterator);

template <class _Iter1, class _Iter2>
inline _LIBCPP_HIDE_FROM_ABI bool operator==(const move_iterator<_Iter1>& __x, const move_iterator<_Iter2>& __y) {
  return __x.base() == __y.base();
}

template <class _Iter1, class _Iter2>
inline _LIBCPP_HIDE_FROM_ABI bool operator!=(const move_iterator<_Iter1>& __x, const move_iterator<_Iter2>& __y) {
  return __x.base() != __y.base();
}

template <class _Iter1, class _Iter2>
inline _LIBCPP_HIDE_FROM_ABI bool operator<(const move_iterator<_Iter1>& __x, const move_iterator<_Iter2>& __y) {
  return __x.base() < __y.base();
}

template <class _Iter1, class _Iter2>
inline _LIBCPP_HIDE_FROM_ABI bool operator>(const move_iterator<_Iter1>& __x, const move_iterator<_Iter2>& __y) {
  return __x.base() > __y.base();
}

template <class _Iter1, class _Iter2>
inline _LIBCPP_HIDE_FROM_ABI bool operator<=(const move_iterator<_Iter1>& __x, const move_iterator<_Iter2>& __y) {
  return __x.base() <= __y.base();
}

template <class _Iter1, class _Iter2>
inline _LIBCPP_HIDE_FROM_ABI bool operator>=(const move_iterator<_Iter1>& __x, const move_iterator<_Iter2>& __y) {
  return __x.base() >= __y.base();
}

template <class _Iter1, class _Iter2>
inline _LIBCPP_HIDE_FROM_ABI typename move_iterator<_Iter1>::difference_type
operator-(const move_iterator<_Iter1>& __x, const move_iterator<_Iter2>& __y) {
  return __x.base() - __y.base();
}

template <class _Iter>
inline _LIBCPP_HIDE_FROM_ABI move_iterator<_Iter>
operator+(typename move_iterator<_Iter>::difference_type __n, const move_iterator<_Iter>& __x) {
  return move_iterator<_Iter>(__x.base() + __n);
}

template <class _Iter>
inline _LIBCPP_HIDE_FROM_ABI move_iterator<_Iter> make_move_iterator(_Iter __i) {
  return move_iterator<_Iter>(std::move(__i));
}

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___CXX03___ITERATOR_MOVE_ITERATOR_H
