// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ITERATOR_WRAP_ITER_H
#define _LIBCPP___ITERATOR_WRAP_ITER_H

#include <__compare/ordering.h>
#include <__compare/three_way_comparable.h>
#include <__config>
#include <__iterator/iterator_traits.h>
#include <__memory/addressof.h>
#include <__memory/pointer_traits.h>
#include <__type_traits/conditional.h>
#include <__type_traits/enable_if.h>
#include <__type_traits/integral_constant.h>
#include <__type_traits/is_convertible.h>
#include <__type_traits/void_t.h>
#include <cstddef>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#ifdef _LIBCPP_ABI_WRAP_ITER_ADL_PROOF
template <class _MutableIter, class _ConstIter>
struct __wrap_iter_impl {
  template <bool _IsConst>
#else
template <class _Iter>
#endif
  class __wrap_iter {

  public:
#ifdef _LIBCPP_ABI_WRAP_ITER_ADL_PROOF
    using _Iter = __conditional_t<_IsConst, _ConstIter, _MutableIter>;
#endif
    typedef _Iter iterator_type;
    typedef typename iterator_traits<iterator_type>::value_type value_type;
    typedef typename iterator_traits<iterator_type>::difference_type difference_type;
    typedef typename iterator_traits<iterator_type>::pointer pointer;
    typedef typename iterator_traits<iterator_type>::reference reference;
    typedef typename iterator_traits<iterator_type>::iterator_category iterator_category;
#if _LIBCPP_STD_VER >= 20
  typedef contiguous_iterator_tag iterator_concept;
#endif

private:
  iterator_type __i_;

public:
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 __wrap_iter() _NOEXCEPT : __i_() {}
#ifdef _LIBCPP_ABI_WRAP_ITER_ADL_PROOF
  template <bool _B, __enable_if_t<!_B, int> = 0>
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 __wrap_iter(const __wrap_iter<_B>& __u) _NOEXCEPT
      : __i_(__u.base()) {}
#else
  template <class _Up, __enable_if_t<is_convertible<_Up, iterator_type>::value, int> = 0>
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 __wrap_iter(const __wrap_iter<_Up>& __u) _NOEXCEPT
      : __i_(__u.base()) {}
#endif
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 reference operator*() const _NOEXCEPT { return *__i_; }
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 pointer operator->() const _NOEXCEPT {
    return std::__to_address(__i_);
  }
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 __wrap_iter& operator++() _NOEXCEPT {
    ++__i_;
    return *this;
  }
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 __wrap_iter operator++(int) _NOEXCEPT {
    __wrap_iter __tmp(*this);
    ++(*this);
    return __tmp;
  }

  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 __wrap_iter& operator--() _NOEXCEPT {
    --__i_;
    return *this;
  }
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 __wrap_iter operator--(int) _NOEXCEPT {
    __wrap_iter __tmp(*this);
    --(*this);
    return __tmp;
  }
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 __wrap_iter operator+(difference_type __n) const _NOEXCEPT {
    __wrap_iter __w(*this);
    __w += __n;
    return __w;
  }
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 __wrap_iter& operator+=(difference_type __n) _NOEXCEPT {
    __i_ += __n;
    return *this;
  }
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 __wrap_iter operator-(difference_type __n) const _NOEXCEPT {
    return *this + (-__n);
  }
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 __wrap_iter& operator-=(difference_type __n) _NOEXCEPT {
    *this += -__n;
    return *this;
  }
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 reference operator[](difference_type __n) const _NOEXCEPT {
    return __i_[__n];
  }

  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 iterator_type base() const _NOEXCEPT { return __i_; }

private:
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 explicit __wrap_iter(iterator_type __x) _NOEXCEPT : __i_(__x) {}

#ifdef _LIBCPP_ABI_WRAP_ITER_ADL_PROOF
  using __is_wrap_iter = void;

  template <bool _OtherConst>
#else
  template <class _Up>
#endif
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

#ifdef _LIBCPP_ABI_WRAP_ITER_ADL_PROOF
#  define _LIBCPP_WRAP_ITER_SINGLE_TEMPLATE_HEAD                                                                       \
    template <bool _Iter1>                                                                                             \
    friend
#  define _LIBCPP_WRAP_ITER_PAIR_TEMPLATE_HEAD                                                                         \
    template <bool _Iter1, bool _Iter2>                                                                                \
    friend
#else
#  define _LIBCPP_WRAP_ITER_SINGLE_TEMPLATE_HEAD template <class _Iter1>
#  define _LIBCPP_WRAP_ITER_PAIR_TEMPLATE_HEAD template <class _Iter1, class _Iter2>
#endif

  _LIBCPP_WRAP_ITER_SINGLE_TEMPLATE_HEAD
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR bool
  operator==(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter1>& __y) _NOEXCEPT {
    return __x.base() == __y.base();
  }

  _LIBCPP_WRAP_ITER_PAIR_TEMPLATE_HEAD
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR bool
  operator==(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter2>& __y) _NOEXCEPT {
    return __x.base() == __y.base();
  }

  _LIBCPP_WRAP_ITER_SINGLE_TEMPLATE_HEAD
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 bool
  operator<(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter1>& __y) _NOEXCEPT {
    return __x.base() < __y.base();
  }

  _LIBCPP_WRAP_ITER_PAIR_TEMPLATE_HEAD
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 bool
  operator<(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter2>& __y) _NOEXCEPT {
    return __x.base() < __y.base();
  }

#if _LIBCPP_STD_VER <= 17
  _LIBCPP_WRAP_ITER_SINGLE_TEMPLATE_HEAD
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR bool
  operator!=(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter1>& __y) _NOEXCEPT {
    return !(__x == __y);
  }

  _LIBCPP_WRAP_ITER_PAIR_TEMPLATE_HEAD
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR bool
  operator!=(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter2>& __y) _NOEXCEPT {
    return !(__x == __y);
  }
  _LIBCPP_WRAP_ITER_SINGLE_TEMPLATE_HEAD
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR bool
  operator>(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter1>& __y) _NOEXCEPT {
    return __y < __x;
  }

  _LIBCPP_WRAP_ITER_PAIR_TEMPLATE_HEAD
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR bool
  operator>(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter2>& __y) _NOEXCEPT {
    return __y < __x;
  }

  _LIBCPP_WRAP_ITER_SINGLE_TEMPLATE_HEAD
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR bool
  operator>=(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter1>& __y) _NOEXCEPT {
    return !(__x < __y);
  }

  _LIBCPP_WRAP_ITER_PAIR_TEMPLATE_HEAD
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR bool
  operator>=(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter2>& __y) _NOEXCEPT {
    return !(__x < __y);
  }

  _LIBCPP_WRAP_ITER_SINGLE_TEMPLATE_HEAD
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR bool
  operator<=(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter1>& __y) _NOEXCEPT {
    return !(__y < __x);
  }

  _LIBCPP_WRAP_ITER_PAIR_TEMPLATE_HEAD
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR bool
  operator<=(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter2>& __y) _NOEXCEPT {
    return !(__y < __x);
  }

#else
_LIBCPP_WRAP_ITER_PAIR_TEMPLATE_HEAD
_LIBCPP_HIDE_FROM_ABI constexpr strong_ordering
operator<=>(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter2>& __y) noexcept {
  if constexpr (three_way_comparable_with<typename __wrap_iter<_Iter1>::iterator_type,
                                          typename __wrap_iter<_Iter2>::iterator_type,
                                          strong_ordering>) {
    return __x.base() <=> __y.base();
  } else {
    if (__x.base() < __y.base())
      return strong_ordering::less;

    if (__x.base() == __y.base())
      return strong_ordering::equal;

    return strong_ordering::greater;
  }
}
#endif // _LIBCPP_STD_VER >= 20

  _LIBCPP_WRAP_ITER_PAIR_TEMPLATE_HEAD
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14
#ifndef _LIBCPP_CXX03_LANG
  auto
  operator-(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter2>& __y) _NOEXCEPT->decltype(__x.base() - __y.base())
#else
typename __wrap_iter<_Iter1>::difference_type
operator-(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter2>& __y) _NOEXCEPT
#endif // C++03
  {
  return __x.base() - __y.base();
  }

  _LIBCPP_WRAP_ITER_SINGLE_TEMPLATE_HEAD
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 __wrap_iter<_Iter1>
  operator+(typename __wrap_iter<_Iter1>::difference_type __n, __wrap_iter<_Iter1> __x) _NOEXCEPT {
    __x += __n;
    return __x;
  }

#ifdef _LIBCPP_ABI_WRAP_ITER_ADL_PROOF
};

template <class _MutableIter, class _ConstIter>
using __wrap_mut_iter = typename __wrap_iter_impl<_MutableIter, _ConstIter>::template __wrap_iter<false>;
template <class _MutableIter, class _ConstIter>
using __wrap_const_iter = typename __wrap_iter_impl<_MutableIter, _ConstIter>::template __wrap_iter<true>;
#else
#  if _LIBCPP_STD_VER <= 17
template <class _It>
struct __libcpp_is_contiguous_iterator<__wrap_iter<_It> > : true_type {};
#  endif

template <class _It>
struct _LIBCPP_TEMPLATE_VIS pointer_traits<__wrap_iter<_It> > {
  typedef __wrap_iter<_It> pointer;
  typedef typename pointer_traits<_It>::element_type element_type;
  typedef typename pointer_traits<_It>::difference_type difference_type;

  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR static element_type* to_address(pointer __w) _NOEXCEPT {
    return std::__to_address(__w.base());
  }
};

template <class _MutableIter, class _ConstIter>
using __wrap_mut_iter = __wrap_iter<_MutableIter>;
template <class _MutableIter, class _ConstIter>
using __wrap_const_iter = __wrap_iter<_ConstIter>;
#endif

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ITERATOR_WRAP_ITER_H
