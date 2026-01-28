// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CXX03___ITERATOR_REVERSE_ITERATOR_H
#define _LIBCPP___CXX03___ITERATOR_REVERSE_ITERATOR_H

#include <__cxx03/__algorithm/unwrap_iter.h>
#include <__cxx03/__config>
#include <__cxx03/__iterator/advance.h>
#include <__cxx03/__iterator/iterator.h>
#include <__cxx03/__iterator/iterator_traits.h>
#include <__cxx03/__iterator/next.h>
#include <__cxx03/__iterator/prev.h>
#include <__cxx03/__iterator/segmented_iterator.h>
#include <__cxx03/__memory/addressof.h>
#include <__cxx03/__type_traits/conditional.h>
#include <__cxx03/__type_traits/enable_if.h>
#include <__cxx03/__type_traits/is_assignable.h>
#include <__cxx03/__type_traits/is_convertible.h>
#include <__cxx03/__type_traits/is_nothrow_constructible.h>
#include <__cxx03/__type_traits/is_pointer.h>
#include <__cxx03/__type_traits/is_same.h>
#include <__cxx03/__utility/declval.h>
#include <__cxx03/__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

_LIBCPP_SUPPRESS_DEPRECATED_PUSH
template <class _Iter>
class _LIBCPP_TEMPLATE_VIS reverse_iterator
    : public iterator<typename iterator_traits<_Iter>::iterator_category,
                      typename iterator_traits<_Iter>::value_type,
                      typename iterator_traits<_Iter>::difference_type,
                      typename iterator_traits<_Iter>::pointer,
                      typename iterator_traits<_Iter>::reference> {
  _LIBCPP_SUPPRESS_DEPRECATED_POP

private:
#ifndef _LIBCPP_ABI_NO_ITERATOR_BASES
  _Iter __t_; // no longer used as of LWG #2360, not removed due to ABI break
#endif

protected:
  _Iter current;

public:
  using iterator_type = _Iter;

  using iterator_category =
      _If<__has_random_access_iterator_category<_Iter>::value,
          random_access_iterator_tag,
          typename iterator_traits<_Iter>::iterator_category>;
  using pointer         = typename iterator_traits<_Iter>::pointer;
  using value_type      = typename iterator_traits<_Iter>::value_type;
  using difference_type = typename iterator_traits<_Iter>::difference_type;
  using reference       = typename iterator_traits<_Iter>::reference;

#ifndef _LIBCPP_ABI_NO_ITERATOR_BASES
  _LIBCPP_HIDE_FROM_ABI reverse_iterator() : __t_(), current() {}

  _LIBCPP_HIDE_FROM_ABI explicit reverse_iterator(_Iter __x) : __t_(__x), current(__x) {}

  template <class _Up, __enable_if_t<!is_same<_Up, _Iter>::value && is_convertible<_Up const&, _Iter>::value, int> = 0>
  _LIBCPP_HIDE_FROM_ABI reverse_iterator(const reverse_iterator<_Up>& __u) : __t_(__u.base()), current(__u.base()) {}

  template <class _Up,
            __enable_if_t<!is_same<_Up, _Iter>::value && is_convertible<_Up const&, _Iter>::value &&
                              is_assignable<_Iter&, _Up const&>::value,
                          int> = 0>
  _LIBCPP_HIDE_FROM_ABI reverse_iterator& operator=(const reverse_iterator<_Up>& __u) {
    __t_ = current = __u.base();
    return *this;
  }
#else
  _LIBCPP_HIDE_FROM_ABI reverse_iterator() : current() {}

  _LIBCPP_HIDE_FROM_ABI explicit reverse_iterator(_Iter __x) : current(__x) {}

  template <class _Up, __enable_if_t<!is_same<_Up, _Iter>::value && is_convertible<_Up const&, _Iter>::value, int> = 0>
  _LIBCPP_HIDE_FROM_ABI reverse_iterator(const reverse_iterator<_Up>& __u) : current(__u.base()) {}

  template <class _Up,
            __enable_if_t<!is_same<_Up, _Iter>::value && is_convertible<_Up const&, _Iter>::value &&
                              is_assignable<_Iter&, _Up const&>::value,
                          int> = 0>
  _LIBCPP_HIDE_FROM_ABI reverse_iterator& operator=(const reverse_iterator<_Up>& __u) {
    current = __u.base();
    return *this;
  }
#endif
  _LIBCPP_HIDE_FROM_ABI _Iter base() const { return current; }
  _LIBCPP_HIDE_FROM_ABI reference operator*() const {
    _Iter __tmp = current;
    return *--__tmp;
  }

  _LIBCPP_HIDE_FROM_ABI pointer operator->() const { return std::addressof(operator*()); }

  _LIBCPP_HIDE_FROM_ABI reverse_iterator& operator++() {
    --current;
    return *this;
  }
  _LIBCPP_HIDE_FROM_ABI reverse_iterator operator++(int) {
    reverse_iterator __tmp(*this);
    --current;
    return __tmp;
  }
  _LIBCPP_HIDE_FROM_ABI reverse_iterator& operator--() {
    ++current;
    return *this;
  }
  _LIBCPP_HIDE_FROM_ABI reverse_iterator operator--(int) {
    reverse_iterator __tmp(*this);
    ++current;
    return __tmp;
  }
  _LIBCPP_HIDE_FROM_ABI reverse_iterator operator+(difference_type __n) const {
    return reverse_iterator(current - __n);
  }
  _LIBCPP_HIDE_FROM_ABI reverse_iterator& operator+=(difference_type __n) {
    current -= __n;
    return *this;
  }
  _LIBCPP_HIDE_FROM_ABI reverse_iterator operator-(difference_type __n) const {
    return reverse_iterator(current + __n);
  }
  _LIBCPP_HIDE_FROM_ABI reverse_iterator& operator-=(difference_type __n) {
    current += __n;
    return *this;
  }
  _LIBCPP_HIDE_FROM_ABI reference operator[](difference_type __n) const { return *(*this + __n); }
};

template <class _Iter1, class _Iter2>
inline _LIBCPP_HIDE_FROM_ABI bool operator==(const reverse_iterator<_Iter1>& __x, const reverse_iterator<_Iter2>& __y) {
  return __x.base() == __y.base();
}

template <class _Iter1, class _Iter2>
inline _LIBCPP_HIDE_FROM_ABI bool operator<(const reverse_iterator<_Iter1>& __x, const reverse_iterator<_Iter2>& __y) {
  return __x.base() > __y.base();
}

template <class _Iter1, class _Iter2>
inline _LIBCPP_HIDE_FROM_ABI bool operator!=(const reverse_iterator<_Iter1>& __x, const reverse_iterator<_Iter2>& __y) {
  return __x.base() != __y.base();
}

template <class _Iter1, class _Iter2>
inline _LIBCPP_HIDE_FROM_ABI bool operator>(const reverse_iterator<_Iter1>& __x, const reverse_iterator<_Iter2>& __y) {
  return __x.base() < __y.base();
}

template <class _Iter1, class _Iter2>
inline _LIBCPP_HIDE_FROM_ABI bool operator>=(const reverse_iterator<_Iter1>& __x, const reverse_iterator<_Iter2>& __y) {
  return __x.base() <= __y.base();
}

template <class _Iter1, class _Iter2>
inline _LIBCPP_HIDE_FROM_ABI bool operator<=(const reverse_iterator<_Iter1>& __x, const reverse_iterator<_Iter2>& __y) {
  return __x.base() >= __y.base();
}

template <class _Iter1, class _Iter2>
inline _LIBCPP_HIDE_FROM_ABI typename reverse_iterator<_Iter1>::difference_type
operator-(const reverse_iterator<_Iter1>& __x, const reverse_iterator<_Iter2>& __y) {
  return __y.base() - __x.base();
}

template <class _Iter>
inline _LIBCPP_HIDE_FROM_ABI reverse_iterator<_Iter>
operator+(typename reverse_iterator<_Iter>::difference_type __n, const reverse_iterator<_Iter>& __x) {
  return reverse_iterator<_Iter>(__x.base() - __n);
}

template <class _Iter, bool __b>
struct __unwrap_iter_impl<reverse_iterator<reverse_iterator<_Iter> >, __b> {
  using _UnwrappedIter  = decltype(__unwrap_iter_impl<_Iter>::__unwrap(std::declval<_Iter>()));
  using _ReverseWrapper = reverse_iterator<reverse_iterator<_Iter> >;

  static _LIBCPP_HIDE_FROM_ABI _ReverseWrapper __rewrap(_ReverseWrapper __orig_iter, _UnwrappedIter __unwrapped_iter) {
    return _ReverseWrapper(
        reverse_iterator<_Iter>(__unwrap_iter_impl<_Iter>::__rewrap(__orig_iter.base().base(), __unwrapped_iter)));
  }

  static _LIBCPP_HIDE_FROM_ABI _UnwrappedIter __unwrap(_ReverseWrapper __i) _NOEXCEPT {
    return __unwrap_iter_impl<_Iter>::__unwrap(__i.base().base());
  }
};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___CXX03___ITERATOR_REVERSE_ITERATOR_H
