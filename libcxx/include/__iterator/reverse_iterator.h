// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ITERATOR_REVERSE_ITERATOR_H
#define _LIBCPP___ITERATOR_REVERSE_ITERATOR_H

#include <__algorithm/unwrap_iter.h>
#include <__compare/compare_three_way_result.h>
#include <__compare/three_way_comparable.h>
#include <__concepts/convertible_to.h>
#include <__config>
#include <__iterator/concepts.h>
#include <__iterator/incrementable_traits.h>
#include <__iterator/iter_move.h>
#include <__iterator/iter_swap.h>
#include <__iterator/iterator.h>
#include <__iterator/iterator_traits.h>
#include <__iterator/prev.h>
#include <__iterator/readable_traits.h>
#include <__memory/addressof.h>
#include <__utility/move.h>
#include <type_traits>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

_LIBCPP_SUPPRESS_DEPRECATED_PUSH
template <class _Iter>
class _LIBCPP_TEMPLATE_VIS reverse_iterator
#if _LIBCPP_STD_VER <= 14 || !defined(_LIBCPP_ABI_NO_ITERATOR_BASES)
    : public iterator<typename iterator_traits<_Iter>::iterator_category,
                      typename iterator_traits<_Iter>::value_type,
                      typename iterator_traits<_Iter>::difference_type,
                      typename iterator_traits<_Iter>::pointer,
                      typename iterator_traits<_Iter>::reference>
#endif
{
_LIBCPP_SUPPRESS_DEPRECATED_POP
private:
#ifndef _LIBCPP_ABI_NO_ITERATOR_BASES
    _Iter __t; // no longer used as of LWG #2360, not removed due to ABI break
#endif

#if _LIBCPP_STD_VER > 17
    static_assert(__is_cpp17_bidirectional_iterator<_Iter>::value || bidirectional_iterator<_Iter>,
        "reverse_iterator<It> requires It to be a bidirectional iterator.");
#endif // _LIBCPP_STD_VER > 17

protected:
    _Iter current;
public:
    using iterator_type = _Iter;

    using iterator_category = _If<__is_cpp17_random_access_iterator<_Iter>::value,
                                  random_access_iterator_tag,
                                  typename iterator_traits<_Iter>::iterator_category>;
    using pointer = typename iterator_traits<_Iter>::pointer;
#if _LIBCPP_STD_VER > 17
    using iterator_concept = _If<__is_cpp17_random_access_iterator<_Iter>::value,
                                  random_access_iterator_tag,
                                  bidirectional_iterator_tag>;
    using value_type = iter_value_t<_Iter>;
    using difference_type = iter_difference_t<_Iter>;
    using reference = iter_reference_t<_Iter>;
#else
    using value_type = typename iterator_traits<_Iter>::value_type;
    using difference_type = typename iterator_traits<_Iter>::difference_type;
    using reference = typename iterator_traits<_Iter>::reference;
#endif

#ifndef _LIBCPP_ABI_NO_ITERATOR_BASES
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX14
    reverse_iterator() : __t(), current() {}

    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX14
    explicit reverse_iterator(_Iter __x) : __t(__x), current(__x) {}

    template <class _Up, class = __enable_if_t<
        !is_same<_Up, _Iter>::value && is_convertible<_Up const&, _Iter>::value
    > >
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX14
    reverse_iterator(const reverse_iterator<_Up>& __u)
        : __t(__u.base()), current(__u.base())
    { }

    template <class _Up, class = __enable_if_t<
        !is_same<_Up, _Iter>::value &&
        is_convertible<_Up const&, _Iter>::value &&
        is_assignable<_Iter&, _Up const&>::value
    > >
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX14
    reverse_iterator& operator=(const reverse_iterator<_Up>& __u) {
        __t = current = __u.base();
        return *this;
    }
#else
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX14
    reverse_iterator() : current() {}

    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX14
    explicit reverse_iterator(_Iter __x) : current(__x) {}

    template <class _Up, class = __enable_if_t<
        !is_same<_Up, _Iter>::value && is_convertible<_Up const&, _Iter>::value
    > >
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX14
    reverse_iterator(const reverse_iterator<_Up>& __u)
        : current(__u.base())
    { }

    template <class _Up, class = __enable_if_t<
        !is_same<_Up, _Iter>::value &&
        is_convertible<_Up const&, _Iter>::value &&
        is_assignable<_Iter&, _Up const&>::value
    > >
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX14
    reverse_iterator& operator=(const reverse_iterator<_Up>& __u) {
        current = __u.base();
        return *this;
    }
#endif
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX14
    _Iter base() const {return current;}
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX14
    reference operator*() const {_Iter __tmp = current; return *--__tmp;}

#if _LIBCPP_STD_VER > 17
    _LIBCPP_INLINE_VISIBILITY
    constexpr pointer operator->() const
      requires is_pointer_v<_Iter> || requires(const _Iter __i) { __i.operator->(); }
    {
      if constexpr (is_pointer_v<_Iter>) {
        return std::prev(current);
      } else {
        return std::prev(current).operator->();
      }
    }
#else
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX14
    pointer operator->() const {
      return std::addressof(operator*());
    }
#endif // _LIBCPP_STD_VER > 17

    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX14
    reverse_iterator& operator++() {--current; return *this;}
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX14
    reverse_iterator operator++(int) {reverse_iterator __tmp(*this); --current; return __tmp;}
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX14
    reverse_iterator& operator--() {++current; return *this;}
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX14
    reverse_iterator operator--(int) {reverse_iterator __tmp(*this); ++current; return __tmp;}
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX14
    reverse_iterator operator+(difference_type __n) const {return reverse_iterator(current - __n);}
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX14
    reverse_iterator& operator+=(difference_type __n) {current -= __n; return *this;}
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX14
    reverse_iterator operator-(difference_type __n) const {return reverse_iterator(current + __n);}
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX14
    reverse_iterator& operator-=(difference_type __n) {current += __n; return *this;}
    _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX14
    reference operator[](difference_type __n) const {return *(*this + __n);}

#if _LIBCPP_STD_VER > 17
    _LIBCPP_HIDE_FROM_ABI friend constexpr
    iter_rvalue_reference_t<_Iter> iter_move(const reverse_iterator& __i)
      noexcept(is_nothrow_copy_constructible_v<_Iter> &&
          noexcept(ranges::iter_move(--declval<_Iter&>()))) {
      auto __tmp = __i.base();
      return ranges::iter_move(--__tmp);
    }

    template <indirectly_swappable<_Iter> _Iter2>
    _LIBCPP_HIDE_FROM_ABI friend constexpr
    void iter_swap(const reverse_iterator& __x, const reverse_iterator<_Iter2>& __y)
      noexcept(is_nothrow_copy_constructible_v<_Iter> &&
          is_nothrow_copy_constructible_v<_Iter2> &&
          noexcept(ranges::iter_swap(--declval<_Iter&>(), --declval<_Iter2&>()))) {
      auto __xtmp = __x.base();
      auto __ytmp = __y.base();
      ranges::iter_swap(--__xtmp, --__ytmp);
    }
#endif // _LIBCPP_STD_VER > 17
};

template <class _Iter1, class _Iter2>
inline _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX14
bool
operator==(const reverse_iterator<_Iter1>& __x, const reverse_iterator<_Iter2>& __y)
#if _LIBCPP_STD_VER > 17
    requires requires {
      { __x.base() == __y.base() } -> convertible_to<bool>;
    }
#endif // _LIBCPP_STD_VER > 17
{
    return __x.base() == __y.base();
}

template <class _Iter1, class _Iter2>
inline _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX14
bool
operator<(const reverse_iterator<_Iter1>& __x, const reverse_iterator<_Iter2>& __y)
#if _LIBCPP_STD_VER > 17
    requires requires {
        { __x.base() > __y.base() } -> convertible_to<bool>;
      }
#endif // _LIBCPP_STD_VER > 17
{
    return __x.base() > __y.base();
}

template <class _Iter1, class _Iter2>
inline _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX14
bool
operator!=(const reverse_iterator<_Iter1>& __x, const reverse_iterator<_Iter2>& __y)
#if _LIBCPP_STD_VER > 17
    requires requires {
      { __x.base() != __y.base() } -> convertible_to<bool>;
    }
#endif // _LIBCPP_STD_VER > 17
{
    return __x.base() != __y.base();
}

template <class _Iter1, class _Iter2>
inline _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX14
bool
operator>(const reverse_iterator<_Iter1>& __x, const reverse_iterator<_Iter2>& __y)
#if _LIBCPP_STD_VER > 17
    requires requires {
        { __x.base() < __y.base() } -> convertible_to<bool>;
      }
#endif // _LIBCPP_STD_VER > 17
{
    return __x.base() < __y.base();
}

template <class _Iter1, class _Iter2>
inline _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX14
bool
operator>=(const reverse_iterator<_Iter1>& __x, const reverse_iterator<_Iter2>& __y)
#if _LIBCPP_STD_VER > 17
    requires requires {
        { __x.base() <= __y.base() } -> convertible_to<bool>;
      }
#endif // _LIBCPP_STD_VER > 17
{
    return __x.base() <= __y.base();
}

template <class _Iter1, class _Iter2>
inline _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX14
bool
operator<=(const reverse_iterator<_Iter1>& __x, const reverse_iterator<_Iter2>& __y)
#if _LIBCPP_STD_VER > 17
    requires requires {
        { __x.base() >= __y.base() } -> convertible_to<bool>;
      }
#endif // _LIBCPP_STD_VER > 17
{
    return __x.base() >= __y.base();
}

#if _LIBCPP_STD_VER > 17
template <class _Iter1, three_way_comparable_with<_Iter1> _Iter2>
_LIBCPP_HIDE_FROM_ABI constexpr
compare_three_way_result_t<_Iter1, _Iter2>
operator<=>(const reverse_iterator<_Iter1>& __x, const reverse_iterator<_Iter2>& __y)
{
    return __y.base() <=> __x.base();
}
#endif // _LIBCPP_STD_VER > 17

#ifndef _LIBCPP_CXX03_LANG
template <class _Iter1, class _Iter2>
inline _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX14
auto
operator-(const reverse_iterator<_Iter1>& __x, const reverse_iterator<_Iter2>& __y)
-> decltype(__y.base() - __x.base())
{
    return __y.base() - __x.base();
}
#else
template <class _Iter1, class _Iter2>
inline _LIBCPP_INLINE_VISIBILITY
typename reverse_iterator<_Iter1>::difference_type
operator-(const reverse_iterator<_Iter1>& __x, const reverse_iterator<_Iter2>& __y)
{
    return __y.base() - __x.base();
}
#endif

template <class _Iter>
inline _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX14
reverse_iterator<_Iter>
operator+(typename reverse_iterator<_Iter>::difference_type __n, const reverse_iterator<_Iter>& __x)
{
    return reverse_iterator<_Iter>(__x.base() - __n);
}

#if _LIBCPP_STD_VER > 17
template <class _Iter1, class _Iter2>
  requires (!sized_sentinel_for<_Iter1, _Iter2>)
inline constexpr bool disable_sized_sentinel_for<reverse_iterator<_Iter1>, reverse_iterator<_Iter2>> = true;
#endif // _LIBCPP_STD_VER > 17

#if _LIBCPP_STD_VER > 11
template <class _Iter>
inline _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX14
reverse_iterator<_Iter> make_reverse_iterator(_Iter __i)
{
    return reverse_iterator<_Iter>(__i);
}
#endif

template <class _Iter>
using _ReverseWrapper = reverse_iterator<reverse_iterator<_Iter> >;

template <class _Iter, bool __b>
struct __unwrap_iter_impl<_ReverseWrapper<_Iter>, __b> {
  static _LIBCPP_CONSTEXPR decltype(std::__unwrap_iter(std::declval<_Iter>()))
  __apply(_ReverseWrapper<_Iter> __i) _NOEXCEPT {
    return std::__unwrap_iter(__i.base().base());
  }
};

template <class _OrigIter, class _UnwrappedIter>
struct __rewrap_iter_impl<_ReverseWrapper<_OrigIter>, _UnwrappedIter> {
  template <class _Iter>
  struct _ReverseWrapperCount {
    static _LIBCPP_CONSTEXPR const size_t value = 1;
  };

  template <class _Iter>
  struct _ReverseWrapperCount<_ReverseWrapper<_Iter> > {
    static _LIBCPP_CONSTEXPR const size_t value = 1 + _ReverseWrapperCount<_Iter>::value;
  };

  template <size_t _RewrapCount, class _OIter, class _UIter, __enable_if_t<_RewrapCount != 0, int> = 0>
  _LIBCPP_HIDE_FROM_ABI static _LIBCPP_CONSTEXPR _ReverseWrapper<_OIter> __rewrap(_ReverseWrapper<_OIter> __iter1,
                                                                                  _UIter __iter2) {
    return _ReverseWrapper<_OIter>(
        reverse_iterator<_OIter>(__rewrap<_RewrapCount - 1>(__iter1.base().base(), __iter2)));
  }

  template <size_t _RewrapCount, class _OIter, class _UIter, __enable_if_t<_RewrapCount == 0, int> = 0>
  _LIBCPP_HIDE_FROM_ABI static _LIBCPP_CONSTEXPR decltype(std::__rewrap_iter(std::declval<_OIter>(),
                                                                             std::declval<_UIter>()))
  __rewrap(_OIter __iter1, _UIter __iter2) {
    return std::__rewrap_iter(__iter1, __iter2);
  }

  _LIBCPP_HIDE_FROM_ABI static _LIBCPP_CONSTEXPR _ReverseWrapper<_OrigIter> __apply(_ReverseWrapper<_OrigIter> __iter1,
                                                                                    _UnwrappedIter __iter2) {
    return __rewrap<_ReverseWrapperCount<_OrigIter>::value>(__iter1, __iter2);
  }
};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ITERATOR_REVERSE_ITERATOR_H
