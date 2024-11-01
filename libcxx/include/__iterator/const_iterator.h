// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ITERATOR_CONST_ITERATOR_H
#define _LIBCPP___ITERATOR_CONST_ITERATOR_H

#include <__compare/three_way_comparable.h>
#include <__concepts/common_with.h>
#include <__concepts/constructible.h>
#include <__concepts/convertible_to.h>
#include <__concepts/different_from.h>
#include <__concepts/same_as.h>
#include <__concepts/semiregular.h>
#include <__concepts/totally_ordered.h>
#include <__iterator/concepts.h>
#include <__iterator/incrementable_traits.h>
#include <__iterator/iter_move.h>
#include <__iterator/iterator_traits.h>
#include <__memory/addressof.h>
#include <__memory/pointer_traits.h>
#include <__type_traits/common_reference.h>
#include <__type_traits/common_type.h>
#include <__type_traits/conditional.h>
#include <__type_traits/integral_constant.h>
#include <__type_traits/is_reference.h>
#include <__type_traits/is_specialization.h>
#include <__type_traits/remove_cvref.h>
#include <__utility/forward.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 23

template <indirectly_readable _Iter>
using iter_const_reference_t = common_reference_t<const iter_value_t<_Iter>&&, iter_reference_t<_Iter>>;

template <class _Iter>
concept __constant_iterator = input_iterator<_Iter> && same_as<iter_const_reference_t<_Iter>, iter_reference_t<_Iter>>;

template <input_iterator _Iter>
class basic_const_iterator;

template <input_iterator _Iter>
using const_iterator = conditional_t<__constant_iterator<_Iter>, _Iter, basic_const_iterator<_Iter>>;

// This doesn't use `conditional_t` to avoid instantiating const_iterator<_Sent> when _Sent is not an input_iterator.
template <class _Sent>
struct __const_sentinel_impl {
  using type = _Sent;
};
template <class _Sent>
  requires input_iterator<_Sent>
struct __const_sentinel_impl<_Sent> {
  using type = const_iterator<_Sent>;
};
template <semiregular _Sent>
using const_sentinel = __const_sentinel_impl<_Sent>::type;

template <class _Iter>
concept __not_a_const_iterator = !__is_specialization_v<_Iter, basic_const_iterator>;

template <indirectly_readable _Iter>
using __iter_const_rvalue_reference_t = common_reference_t<const iter_value_t<_Iter>&&, iter_rvalue_reference_t<_Iter>>;

template <class _Iter>
struct __basic_const_iterator_concept {
  // clang-format off
  using iterator_concept =
    conditional_t<contiguous_iterator<_Iter>,
      contiguous_iterator_tag,
    conditional_t<random_access_iterator<_Iter>,
      random_access_iterator_tag,
    conditional_t<bidirectional_iterator<_Iter>,
      bidirectional_iterator_tag,
    conditional_t<forward_iterator<_Iter>,
      forward_iterator_tag,
    // else
      input_iterator_tag>>>>;
  // clang-format on
};

template <class _Iter>
struct __basic_const_iterator_category : __basic_const_iterator_concept<_Iter> {};
template <forward_iterator _Iter>
struct __basic_const_iterator_category<_Iter> : __basic_const_iterator_concept<_Iter> {
  using iterator_category = std::iterator_traits<_Iter>::iterator_category;
};

template <input_iterator _Iter>
class _LIBCPP_TEMPLATE_VIS basic_const_iterator : public __basic_const_iterator_category<_Iter> {
  _Iter __current_ = _Iter();

  using __reference        = iter_const_reference_t<_Iter>;
  using __rvalue_reference = __iter_const_rvalue_reference_t<_Iter>;

public:
  using value_type      = iter_value_t<_Iter>;
  using difference_type = iter_difference_t<_Iter>;

  _LIBCPP_HIDE_FROM_ABI basic_const_iterator()
    requires default_initializable<_Iter>
  = default;

  _LIBCPP_HIDE_FROM_ABI constexpr basic_const_iterator(_Iter __cur) : __current_(std::move(__cur)) {}
  template <convertible_to<_Iter> _Type>
  _LIBCPP_HIDE_FROM_ABI constexpr basic_const_iterator(basic_const_iterator<_Type> __cur)
      : __current_(std::move(__cur.__current_)) {}
  template <__different_from<basic_const_iterator> _Type>
    requires convertible_to<_Type, _Iter>
  _LIBCPP_HIDE_FROM_ABI constexpr basic_const_iterator(_Type&& __cur) : __current_(std::forward<_Type>(__cur)) {}

  _LIBCPP_HIDE_FROM_ABI constexpr const _Iter& base() const& noexcept { return __current_; }
  _LIBCPP_HIDE_FROM_ABI constexpr _Iter base() && { return std::move(__current_); }

  _LIBCPP_HIDE_FROM_ABI constexpr __reference operator*() const { return static_cast<__reference>(*__current_); }
  _LIBCPP_HIDE_FROM_ABI constexpr const auto* operator->() const
    requires is_lvalue_reference_v<iter_reference_t<_Iter>> &&
             same_as<remove_cvref_t<iter_reference_t<_Iter>>, value_type>
  {
    if constexpr (contiguous_iterator<_Iter>) {
      return std::to_address(__current_);
    } else {
      return std::addressof(*__current_);
    }
  }

  _LIBCPP_HIDE_FROM_ABI constexpr basic_const_iterator& operator++() {
    ++__current_;
    return *this;
  }
  _LIBCPP_HIDE_FROM_ABI constexpr void operator++(int) { ++__current_; }
  _LIBCPP_HIDE_FROM_ABI constexpr basic_const_iterator operator++(int)
    requires forward_iterator<_Iter>
  {
    auto __tmp = *this;
    ++__current_;
    return __tmp;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr basic_const_iterator& operator--()
    requires bidirectional_iterator<_Iter>
  {
    --__current_;
    return *this;
  }
  _LIBCPP_HIDE_FROM_ABI constexpr basic_const_iterator operator--(int)
    requires bidirectional_iterator<_Iter>
  {
    auto __tmp = *this;
    --__current_;
    return __tmp;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr basic_const_iterator& operator+=(difference_type __n)
    requires random_access_iterator<_Iter>
  {
    __current_ += __n;
    return *this;
  }
  _LIBCPP_HIDE_FROM_ABI constexpr basic_const_iterator& operator-=(difference_type __n)
    requires random_access_iterator<_Iter>
  {
    __current_ -= __n;
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __reference operator[](difference_type __n) const
    requires random_access_iterator<_Iter>
  {
    return static_cast<__reference>(__current_[__n]);
  }

  template <sentinel_for<_Iter> _Sent>
  _LIBCPP_HIDE_FROM_ABI constexpr bool operator==(const _Sent& __sent) const {
    return __current_ == __sent;
  }

  template <__not_a_const_iterator _ConstIt>
    requires __constant_iterator<_ConstIt> && convertible_to<_Iter const&, _ConstIt>
  _LIBCPP_HIDE_FROM_ABI constexpr operator _ConstIt() const& {
    return __current_;
  }
  template <__not_a_const_iterator _ConstIt>
    requires __constant_iterator<_ConstIt> && convertible_to<_Iter, _ConstIt>
  _LIBCPP_HIDE_FROM_ABI constexpr operator _ConstIt() && {
    return std::move(__current_);
  }

  _LIBCPP_HIDE_FROM_ABI constexpr bool operator<(const basic_const_iterator& __rhs) const
    requires random_access_iterator<_Iter>
  {
    return __current_ < __rhs.__current_;
  }
  _LIBCPP_HIDE_FROM_ABI constexpr bool operator>(const basic_const_iterator& __rhs) const
    requires random_access_iterator<_Iter>
  {
    return __current_ > __rhs.__current_;
  }
  _LIBCPP_HIDE_FROM_ABI constexpr bool operator<=(const basic_const_iterator& __rhs) const
    requires random_access_iterator<_Iter>
  {
    return __current_ <= __rhs.__current_;
  }
  _LIBCPP_HIDE_FROM_ABI constexpr bool operator>=(const basic_const_iterator& __rhs) const
    requires random_access_iterator<_Iter>
  {
    return __current_ >= __rhs.__current_;
  }
  _LIBCPP_HIDE_FROM_ABI constexpr auto operator<=>(const basic_const_iterator& __rhs) const
    requires random_access_iterator<_Iter> && three_way_comparable<_Iter>
  {
    return __current_ <=> __rhs.__current_;
  }

  template <__different_from<basic_const_iterator> _Iter2>
  _LIBCPP_HIDE_FROM_ABI constexpr bool operator<(const _Iter2& __rhs) const
    requires random_access_iterator<_Iter> && totally_ordered_with<_Iter, _Iter2>
  {
    return __current_ < __rhs;
  }
  template <__different_from<basic_const_iterator> _Iter2>
  _LIBCPP_HIDE_FROM_ABI constexpr bool operator>(const _Iter2& __rhs) const
    requires random_access_iterator<_Iter> && totally_ordered_with<_Iter, _Iter2>
  {
    return __current_ > __rhs;
  }
  template <__different_from<basic_const_iterator> _Iter2>
  _LIBCPP_HIDE_FROM_ABI constexpr bool operator<=(const _Iter2& __rhs) const
    requires random_access_iterator<_Iter> && totally_ordered_with<_Iter, _Iter2>
  {
    return __current_ <= __rhs;
  }
  template <__different_from<basic_const_iterator> _Iter2>
  _LIBCPP_HIDE_FROM_ABI constexpr bool operator>=(const _Iter2& __rhs) const
    requires random_access_iterator<_Iter> && totally_ordered_with<_Iter, _Iter2>
  {
    return __current_ >= __rhs;
  }
  template <__different_from<basic_const_iterator> _Iter2>
  _LIBCPP_HIDE_FROM_ABI constexpr auto operator<=>(const _Iter2& __rhs) const
    requires random_access_iterator<_Iter> && totally_ordered_with<_Iter, _Iter2> &&
             three_way_comparable_with<_Iter, _Iter2>
  {
    return __current_ <=> __rhs;
  }

  template <__not_a_const_iterator _Iter2>
  friend _LIBCPP_HIDE_FROM_ABI constexpr bool operator<(const _Iter2& __lhs, const basic_const_iterator& __rhs)
    requires random_access_iterator<_Iter> && totally_ordered_with<_Iter, _Iter2>
  {
    return __lhs < __rhs.__current_;
  }
  template <__not_a_const_iterator _Iter2>
  friend _LIBCPP_HIDE_FROM_ABI constexpr bool operator>(const _Iter2& __lhs, const basic_const_iterator& __rhs)
    requires random_access_iterator<_Iter> && totally_ordered_with<_Iter, _Iter2>
  {
    return __lhs > __rhs.__current_;
  }
  template <__not_a_const_iterator _Iter2>
  friend _LIBCPP_HIDE_FROM_ABI constexpr bool operator<=(const _Iter2& __lhs, const basic_const_iterator& __rhs)
    requires random_access_iterator<_Iter> && totally_ordered_with<_Iter, _Iter2>
  {
    return __lhs <= __rhs.__current_;
  }
  template <__not_a_const_iterator _Iter2>
  friend _LIBCPP_HIDE_FROM_ABI constexpr bool operator>=(const _Iter2& __lhs, const basic_const_iterator& __rhs)
    requires random_access_iterator<_Iter> && totally_ordered_with<_Iter, _Iter2>
  {
    return __lhs >= __rhs.__current_;
  }

  friend _LIBCPP_HIDE_FROM_ABI constexpr basic_const_iterator
  operator+(const basic_const_iterator& __it, difference_type __n)
    requires random_access_iterator<_Iter>
  {
    return basic_const_iterator(__it.__current_ + __n);
  }
  friend _LIBCPP_HIDE_FROM_ABI constexpr basic_const_iterator
  operator+(difference_type __n, const basic_const_iterator& __it)
    requires random_access_iterator<_Iter>
  {
    return basic_const_iterator(__it.__current_ + __n);
  }

  friend _LIBCPP_HIDE_FROM_ABI constexpr basic_const_iterator
  operator-(const basic_const_iterator& __it, difference_type __n)
    requires random_access_iterator<_Iter>
  {
    return basic_const_iterator(__it.__current_ - __n);
  }
  template <sized_sentinel_for<_Iter> _Sent>
  _LIBCPP_HIDE_FROM_ABI constexpr difference_type operator-(const _Sent& __rhs) const {
    return __current_ - __rhs;
  }
  template <__not_a_const_iterator _Sent>
    requires sized_sentinel_for<_Sent, _Iter>
  friend _LIBCPP_HIDE_FROM_ABI constexpr difference_type
  operator-(const _Sent& __lhs, const basic_const_iterator& __rhs) {
    return __lhs - __rhs;
  }

  friend _LIBCPP_HIDE_FROM_ABI constexpr __rvalue_reference iter_move(const basic_const_iterator& __it) noexcept(
      noexcept(static_cast<__rvalue_reference>(ranges::iter_move(__it.__current_)))) {
    return static_cast<__rvalue_reference>(ranges::iter_move(__it.__current_));
  }
};

template <class _Type1, common_with<_Type1> _Type2>
  requires input_iterator<common_type_t<_Type1, _Type2>>
struct common_type<basic_const_iterator<_Type1>, _Type2> {
  using type = basic_const_iterator<common_type_t<_Type1, _Type2>>;
};
template <class _Type1, common_with<_Type1> _Type2>
  requires input_iterator<common_type_t<_Type1, _Type2>>
struct common_type<_Type2, basic_const_iterator<_Type1>> {
  using type = basic_const_iterator<common_type_t<_Type1, _Type2>>;
};
template <class _Type1, common_with<_Type1> _Type2>
  requires input_iterator<common_type_t<_Type1, _Type2>>
struct common_type<basic_const_iterator<_Type1>, basic_const_iterator<_Type2>> {
  using type = basic_const_iterator<common_type_t<_Type1, _Type2>>;
};

template <input_iterator _Iter>
_LIBCPP_HIDE_FROM_ABI constexpr const_iterator<_Iter> make_const_iterator(_Iter __it) {
  return __it;
}
template <semiregular _Sent>
_LIBCPP_HIDE_FROM_ABI constexpr const_sentinel<_Sent> make_const_sentinel(_Sent __sent) {
  return __sent;
}

#endif // _LIBCPP_STD_VER >= 23

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ITERATOR_CONST_ITERATOR_H
