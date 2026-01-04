//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CXX03___UTILITY_PAIR_H
#define _LIBCPP___CXX03___UTILITY_PAIR_H

#include <__cxx03/__config>
#include <__cxx03/__fwd/array.h>
#include <__cxx03/__fwd/pair.h>
#include <__cxx03/__fwd/tuple.h>
#include <__cxx03/__tuple/sfinae_helpers.h>
#include <__cxx03/__tuple/tuple_element.h>
#include <__cxx03/__tuple/tuple_indices.h>
#include <__cxx03/__tuple/tuple_like_no_subrange.h>
#include <__cxx03/__tuple/tuple_size.h>
#include <__cxx03/__type_traits/common_type.h>
#include <__cxx03/__type_traits/conditional.h>
#include <__cxx03/__type_traits/decay.h>
#include <__cxx03/__type_traits/integral_constant.h>
#include <__cxx03/__type_traits/is_assignable.h>
#include <__cxx03/__type_traits/is_constructible.h>
#include <__cxx03/__type_traits/is_convertible.h>
#include <__cxx03/__type_traits/is_implicitly_default_constructible.h>
#include <__cxx03/__type_traits/is_nothrow_assignable.h>
#include <__cxx03/__type_traits/is_nothrow_constructible.h>
#include <__cxx03/__type_traits/is_reference.h>
#include <__cxx03/__type_traits/is_same.h>
#include <__cxx03/__type_traits/is_swappable.h>
#include <__cxx03/__type_traits/is_trivially_relocatable.h>
#include <__cxx03/__type_traits/nat.h>
#include <__cxx03/__type_traits/remove_cvref.h>
#include <__cxx03/__type_traits/unwrap_ref.h>
#include <__cxx03/__utility/declval.h>
#include <__cxx03/__utility/forward.h>
#include <__cxx03/__utility/move.h>
#include <__cxx03/__utility/piecewise_construct.h>
#include <__cxx03/cstddef>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__cxx03/__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

template <class, class>
struct __non_trivially_copyable_base {
  _LIBCPP_HIDE_FROM_ABI __non_trivially_copyable_base() _NOEXCEPT {}
  _LIBCPP_HIDE_FROM_ABI __non_trivially_copyable_base(__non_trivially_copyable_base const&) _NOEXCEPT {}
};

template <class _T1, class _T2>
struct _LIBCPP_TEMPLATE_VIS pair
#if defined(_LIBCPP_DEPRECATED_ABI_DISABLE_PAIR_TRIVIAL_COPY_CTOR)
    : private __non_trivially_copyable_base<_T1, _T2>
#endif
{
  using first_type  = _T1;
  using second_type = _T2;

  _T1 first;
  _T2 second;

  using __trivially_relocatable =
      __conditional_t<__libcpp_is_trivially_relocatable<_T1>::value && __libcpp_is_trivially_relocatable<_T2>::value,
                      pair,
                      void>;

  _LIBCPP_HIDE_FROM_ABI pair(pair const&) = default;
  _LIBCPP_HIDE_FROM_ABI pair(pair&&)      = default;

  // When we are requested for pair to be trivially copyable by the ABI macro, we use defaulted members
  // if it is both legal to do it (i.e. no references) and we have a way to actually implement it, which requires
  // the __enable_if__ attribute before C++20.
#ifdef _LIBCPP_ABI_TRIVIALLY_COPYABLE_PAIR
  // FIXME: This should really just be a static constexpr variable. It's in a struct to avoid gdb printing the value
  // when printing a pair
  struct __has_defaulted_members {
    static const bool value = !is_reference<first_type>::value && !is_reference<second_type>::value;
  };
#  if __has_attribute(__enable_if__)
  _LIBCPP_HIDE_FROM_ABI pair& operator=(const pair&)
      __attribute__((__enable_if__(__has_defaulted_members::value, ""))) = default;

  _LIBCPP_HIDE_FROM_ABI pair& operator=(pair&&)
      __attribute__((__enable_if__(__has_defaulted_members::value, ""))) = default;
#  else
#    error "_LIBCPP_ABI_TRIVIALLY_COPYABLE_PAIR isn't supported with this compiler"
#  endif
#else
  struct __has_defaulted_members {
    static const bool value = false;
  };
#endif // defined(_LIBCPP_ABI_TRIVIALLY_COPYABLE_PAIR) && __has_attribute(__enable_if__)

  _LIBCPP_HIDE_FROM_ABI pair() : first(), second() {}

  _LIBCPP_HIDE_FROM_ABI pair(_T1 const& __t1, _T2 const& __t2) : first(__t1), second(__t2) {}

  template <class _U1, class _U2>
  _LIBCPP_HIDE_FROM_ABI pair(const pair<_U1, _U2>& __p) : first(__p.first), second(__p.second) {}

  _LIBCPP_HIDE_FROM_ABI pair& operator=(pair const& __p) {
    first  = __p.first;
    second = __p.second;
    return *this;
  }

  // Extension: This is provided in C++03 because it allows properly handling the
  //            assignment to a pair containing references, which would be a hard
  //            error otherwise.
  template <
      class _U1,
      class _U2,
      __enable_if_t<is_assignable<first_type&, _U1 const&>::value && is_assignable<second_type&, _U2 const&>::value,
                    int> = 0>
  _LIBCPP_HIDE_FROM_ABI pair& operator=(pair<_U1, _U2> const& __p) {
    first  = __p.first;
    second = __p.second;
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI void swap(pair& __p) {
    using std::swap;
    swap(first, __p.first);
    swap(second, __p.second);
  }
};

// [pairs.spec], specialized algorithms

template <class _T1, class _T2, class _U1, class _U2>
inline _LIBCPP_HIDE_FROM_ABI bool operator==(const pair<_T1, _T2>& __x, const pair<_U1, _U2>& __y) {
  return __x.first == __y.first && __x.second == __y.second;
}

template <class _T1, class _T2, class _U1, class _U2>
inline _LIBCPP_HIDE_FROM_ABI bool operator!=(const pair<_T1, _T2>& __x, const pair<_U1, _U2>& __y) {
  return !(__x == __y);
}

template <class _T1, class _T2, class _U1, class _U2>
inline _LIBCPP_HIDE_FROM_ABI bool operator<(const pair<_T1, _T2>& __x, const pair<_U1, _U2>& __y) {
  return __x.first < __y.first || (!(__y.first < __x.first) && __x.second < __y.second);
}

template <class _T1, class _T2, class _U1, class _U2>
inline _LIBCPP_HIDE_FROM_ABI bool operator>(const pair<_T1, _T2>& __x, const pair<_U1, _U2>& __y) {
  return __y < __x;
}

template <class _T1, class _T2, class _U1, class _U2>
inline _LIBCPP_HIDE_FROM_ABI bool operator>=(const pair<_T1, _T2>& __x, const pair<_U1, _U2>& __y) {
  return !(__x < __y);
}

template <class _T1, class _T2, class _U1, class _U2>
inline _LIBCPP_HIDE_FROM_ABI bool operator<=(const pair<_T1, _T2>& __x, const pair<_U1, _U2>& __y) {
  return !(__y < __x);
}

template <class _T1, class _T2, __enable_if_t<__is_swappable_v<_T1> && __is_swappable_v<_T2>, int> = 0>
inline _LIBCPP_HIDE_FROM_ABI void swap(pair<_T1, _T2>& __x, pair<_T1, _T2>& __y) {
  __x.swap(__y);
}

template <class _T1, class _T2>
inline _LIBCPP_HIDE_FROM_ABI pair<typename __unwrap_ref_decay<_T1>::type, typename __unwrap_ref_decay<_T2>::type>
make_pair(_T1&& __t1, _T2&& __t2) {
  return pair<typename __unwrap_ref_decay<_T1>::type, typename __unwrap_ref_decay<_T2>::type>(
      std::forward<_T1>(__t1), std::forward<_T2>(__t2));
}

template <class _T1, class _T2>
struct _LIBCPP_TEMPLATE_VIS tuple_size<pair<_T1, _T2> > : public integral_constant<size_t, 2> {};

template <size_t _Ip, class _T1, class _T2>
struct _LIBCPP_TEMPLATE_VIS tuple_element<_Ip, pair<_T1, _T2> > {
  static_assert(_Ip < 2, "Index out of bounds in std::tuple_element<std::pair<T1, T2>>");
};

template <class _T1, class _T2>
struct _LIBCPP_TEMPLATE_VIS tuple_element<0, pair<_T1, _T2> > {
  using type _LIBCPP_NODEBUG = _T1;
};

template <class _T1, class _T2>
struct _LIBCPP_TEMPLATE_VIS tuple_element<1, pair<_T1, _T2> > {
  using type _LIBCPP_NODEBUG = _T2;
};

template <size_t _Ip>
struct __get_pair;

template <>
struct __get_pair<0> {
  template <class _T1, class _T2>
  static _LIBCPP_HIDE_FROM_ABI _T1& get(pair<_T1, _T2>& __p) _NOEXCEPT {
    return __p.first;
  }

  template <class _T1, class _T2>
  static _LIBCPP_HIDE_FROM_ABI const _T1& get(const pair<_T1, _T2>& __p) _NOEXCEPT {
    return __p.first;
  }

  template <class _T1, class _T2>
  static _LIBCPP_HIDE_FROM_ABI _T1&& get(pair<_T1, _T2>&& __p) _NOEXCEPT {
    return std::forward<_T1>(__p.first);
  }

  template <class _T1, class _T2>
  static _LIBCPP_HIDE_FROM_ABI const _T1&& get(const pair<_T1, _T2>&& __p) _NOEXCEPT {
    return std::forward<const _T1>(__p.first);
  }
};

template <>
struct __get_pair<1> {
  template <class _T1, class _T2>
  static _LIBCPP_HIDE_FROM_ABI _T2& get(pair<_T1, _T2>& __p) _NOEXCEPT {
    return __p.second;
  }

  template <class _T1, class _T2>
  static _LIBCPP_HIDE_FROM_ABI const _T2& get(const pair<_T1, _T2>& __p) _NOEXCEPT {
    return __p.second;
  }

  template <class _T1, class _T2>
  static _LIBCPP_HIDE_FROM_ABI _T2&& get(pair<_T1, _T2>&& __p) _NOEXCEPT {
    return std::forward<_T2>(__p.second);
  }

  template <class _T1, class _T2>
  static _LIBCPP_HIDE_FROM_ABI const _T2&& get(const pair<_T1, _T2>&& __p) _NOEXCEPT {
    return std::forward<const _T2>(__p.second);
  }
};

template <size_t _Ip, class _T1, class _T2>
inline _LIBCPP_HIDE_FROM_ABI typename tuple_element<_Ip, pair<_T1, _T2> >::type& get(pair<_T1, _T2>& __p) _NOEXCEPT {
  return __get_pair<_Ip>::get(__p);
}

template <size_t _Ip, class _T1, class _T2>
inline _LIBCPP_HIDE_FROM_ABI const typename tuple_element<_Ip, pair<_T1, _T2> >::type&
get(const pair<_T1, _T2>& __p) _NOEXCEPT {
  return __get_pair<_Ip>::get(__p);
}

template <size_t _Ip, class _T1, class _T2>
inline _LIBCPP_HIDE_FROM_ABI typename tuple_element<_Ip, pair<_T1, _T2> >::type&& get(pair<_T1, _T2>&& __p) _NOEXCEPT {
  return __get_pair<_Ip>::get(std::move(__p));
}

template <size_t _Ip, class _T1, class _T2>
inline _LIBCPP_HIDE_FROM_ABI const typename tuple_element<_Ip, pair<_T1, _T2> >::type&&
get(const pair<_T1, _T2>&& __p) _NOEXCEPT {
  return __get_pair<_Ip>::get(std::move(__p));
}

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___CXX03___UTILITY_PAIR_H
