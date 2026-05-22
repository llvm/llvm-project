//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_OPTIONAL_COMPARISON_H
#define _LIBCPP_OPTIONAL_COMPARISON_H

#include <__compare/compare_three_way_result.h>
#include <__compare/ordering.h>
#include <__compare/three_way_comparable.h>
#include <__config>
#include <__fwd/optional.h>
#include <__optional/nullopt_t.h>
#include <__type_traits/enable_if.h>
#include <__type_traits/is_constructible.h>
#include <__type_traits/is_core_convertible.h>
#include <__type_traits/is_swappable.h>
#include <__utility/declval.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD

template <
    class _Tp,
    class _Up,
    enable_if_t<__is_core_convertible_v<decltype(std::declval<const _Tp&>() == std::declval<const _Up&>()), bool>,
                int> = 0>
_LIBCPP_HIDE_FROM_ABI constexpr bool operator==(const optional<_Tp>& __x, const optional<_Up>& __y) {
  if (static_cast<bool>(__x) != static_cast<bool>(__y))
    return false;
  if (!static_cast<bool>(__x))
    return true;
  return *__x == *__y;
}

template <
    class _Tp,
    class _Up,
    enable_if_t<__is_core_convertible_v<decltype(std::declval<const _Tp&>() != std::declval<const _Up&>()), bool>,
                int> = 0>
_LIBCPP_HIDE_FROM_ABI constexpr bool operator!=(const optional<_Tp>& __x, const optional<_Up>& __y) {
  if (static_cast<bool>(__x) != static_cast<bool>(__y))
    return true;
  if (!static_cast<bool>(__x))
    return false;
  return *__x != *__y;
}

template < class _Tp,
           class _Up,
           enable_if_t<__is_core_convertible_v<decltype(std::declval<const _Tp&>() < std::declval<const _Up&>()), bool>,
                       int> = 0>
_LIBCPP_HIDE_FROM_ABI constexpr bool operator<(const optional<_Tp>& __x, const optional<_Up>& __y) {
  if (!static_cast<bool>(__y))
    return false;
  if (!static_cast<bool>(__x))
    return true;
  return *__x < *__y;
}

template < class _Tp,
           class _Up,
           enable_if_t<__is_core_convertible_v<decltype(std::declval<const _Tp&>() > std::declval<const _Up&>()), bool>,
                       int> = 0>
_LIBCPP_HIDE_FROM_ABI constexpr bool operator>(const optional<_Tp>& __x, const optional<_Up>& __y) {
  if (!static_cast<bool>(__x))
    return false;
  if (!static_cast<bool>(__y))
    return true;
  return *__x > *__y;
}

template <
    class _Tp,
    class _Up,
    enable_if_t<__is_core_convertible_v<decltype(std::declval<const _Tp&>() <= std::declval<const _Up&>()), bool>,
                int> = 0>
_LIBCPP_HIDE_FROM_ABI constexpr bool operator<=(const optional<_Tp>& __x, const optional<_Up>& __y) {
  if (!static_cast<bool>(__x))
    return true;
  if (!static_cast<bool>(__y))
    return false;
  return *__x <= *__y;
}

template <
    class _Tp,
    class _Up,
    enable_if_t<__is_core_convertible_v<decltype(std::declval<const _Tp&>() >= std::declval<const _Up&>()), bool>,
                int> = 0>
_LIBCPP_HIDE_FROM_ABI constexpr bool operator>=(const optional<_Tp>& __x, const optional<_Up>& __y) {
  if (!static_cast<bool>(__y))
    return true;
  if (!static_cast<bool>(__x))
    return false;
  return *__x >= *__y;
}

#  if _LIBCPP_STD_VER >= 20

template <class _Tp, three_way_comparable_with<_Tp> _Up>
_LIBCPP_HIDE_FROM_ABI constexpr compare_three_way_result_t<_Tp, _Up>
operator<=>(const optional<_Tp>& __x, const optional<_Up>& __y) {
  if (__x && __y)
    return *__x <=> *__y;
  return __x.has_value() <=> __y.has_value();
}

#  endif // _LIBCPP_STD_VER >= 20

// [optional.nullops] Comparison with nullopt

template <class _Tp>
_LIBCPP_HIDE_FROM_ABI constexpr bool operator==(const optional<_Tp>& __x, std::nullopt_t) noexcept {
  return !static_cast<bool>(__x);
}

#  if _LIBCPP_STD_VER <= 17

template <class _Tp>
_LIBCPP_HIDE_FROM_ABI constexpr bool operator==(nullopt_t, const optional<_Tp>& __x) noexcept {
  return !static_cast<bool>(__x);
}

template <class _Tp>
_LIBCPP_HIDE_FROM_ABI constexpr bool operator!=(const optional<_Tp>& __x, nullopt_t) noexcept {
  return static_cast<bool>(__x);
}

template <class _Tp>
_LIBCPP_HIDE_FROM_ABI constexpr bool operator!=(nullopt_t, const optional<_Tp>& __x) noexcept {
  return static_cast<bool>(__x);
}

template <class _Tp>
_LIBCPP_HIDE_FROM_ABI constexpr bool operator<(const optional<_Tp>&, nullopt_t) noexcept {
  return false;
}

template <class _Tp>
_LIBCPP_HIDE_FROM_ABI constexpr bool operator<(nullopt_t, const optional<_Tp>& __x) noexcept {
  return static_cast<bool>(__x);
}

template <class _Tp>
_LIBCPP_HIDE_FROM_ABI constexpr bool operator<=(const optional<_Tp>& __x, nullopt_t) noexcept {
  return !static_cast<bool>(__x);
}

template <class _Tp>
_LIBCPP_HIDE_FROM_ABI constexpr bool operator<=(nullopt_t, const optional<_Tp>&) noexcept {
  return true;
}

template <class _Tp>
_LIBCPP_HIDE_FROM_ABI constexpr bool operator>(const optional<_Tp>& __x, nullopt_t) noexcept {
  return static_cast<bool>(__x);
}

template <class _Tp>
_LIBCPP_HIDE_FROM_ABI constexpr bool operator>(nullopt_t, const optional<_Tp>&) noexcept {
  return false;
}

template <class _Tp>
_LIBCPP_HIDE_FROM_ABI constexpr bool operator>=(const optional<_Tp>&, nullopt_t) noexcept {
  return true;
}

template <class _Tp>
_LIBCPP_HIDE_FROM_ABI constexpr bool operator>=(nullopt_t, const optional<_Tp>& __x) noexcept {
  return !static_cast<bool>(__x);
}

#  else // _LIBCPP_STD_VER <= 17

template <class _Tp>
_LIBCPP_HIDE_FROM_ABI constexpr strong_ordering operator<=>(const optional<_Tp>& __x, nullopt_t) noexcept {
  return __x.has_value() <=> false;
}

#  endif // _LIBCPP_STD_VER <= 17

// [optional.comp.with.t] Comparison with T

template <
    class _Tp,
    class _Up,
    enable_if_t<__is_core_convertible_v<decltype(std::declval<const _Tp&>() == std::declval<const _Up&>()), bool>,
                int> = 0>
_LIBCPP_HIDE_FROM_ABI constexpr bool operator==(const optional<_Tp>& __x, const _Up& __v) {
  if (__x.has_value())
    return *__x == __v;
  return false;
}

template <
    class _Tp,
    class _Up,
    enable_if_t<__is_core_convertible_v<decltype(std::declval<const _Tp&>() == std::declval<const _Up&>()), bool>,
                int> = 0>
_LIBCPP_HIDE_FROM_ABI constexpr bool operator==(const _Tp& __v, const optional<_Up>& __x) {
  if (__x.has_value())
    return __v == *__x;
  return false;
}

template <
    class _Tp,
    class _Up,
    enable_if_t<__is_core_convertible_v<decltype(std::declval<const _Tp&>() != std::declval<const _Up&>()), bool>,
                int> = 0>
_LIBCPP_HIDE_FROM_ABI constexpr bool operator!=(const optional<_Tp>& __x, const _Up& __v) {
  if (__x.has_value())
    return *__x != __v;
  return true;
}

template <
    class _Tp,
    class _Up,
    enable_if_t<__is_core_convertible_v<decltype(std::declval<const _Tp&>() != std::declval<const _Up&>()), bool>,
                int> = 0>
_LIBCPP_HIDE_FROM_ABI constexpr bool operator!=(const _Tp& __v, const optional<_Up>& __x) {
  if (__x.has_value())
    return __v != *__x;
  return true;
}

template < class _Tp,
           class _Up,
           enable_if_t<__is_core_convertible_v<decltype(std::declval<const _Tp&>() < std::declval<const _Up&>()), bool>,
                       int> = 0>
_LIBCPP_HIDE_FROM_ABI constexpr bool operator<(const optional<_Tp>& __x, const _Up& __v) {
  if (__x.has_value())
    return *__x < __v;
  return true;
}

template < class _Tp,
           class _Up,
           enable_if_t<__is_core_convertible_v<decltype(std::declval<const _Tp&>() < std::declval<const _Up&>()), bool>,
                       int> = 0>
_LIBCPP_HIDE_FROM_ABI constexpr bool operator<(const _Tp& __v, const optional<_Up>& __x) {
  if (__x.has_value())
    return __v < *__x;
  return false;
}

template <
    class _Tp,
    class _Up,
    enable_if_t<__is_core_convertible_v<decltype(std::declval<const _Tp&>() <= std::declval<const _Up&>()), bool>,
                int> = 0>
_LIBCPP_HIDE_FROM_ABI constexpr bool operator<=(const optional<_Tp>& __x, const _Up& __v) {
  if (__x.has_value())
    return *__x <= __v;
  return true;
}

template <
    class _Tp,
    class _Up,
    enable_if_t<__is_core_convertible_v<decltype(std::declval<const _Tp&>() <= std::declval<const _Up&>()), bool>,
                int> = 0>
_LIBCPP_HIDE_FROM_ABI constexpr bool operator<=(const _Tp& __v, const optional<_Up>& __x) {
  if (__x.has_value())
    return __v <= *__x;
  return false;
}

template < class _Tp,
           class _Up,
           enable_if_t<__is_core_convertible_v<decltype(std::declval<const _Tp&>() > std::declval<const _Up&>()), bool>,
                       int> = 0>
_LIBCPP_HIDE_FROM_ABI constexpr bool operator>(const optional<_Tp>& __x, const _Up& __v) {
  if (__x.has_value())
    return *__x > __v;
  return false;
}

template < class _Tp,
           class _Up,
           enable_if_t<__is_core_convertible_v<decltype(std::declval<const _Tp&>() > std::declval<const _Up&>()), bool>,
                       int> = 0>
_LIBCPP_HIDE_FROM_ABI constexpr bool operator>(const _Tp& __v, const optional<_Up>& __x) {
  if (__x.has_value())
    return __v > *__x;
  return true;
}

template <
    class _Tp,
    class _Up,
    enable_if_t<__is_core_convertible_v<decltype(std::declval<const _Tp&>() >= std::declval<const _Up&>()), bool>,
                int> = 0>
_LIBCPP_HIDE_FROM_ABI constexpr bool operator>=(const optional<_Tp>& __x, const _Up& __v) {
  if (__x.has_value())
    return *__x >= __v;
  return false;
}

template <
    class _Tp,
    class _Up,
    enable_if_t<__is_core_convertible_v<decltype(std::declval<const _Tp&>() >= std::declval<const _Up&>()), bool>,
                int> = 0>
_LIBCPP_HIDE_FROM_ABI constexpr bool operator>=(const _Tp& __v, const optional<_Up>& __x) {
  if (__x.has_value())
    return __v >= *__x;
  return true;
}

#  if _LIBCPP_STD_VER >= 20

template <class _Tp>
concept __is_derived_from_optional = requires(const _Tp& __t) { []<class _Up>(const optional<_Up>&) {}(__t); };

template <class _Tp, class _Up>
  requires(!__is_derived_from_optional<_Up>) && three_way_comparable_with<_Tp, _Up>
_LIBCPP_HIDE_FROM_ABI constexpr compare_three_way_result_t<_Tp, _Up>
operator<=>(const optional<_Tp>& __x, const _Up& __v) {
  return __x.has_value() ? *__x <=> __v : strong_ordering::less;
}

#  endif // _LIBCPP_STD_VER >= 20

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 17

_LIBCPP_POP_MACROS

#endif
