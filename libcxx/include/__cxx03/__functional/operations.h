// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CXX03___FUNCTIONAL_OPERATIONS_H
#define _LIBCPP___CXX03___FUNCTIONAL_OPERATIONS_H

#include <__cxx03/__config>
#include <__cxx03/__functional/binary_function.h>
#include <__cxx03/__functional/unary_function.h>
#include <__cxx03/__type_traits/desugars_to.h>
#include <__cxx03/__utility/forward.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

// Arithmetic operations

template <class _Tp>
struct _LIBCPP_TEMPLATE_VIS plus : __binary_function<_Tp, _Tp, _Tp> {
  typedef _Tp __result_type; // used by valarray
  _LIBCPP_HIDE_FROM_ABI _Tp operator()(const _Tp& __x, const _Tp& __y) const { return __x + __y; }
};
_LIBCPP_CTAD_SUPPORTED_FOR_TYPE(plus);

// The non-transparent std::plus specialization is only equivalent to a raw plus
// operator when we don't perform an implicit conversion when calling it.
template <class _Tp>
inline const bool __desugars_to_v<__plus_tag, plus<_Tp>, _Tp, _Tp> = true;

template <class _Tp, class _Up>
inline const bool __desugars_to_v<__plus_tag, plus<void>, _Tp, _Up> = true;

template <class _Tp>
struct _LIBCPP_TEMPLATE_VIS minus : __binary_function<_Tp, _Tp, _Tp> {
  typedef _Tp __result_type; // used by valarray
  _LIBCPP_HIDE_FROM_ABI _Tp operator()(const _Tp& __x, const _Tp& __y) const { return __x - __y; }
};
_LIBCPP_CTAD_SUPPORTED_FOR_TYPE(minus);

template <class _Tp>
struct _LIBCPP_TEMPLATE_VIS multiplies : __binary_function<_Tp, _Tp, _Tp> {
  typedef _Tp __result_type; // used by valarray
  _LIBCPP_HIDE_FROM_ABI _Tp operator()(const _Tp& __x, const _Tp& __y) const { return __x * __y; }
};
_LIBCPP_CTAD_SUPPORTED_FOR_TYPE(multiplies);

template <class _Tp>
struct _LIBCPP_TEMPLATE_VIS divides : __binary_function<_Tp, _Tp, _Tp> {
  typedef _Tp __result_type; // used by valarray
  _LIBCPP_HIDE_FROM_ABI _Tp operator()(const _Tp& __x, const _Tp& __y) const { return __x / __y; }
};
_LIBCPP_CTAD_SUPPORTED_FOR_TYPE(divides);

template <class _Tp>
struct _LIBCPP_TEMPLATE_VIS modulus : __binary_function<_Tp, _Tp, _Tp> {
  typedef _Tp __result_type; // used by valarray
  _LIBCPP_HIDE_FROM_ABI _Tp operator()(const _Tp& __x, const _Tp& __y) const { return __x % __y; }
};
_LIBCPP_CTAD_SUPPORTED_FOR_TYPE(modulus);

template <class _Tp = void>
struct _LIBCPP_TEMPLATE_VIS negate : __unary_function<_Tp, _Tp> {
  typedef _Tp __result_type; // used by valarray
  _LIBCPP_HIDE_FROM_ABI _Tp operator()(const _Tp& __x) const { return -__x; }
};
_LIBCPP_CTAD_SUPPORTED_FOR_TYPE(negate);

// Bitwise operations

template <class _Tp>
struct _LIBCPP_TEMPLATE_VIS bit_and : __binary_function<_Tp, _Tp, _Tp> {
  typedef _Tp __result_type; // used by valarray
  _LIBCPP_HIDE_FROM_ABI _Tp operator()(const _Tp& __x, const _Tp& __y) const { return __x & __y; }
};
_LIBCPP_CTAD_SUPPORTED_FOR_TYPE(bit_and);

template <class _Tp>
struct _LIBCPP_TEMPLATE_VIS bit_or : __binary_function<_Tp, _Tp, _Tp> {
  typedef _Tp __result_type; // used by valarray
  _LIBCPP_HIDE_FROM_ABI _Tp operator()(const _Tp& __x, const _Tp& __y) const { return __x | __y; }
};
_LIBCPP_CTAD_SUPPORTED_FOR_TYPE(bit_or);

template <class _Tp = void>
struct _LIBCPP_TEMPLATE_VIS bit_xor : __binary_function<_Tp, _Tp, _Tp> {
  typedef _Tp __result_type; // used by valarray
  _LIBCPP_HIDE_FROM_ABI _Tp operator()(const _Tp& __x, const _Tp& __y) const { return __x ^ __y; }
};
_LIBCPP_CTAD_SUPPORTED_FOR_TYPE(bit_xor);

// Comparison operations

template <class _Tp>
struct _LIBCPP_TEMPLATE_VIS equal_to : __binary_function<_Tp, _Tp, bool> {
  typedef bool __result_type; // used by valarray
  _LIBCPP_HIDE_FROM_ABI bool operator()(const _Tp& __x, const _Tp& __y) const { return __x == __y; }
};
_LIBCPP_CTAD_SUPPORTED_FOR_TYPE(equal_to);

// The non-transparent std::equal_to specialization is only equivalent to a raw equality
// comparison when we don't perform an implicit conversion when calling it.
template <class _Tp>
inline const bool __desugars_to_v<__equal_tag, equal_to<_Tp>, _Tp, _Tp> = true;

// In the transparent case, we do not enforce that
template <class _Tp, class _Up>
inline const bool __desugars_to_v<__equal_tag, equal_to<void>, _Tp, _Up> = true;

template <class _Tp>
struct _LIBCPP_TEMPLATE_VIS not_equal_to : __binary_function<_Tp, _Tp, bool> {
  typedef bool __result_type; // used by valarray
  _LIBCPP_HIDE_FROM_ABI bool operator()(const _Tp& __x, const _Tp& __y) const { return __x != __y; }
};
_LIBCPP_CTAD_SUPPORTED_FOR_TYPE(not_equal_to);

template <class _Tp>
struct _LIBCPP_TEMPLATE_VIS less : __binary_function<_Tp, _Tp, bool> {
  typedef bool __result_type; // used by valarray
  _LIBCPP_HIDE_FROM_ABI bool operator()(const _Tp& __x, const _Tp& __y) const { return __x < __y; }
};
_LIBCPP_CTAD_SUPPORTED_FOR_TYPE(less);

template <class _Tp>
inline const bool __desugars_to_v<__less_tag, less<_Tp>, _Tp, _Tp> = true;

template <class _Tp>
struct _LIBCPP_TEMPLATE_VIS less_equal : __binary_function<_Tp, _Tp, bool> {
  typedef bool __result_type; // used by valarray
  _LIBCPP_HIDE_FROM_ABI bool operator()(const _Tp& __x, const _Tp& __y) const { return __x <= __y; }
};
_LIBCPP_CTAD_SUPPORTED_FOR_TYPE(less_equal);

template <class _Tp>
struct _LIBCPP_TEMPLATE_VIS greater_equal : __binary_function<_Tp, _Tp, bool> {
  typedef bool __result_type; // used by valarray
  _LIBCPP_HIDE_FROM_ABI bool operator()(const _Tp& __x, const _Tp& __y) const { return __x >= __y; }
};
_LIBCPP_CTAD_SUPPORTED_FOR_TYPE(greater_equal);

template <class _Tp>
struct _LIBCPP_TEMPLATE_VIS greater : __binary_function<_Tp, _Tp, bool> {
  typedef bool __result_type; // used by valarray
  _LIBCPP_HIDE_FROM_ABI bool operator()(const _Tp& __x, const _Tp& __y) const { return __x > __y; }
};
_LIBCPP_CTAD_SUPPORTED_FOR_TYPE(greater);

// Logical operations

template <class _Tp>
struct _LIBCPP_TEMPLATE_VIS logical_and : __binary_function<_Tp, _Tp, bool> {
  typedef bool __result_type; // used by valarray
  _LIBCPP_HIDE_FROM_ABI bool operator()(const _Tp& __x, const _Tp& __y) const { return __x && __y; }
};
_LIBCPP_CTAD_SUPPORTED_FOR_TYPE(logical_and);

template <class _Tp>
struct _LIBCPP_TEMPLATE_VIS logical_not : __unary_function<_Tp, bool> {
  typedef bool __result_type; // used by valarray
  _LIBCPP_HIDE_FROM_ABI bool operator()(const _Tp& __x) const { return !__x; }
};
_LIBCPP_CTAD_SUPPORTED_FOR_TYPE(logical_not);

template <class _Tp>
struct _LIBCPP_TEMPLATE_VIS logical_or : __binary_function<_Tp, _Tp, bool> {
  typedef bool __result_type; // used by valarray
  _LIBCPP_HIDE_FROM_ABI bool operator()(const _Tp& __x, const _Tp& __y) const { return __x || __y; }
};
_LIBCPP_CTAD_SUPPORTED_FOR_TYPE(logical_or);

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___CXX03___FUNCTIONAL_OPERATIONS_H
