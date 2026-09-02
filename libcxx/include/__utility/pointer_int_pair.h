//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___UTILITY_POINTER_INT_PAIR_H
#define _LIBCPP___UTILITY_POINTER_INT_PAIR_H

#include <__assert>
#include <__config>
#include <__cstddef/size_t.h>
#include <__fwd/tuple.h>
#include <__type_traits/enable_if.h>
#include <__type_traits/is_integral.h>
#include <__type_traits/is_unsigned.h>
#include <__type_traits/is_void.h>
#include <cstdint>
#include <limits>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

// A __pointer_int_pair is a pair of a pointer and an integral type. The lower bits of the pointer that are free
// due to the alignment requirement of the pointee are used to store the integral type.
//
// This imposes a constraint on the number of bits available for the integral type -- the integral type can use
// at most log2(alignof(T)) bits. This technique allows storing the integral type without additional storage
// beyond that of the pointer itself, at the cost of some bit twiddling.

_LIBCPP_BEGIN_NAMESPACE_STD

template <class, class = void>
struct _PointerLikeTraits;

template <class _Tp>
struct _PointerLikeTraits<_Tp*, __enable_if_t<!is_void<_Tp>::value> > {
  // This is really `__bit_log2`, but we need this to be a constant expression even in C++03, so we can't use that
  static const size_t __low_bits_available = numeric_limits<size_t>::digits - 1 - __builtin_clzg(_LIBCPP_ALIGNOF(_Tp));

  static _LIBCPP_HIDE_FROM_ABI uintptr_t __to_uintptr(_Tp* __ptr) { return reinterpret_cast<uintptr_t>(__ptr); }
  static _LIBCPP_HIDE_FROM_ABI _Tp* __to_pointer(uintptr_t __ptr) { return reinterpret_cast<_Tp*>(__ptr); }
};

template <class _Tp>
struct _PointerLikeTraits<_Tp*, __enable_if_t<is_void<_Tp>::value> > {
  static const size_t __low_bits_available = 0;

  static _LIBCPP_HIDE_FROM_ABI uintptr_t __to_uintptr(_Tp* __ptr) { return reinterpret_cast<uintptr_t>(__ptr); }
  static _LIBCPP_HIDE_FROM_ABI _Tp* __to_pointer(uintptr_t __ptr) { return reinterpret_cast<_Tp*>(__ptr); }
};

enum __integer_width : size_t {};

template <class _Pointer, class _IntType, __integer_width __int_bit_count>
class __pointer_int_pair {
  using _PointerTraits = _PointerLikeTraits<_Pointer>;

  static const auto __int_width = static_cast<size_t>(__int_bit_count);

  static_assert(__int_width <= _PointerTraits::__low_bits_available,
                "Not enough bits available for requested bit count");
  static_assert(is_integral<_IntType>::value, "_IntType has to be an integral type");
  static_assert(is_unsigned<_IntType>::value,
                "__pointer_int_pair doesn't work for signed types since that would require handling the sign bit");

  static const size_t __extra_bits  = _PointerTraits::__low_bits_available - __int_width;
  static const uintptr_t __int_mask = static_cast<uintptr_t>(1 << _PointerTraits::__low_bits_available) - 1;
  static const uintptr_t __ptr_mask = ~__int_mask;

  uintptr_t __value_ = 0;

public:
  __pointer_int_pair() = default;

  _LIBCPP_HIDE_FROM_ABI __pointer_int_pair(_Pointer __ptr_value, _IntType __int_value)
      : __value_(_PointerTraits::__to_uintptr(__ptr_value) | (__int_value << __extra_bits)) {
    _LIBCPP_ASSERT_INTERNAL((__int_value & (__int_mask >> __extra_bits)) == __int_value, "integer is too large!");
    _LIBCPP_ASSERT_INTERNAL(
        (_PointerTraits::__to_uintptr(__ptr_value) & __ptr_mask) == _PointerTraits::__to_uintptr(__ptr_value),
        "Pointer alignment is too low!");
  }

  _LIBCPP_HIDE_FROM_ABI _IntType __get_value() const { return (__value_ & __int_mask) >> __extra_bits; }
  _LIBCPP_HIDE_FROM_ABI _Pointer __get_ptr() const { return _PointerTraits::__to_pointer(__value_ & __ptr_mask); }

  template <class, class>
  friend struct _PointerLikeTraits;
};

template <class _Pointer, __integer_width __int_bit_count, class _IntType>
struct _PointerLikeTraits<__pointer_int_pair<_Pointer, _IntType, __int_bit_count> > {
private:
  using _PointerIntPair = __pointer_int_pair<_Pointer, _IntType, __int_bit_count>;

public:
  static inline const size_t __low_bits_available = _PointerIntPair::__extra_bits;

  static _LIBCPP_HIDE_FROM_ABI uintptr_t __to_uintptr(_PointerIntPair __ptr) { return __ptr.__value_; }

  static _LIBCPP_HIDE_FROM_ABI _PointerIntPair __to_pointer(uintptr_t __ptr) {
    _PointerIntPair __tmp;
    __tmp.__value_ = __ptr;
    return __tmp;
  }
};

#ifndef _LIBCPP_CXX03_LANG

// Make __pointer_int_pair tuple-like

template <class _Pointer, class _IntType, __integer_width __int_bit_count>
struct tuple_size<__pointer_int_pair<_Pointer, _IntType, __int_bit_count> > : integral_constant<size_t, 2> {};

template <class _Pointer, class _IntType, __integer_width __int_bit_count>
struct tuple_element<0, __pointer_int_pair<_Pointer, _IntType, __int_bit_count> > {
  using type = _Pointer;
};

template <class _Pointer, class _IntType, __integer_width __int_bit_count>
struct tuple_element<1, __pointer_int_pair<_Pointer, _IntType, __int_bit_count> > {
  using type = _IntType;
};

template <size_t __i>
struct __pointer_int_pair_getter;

template <>
struct __pointer_int_pair_getter<0> {
  template <class _Pointer, class _IntType, __integer_width __int_bit_count>
  static _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR _Pointer
  __get(__pointer_int_pair<_Pointer, _IntType, __int_bit_count> __pair) {
    return __pair.__get_ptr();
  }
};

template <>
struct __pointer_int_pair_getter<1> {
  template <class _Pointer, class _IntType, __integer_width __int_bit_count>
  static _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR _IntType
  __get(__pointer_int_pair<_Pointer, _IntType, __int_bit_count> __pair) {
    return __pair.__get_value();
  }
};

template <size_t __i, class _Pointer, class _IntType, __integer_width __int_bit_count>
_LIBCPP_HIDE_FROM_ABI typename tuple_element<__i, __pointer_int_pair<_Pointer, _IntType, __int_bit_count> >::type
get(__pointer_int_pair<_Pointer, _IntType, __int_bit_count> __pair) {
  return __pointer_int_pair_getter<__i>::__get(__pair);
}

#endif // _LIBCPP_CXX03_LANG

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___UTILITY_POINTER_INT_PAIR_H
