//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___SIMD_ABI_H
#define _LIBCPP___SIMD_ABI_H

#include <__concepts/convertible_to.h>
#include <__concepts/equality_comparable.h>
#include <__config>
#include <__cstddef/size_t.h>
#include <__type_traits/standard_types.h>
#include <__utility/integer_sequence.h>
#include <cstdint>

#if _LIBCPP_STD_VER >= 26

_LIBCPP_BEGIN_NAMESPACE_STD
namespace datapar {

template <class _Tp>
inline constexpr bool __is_vectorizable_type_v = __is_standard_integer_type_v<_Tp> || __is_character_type_v<_Tp>;

template <>
inline constexpr bool __is_vectorizable_type_v<float> = true;

template <>
inline constexpr bool __is_vectorizable_type_v<double> = true;

template <class _From, class _To>
concept __value_preserving_convertible = requires(_From __from) { _To{__from}; };

template <class _Tp>
concept __constexpr_wrapper_like =
    convertible_to<_Tp, decltype(_Tp::value)> && equality_comparable_with<_Tp, decltype(_Tp::value)> &&
    bool_constant<_Tp() == _Tp::value>::value &&
    bool_constant<static_cast<decltype(_Tp::value)>(_Tp()) == _Tp::value>::value;

// [simd.expos]
using __simd_size_type = int;

template <class _Tp>
struct __deduce_abi;

template <class _Tp, __simd_size_type _Np>
  requires __is_vectorizable_type_v<_Tp> && (_Np <= 64)
using __deduce_abi_t = __deduce_abi<_Tp>::template __apply<_Np>;

template <class _Tp>
using __native_abi = __deduce_abi<_Tp>::template __apply<4>;

template <class _Tp, class _Abi>
inline constexpr __simd_size_type __simd_size_v = 0;

template <size_t>
struct __integer_from_impl;

template <>
struct __integer_from_impl<1> {
  using type = uint8_t;
};

template <>
struct __integer_from_impl<2> {
  using type = uint16_t;
};

template <>
struct __integer_from_impl<4> {
  using type = uint32_t;
};

template <>
struct __integer_from_impl<8> {
  using type = uint64_t;
};

template <size_t _Bytes>
using __integer_from = __integer_from_impl<_Bytes>::type;

// ABI Types

template <class _Tp, __simd_size_type _Np>
struct __vector_size_abi {
  using _SimdT [[__gnu__::__vector_size__(_Np * sizeof(_Tp))]] = _Tp;
  using _MaskT [[__gnu__::__vector_size__(_Np * sizeof(_Tp))]] = __integer_from<sizeof(_Tp)>;

  _LIBCPP_ALWAYS_INLINE constexpr _SimdT __select(_MaskT __mask, _SimdT __true, _SimdT __false) {
    return __mask ? __true : __false;
  }

#  ifdef _LIBCPP_COMPILER_CLANG_BASED
  using _BoolVec __attribute__((__ext_vector_type__(_Np))) = bool;

  static constexpr auto __int_size = _Np <= 8 ? 8 : _Np <= 16 ? 16 : _Np <= 32 ? 32 : 64;
  static_assert(__int_size >= _Np);

  using _IntSizeBoolVec __attribute__((__ext_vector_type__(__int_size))) = bool;

  _LIBCPP_ALWAYS_INLINE static constexpr auto __mask_to_int(_BoolVec __mask) noexcept {
    return [&]<size_t... _Origs, size_t... _Fillers>(index_sequence<_Origs...>, index_sequence<_Fillers...>)
               _LIBCPP_ALWAYS_INLINE {
                 auto __vec = __builtin_convertvector(
                     __builtin_shufflevector(__mask, _BoolVec{}, _Origs..., ((void)_Fillers, _Np)...), _IntSizeBoolVec);
                 if constexpr (_Np <= 8)
                   return __builtin_bit_cast(unsigned char, __vec);
                 else if constexpr (_Np <= 16)
                   return __builtin_bit_cast(unsigned short, __vec);
                 else if constexpr (_Np <= 32)
                   return __builtin_bit_cast(unsigned int, __vec);
                 else
                   return __builtin_bit_cast(unsigned long long, __vec);
               }(make_index_sequence<_Np>{}, make_index_sequence<__int_size - _Np>{});
  }

  _LIBCPP_ALWAYS_INLINE static constexpr bool __any_of(_MaskT __mask) noexcept {
    return __builtin_reduce_or(__builtin_convertvector(__mask, _BoolVec));
  }

  _LIBCPP_ALWAYS_INLINE static constexpr bool __all_of(_MaskT __mask) noexcept {
    return __builtin_reduce_and(__builtin_convertvector(__mask, _BoolVec));
  }

  _LIBCPP_ALWAYS_INLINE static constexpr __simd_size_type __reduce_count(_MaskT __mask) noexcept {
    return __builtin_reduce_add(__builtin_convertvector(__builtin_convertvector(__mask, _BoolVec), _MaskT));
  }

  _LIBCPP_ALWAYS_INLINE static constexpr __simd_size_type __reduce_min_index(_MaskT __mask) noexcept {
    return __builtin_ctzg(__mask_to_int(__builtin_convertvector(__mask, _BoolVec)));
  }

  _LIBCPP_ALWAYS_INLINE static constexpr __simd_size_type __reduce_max_index(_MaskT __mask) noexcept {
    return __int_size - 1 - __builtin_clzg(__mask_to_int(__builtin_convertvector(__mask, _BoolVec)));
  }
#  else
  _LIBCPP_ALWAYS_INLINE constexpr bool __any_of(_MaskT __mask) noexcept {
    for (size_t __i = 0; __i != _Np; ++__i) {
      if (__mask[__i])
        return true;
    }
    return false;
  }
#  endif
};

template <class _Tp>
  requires __is_vectorizable_type_v<_Tp>
struct __deduce_abi<_Tp> {
  template <__simd_size_type _Np>
  using __apply = __vector_size_abi<_Tp, _Np>;
};

template <class _Tp, __simd_size_type _Np>
inline constexpr __simd_size_type __simd_size_v<_Tp, __vector_size_abi<_Tp, _Np>> = _Np;

} // namespace datapar
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 26

#endif // _LIBCPP___SIMD_ABI_H
