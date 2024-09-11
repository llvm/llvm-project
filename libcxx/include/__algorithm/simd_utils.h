//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_SIMD_UTILS_H
#define _LIBCPP___ALGORITHM_SIMD_UTILS_H

#include <__algorithm/min.h>
#include <__bit/bit_cast.h>
#include <__bit/countl.h>
#include <__bit/countr.h>
#include <__config>
#include <__type_traits/is_arithmetic.h>
#include <__type_traits/is_same.h>
#include <__utility/integer_sequence.h>
#include <cstddef>
#include <cstdint>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

// TODO: Find out how altivec changes things and allow vectorizations there too.
#if _LIBCPP_STD_VER >= 14 && defined(_LIBCPP_CLANG_VER) && !defined(__ALTIVEC__)
#  define _LIBCPP_HAS_ALGORITHM_VECTOR_UTILS 1
#else
#  define _LIBCPP_HAS_ALGORITHM_VECTOR_UTILS 0
#endif

#if _LIBCPP_HAS_ALGORITHM_VECTOR_UTILS && !defined(__OPTIMIZE_SIZE__)
#  define _LIBCPP_VECTORIZE_ALGORITHMS 1
#else
#  define _LIBCPP_VECTORIZE_ALGORITHMS 0
#endif

#if _LIBCPP_HAS_ALGORITHM_VECTOR_UTILS

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Tp>
inline constexpr bool __can_map_to_integer_v =
    sizeof(_Tp) == alignof(_Tp) && (sizeof(_Tp) == 1 || sizeof(_Tp) == 2 || sizeof(_Tp) == 4 || sizeof(_Tp) == 8);

template <size_t _TypeSize>
struct __get_as_integer_type_impl;

template <>
struct __get_as_integer_type_impl<1> {
  using type = uint8_t;
};

template <>
struct __get_as_integer_type_impl<2> {
  using type = uint16_t;
};
template <>
struct __get_as_integer_type_impl<4> {
  using type = uint32_t;
};
template <>
struct __get_as_integer_type_impl<8> {
  using type = uint64_t;
};

template <class _Tp>
using __get_as_integer_type_t = typename __get_as_integer_type_impl<sizeof(_Tp)>::type;

// This isn't specialized for 64 byte vectors on purpose. They have the potential to significantly reduce performance
// in mixed simd/non-simd workloads and don't provide any performance improvement for currently vectorized algorithms
// as far as benchmarks are concerned.
#  if defined(__AVX__) || defined(__MVS__)
template <class _Tp>
inline constexpr size_t __native_vector_size = 32 / sizeof(_Tp);
#  elif defined(__SSE__) || defined(__ARM_NEON__)
template <class _Tp>
inline constexpr size_t __native_vector_size = 16 / sizeof(_Tp);
#  elif defined(__MMX__)
template <class _Tp>
inline constexpr size_t __native_vector_size = 8 / sizeof(_Tp);
#  else
template <class _Tp>
inline constexpr size_t __native_vector_size = 1;
#  endif

template <class _ArithmeticT, size_t _Np>
using __simd_vector __attribute__((__ext_vector_type__(_Np))) = _ArithmeticT;

template <class _VecT>
inline constexpr size_t __simd_vector_size_v = []<bool _False = false>() -> size_t {
  static_assert(_False, "Not a vector!");
}();

template <class _Tp, size_t _Np>
inline constexpr size_t __simd_vector_size_v<__simd_vector<_Tp, _Np>> = _Np;

template <class _Tp, size_t _Np>
_LIBCPP_HIDE_FROM_ABI _Tp __simd_vector_underlying_type_impl(__simd_vector<_Tp, _Np>) {
  return _Tp{};
}

template <class _VecT>
using __simd_vector_underlying_type_t = decltype(std::__simd_vector_underlying_type_impl(_VecT{}));

template <class _VecT>
_LIBCPP_NODISCARD _LIBCPP_HIDE_FROM_ABI _VecT __broadcast(__simd_vector_underlying_type_t<_VecT> __value) noexcept {
  return [=]<size_t... _Indices>(index_sequence<_Indices...>) _LIBCPP_ALWAYS_INLINE noexcept {
    return _VecT{((void)_Indices, __value)...};
  }(make_index_sequence<__simd_vector_size_v<_VecT>>{});
}

// This isn't inlined without always_inline when loading chars.
template <class _VecT, class _Iter>
_LIBCPP_NODISCARD _LIBCPP_ALWAYS_INLINE _LIBCPP_HIDE_FROM_ABI _VecT __load_vector(_Iter __iter) noexcept {
  return [=]<size_t... _Indices>(index_sequence<_Indices...>) _LIBCPP_ALWAYS_INLINE noexcept {
    return _VecT{__iter[_Indices]...};
  }(make_index_sequence<__simd_vector_size_v<_VecT>>{});
}

template <size_t _Np>
_LIBCPP_NODISCARD _LIBCPP_HIDE_FROM_ABI auto __extend_vector(__simd_vector<bool, _Np> __vec) noexcept {
  using _VecT = __simd_vector<bool, _Np>;
  static_assert(_Np <= 8, "Unexpected vector size");
  if constexpr (_Np >= 4) {
    return __builtin_shufflevector(__vec, _VecT{}, 0, 1, 2, 3, 4, 5, 6, 7);
  } else if constexpr (_Np >= 2) {
    return std::__extend_vector(__builtin_shufflevector(__vec, _VecT{}, 0, 1, 2, 3));
  } else {
    return std::__extend_vector(__builtin_shufflevector(__vec, _VecT{}, 0, 1));
  }
}

template <class _Tp, size_t _Np>
_LIBCPP_NODISCARD _LIBCPP_HIDE_FROM_ABI auto __to_int_mask(__simd_vector<_Tp, _Np> __in_vec) noexcept {
  auto __vec = __builtin_convertvector(__in_vec, __simd_vector<bool, _Np>);
  if constexpr (_Np <= 8) {
    return std::__bit_cast<uint8_t>(std::__extend_vector(__vec));
  } else if constexpr (_Np <= 16) {
    return std::__bit_cast<uint16_t>(__vec);
  } else if constexpr (_Np <= 32) {
    return std::__bit_cast<uint32_t>(__vec);
  } else if constexpr (_Np <= 64) {
    return std::__bit_cast<uint64_t>(__vec);
  } else {
    static_assert(sizeof(__simd_vector<bool, _Np>) == 0, "unexpected vector size");
  }
}

template <size_t _Np>
__simd_vector<bool, _Np> __popcount(__simd_vector<bool, _Np> __vec) {
  return std::__popcount(std::__to_int_mask(__vec));
}

template <class _Vector, class = void>
inline constexpr bool __has_compressstore = false;

void __compressstore() = delete;

#  if defined(__BMI2__) && __has_builtin(__builtin_ia32_bzhi_di) && defined(__AVX512VL__)

// 8 bit elements
#    if defined(__AVX512BW__) && defined(__AVX512VBMI2__)
#      if __has_builtin(__builtin_ia32_compressqi256_mask) && __has_builtin(__builtin_ia32_storedquqi256_mask)
template <class _Tp>
inline constexpr bool __has_compressstore<__simd_vector<_Tp, 32>, __enable_if_t<sizeof(_Tp) == 1>> = true;

template <class _Tp, __enable_if_t<sizeof(_Tp) == 1, int> = 0>
void __compressstore(_Tp* __dest, __simd_vector<_Tp, 32> __vec, __simd_vector<_Tp, 32> __mask) {
  auto __storemask = __builtin_ia32_bzhi_di(-1, std::__popcount(std::__to_int_mask(__mask)));
  __builtin_ia32_storedquqi256_mask(
      (__simd_vector<char, 32>*)__dest,
      __builtin_ia32_compressqi256_mask(__vec, {}, std::__to_int_mask(__mask)),
      __storemask);
}
#      endif // __has_builtin(__builtin_ia32_compressqi256_mask) && __has_builtin(__builtin_ia32_storedquqi256_mask)

// 16 bit elements
#      if __has_builtin(__builtin_ia32_compresshi256_mask) && __has_builtin(__builtin_ia32_storedquhi256_mask)
template <class _Tp>
inline constexpr bool __has_compressstore<__simd_vector<_Tp, 16>, __enable_if_t<sizeof(_Tp) == 2>> = true;

template <class _Tp, __enable_if_t<sizeof(_Tp) == 2, int> = 0>
void __compressstore(_Tp* __dest, __simd_vector<_Tp, 16> __vec, __simd_vector<_Tp, 16> __mask) {
  auto __storemask = __builtin_ia32_bzhi_di(-1, std::__popcount(std::__to_int_mask(__mask)));
  __builtin_ia32_storedquhi256_mask(
      (__simd_vector<char, 32>*)__dest,
      __builtin_ia32_compresshi256_mask(__vec, {}, std::__to_int_mask(__mask)),
      __storemask);
}
#      endif // __has_builtin(__builtin_ia32_compresshi256_mask) && __has_builtin(__builtin_ia32_storedquhi256_mask)
#    endif   // defined(__AVX512BW__) && defined(__AVX512VBMI2__)

// 32 bit elements
#    if __has_builtin(__builtin_ia32_compresssi256_mask) && __has_builtin(__builtin_ia32_movdqa32store256_mask)
template <class _Tp>
inline constexpr bool __has_compressstore<__simd_vector<_Tp, 8>, __enable_if_t<sizeof(_Tp) == 4>> = true;

template <class _Tp, __enable_if_t<sizeof(_Tp) == 4, int> = 0>
void __compressstore(_Tp* __dest, __simd_vector<_Tp, 8> __vec, __simd_vector<_Tp, 8> __mask) {
  auto __storemask = __builtin_ia32_bzhi_di(-1, std::__popcount(std::__to_int_mask(__mask)));
  __builtin_ia32_movdqa32store256_mask(
      (__simd_vector<char, 32>*)__dest,
      __builtin_ia32_compresssi256_mask(__vec, {}, std::__to_int_mask(__mask)),
      __storemask);
}
#    endif // __has_builtin(__builtin_ia32_compresssi256_mask) && __has_builtin(__builtin_ia32_movdqa32store256_mask)

// 64 bit elements
#    if __has_builtin(__builtin_ia32_compresssi256_mask) && __has_builtin(__builtin_ia32_movdqa64store256_mask)
template <class _Tp>
inline constexpr bool __has_compressstore<__simd_vector<_Tp, 4>, __enable_if_t<sizeof(_Tp) == 8>> = true;

template <class _Tp, __enable_if_t<sizeof(_Tp) == 8, int> = 0>
void __compressstore(_Tp* __dest, __simd_vector<_Tp, 4> __vec, __simd_vector<_Tp, 4> __mask) {
  auto __storemask = __builtin_ia32_bzhi_di(-1, std::__popcount(std::__to_int_mask(__mask)));
  __builtin_ia32_movdqa64store256_mask(
      (__simd_vector<char, 32>*)__dest,
      __builtin_ia32_compresssi256_mask(__vec, {}, std::__to_int_mask(__mask)),
      __storemask);
}
#    endif // __has_builtin(__builtin_ia32_compresssi256_mask) && __has_builtin(__builtin_ia32_movdqa64store256_mask)

#  endif

template <class _Tp, size_t _Np>
_LIBCPP_NODISCARD _LIBCPP_HIDE_FROM_ABI bool __all_of(__simd_vector<_Tp, _Np> __vec) noexcept {
  return __builtin_reduce_and(__builtin_convertvector(__vec, __simd_vector<bool, _Np>));
}

template <class _Tp, size_t _Np>
_LIBCPP_NODISCARD _LIBCPP_HIDE_FROM_ABI size_t __find_first_set(__simd_vector<_Tp, _Np> __vec) noexcept {
#  if defined(_LIBCPP_BIG_ENDIAN)
  return std::min<size_t>(_Np, std::__countl_zero(std::__to_int_mask(__vec)));
#  else
  return std::min<size_t>(_Np, std::__countr_zero(std::__to_int_mask(__vec)));
#  endif
}

template <class _Tp, size_t _Np>
_LIBCPP_NODISCARD _LIBCPP_HIDE_FROM_ABI size_t __find_first_not_set(__simd_vector<_Tp, _Np> __vec) noexcept {
  return std::__find_first_set(~__vec);
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_HAS_ALGORITHM_VECTOR_UTILS

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ALGORITHM_SIMD_UTILS_H
