// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_EXPERIMENTAL___SIMD_TRAITS_H
#define _LIBCPP_EXPERIMENTAL___SIMD_TRAITS_H

#include <experimental/__simd/abi_tag.h>
#include <experimental/__simd/aligned_tag.h>
#include <experimental/__simd/declaration.h>
#include <experimental/__simd/utility.h>

#if _LIBCPP_STD_VER >= 17 && defined(_LIBCPP_ENABLE_EXPERIMENTAL)

_LIBCPP_BEGIN_NAMESPACE_EXPERIMENTAL
inline namespace parallelism_v2 {

// traits [simd.traits]
template <class _Tp>
inline constexpr bool is_abi_tag_v = false;

template <>
inline constexpr bool is_abi_tag_v<simd_abi::__scalar> = true;

template <int _Np>
inline constexpr bool is_abi_tag_v<simd_abi::__vec_ext<_Np>> = _Np > 0 && _Np <= 32;

template <class _Tp>
struct is_abi_tag : bool_constant<is_abi_tag_v<_Tp>> {};

template <class _Tp>
inline constexpr bool is_simd_v = false;

template <class _Tp, class _Abi>
inline constexpr bool is_simd_v<simd<_Tp, _Abi>> = true;

template <class _Tp>
struct is_simd : bool_constant<is_simd_v<_Tp>> {};

template <class _Tp>
inline constexpr bool is_simd_mask_v = false;

template <class _Tp, class _Abi>
inline constexpr bool is_simd_mask_v<simd_mask<_Tp, _Abi>> = true;

template <class _Tp>
struct is_simd_mask : bool_constant<is_simd_mask_v<_Tp>> {};

template <class _Tp>
inline constexpr bool is_simd_flag_type_v = false;

template <>
inline constexpr bool is_simd_flag_type_v<element_aligned_tag> = true;

template <>
inline constexpr bool is_simd_flag_type_v<vector_aligned_tag> = true;

template <size_t _Np>
inline constexpr bool is_simd_flag_type_v<overaligned_tag<_Np>> = true;

template <class _Tp>
struct is_simd_flag_type : bool_constant<is_simd_flag_type_v<_Tp>> {};

template <class _Tp, class _Abi = simd_abi::compatible<_Tp>, bool = (__is_vectorizable_v<_Tp> && is_abi_tag_v<_Abi>)>
struct simd_size : integral_constant<size_t, _Abi::__simd_size> {};

template <class _Tp, class _Abi>
struct simd_size<_Tp, _Abi, false> {};

template <class _Tp, class _Abi = simd_abi::compatible<_Tp>>
inline constexpr size_t simd_size_v = simd_size<_Tp, _Abi>::value;

template <class _Tp,
          class _Up = typename _Tp::value_type,
          bool      = (is_simd_v<_Tp> && __is_vectorizable_v<_Up>) || (is_simd_mask_v<_Tp> && is_same_v<_Up, bool>)>
struct memory_alignment : integral_constant<size_t, vector_aligned_tag::__alignment<_Tp, _Up>> {};

template <class _Tp, class _Up>
struct memory_alignment<_Tp, _Up, false> {};

template <class _Tp, class _Up = typename _Tp::value_type>
inline constexpr size_t memory_alignment_v = memory_alignment<_Tp, _Up>::value;

} // namespace parallelism_v2
_LIBCPP_END_NAMESPACE_EXPERIMENTAL

#endif // _LIBCPP_STD_VER >= 17 && defined(_LIBCPP_ENABLE_EXPERIMENTAL)
#endif // _LIBCPP_EXPERIMENTAL___SIMD_TRAITS_H
