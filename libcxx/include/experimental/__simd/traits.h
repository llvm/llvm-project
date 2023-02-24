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
#include <experimental/__simd/simd.h>
#include <experimental/__simd/simd_mask.h>

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

} // namespace parallelism_v2
_LIBCPP_END_NAMESPACE_EXPERIMENTAL

#endif // _LIBCPP_STD_VER >= 17 && defined(_LIBCPP_ENABLE_EXPERIMENTAL)
#endif // _LIBCPP_EXPERIMENTAL___SIMD_TRAITS_H
